"""AutoAce trial dashboard: login, batch upload, progress, results, download.

Run locally:
    uvicorn app.main:app --port 8000
Credentials and keys come from the environment (.env in development).
"""

from __future__ import annotations

import csv
import io
import json
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: E402
from starlette.datastructures import UploadFile  # noqa: E402
from fastapi.responses import (FileResponse, JSONResponse, RedirectResponse,  # noqa: E402
                               Response)
from fastapi.staticfiles import StaticFiles  # noqa: E402
from itsdangerous import BadSignature, URLSafeTimedSerializer  # noqa: E402

from . import batches, db, worker  # noqa: E402

def _session_secret() -> str:
    """A secret that survives restarts.

    Generating one per process signs every deploy's cookies with a different
    key, so users are silently logged out on each restart (and randomly, if
    more than one replica serves traffic). SESSION_SECRET is the correct
    answer; falling back to a file under DATA_DIR keeps sessions alive when
    it is unset, provided the data volume is persistent and shared.
    """
    if env := os.environ.get("SESSION_SECRET"):
        return env
    path = Path(os.environ.get("DATA_DIR", "data")) / ".session_secret"
    try:
        if path.exists():
            return path.read_text().strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        value = secrets.token_hex(32)
        path.write_text(value)
        print("[auth] SESSION_SECRET unset; persisted a generated secret to "
              f"{path}. Set SESSION_SECRET to survive volume loss and to keep "
              "sessions valid across replicas.", flush=True)
        return value
    except OSError as e:
        print(f"[auth] WARNING: cannot persist a session secret ({e}); "
              "sessions will not survive a restart.", flush=True)
        return secrets.token_hex(32)


def _flag(name: str) -> bool:
    """Env flags: only 1/true/yes enable. bool('0') is True in Python."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


SECRET = _session_secret()
USERNAME = os.environ.get("DASH_USERNAME", "autoace")
PASSWORD = os.environ.get("DASH_PASSWORD", "")
SESSION_HOURS = 24 * 7

signer = URLSafeTimedSerializer(SECRET, salt="session")
app = FastAPI(title="AutoAce Voice Analysis", docs_url=None, redoc_url=None)
STATIC = Path(__file__).parent / "static"


@app.on_event("startup")
def _startup() -> None:
    db.con()
    worker.start(workers=int(os.environ.get("WORKERS", "1")))


def _session_user(request: Request) -> str | None:
    cookie = request.cookies.get("session")
    if not cookie:
        return None
    try:
        return signer.loads(cookie, max_age=SESSION_HOURS * 3600)["u"]
    except (BadSignature, Exception):
        return None


def require_auth(request: Request) -> str:
    user = _session_user(request)
    if not user:
        raise HTTPException(401, "not authenticated")
    return user


# ---------------------------------------------------------------- auth & pages

@app.post("/api/login")
async def login(request: Request):
    body = await request.json()
    ok = (secrets.compare_digest(body.get("username", ""), USERNAME)
          and PASSWORD and secrets.compare_digest(body.get("password", ""), PASSWORD))
    if not ok:
        raise HTTPException(401, "invalid credentials")
    resp = JSONResponse({"ok": True})
    resp.set_cookie("session", signer.dumps({"u": USERNAME}), httponly=True,
                    samesite="lax", max_age=SESSION_HOURS * 3600,
                    secure=_flag("COOKIE_SECURE"))
    return resp


@app.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("session")
    return resp


@app.get("/")
def index(request: Request):
    if not _session_user(request):
        return RedirectResponse("/login")
    return FileResponse(STATIC / "index.html")


@app.get("/login")
def login_page():
    return FileResponse(STATIC / "login.html")


# ------------------------------------------------------------------- batches

@app.post("/api/batches")
async def upload_batch(request: Request, user: str = Depends(require_auth)):
    form = await request.form()
    uploads = [v for v in form.getlist("files") if isinstance(v, UploadFile)]
    if not uploads:
        raise HTTPException(400, "no files in upload")
    pairs = [(u.filename or "file", await u.read()) for u in uploads]
    name = str(form.get("name") or "")
    try:
        batch_id = batches.create_batch_from_upload(pairs, name)
    except Exception as e:  # zip bombs, malformed archives
        raise HTTPException(400, f"could not read upload: {e}") from e
    return {"batch_id": batch_id}


@app.get("/api/batches")
def get_batches(user: str = Depends(require_auth)):
    return db.list_batches()


@app.get("/api/batches/{batch_id}")
def get_batch(batch_id: str, user: str = Depends(require_auth)):
    b = db.batch_summary(batch_id)
    if not b:
        raise HTTPException(404)
    files = db.batch_files(batch_id)
    for f in files:
        f["result"] = json.loads(f["result_json"]) if f["result_json"] else None
        f["expected"] = json.loads(f["expected_json"]) if f["expected_json"] else None
        del f["result_json"], f["expected_json"]
    return {**b, "files": files, "worker_ready": worker.is_ready()}


@app.get("/api/files/{file_id}")
def get_file(file_id: int, user: str = Depends(require_auth)):
    f = db.file_detail(file_id)
    if not f:
        raise HTTPException(404)
    for k in ("result_json", "detail_json", "expected_json"):
        f[k.replace("_json", "")] = json.loads(f[k]) if f[k] else None
        del f[k]
    f.pop("path", None)
    return f


@app.get("/api/batches/{batch_id}/download.{fmt}")
def download(batch_id: str, fmt: str, user: str = Depends(require_auth)):
    if fmt not in ("csv", "json"):
        raise HTTPException(404)
    files = db.batch_files(batch_id)
    if fmt == "json":
        payload = [{"name": f["filename"],
                    "result_json": json.loads(f["result_json"]) if f["result_json"] else None,
                    "status": f["status"], "error": f["error"]} for f in files]
        return Response(json.dumps(payload, indent=1), media_type="application/json",
                        headers={"Content-Disposition":
                                 f'attachment; filename="results_{batch_id}.json"'})
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["name", "result_json"])
    for f in files:
        w.writerow([f["filename"], f["result_json"] or f'ERROR: {f["error"] or "unprocessed"}'])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition":
                             f'attachment; filename="results_{batch_id}.csv"'})


app.mount("/static", StaticFiles(directory=STATIC), name="static")

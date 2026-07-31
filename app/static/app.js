/* Dashboard: upload with progress, live batch view with FLIP reordering,
   boot-status banner, unified download menu. Rows are persistent DOM nodes
   updated in place so reorders can animate. */

const $ = (s, el = document) => el.querySelector(s);
let currentBatch = null;
let pollTimer = null;
const openDetails = new Set();
const rowCache = new Map(); // file id -> row element

const STATUS_RANK = { processing: 0, queued: 1, failed: 2, done: 3 };
const QUAL_TONE = { clear: 'green', slightly_impaired: 'amber', severely_impaired: 'red' };
const TONE_TONE = { neutral: 'plain', satisfied: 'green', frustrated: 'amber', upset: 'red', distressed: 'red' };

function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.remove('show'), 3800);
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (r.status === 401) { location.href = '/login'; throw new Error('auth'); }
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

/* ---------------- row rendering ---------------- */

const badge = (text, tone = '', dot = true) =>
  `<span class="badge ${tone}">${dot ? '<i></i>' : ''}${text}</span>`;

function meterCell(label, level, max) {
  const dots = Array.from({ length: max }, (_, i) =>
    `<span class="badge plain" style="padding:2px 6px;border:none;background:none;opacity:${i < level ? 1 : 0.25}">●</span>`).join('');
  return `<span class="cell-text" title="${label}">${label}</span>`;
}

const CHEV = '<svg class="chev" width="9" height="9" viewBox="0 0 10 10" fill="none" aria-hidden="true"><path d="M3.5 1 7.5 5 3.5 9" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';

const detailSlot = (id, open) =>
  `<div class="detail-wrap${open ? ' open' : ''}" data-detail="${id}">
     <div class="detail-inner"></div>
   </div>`;

function rowContent(f) {
  const r = f.result;
  const open = openDetails.has(f.id);
  const status = `<span class="status ${f.status}">${f.status === 'processing'
    ? '<span class="spinner"></span>' : '<i></i>'}${f.status}</span>`;
  if (!r) {
    return `<button class="strip" data-id="${f.id}" aria-expanded="${open}">
      <span class="fname">${CHEV}<span>${f.filename}</span></span>
      <span></span><span></span><span></span><span></span><span></span><span></span>
      ${status}
    </button>
    ${f.error ? `<div class="file-error">${f.error}</div>` : ''}
    ${detailSlot(f.id, open)}`;
  }
  const noise = r.background_noise_present
    ? badge(`${r.background_noise_type || 'noise'} · ${r.background_noise_severity}`,
            r.background_noise_severity === 'high' ? 'red'
            : r.background_noise_severity === 'medium' ? 'amber' : 'plain')
    : '<span class="cell-text">no noise</span>';
  const flags = [
    r.speaker_overlap_present ? badge('overlap', 'blue') : '',
    r.long_silence_present ? badge('silence', 'blue') : '',
  ].join('') || '<span class="cell-text">—</span>';
  return `<button class="strip" data-id="${f.id}" aria-expanded="${open}">
    <span class="fname">${CHEV}<span>${f.filename}</span></span>
    ${badge(r.emotional_tone, TONE_TONE[r.emotional_tone] || 'plain')}
    <span class="cell-text">${r.emotional_intensity}</span>
    ${noise}
    ${badge(r.audio_quality.replace(/_/g, ' '), QUAL_TONE[r.audio_quality] || 'plain')}
    <span class="flags">${flags}</span>
    <span class="conf">${(r.confidence ?? 0).toFixed(2)}</span>
    ${status}
  </button>
  ${detailSlot(f.id, open)}`;
}

function ensureRow(f) {
  let row = rowCache.get(f.id);
  const signature = JSON.stringify([f.status, f.result, f.error]);
  if (!row) {
    row = document.createElement('div');
    row.className = 'file-row';
    row.dataset.fid = f.id;
    rowCache.set(f.id, row);
  }
  if (row._sig !== signature) {
    // status/result changed, so any cached detail is stale
    if (row._sig !== undefined) detailCache.delete(f.id);
    row._sig = signature;
    row.innerHTML = rowContent(f);
    const strip = row.querySelector('.strip');
    strip.addEventListener('click', () => toggleDetail(f.id));
    if (f.status === 'done' || f.status === 'failed') {
      strip.addEventListener('mouseenter', () => prefetchDetail(f.id));
      strip.addEventListener('focus', () => prefetchDetail(f.id));
    }
    if (openDetails.has(f.id)) {
      restoreDetail(f.id, row.querySelector(`[data-detail="${f.id}"] .detail-inner`));
    }
  }
  return row;
}

/* FLIP: capture positions, reorder DOM, animate the delta. */
function reorderRows(container, orderedRows) {
  const before = new Map();
  orderedRows.forEach(el => { if (el.parentNode) before.set(el, el.getBoundingClientRect().top); });
  orderedRows.forEach(el => container.appendChild(el));
  orderedRows.forEach(el => {
    const prev = before.get(el);
    if (prev == null) return;
    const dy = prev - el.getBoundingClientRect().top;
    if (Math.abs(dy) > 2) {
      el.animate(
        [{ transform: `translateY(${dy}px)` }, { transform: 'translateY(0)' }],
        { duration: 320, easing: 'cubic-bezier(0.2, 0.7, 0.3, 1)' });
    }
  });
}

/* ---------------- batch view ---------------- */

function batchShell(b) {
  return `
    <div class="batch-head">
      <h2>${b.name}</h2>
      <span class="label" id="bCount"></span>
      <span class="spacer"></span>
      <div class="menu-wrap">
        <button class="btn" id="dlBtn" aria-haspopup="menu">
          <svg width="13" height="13" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <path d="M7 1.75v6.5m0 0L4.5 5.75M7 8.25l2.5-2.5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M2 10v1.5a.75.75 0 0 0 .75.75h8.5a.75.75 0 0 0 .75-.75V10" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
          </svg>
          Download
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true"><path d="M1 3.5 5 7.5 9 3.5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>
        </button>
        <div class="menu" id="dlMenu" role="menu">
          <button data-fmt="csv" role="menuitem">CSV<span>name,result_json — same shape as the manifest</span></button>
          <button data-fmt="json" role="menuitem">JSON<span>full results with per-file status</span></button>
        </div>
      </div>
    </div>
    <div class="batch-meta" id="bMeta"></div>
    <div class="banner" id="bBanner" hidden></div>
    <ul class="warnings" id="bWarnings"></ul>
    <div class="files">
      <div class="thead">
        <span class="label">File</span><span class="label">Tone</span>
        <span class="label">Intensity</span><span class="label">Noise</span>
        <span class="label">Quality</span><span class="label">Flags</span>
        <span class="label">Conf</span><span class="label">Status</span>
      </div>
      <div id="rows"></div>
    </div>`;
}

function updateBatchView(b) {
  const view = $('#batchView');
  if (view.dataset.batch !== b.id) {
    view.dataset.batch = b.id;
    rowCache.clear();
    view.innerHTML = batchShell(b);
    const btn = $('#dlBtn'), menu = $('#dlMenu');
    btn.addEventListener('click', (e) => { e.stopPropagation(); menu.classList.toggle('open'); });
    document.addEventListener('click', () => menu.classList.remove('open'));
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') menu.classList.remove('open'); });
    menu.querySelectorAll('button').forEach(mi => mi.addEventListener('click', () =>
      location.href = `/api/batches/${b.id}/download.${mi.dataset.fmt}`));
  }

  const done = (b.counts.done || 0), failed = (b.counts.failed || 0);
  const processed = done + failed;
  $('#bCount').textContent = `${processed}/${b.total} processed`;

  const cost = b.files.reduce((s, f) => s + (f.cost_usd || 0), 0);
  const dur = b.files.reduce((s, f) => s + (f.duration_s || 0), 0);
  $('#bMeta').textContent = [
    failed ? `${failed} failed` : '',
    dur ? `${(dur / 60).toFixed(1)} min of audio` : '',
    cost ? `API cost $${cost.toFixed(4)} ($${(cost / (dur / 60)).toFixed(4)}/min)` : '',
  ].filter(Boolean).join(' · ');

  const banner = $('#bBanner');
  const processing = b.files.filter(f => f.status === 'processing').map(f => f.filename);
  const queued = (b.counts.queued || 0);
  if (!b.worker_ready && (queued || processing.length)) {
    banner.hidden = false;
    banner.className = 'banner info';
    banner.innerHTML = `<span class="spinner"></span>
      Warming up the analysis engine — three audio models are loading (~1 min).
      This happens once per deployment; your files start right after.`;
  } else if (processing.length) {
    banner.hidden = false;
    banner.className = 'banner';
    banner.innerHTML = `<span class="spinner"></span>
      Analyzing <b style="font-family:var(--mono);font-size:12.5px">&nbsp;${processing.join(', ')}&nbsp;</b>
      · ${queued} queued · results appear as each file finishes.`;
  } else {
    banner.hidden = true;
  }

  $('#bWarnings').innerHTML = (b.warnings || []).map(w =>
    `<li class="${w.startsWith('Note:') ? 'note' : ''}">${w}</li>`).join('');

  const ordered = [...b.files].sort((a, z) =>
    (STATUS_RANK[a.status] ?? 9) - (STATUS_RANK[z.status] ?? 9)
    || a.filename.localeCompare(z.filename));
  reorderRows($('#rows'), ordered.map(ensureRow));

  // Warm finished rows while the browser is idle so the first expand of any
  // row is instant, including for keyboard and touch users who never hover.
  const idle = window.requestIdleCallback || (fn => setTimeout(fn, 400));
  idle(() => ordered
    .filter(f => (f.status === 'done' || f.status === 'failed') && !detailCache.has(f.id))
    .slice(0, 12)
    .forEach(f => prefetchDetail(f.id)));
}

/* ---------------- detail ---------------- */

/* Detail payloads are immutable once a file is done, so they are fetched
   at most once and kept. Content is rendered BEFORE the panel opens: a
   panel that expands to a spinner and then jumps to full height animates
   twice and reads as broken. */
const detailCache = new Map(); // id -> Promise<detail>

function fetchDetail(id) {
  if (!detailCache.has(id)) {
    detailCache.set(id, api(`/api/files/${id}`).catch(e => {
      detailCache.delete(id); // a failed fetch must not poison later attempts
      throw e;
    }));
  }
  return detailCache.get(id);
}

/* Warm the cache while the pointer is still travelling to the row. */
function prefetchDetail(id) {
  fetchDetail(id).catch(() => {});
}

function detailHTML(f) {
  const ft = f.detail?.features || {};
  const llm = f.detail?.llm || {};
  const trace = f.detail?.trace || {};
  const kv = (label, v) => `<div class="kv"><b>${label}</b>${v ?? '—'}</div>`;
  return `<div class="detail-body">
    <div class="detail-grid">
      ${kv('duration', ft.duration_s + ' s')}
      ${kv('snr', ft.snr_db + ' dB')}
      ${kv('pause floor', (ft.pause_floor_db ?? '—') + ' dBFS')}
      ${kv('max gap', ft.max_gap_s + ' s @ ' + (ft.max_gap_at_s ?? '—') + ' s')}
      ${kv('clip runs ≥3', ft.clip_runs_ge3)}
      ${kv('clicks/min (pause)', ft.clicks_per_min_pause)}
      ${kv('dropouts/min', ft.dropouts_per_min)}
      ${kv('dual mono', ft.is_dual_mono)}
      ${kv('overlap s', f.detail?.overlap?.overlap_total_s ?? '—')}
      ${kv('ser dominant', f.detail?.ser?.dominant ?? '—')}
      ${kv('llm model', llm.model)}
      ${kv('llm latency', llm.latency_s + ' s')}
      ${kv('analysis wall', f.wall_s + ' s')}
      ${kv('api cost', '$' + (f.cost_usd ?? 0))}
    </div>
    <div class="evidence">
      ${llm.tone_evidence ? `<p><b>Tone evidence:</b> ${llm.tone_evidence}</p>` : ''}
      ${llm.noise_evidence ? `<p><b>Noise evidence:</b> ${llm.noise_evidence}</p>` : ''}
      ${trace.noise_rule ? `<p><b>Noise decision:</b> ${trace.noise_rule}</p>` : ''}
      ${(trace.ser_notes || []).map(n => `<p><b>SER:</b> ${n}</p>`).join('')}
      ${f.expected ? `<p>${agreementNote(f.result, f.expected)}</p>` : ''}
    </div>
  </div>`;
}

async function toggleDetail(id) {
  const wrap = document.querySelector(`[data-detail="${id}"]`);
  if (!wrap) return;
  const strip = wrap.parentNode.querySelector('.strip');
  if (openDetails.has(id)) {
    openDetails.delete(id);
    wrap.classList.remove('open');
    strip.setAttribute('aria-expanded', 'false');
    return;
  }
  // On a cache hit this resolves within the same frame, so filling the
  // panel and opening it land together and the height animates once.
  strip.classList.add('loading');
  try {
    wrap.firstElementChild.innerHTML = detailHTML(await fetchDetail(id));
  } catch (e) {
    wrap.firstElementChild.innerHTML =
      `<div class="detail-body"><span class="mismatch">could not load detail: ${e.message}</span></div>`;
  } finally {
    strip.classList.remove('loading');
  }
  openDetails.add(id);
  wrap.classList.add('open');
  strip.setAttribute('aria-expanded', 'true');
}

/* Re-fill an already-open panel after its row re-renders during polling. */
async function restoreDetail(id, el) {
  if (!el) return;
  try {
    el.innerHTML = detailHTML(await fetchDetail(id));
  } catch { /* leave the panel empty rather than flashing an error */ }
}

function agreementNote(result, expected) {
  if (!result || !expected) return '';
  const keys = ['emotional_tone', 'emotional_intensity', 'background_noise_present',
    'background_noise_severity', 'audio_quality', 'speaker_overlap_present',
    'long_silence_present'];
  const diff = keys.filter(k => k in expected && String(result[k]) !== String(expected[k]));
  if (!diff.length) return '<span class="match">matches provided label</span>';
  return `<span class="mismatch">differs from label: ${diff.map(k =>
    `${k.replace(/_/g, ' ')} (${result[k]} vs ${expected[k]})`).join(', ')}</span>`;
}

/* ---------------- batches list + polling ---------------- */

function renderList(items) {
  $('#batchList').innerHTML = items.map(b => {
    const done = (b.counts.done || 0) + (b.counts.failed || 0);
    return `<button class="batch-item ${b.id === currentBatch ? 'active' : ''}" data-id="${b.id}">
      <div class="b-name">${b.name}</div>
      <div class="b-meta">${done}/${b.total} · ${b.status}${b.counts.failed ? ` · ${b.counts.failed} failed` : ''}</div>
    </button>`;
  }).join('') || '<div class="empty">No batches yet.</div>';
  document.querySelectorAll('.batch-item').forEach(el =>
    el.addEventListener('click', () => selectBatch(el.dataset.id)));
}

async function refreshList(selectId) {
  renderList(await api('/api/batches'));
  if (selectId) selectBatch(selectId);
}

async function selectBatch(id) {
  currentBatch = id;
  document.querySelectorAll('.batch-item').forEach(el =>
    el.classList.toggle('active', el.dataset.id === id));
  await pollBatch();
}

async function pollBatch() {
  clearTimeout(pollTimer);
  if (!currentBatch) return;
  const b = await api(`/api/batches/${currentBatch}`);
  updateBatchView(b);
  if (b.status !== 'done') pollTimer = setTimeout(pollBatch, 2000);
  else renderList(await api('/api/batches'));
}

/* ---------------- upload with progress ---------------- */

function uploadXHR(fd) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/batches');
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const pct = Math.round(100 * e.loaded / e.total);
      $('#upPct').textContent = pct + '%';
      $('#upBar').classList.remove('indeterminate');
      $('#upBar').firstElementChild.style.width = pct + '%';
      if (pct >= 100) {
        $('#upLabel').textContent = 'Validating batch — checking the manifest against the audio files…';
        $('#upPct').textContent = '';
        $('#upBar').classList.add('indeterminate');
      }
    };
    xhr.onload = () => {
      if (xhr.status === 401) { location.href = '/login'; return; }
      try {
        const body = JSON.parse(xhr.responseText || '{}');
        xhr.status < 300 ? resolve(body) : reject(new Error(body.detail || xhr.statusText));
      } catch { reject(new Error(xhr.statusText)); }
    };
    xhr.onerror = () => reject(new Error('network error'));
    xhr.send(fd);
  });
}

async function upload(files) {
  const drop = $('#drop');
  if (drop.classList.contains('busy')) return;
  drop.classList.add('busy');
  $('#dropIdle').hidden = true;
  $('#dropBusy').hidden = false;
  $('#upLabel').textContent = `Uploading ${files.length} file${files.length > 1 ? 's' : ''}…`;
  $('#upBar').firstElementChild.style.width = '0%';

  const fd = new FormData();
  [...files].forEach(f => fd.append('files', f));
  fd.append('name', files.length === 1
    ? files[0].name.replace(/\.zip$/i, '')
    : `batch ${new Date().toLocaleTimeString()}`);
  try {
    const r = await uploadXHR(fd);
    toast('Batch created — analysis starts now.');
    await refreshList(r.batch_id);
  } catch (e) {
    toast(`Upload failed: ${e.message}`);
  } finally {
    drop.classList.remove('busy');
    $('#dropIdle').hidden = false;
    $('#dropBusy').hidden = true;
    $('#fileInput').value = '';
  }
}

const drop = $('#drop');
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('over');
  if (e.dataTransfer.files.length) upload(e.dataTransfer.files);
});
$('#fileInput').addEventListener('change', e => {
  if (e.target.files.length) upload(e.target.files);
});
$('#logout').addEventListener('click', async () => {
  await fetch('/api/logout', { method: 'POST' });
  location.href = '/login';
});

refreshList();

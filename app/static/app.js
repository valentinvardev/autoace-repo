/* Dashboard logic: upload, poll, render channel-strip rows, download. */

const $ = (s, el = document) => el.querySelector(s);
let currentBatch = null;
let pollTimer = null;
const openDetails = new Set();

const SEV = { none: 0, low: 1, medium: 2, high: 3 };
const QUAL = { clear: 0, slightly_impaired: 1, severely_impaired: 2 };
const INT = { low: 1, medium: 2, high: 3 };

function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3500);
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (r.status === 401) { location.href = '/login'; throw new Error('auth'); }
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

/* ---------------- rendering ---------------- */

function meter(label, level, max) {
  const segs = Array.from({ length: max }, (_, i) =>
    `<i class="${i < level ? 'on-' + Math.min(level, 3) : ''}"></i>`).join('');
  return `<div class="meter"><span class="console-label">${label}</span>
    <span class="segs">${segs}</span></div>`;
}

function led(label, on) {
  return `<span class="led"><i class="${on ? 'on' : ''}"></i>${label}</span>`;
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

function stripRow(f) {
  const r = f.result;
  const status = `<span class="status-tag status-${f.status}">${f.status}</span>`;
  if (!r) {
    return `<div class="file-row">
      <button class="strip" data-id="${f.id}">
        <span class="fname">${f.filename}</span>
        <span></span>${status}<span></span><span></span><span></span><span></span><span></span>
      </button>
      ${f.error ? `<div class="file-error">${f.error}</div>` : ''}
    </div>`;
  }
  const noiseLabel = r.background_noise_present
    ? (r.background_noise_type || 'noise') : 'no noise';
  return `<div class="file-row">
    <button class="strip" data-id="${f.id}" aria-expanded="${openDetails.has(f.id)}">
      <span class="fname">${f.filename}</span>
      <span class="tone-chip tone-${r.emotional_tone}">${r.emotional_tone}</span>
      ${status}
      ${meter('intensity', INT[r.emotional_intensity] ?? 0, 3)}
      ${meter(noiseLabel, SEV[r.background_noise_severity] ?? 0, 3)}
      <span class="tone-chip qual-${r.audio_quality}">${r.audio_quality.replace(/_/g, ' ')}</span>
      <span class="led-group">${led('ovl', r.speaker_overlap_present)}${led('sil', r.long_silence_present)}</span>
      <span class="conf">${(r.confidence ?? 0).toFixed(2)}</span>
    </button>
    <div class="detail" data-detail="${f.id}" hidden></div>
  </div>`;
}

function renderBatch(b) {
  const total = b.total || b.files.length || 1;
  const segments = b.files.map(f =>
    `<span class="${f.status === 'done' ? 'done' : f.status === 'failed' ? 'failed'
      : f.status === 'processing' ? 'processing' : ''}"></span>`).join('');
  const doneCount = (b.counts.done || 0) + (b.counts.failed || 0);
  const cost = b.files.reduce((s, f) => s + (f.cost_usd || 0), 0);
  const dur = b.files.reduce((s, f) => s + (f.duration_s || 0), 0);
  const costPerMin = dur > 0 ? (cost / (dur / 60)) : 0;

  $('#batchView').innerHTML = `
    <div class="batch-head">
      <h2>${b.name}</h2>
      <span class="console-label">${doneCount}/${total} processed</span>
      <span class="spacer"></span>
      <button class="btn" id="dlCsv">Download CSV</button>
      <button class="btn" id="dlJson">Download JSON</button>
    </div>
    <div class="progress-meter">${segments}</div>
    <div class="batch-meta">
      ${b.counts.failed ? `${b.counts.failed} failed · ` : ''}
      ${dur ? `${(dur / 60).toFixed(1)} min of audio · API cost $${cost.toFixed(4)}
        ($${costPerMin.toFixed(4)}/min)` : ''}
    </div>
    ${b.warnings.length ? `<ul class="warnings">${b.warnings.map(w => `<li>${w}</li>`).join('')}</ul>` : ''}
    <div class="files">${b.files.map(stripRow).join('')}</div>`;

  $('#dlCsv').onclick = () => location.href = `/api/batches/${b.id}/download.csv`;
  $('#dlJson').onclick = () => location.href = `/api/batches/${b.id}/download.json`;
  document.querySelectorAll('.strip').forEach(el =>
    el.addEventListener('click', () => toggleDetail(+el.dataset.id)));
  openDetails.forEach(id => { const el = $(`[data-detail="${id}"]`); if (el) loadDetail(id, el); });
}

async function toggleDetail(id) {
  const el = $(`[data-detail="${id}"]`);
  if (!el) return;
  if (openDetails.has(id)) { openDetails.delete(id); el.hidden = true; return; }
  openDetails.add(id);
  await loadDetail(id, el);
}

async function loadDetail(id, el) {
  el.hidden = false;
  el.innerHTML = '<span class="console-label">loading…</span>';
  try {
    const f = await api(`/api/files/${id}`);
    const ft = f.detail?.features || {};
    const llm = f.detail?.llm || {};
    const trace = f.detail?.trace || {};
    const kv = (label, v) => `<div class="kv"><b>${label}</b>${v ?? '—'}</div>`;
    el.innerHTML = `
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
      </div>`;
  } catch (e) {
    el.innerHTML = `<span class="mismatch">could not load detail: ${e.message}</span>`;
  }
}

/* ---------------- batches list + polling ---------------- */

async function refreshList(selectId) {
  const items = await api('/api/batches');
  $('#batchList').innerHTML = items.map(b => {
    const done = (b.counts.done || 0) + (b.counts.failed || 0);
    return `<button class="batch-item ${b.id === currentBatch ? 'active' : ''}" data-id="${b.id}">
      <div class="b-name">${b.name}</div>
      <div class="b-meta">${done}/${b.total} · ${b.status}${b.counts.failed ? ` · ${b.counts.failed} failed` : ''}</div>
    </button>`;
  }).join('') || '<div class="empty">No batches yet.</div>';
  document.querySelectorAll('.batch-item').forEach(el =>
    el.addEventListener('click', () => selectBatch(el.dataset.id)));
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
  renderBatch(b);
  if (b.status !== 'done') pollTimer = setTimeout(pollBatch, 2000);
  else refreshListQuiet();
}

async function refreshListQuiet() {
  const active = currentBatch;
  const items = await api('/api/batches');
  $('#batchList').innerHTML = items.map(b => {
    const done = (b.counts.done || 0) + (b.counts.failed || 0);
    return `<button class="batch-item ${b.id === active ? 'active' : ''}" data-id="${b.id}">
      <div class="b-name">${b.name}</div>
      <div class="b-meta">${done}/${b.total} · ${b.status}${b.counts.failed ? ` · ${b.counts.failed} failed` : ''}</div>
    </button>`;
  }).join('');
  document.querySelectorAll('.batch-item').forEach(el =>
    el.addEventListener('click', () => selectBatch(el.dataset.id)));
}

/* ---------------- upload ---------------- */

async function upload(files) {
  const fd = new FormData();
  [...files].forEach(f => fd.append('files', f));
  fd.append('name', files.length === 1 ? files[0].name.replace(/\.zip$/i, '') : `batch ${new Date().toLocaleTimeString()}`);
  toast(`Uploading ${files.length} file(s)…`);
  try {
    const r = await api('/api/batches', { method: 'POST', body: fd });
    toast('Batch created — processing.');
    await refreshList(r.batch_id);
  } catch (e) {
    toast(`Upload failed: ${e.message}`);
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

/*
 * JARVIS · floating bar behaviour.
 *
 * Everything this file does is real: it talks to the local core over HTTP + the same WebSocket the
 * HUD uses, records microphone audio and streams it to /api/listen, and drives the always-on
 * wake-word ear through /api/listening.  No framework, no build step, no CDN - the bar has to come
 * up in well under a second on an offline laptop.
 */
'use strict';

const STATE = {
  busy: false,
  recording: false,
  offline: false,
  earOn: false,
  history: [],
  historyAt: -1,
  recorder: null,
  chunks: [],
  stream: null,
  meter: null,
  socket: null,
  retry: 1000,
};

const $ = (id) => document.getElementById(id);
const entry = $('entry');
const replyBox = $('reply');
const modeBox = $('mode');
const liveFill = document.querySelector('#live i');
const earBtn = $('ear');
const micBtn = $('mic');
const goBtn = $('go');
const hideBtn = $('hide');

/* ---------------------------------------------------------------- transport */
function origin() {
  const boot = window.JARVIS_BOOT || {};
  if (boot.backend) return String(boot.backend).replace(/\/$/, '');
  if (/^https?:$/.test(location.protocol)) return location.origin;
  // Opened straight from disk: fall back to the default JARVIS_PORT.
  const port = (window.JARVIS_PORT || '8760').toString();
  return `http://127.0.0.1:${port}`;
}

function wsUrl() {
  return origin().replace(/^http/, 'ws') + '/ws';
}

async function api(path, body, timeoutMs = 15000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const init = { method: body ? 'POST' : 'GET', signal: controller.signal, cache: 'no-store' };
  if (body) {
    init.headers = { 'content-type': 'application/json' };
    init.body = JSON.stringify(body);
  }
  try {
    const response = await fetch(origin() + path, init);
    const text = await response.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch { data = { ok: false, raw: text.slice(0, 160) }; }
    setOffline(false);
    if (!response.ok && data.ok !== false) data.ok = false;
    if (!response.ok) data.error = data.error || `http ${response.status}`;
    return data;
  } catch (error) {
    setOffline(true);
    return { ok: false, error: error.name === 'AbortError' ? 'the core is still thinking' : 'core unreachable' };
  } finally {
    clearTimeout(timer);
  }
}

function setOffline(flag) {
  if (STATE.offline === flag) return;
  STATE.offline = flag;
  document.body.dataset.offline = flag ? '1' : '0';
  if (flag) say('Core unreachable — is <b>python main.py</b> running?', 'fail');
}

function connect() {
  try {
    STATE.socket = new WebSocket(wsUrl());
  } catch (error) {
    scheduleReconnect();
    return;
  }
  STATE.socket.onopen = () => { STATE.retry = 1000; };
  STATE.socket.onmessage = (event) => {
    let payload;
    try { payload = JSON.parse(event.data); } catch { return; }
    applyEvent(payload);
  };
  STATE.socket.onclose = scheduleReconnect;
  STATE.socket.onerror = () => { try { STATE.socket.close(); } catch { /* already down */ } };
}

function scheduleReconnect() {
  setTimeout(connect, STATE.retry);
  STATE.retry = Math.min(8000, Math.round(STATE.retry * 1.7));
}

function applyEvent(payload) {
  switch (payload.type) {
    case 'hello':
      if (payload.mode) setMode(payload.mode);
      refresh();
      break;
    case 'state':
      setMode(payload.mode, payload.detail);
      break;
    case 'reply':
      if (typeof payload.latency_ms === 'number') goBtn.dataset.busy = '0';
      STATE.busy = false;
      say(formatAnswer(payload), payload.ok === false ? 'fail' : 'ok');
      break;
    case 'transcript':
      if (!entry.value.trim()) entry.value = payload.text || '';
      say(`Heard <b>${escapeHtml(payload.text || '')}</b>${payload.engine ? ` · ${escapeHtml(payload.engine)}` : ''}`, 'ok');
      break;
    case 'wake':
      applyEar(payload);
      break;
    case 'error':
      STATE.busy = false;
      say(escapeHtml(payload.message || 'the core reported an error'), 'fail');
      break;
    default:
      break;
  }
}

/* ------------------------------------------------------------------- output */
function escapeHtml(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function formatAnswer(payload) {
  const answer = String(payload.answer ?? payload.text ?? '').trim();
  const error = String(payload.error ?? payload.message ?? '').trim();
  const tools = Array.isArray(payload.tool_calls) ? payload.tool_calls.length : 0;
  const bits = [];
  if (answer) bits.push(escapeHtml(answer));
  else bits.push(error ? escapeHtml(error) : (payload.ok === false ? 'Failed.' : 'Done.'));
  const tags = [];
  if (payload.track) tags.push(payload.track);
  if (tools) tags.push(`${tools} tool${tools === 1 ? '' : 's'}`);
  if (typeof payload.latency_ms === 'number' && payload.latency_ms > 0) tags.push(`${payload.latency_ms}ms`);
  if (tags.length) bits.push(`<span style="opacity:.6">${tags.join(' · ')}</span>`);
  return bits.join(' ');
}

function say(html, kind) {
  replyBox.dataset.kind = kind === 'fail' ? 'fail' : 'ok';
  replyBox.innerHTML = html || '';
}

function setMode(mode, detail) {
  const clean = String(mode || 'idle').toLowerCase();
  document.body.dataset.mode = clean === 'idle' ? 'idle' : clean;
  modeBox.textContent = detail ? `${clean} · ${String(detail).slice(0, 34)}` : clean;
  if (clean === 'thinking') { goBtn.dataset.busy = '1'; liveFill.style.width = '55%'; }
  else if (clean === 'speaking') { liveFill.style.width = '100%'; }
  else { goBtn.dataset.busy = STATE.busy ? '1' : '0'; liveFill.style.width = '0'; }
}

/* --------------------------------------------------------------------- send */
async function submit(raw) {
  const text = String(raw == null ? entry.value : raw).trim();
  if (!text || STATE.busy) return;
  remember(text);
  entry.value = '';
  STATE.busy = true;
  goBtn.dataset.busy = '1';
  setMode('thinking', text.slice(0, 34));
  say(`Working on <b>${escapeHtml(text)}</b>…`, 'ok');
  const result = await api('/api/command', { text, speak: true, source: 'bar' }, 180000);
  STATE.busy = false;
  goBtn.dataset.busy = '0';
  if (result && (result.answer || result.error)) {
    say(formatAnswer(result), result.ok === false ? 'fail' : 'ok');
    setMode(result.ok === false ? 'error' : 'idle');
  } else {
    say('No reply came back — check the HUD terminal for the reason.', 'fail');
    setMode('error', 'no reply');
  }
}

function remember(text) {
  STATE.history = [text].concat(STATE.history.filter((t) => t !== text)).slice(0, 30);
  STATE.historyAt = -1;
  try { localStorage.setItem('jarvis.bar.history', JSON.stringify(STATE.history)); } catch { /* private mode */ }
}

function recall(delta) {
  if (!STATE.history.length) return;
  STATE.historyAt = Math.max(-1, Math.min(STATE.history.length - 1, STATE.historyAt + delta));
  entry.value = STATE.historyAt < 0 ? '' : STATE.history[STATE.historyAt];
  requestAnimationFrame(() => entry.setSelectionRange(entry.value.length, entry.value.length));
}

/* ---------------------------------------------------------------------- mic */
async function toggleMic() {
  if (STATE.recording) { stopMic(); return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  } catch (error) {
    say('Microphone blocked — allow it in the browser/WebView2 site settings '
        + `(${escapeHtml(error.name || 'error')}). Or say <b>Jarvis</b> and speak.`, 'fail');
    setMode('error', 'mic denied');
    return;
  }
  STATE.stream = stream;
  STATE.chunks = [];
  const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', '']
    .find((type) => !type || (window.MediaRecorder && MediaRecorder.isTypeSupported(type)));
  try {
    STATE.recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  } catch (error) {
    say(`This WebView cannot record audio (${escapeHtml(error.name)}). Use the HUD mic or the wake word.`, 'fail');
    stream.getTracks().forEach((t) => t.stop());
    return;
  }
  STATE.recorder.ondataavailable = (event) => { if (event.data && event.data.size) STATE.chunks.push(event.data); };
  STATE.recorder.onstop = flushMic;
  STATE.recorder.start(250);
  STATE.recording = true;
  micBtn.dataset.rec = '1';
  micBtn.textContent = 'STOP';
  setMode('listening', 'speak now');
  startMeter(stream);
}

function stopMic() {
  if (!STATE.recording) return;
  STATE.recording = false;
  try { STATE.recorder && STATE.recorder.state !== 'inactive' && STATE.recorder.stop(); } catch { /* already stopped */ }
  micBtn.dataset.rec = '0';
  micBtn.textContent = 'LISTEN';
  stopMeter();
}

async function flushMic() {
  if (STATE.stream) { STATE.stream.getTracks().forEach((t) => t.stop()); STATE.stream = null; }
  const blob = new Blob(STATE.chunks, { type: STATE.recorder && STATE.recorder.mimeType ? STATE.recorder.mimeType : 'audio/webm' });
  STATE.chunks = [];
  if (!blob.size) { say('Nothing was recorded.', 'fail'); setMode('idle'); return; }
  setMode('thinking', `${(blob.size / 1024).toFixed(0)} kB of audio`);
  say('Transcribing…', 'ok');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch(`${origin()}/api/listen`, {
      method: 'POST',
      headers: { 'content-type': blob.type || 'audio/webm' },
      body: blob,
      signal: controller.signal,
    });
    const text = await response.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch { data = { ok: false, raw: text.slice(0, 120) }; }
    if (data.text && !data.answer) entry.value = data.text;
    say(formatAnswer({ answer: data.answer || data.text || '', ok: data.ok !== false,
                       track: 'voice', latency_ms: data.latency_ms }), 'ok');
    setMode('idle');
  } catch (error) {
    say(`Upload failed: ${escapeHtml(error.message || error.name)}`, 'fail');
    setMode('error', 'mic upload');
  } finally {
    clearTimeout(timer);
  }
}

/* A real meter — AnalyserNode on the live mic, not a fake animation. */
function startMeter(stream) {
  const AudioCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtor) return;
  try {
    const ctx = new AudioCtor();
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    const buffer = new Uint8Array(analyser.frequencyBinCount);
    STATE.meter = { ctx, source, analyser, raf: 0 };
    const tick = () => {
      if (!STATE.meter) return;
      analyser.getByteTimeDomainData(buffer);
      let peak = 0;
      for (let i = 0; i < buffer.length; i += 2) peak = Math.max(peak, Math.abs(buffer[i] - 128) / 128);
      liveFill.style.width = `${Math.min(100, Math.round(peak * 160))}%`;
      STATE.meter.raf = requestAnimationFrame(tick);
    };
    tick();
  } catch { /* no WebAudio here: the meter simply stays flat */ }
}

function stopMeter() {
  if (!STATE.meter) return;
  cancelAnimationFrame(STATE.meter.raf);
  try { STATE.meter.source.disconnect(); STATE.meter.ctx.close(); } catch { /* already closed */ }
  STATE.meter = null;
  liveFill.style.width = '0';
}

/* ---------------------------------------------------------------------- ear */
async function toggleEar() {
  const action = STATE.earOn ? 'stop' : 'start';
  earBtn.disabled = true;
  const result = await api('/api/listening', { action }, 45000);
  earBtn.disabled = false;
  if (result && Object.prototype.hasOwnProperty.call(result, 'running')) applyEar(result);
  else if (result && result.status) applyEar(result.status);
  const message = result && (result.message || result.error);
  if (message) say(escapeHtml(message), result.ok === false ? 'fail' : 'ok');
  if (action === 'start' && result && result.ok !== false) entry.blur();
}

function applyEar(status) {
  const running = !!(status && status.running);
  STATE.earOn = running;
  earBtn.setAttribute('aria-pressed', running ? 'true' : 'false');
  earBtn.textContent = running ? 'ear on' : 'ear off';
  earBtn.title = running
    ? `Say “${status.wake_word || 'Jarvis'}” from any app. Reason: ${status.detail || status.reason || 'listening'}`
    : (status && (status.detail || status.reason)) || 'Wake-word ear is off — click to start it';
}

/* ------------------------------------------------------------------ overlay */
async function hideBar() {
  stopMic();
  const out = await api('/api/bar', { action: 'hide' }, 4000);
  if (!out || out.ok === false) {
    // No controller (opened in a plain browser tab): at least clear the field.
    entry.value = '';
    entry.blur();
    say('Hiding is wired through the launcher (main.py). In a browser tab, just close the tab.', 'ok');
  }
}

async function refresh() {
  const [status, desktop, reminders] = await Promise.all([
    api('/api/status', null, 4000),
    api('/api/desktop', null, 6000),
    api('/api/reminders', null, 4000),
  ]);
  if (status && status.mode) setMode(status.mode);
  if (desktop && desktop.listening) applyEar(desktop.listening);
  if (reminders && typeof reminders.count === 'number' && reminders.count) {
    const next = (reminders.reminders || [])[0];
    if (next) {
      earBtn.insertAdjacentHTML('afterend',
        `<span id="sched" title="next scheduled action" style="color:var(--amber)">⏰ ${escapeHtml(next.next_human || next.run_at || '')}</span>`);
    }
  }
}

/* ----------------------------------------------------------------- wire up */
goBtn.addEventListener('click', () => submit());
hideBtn.addEventListener('click', hideBar);
micBtn.addEventListener('click', toggleMic);
earBtn.addEventListener('click', toggleEar);
entry.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') { event.preventDefault(); submit(); }
  else if (event.key === 'Escape') { event.preventDefault(); hideBar(); }
  else if (event.key === 'ArrowUp') { event.preventDefault(); recall(1); }
  else if (event.key === 'ArrowDown') { event.preventDefault(); recall(-1); }
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') { hideBar(); return; }
  if (event.key === 'F2') { event.preventDefault(); toggleMic(); return; }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); entry.focus(); entry.select(); }
});
document.querySelectorAll('.chip[data-say]').forEach((chip) => {
  chip.addEventListener('click', () => submit(chip.dataset.say));
});
// The overlay is shown by the hotkey with no focus; take the caret when it becomes visible.
document.addEventListener('visibilitychange', () => { if (!document.hidden) entry.focus({ preventScroll: true }); });
window.addEventListener('focus', () => entry.focus({ preventScroll: true }));

try {
  const stored = JSON.parse(localStorage.getItem('jarvis.bar.history') || '[]');
  if (Array.isArray(stored)) STATE.history = stored.slice(0, 30);
} catch { /* nothing stored yet */ }

if (window.JARVIS_BOOT && JARVIS_BOOT.backend) {
  /* main.py can inject the real port; nothing else to do. */
}
connect();
refresh();
setInterval(refresh, 20000);
setInterval(() => { if (document.hidden) return; api('/api/status', null, 3000); }, 45000);
entry.focus({ preventScroll: true });

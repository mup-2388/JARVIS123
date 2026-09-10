/* ==========================================================================
   J.A.R.V.I.S.  --  arc_reactor.js
   --------------------------------------------------------------------------
   Two cooperating layers in one module:

   1. WebGL layer (Three.js)  : a glowing cyan wireframe torus + emissive
      sphere core, ambient + blue point light, bloom, particles and a render
      loop whose energy/pulse is driven by the assistant's state machine
      (idle / listening / thinking / speaking / error).
   2. HUD layer               : WebSocket client for the FastAPI core
      (/ws), live CPU/RAM/VRAM telemetry panels, a scrolling terminal fed by
      server log lines, structured tool "cards", mic capture (WebM clip or
      live 16 kHz PCM stream), TTS playback and keyboard control.

   Loaded as an ES module from static/index.html through an import map:
       "three"          -> https://cdn.jsdelivr.net/npm/three@0.160.1/...
       "three/addons/*" -> examples/jsm (postprocessing)
   If a CDN is unreachable the HUD still works: rendering falls back to the
   CSS grid backdrop and every REST call is attempted through pywebview's
   js_api bridge, so the panel degrades instead of dying.
   ========================================================================== */

/* Three.js is optional and loads from the CDN at runtime. If the CDN is
   unreachable (e.g. an offline laptop) or WebGL cannot be created, the HUD
   degrades to the CSS backdrop + terminal + telemetry + mic/TTS path instead
   of crashing the whole module. */
let THREE = null;
try {
  THREE = await import('three');
} catch (error) {
  console.warn('[JARVIS] Three.js unavailable — running the 2D HUD without the reactor renderer.', error);
}

/* ------------------------------------------------------------------ helpers */
const $ = (id) => document.getElementById(id);
const num = (value, fallback = 0) => (Number.isFinite(+value) ? +value : fallback);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const pct = (v) => `${clamp(num(v), 0, 100).toFixed(0)}%`;
const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

const BOOT = window.JARVIS_BOOT || {};
const isHttp = location.protocol === 'http:' || location.protocol === 'https:';

function backendOrigin() {
  if (isHttp) return location.origin;
  return BOOT.backend || window.JARVIS_BACKEND || 'http://127.0.0.1:8760';
}
function socketUrl() {
  if (isHttp) {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    return `${scheme}://${location.host}/ws`;
  }
  return BOOT.socket || window.JARVIS_SOCKET || `${backendOrigin().replace(/^http/, 'ws')}/ws`;
}

const state = {
  mode: 'boot',
  energy: 0.12,
  targetEnergy: 0.12,
  audioLevel: 0,
  serverLevel: 0,
  net: 'offline',
  latency: 0,
  voiceOn: BOOT.speak_replies !== false,
  autoscroll: true,
  listening: false,
  liveStream: false,
  history: { cpu: [], ram: [], gpu: [] },
  prompts: [],
  promptIndex: -1,
  lastAnswer: '',
};

/* ---- state-driven colour + energy (shared by renderer + HUD) ------------ */
let wantedColor = null;   // THREE.Color once the renderer is up; null in 2D fallback

const MODE_STYLE = {
  idle:      { energy: 0.18, color: 0x22d3ee, label: 'arc reactor nominal' },
  listening: { energy: 0.6,  color: 0x34d399, label: 'audio ingest · whisper listening' },
  thinking:  { energy: 0.95, color: 0xfbbf24, label: 'routing · executing tools' },
  speaking:  { energy: 0.72, color: 0x60a5fa, label: 'xtts synthesis · voice out' },
  error:     { energy: 1.25, color: 0xfb7185, label: 'fault reported — see terminal' },
  boot:      { energy: 0.4,  color: 0x22d3ee, label: 'cold boot · loading models' },
};

function setMode(mode, detail = '') {
  const key = MODE_STYLE[mode] ? mode : 'idle';
  state.mode = key;
  const style = MODE_STYLE[key];
  state.targetEnergy = reducedMotion ? style.energy * 0.5 : style.energy;
  if (wantedColor) wantedColor.setHex(style.color);
  const pill = $('pill-mode');
  if (pill) {
    pill.textContent = key.toUpperCase();
    pill.dataset.mode = key;
    pill.className = `state-pill pill-${key}`;
  }
  const caption = $('reactor-caption');
  if (caption) caption.textContent = detail ? `${style.label} · ${detail}`.slice(0, 84) : style.label;
  if (key === 'error') flash();
}

function flash() {
  const el = $('flash');
  if (!el) return;
  el.classList.add('on');
  setTimeout(() => el.classList.remove('on'), 90);
}

/* ==========================================================================
   1 · WEBGL — ARC REACTOR  (optional — guarded so the HUD survives a missing
      CDN or a WebGL context failure)
   ========================================================================== */

let renderer = null;
let scene = null;
let camera = null;
let reactor = null;
let torus = null;
let core = null;
let composer = null;

if (THREE) {
  const canvas = $('reactor-canvas');
  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
  } catch (error) {
    console.warn('[JARVIS] WebGL context creation failed — HUD continues in 2D mode.', error);
    renderer = null;
  }
  if (renderer) {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.16;
    if ('outputColorSpace' in renderer) renderer.outputColorSpace = THREE.SRGBColorSpace;
  }

  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x04070d, 0.055);

  camera = new THREE.PerspectiveCamera(55, window.innerWidth / Math.max(1, window.innerHeight), 0.1, 120);
  camera.position.set(0, 0.9, 9.4);

  /* ---- lights (spec: AmbientLight + blue PointLight) ---------------------- */
  const ambient = new THREE.AmbientLight(0x38bdf8, 0.42);
  scene.add(ambient);

  const keyLight = new THREE.PointLight(0x2563eb, 7.5, 46, 2.0);   // the blue point light
  keyLight.position.set(2.6, 3.4, 5.2);
  scene.add(keyLight);

  const coreLight = new THREE.PointLight(0x22d3ee, 4.2, 20, 2.0); // cyan spill from the core
  coreLight.position.set(0, 0, 0.4);
  scene.add(coreLight);

  const rimLight = new THREE.DirectionalLight(0x67e8f9, 0.28);
  rimLight.position.set(-4, -2, -3);
  scene.add(rimLight);

  /* ---- the reactor --------------------------------------------------------- */
  reactor = new THREE.Group();
  scene.add(reactor);

  const torusMaterial = new THREE.MeshStandardMaterial({
    color: 0x22d3ee,
    emissive: 0x0891b2,
    emissiveIntensity: 1.35,
    metalness: 0.35,
    roughness: 0.28,
    wireframe: true,
    transparent: true,
    opacity: 0.94,
  });
  torus = new THREE.Mesh(new THREE.TorusGeometry(3.05, 0.34, 14, 96), torusMaterial);
  reactor.add(torus);

  const innerRing = new THREE.Mesh(
    new THREE.TorusGeometry(2.18, 0.075, 8, 72),
    new THREE.MeshStandardMaterial({
      color: 0x67e8f9, emissive: 0x22d3ee, emissiveIntensity: 2.1,
      wireframe: true, transparent: true, opacity: 0.7,
    })
  );
  innerRing.rotation.x = Math.PI / 2;
  reactor.add(innerRing);

  const outerRing = new THREE.Mesh(
    new THREE.TorusGeometry(3.85, 0.03, 6, 128),
    new THREE.MeshBasicMaterial({ color: 0x1e6fa8, wireframe: true, transparent: true, opacity: 0.55 })
  );
  outerRing.rotation.x = Math.PI * 0.5;
  reactor.add(outerRing);

  /* electromagnetic coil ticks around the rim */
  const coils = [];
  const coilGeometry = new THREE.BoxGeometry(0.1, 0.42, 0.26);
  for (let i = 0; i < 16; i += 1) {
    const material = new THREE.MeshStandardMaterial({
      color: 0x0e7490, emissive: 0x22d3ee, emissiveIntensity: 0.6,
      metalness: 0.8, roughness: 0.35, transparent: true, opacity: 0.9,
    });
    const coil = new THREE.Mesh(coilGeometry, material);
    const angle = (i / 16) * Math.PI * 2;
    coil.position.set(Math.cos(angle) * 3.05, Math.sin(angle) * 3.05, 0);
    coil.rotation.z = angle + Math.PI / 2;
    coil.userData.phase = i * 0.4;
    reactor.add(coil);
    coils.push(coil);
  }

  /* glowing plasma core */
  const coreMaterial = new THREE.MeshStandardMaterial({
    color: 0xe0fbff,
    emissive: 0x22d3ee,
    emissiveIntensity: 2.4,
    metalness: 0.1,
    roughness: 0.08,
  });
  core = new THREE.Mesh(new THREE.SphereGeometry(1.02, 48, 32), coreMaterial);
  reactor.add(core);

  const coreShell = new THREE.Mesh(
    new THREE.SphereGeometry(1.42, 28, 18),
    new THREE.MeshBasicMaterial({ color: 0x22d3ee, wireframe: true, transparent: true, opacity: 0.16 })
  );
  reactor.add(coreShell);

  /* radial halo sprite (canvas gradient, no texture download needed) */
  function haloTexture() {
    const size = 256;
    const cnv = document.createElement('canvas');
    cnv.width = cnv.height = size;
    const ctx = cnv.getContext('2d');
    const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    gradient.addColorStop(0.0, 'rgba(224,251,255,0.95)');
    gradient.addColorStop(0.28, 'rgba(34,211,238,0.45)');
    gradient.addColorStop(0.62, 'rgba(37,99,235,0.14)');
    gradient.addColorStop(1.0, 'rgba(0,0,0,0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);
    const texture = new THREE.CanvasTexture(cnv);
    if ('colorSpace' in texture) texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
  }
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({
    map: haloTexture(),
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    transparent: true,
    opacity: 0.85,
  }));
  halo.scale.set(11, 11, 1);
  scene.add(halo);

  /* drifting particle field */
  const PARTICLES = 520;
  const positions = new Float32Array(PARTICLES * 3);
  const seeds = new Float32Array(PARTICLES);
  for (let i = 0; i < PARTICLES; i += 1) {
    const radius = 4.6 + Math.random() * 9.5;
    const angle = Math.random() * Math.PI * 2;
    positions[i * 3] = Math.cos(angle) * radius;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 8.5;
    positions[i * 3 + 2] = Math.sin(angle) * radius - 2;
    seeds[i] = Math.random() * Math.PI * 2;
  }
  const particleGeometry = new THREE.BufferGeometry();
  particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const particles = new THREE.Points(particleGeometry, new THREE.PointsMaterial({
    color: 0x7dd3fc, size: 0.055, sizeAttenuation: true,
    transparent: true, opacity: 0.65, blending: THREE.AdditiveBlending, depthWrite: false,
  }));
  scene.add(particles);

  /* ---- optional bloom (fails soft when the CDN path changes) ------------- */
  async function initBloom() {
    if (!renderer) { composer = null; return; }
    try {
      const [{ EffectComposer }, { RenderPass }, { UnrealBloomPass }, { OutputPass }] = await Promise.all([
        import('three/addons/postprocessing/EffectComposer.js'),
        import('three/addons/postprocessing/RenderPass.js'),
        import('three/addons/postprocessing/UnrealBloomPass.js'),
        import('three/addons/postprocessing/OutputPass.js'),
      ]);
      const next = new EffectComposer(renderer);
      next.addPass(new RenderPass(scene, camera));
      const bloom = new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        reducedMotion ? 0.35 : 0.72,   // strength
        0.75,                          // radius
        0.18                           // threshold
      );
      next.addPass(bloom);
      next.addPass(new OutputPass());
      composer = next;
      composer.bloom = bloom;
      log('sys', 'post-processing: unrealbloom online');
    } catch (error) {
      composer = null;
      log('warn', `bloom pass unavailable (${error.message}) — rendering without post-fx`);
    }
  }

  /* ---- shared renderer colour state --------------------------------------- */
  const modeColor = new THREE.Color(0x22d3ee);
  wantedColor = new THREE.Color(0x22d3ee);

  /* ---- pointer parallax --------------------------------------------------- */
  const pointer = { x: 0, y: 0, tx: 0, ty: 0 };
  window.addEventListener('pointermove', (event) => {
    pointer.tx = (event.clientX / window.innerWidth - 0.5) * 2;
    pointer.ty = (event.clientY / window.innerHeight - 0.5) * 2;
  }, { passive: true });
  window.addEventListener('blur', () => { pointer.tx = 0; pointer.ty = 0; });

  /* ---- resize ------------------------------------------------------------- */
  function resize() {
    const width = Math.max(1, window.innerWidth);
    const height = Math.max(1, window.innerHeight);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    if (renderer) renderer.setSize(width, height, false);
    if (composer) composer.setSize(width, height);
    const wave = $('waveform');
    if (wave) {
      wave.width = Math.floor(wave.clientWidth * Math.min(2, window.devicePixelRatio || 1));
      wave.height = Math.floor(wave.clientHeight * Math.min(2, window.devicePixelRatio || 1));
    }
  }
  window.addEventListener('resize', resize);
  if (window.ResizeObserver) new ResizeObserver(resize).observe(document.body);

  /* ---- the render loop ---------------------------------------------------- */
  const clock = new THREE.Clock();
  let frame = 0;

  function tick() {
    const dt = Math.min(0.05, clock.getDelta());
    const t = clock.elapsedTime;
    frame += 1;

    // energy + colour ease toward the assistant state
    state.energy += (state.targetEnergy - state.energy) * clamp(dt * 3.4, 0, 1);
    modeColor.lerp(wantedColor, clamp(dt * 2.2, 0, 1));

    const pulse = reducedMotion ? 0 : Math.sin(t * (2.6 + state.energy * 5.5));
    const breathe = 0.5 + 0.5 * pulse;
    const level = 0.35 + state.audioLevel * 0.9;             // live microphone gain

    // continuous rotation: the signature idle spin, amplified while working
    reactor.rotation.y += dt * (0.16 + state.energy * 0.75);
    reactor.rotation.x = Math.sin(t * 0.22) * 0.14;
    torus.rotation.z -= dt * (0.1 + state.energy * 0.35);
    innerRing.rotation.z += dt * (0.5 + state.energy * 2.1);
    outerRing.rotation.y -= dt * (0.25 + state.energy * 0.6);
    coreShell.rotation.y -= dt * 0.55;
    coreShell.rotation.x += dt * 0.2;

    // core + material response
    const glow = 1.6 + state.energy * 2.6 + breathe * 0.5 * (0.4 + state.energy) + level * 1.4;
    coreMaterial.emissive.copy(modeColor);
    coreMaterial.emissiveIntensity = glow;
    core.scale.setScalar(1 + state.energy * 0.05 + breathe * 0.012 + level * 0.05);
    torusMaterial.emissive.copy(modeColor).multiplyScalar(0.62);
    torusMaterial.emissiveIntensity = 0.9 + state.energy * 1.5 + level;

    coils.forEach((coil, i) => {
      const local = Math.sin(t * (2.2 + state.energy * 3.2) + coil.userData.phase);
      coil.material.emissiveIntensity = 0.35 + Math.max(0, local) * (0.8 + state.energy * 2.4);
      coil.scale.z = 1 + Math.max(0, local) * 0.28 * (0.4 + state.energy);
      coil.material.emissive.copy(modeColor);
      if (i === 0) coil.visible = true;
    });

    coreLight.color.copy(modeColor);
    coreLight.intensity = 2.6 + state.energy * 4.6 + breathe * 0.9 + level * 3.2;
    keyLight.intensity = 5.4 + state.energy * 3.4;
    ambient.intensity = 0.34 + state.energy * 0.22;

    halo.material.opacity = 0.42 + state.energy * 0.4 + breathe * 0.06;
    const haloScale = 9.6 + state.energy * 3.4 + breathe * 0.5;
    halo.scale.set(haloScale, haloScale, 1);

    // particles drift
    const pos = particleGeometry.attributes.position;
    if (!reducedMotion && frame % 2 === 0) {
      for (let i = 0; i < PARTICLES; i += 1) {
        const y = pos.array[i * 3 + 1];
        pos.array[i * 3 + 1] = y + Math.sin(t * 0.6 + seeds[i]) * 0.0016;
        const x = pos.array[i * 3];
        const z = pos.array[i * 3 + 2];
        const angle = 0.0009 + state.energy * 0.0016;
        pos.array[i * 3] = x * Math.cos(angle) - z * Math.sin(angle);
        pos.array[i * 3 + 2] = x * Math.sin(angle) + z * Math.cos(angle);
      }
      pos.needsUpdate = true;
    }

    // camera parallax
    pointer.x += (pointer.tx - pointer.x) * clamp(dt * 2.4, 0, 1);
    pointer.y += (pointer.ty - pointer.y) * clamp(dt * 2.4, 0, 1);
    camera.position.x = pointer.x * 1.15;
    camera.position.y = 0.9 - pointer.y * 0.85;
    camera.position.z = 9.4 - state.energy * 0.65;
    camera.lookAt(0, 0, 0);

    if (composer) composer.render(dt);
    else if (renderer) renderer.render(scene, camera);
  }
  if (renderer) renderer.setAnimationLoop(tick);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { if (renderer) renderer.setAnimationLoop(null); }
    else { clock.getDelta(); if (renderer) renderer.setAnimationLoop(tick); }
  });
  resize();
  initBloom();
}

/* ==========================================================================
   2 · TERMINAL (WebSocket-fed log stream)
   ========================================================================== */

const terminal = $('terminal');
let terminalLines = 0;

function log(level, text, stamp) {
  if (!terminal) return;
  const line = document.createElement('div');
  line.className = `terminal-line lv-${level || 'raw'}`;
  const time = document.createElement('time');
  time.textContent = stamp || new Date().toLocaleTimeString('en-GB', { hour12: false });
  const body = document.createElement('span');
  body.className = 'body';
  body.textContent = String(text ?? '');
  line.append(time, body);
  terminal.appendChild(line);
  terminalLines += 1;
  while (terminalLines > 420 && terminal.firstChild) {
    terminal.removeChild(terminal.firstChild);
    terminalLines -= 1;
  }
  const counter = $('term-count');
  if (counter) counter.textContent = String(terminalLines);
  if (state.autoscroll) terminal.scrollTop = terminal.scrollHeight;
}
const logEntry = (entry) => log(entry.level || 'raw', entry.text, entry.at);

/* ==========================================================================
   3 · WEBSOCKET LINK
   ========================================================================== */

let socket = null;
let reconnectDelay = 800;
let reconnectTimer = 0;

function connect() {
  const url = socketUrl();
  try { socket = new WebSocket(url); } catch (error) {
    scheduleReconnect(`cannot open ${url}: ${error.message}`);
    return;
  }
  setNet('connecting', 'amber');

  socket.onopen = () => {
    reconnectDelay = 800;
    setNet('linked', 'cyan');
    log('sys', `core link up · ${url}`);
    send({ type: 'hello', from: 'hud', ua: navigator.userAgent.slice(0, 60) });
  };
  socket.onclose = () => {
    if (state.net !== 'offline') log('warn', 'core link dropped');
    setNet('offline', 'rose');
    scheduleReconnect('socket closed');
  };
  socket.onerror = () => setNet('fault', 'rose');
  socket.onmessage = (event) => {
    let payload;
    try { payload = JSON.parse(event.data); } catch { log('raw', String(event.data).slice(0, 300)); return; }
    handlePayload(payload);
  };
}

function scheduleReconnect(reason) {
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connect, reconnectDelay);
  reconnectDelay = Math.min(15000, Math.round(reconnectDelay * 1.6));
  if (reason) log('warn', `retrying core link in ${Math.round(reconnectDelay / 100) / 10}s (${reason})`);
}

function setNet(label, tone) {
  state.net = label;
  const dot = $('net-dot');
  const text = $('net-label');
  const colour = { cyan: '#22d3ee', amber: '#fbbf24', rose: '#fb7185' }[tone] || '#22d3ee';
  if (dot) { dot.style.background = colour; dot.style.boxShadow = `0 0 10px ${colour}`; }
  if (text) text.textContent = `link ${label}`;
}

function send(payload) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
    return true;
  }
  return false;
}

async function rest(path, body) {
  const init = body
    ? { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }
    : { method: 'GET' };
  const response = await fetch(`${backendOrigin()}${path}`, init);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

/** Command dispatch: WebSocket first, REST + pywebview bridge as fallbacks. */
async function sendCommand(rawText) {
  const text = String(rawText || '').trim();
  if (!text) return;
  state.prompts.push(text);
  if (state.prompts.length > 40) state.prompts.shift();
  state.promptIndex = -1;
  setMode('thinking', 'queued');
  if (send({ type: 'command', text, speak: state.voiceOn, source: 'hud' })) return;
  log('in', `» ${text}  (via fallback path)`);

  try {
    if (window.pywebview?.api?.command) {
      const reply = await window.pywebview.api.command(text);
      handlePayload({ type: 'reply', ...reply });
      return;
    }
    const reply = await rest('/api/command', { text, source: 'hud', speak: state.voiceOn });
    handlePayload({ type: 'reply', ...reply });
  } catch (error) {
    setMode('error');
    log('err', `core unreachable: ${error.message}`);
  }
}

/* ---- payload router ---------------------------------------------------- */
const audio = new Audio();
audio.preload = 'auto';
audio.addEventListener('ended', () => setMode('idle'));
audio.addEventListener('error', () => setMode('idle'));

function handlePayload(payload) {
  switch (payload.type) {
    case 'hello': {
      const cfg = payload.config || {};
      log('sys', `JARVIS core v${payload.version || '?'} · ${payload.server_time || ''}`.trim());
      log('sys', `track1 rules: ${(cfg.instant_rules || []).length} · tools: ${(cfg.tools || []).length || (cfg.brain?.tools_bound ?? '—')} · providers: ${(cfg.brain?.available || cfg.brain?.configured || ['none']).join('/')}`);
      if (Array.isArray(cfg.log) && cfg.log.length) {
        log('sys', '— replaying recent core log —');
        cfg.log.slice(-24).forEach(logEntry);
      }
      if (cfg.speak_replies === false) {
        state.voiceOn = false;
        const chip = $('btn-voice');
        chip?.classList.remove('is-on');
        chip?.classList.add('is-off');
      }
      applyVoiceState(cfg.voice, cfg.brain);
      const stamp = $('build-stamp');
      if (stamp) stamp.textContent = `build v${payload.version || '—'} · ${isHttp ? 'http' : 'webview'}`;
      setMode('idle');
      break;
    }
    case 'telemetry': applyTelemetry(payload.data || {}); break;
    case 'log': logEntry(payload); break;
    case 'cleared': if (terminal) terminal.replaceChildren(); terminalLines = 0; break;
    case 'state': setMode(payload.mode, payload.detail || ''); break;
    case 'reply': {
      const answer = payload.answer || '(no answer)';
      state.lastAnswer = answer;
      const line = $('jarvis-line');
      if (line) line.textContent = answer;
      const track = $('pill-track');
      if (track) track.textContent = payload.track || '—';
      const latency = $('pill-latency');
      if (latency) latency.textContent = `${payload.latency_ms ?? 0} ms`;
      const brainPill = $('pill-model');
      if (brainPill) {
        const model = String(payload.model || '');
        brainPill.textContent = model || '—';
        brainPill.classList.toggle('pill-model-local', /^(local-regex|heuristic)/.test(model));
        brainPill.title = model
          ? `answered by ${model}${String(payload.track || '') === 'agent' ? ' (Track 2, agentic)' : ' (Track 1, no AI call)'}`
          : 'no brain reported a model for that reply';
      }
      state.latency = num(payload.latency_ms);
      const instant = $('hit-instant');
      const agent = $('hit-agent');
      if (instant && agent) {
        const isInstant = String(payload.track || '').startsWith('instant');
        const target = isInstant ? instant : agent;
        target.textContent = String(num(target.textContent) + 1);
      }
      (payload.tool_calls || []).forEach((call) => {
        const mark = call.ok === false ? 'err' : 'tool';
        log(mark, `⚙ ${call.tool}(${Object.entries(call.arguments || {}).map(([k, v]) => `${k}=${typeof v === 'string' ? JSON.stringify(v) : v}`).join(', ')}) → ${String(call.message || (call.ok === false ? 'failed' : 'done')).slice(0, 160)}`);
      });
      if (payload.model) log('sys', `brain: ${payload.model}${payload.tier ? ` · ${payload.tier} tier` : ''}`);
      if (String(payload.track || '') === 'agent') refreshProviders();
      refreshDesktop();
      if (payload.error) log('err', String(payload.error).slice(0, 200));
      if (payload.card) renderCard(payload.card);
      if (payload.answer) setMode(payload.speak ? 'speaking' : 'idle');
      break;
    }
    case 'card': renderCard(payload); break;
    case 'wake':
      window.__jarvisEarOn = Boolean(payload.running);
      renderDesktop({ apps: (window.__jarvisDesktop || {}).apps, files: (window.__jarvisDesktop || {}).files,
                    screen: (window.__jarvisDesktop || {}).screen, reminders: (window.__jarvisDesktop || {}).reminders,
                    listening: payload, windows: (window.__jarvisDesktop || {}).windows });
      break;
    case 'transcript':
      log('tool', `≋ heard ${(num(payload.confidence) * 100).toFixed(0)}% in ${num(payload.latency_ms)} ms: ${payload.text}`);
      if (payload.text) $('cmd-input') && ($('cmd-input').placeholder = `jarvis, ${payload.text.slice(0, 60)}`);
      break;
    case 'transcript-complete': {
      const transcript = payload.transcript || {};
      if (payload.ok) log('tool', `≋ clip → ${transcript.text || '(empty)'}`);
      else log('warn', `microphone clip produced no speech (${payload.error || 'silent'})`);
      $('btn-mic')?.classList.remove('busy');
      setMode('idle');
      break;
    }
    case 'segment':
      log('tool', `≋ stream segment: ${payload.text}`);
      break;
    case 'stream':
      if (payload.state === 'silence-dropped') log('sys', `stream: ${payload.duration_s}s of silence discarded`);
      break;
    case 'audio':
      playUrl(payload.url, payload.engine);
      break;
    case 'ack': setMode('thinking', payload.echo ? `“${payload.echo}”`.slice(0, 60) : ''); break;
    case 'error': log('err', payload.message || 'unknown core error'); setMode('error'); break;
    case 'pong': break;
    case 'level': state.serverLevel = num(payload.rms); break;
    case 'config': applyVoiceState(payload.voice, payload.brain); break;
    default:
      log('raw', `${payload.type || 'message'} ${JSON.stringify(payload).slice(0, 240)}`);
  }
}

function playUrl(url, engine) {
  if (!url) return;
  const absolute = /^https?:/.test(url) ? url : `${backendOrigin()}${url}`;
  audio.src = absolute;
  audio.volume = 1;
  audio.play().then(() => {
    setMode('speaking', engine ? `voice · ${engine}` : '');
    log('tts', `▶ speaking${engine ? ` via ${engine}` : ''}`);
    attachPlaybackAnalyser();
  }).catch((error) => {
    log('warn', `playback blocked: ${error.message} (interact with the page once to unlock audio)`);
    setMode('idle');
  });
}

/* ==========================================================================
   4 · TELEMETRY PANELS
   ========================================================================== */

function applyVoiceState(voice, brain) {
  const stt = $('voice-state');
  if (stt && voice) {
    stt.textContent = `stt ${voice.stt?.state || '—'}/${voice.stt?.device || '?'} · tts ${voice.tts?.state || '—'}`;
    if (voice.stt?.error) stt.title = voice.stt.error;
    if (voice.tts?.error) stt.title = voice.tts.error;
    if (voice.stt?.state === 'failed' || voice.tts?.state === 'failed') {
      stt.classList.add('text-amber-200');
    }
  }
  renderProviders(brain);
  const llm = $('llm-state');
  if (llm && brain) {
    const live = brain.available || [];
    const cooling = (brain.providers || []).filter((p) => p.cooling);
    const active = brain.active || live[0] || '';
    const model = ((brain.providers || []).find((p) => p.key === active) || {}).model || '';
    llm.textContent = live.length
      ? `llm ${active || 'auto'} · ${model}${cooling.length ? ` · ${cooling.length} cooling` : ''}`
      : (brain.configured && brain.configured.length ? 'llm all providers cooling' : 'llm no provider key in .env');
    llm.classList.toggle('text-amber-200', !live.length);
    llm.title = [
      `configured: ${(brain.configured || []).join(', ') || 'none'}`,
      cooling.length ? `cooling: ${cooling.map((p) => `${p.key} ${Math.max(1, Math.round((p.cool_left_s || 0) / 60))}m`).join(', ')}` : '',
      `avg ${brain.avg_ms || 0}ms over ${brain.calls || 0} calls (${brain.fast_calls || 0} fast / ${brain.smart_calls || 0} smart)`,
      brain.last_error || '',
      live.length ? '' : 'add GROQ_API_KEY (or CEREBRAS/CLOUDFLARE/GEMINI) to .env, then restart',
    ].filter(Boolean).join('\n');
  }
}

/* ---- AI provider ring -----------------------------------------------------
   Fed by the same `brain` object the core pushes on hello/status, plus a 15 s
   poll of /api/llm so a cooldown you just hit shows up without a restart. */
function humanise(seconds) {
  const left = Math.max(0, Number(seconds) || 0);
  if (left < 90) return `${Math.round(left)}s`;
  if (left < 5400) return `${Math.round(left / 60)}m`;
  if (left < 172800) return `${(left / 3600).toFixed(1)}h`;
  return `${Math.round(left / 86400)}d`;
}

function providerRow(p, brain) {
  const li = document.createElement('li');
  const ready = !!p.key_set && !p.cooling && (!p.needs_account || p.account_set);
  li.className = 'flex items-baseline justify-between gap-2 '
    + (p.cooling ? 'text-amber-200' : ready ? 'text-cyan-100/85' : 'text-cyan-100/35');
  const name = document.createElement('span');
  name.className = 'min-w-0 truncate';
  name.textContent = `${p.key === brain.active ? '▶' : p.cooling ? '▲' : p.key_set ? '●' : '○'} ${p.label}`;
  const right = document.createElement('span');
  right.className = 'shrink-0 tabular-nums';
  if (p.cooling) right.textContent = `out ${humanise(p.cool_left_s)}`;
  else if (!p.key_set) right.textContent = 'no key';
  else if (p.needs_account && !p.account_set) right.textContent = 'no acct id';
  else right.textContent = `${p.ok || 0} ok · ${p.avg_ms || 0}ms`;
  const model = (brain.tier === 'smart' && p.model) || p.fast_model_used || p.model || '';
  li.title = [
    `${p.key} · ${model || 'no model'}`,
    p.quota ? `free tier: ${p.quota}` : '',
    p.reason ? `last problem: ${p.reason}` : '',
    p.models_seen ? `${p.models_seen} models visible to this key` : '',
    (p.discovered && p.discovered.smart && p.discovered.smart !== p.model)
      ? `catalogue id refused; using ${p.discovered.fast || ''} / ${p.discovered.smart}` : '',
    (p.rejected_models && p.rejected_models.length) ? `refused: ${p.rejected_models.join(', ')}` : '',
    `reset window: ${p.reset || '—'}`,
    p.key_set ? '' : `set ${p.key_env} in .env  (${p.key_url || 'get a key'})`,
    p.needs_account && !p.account_set ? `also set ${p.account_env}` : '',
  ].filter(Boolean).join('\n');
  li.append(name, right);
  return li;
}

function renderProviders(brain) {
  if (!brain) return;
  const list = $('llm-providers');
  if (list) {
    const providers = brain.providers || [];
    list.replaceChildren(...providers.map((p) => providerRow(p, brain)));
  }
  const tier = $('llm-tier');
  if (tier) {
    const used = brain.tier_used ? ` · ${brain.tier_used}` : '';
    tier.textContent = `${brain.tier || 'auto'}${used}`;
    tier.classList.toggle('text-amber-200', brain.tier_used === 'smart');
    tier.title = `LLM_TIER_MODE=${brain.tier || 'auto'} — cheap model for chat and tool calls, big model for reasoning-heavy asks. ${brain.fast_calls || 0} fast / ${brain.smart_calls || 0} smart so far.`;
  }
  const summary = $('llm-summary');
  if (summary) {
    const live = (brain.available || []).length;
    const cooling = (brain.providers || []).filter((p) => p.cooling);
    const bits = [`${live}/${(brain.configured || []).length || 0} usable`,
                  `${brain.calls || 0} calls · avg ${brain.avg_ms || 0}ms`];
    if (brain.failures) bits.push(`${brain.failures} failed`);
    if (cooling.length) bits.push(`${cooling.length} cooling until reset`);
    if (brain.offline_rest_s) bits.push(`no egress, resting ${humanise(brain.offline_rest_s)}`);
    summary.textContent = bits.join(' · ');
    summary.className = 'mt-2 border-t border-cyan-500/20 pt-2 font-mono text-[9.5px] leading-snug '
      + (live ? 'text-cyan-100/55' : 'text-amber-200/80');
    if (!live && brain.last_error) summary.title = brain.last_error;
  }
}

let providersBusy = false;
let providersWarned = false;
function renderDesktop(data) {
  if (!data) return;
  window.__jarvisDesktop = data;
  window.__jarvisEarOn = Boolean((data.listening || {}).running);
  const sum = $('desktop-summary');
  const roots = $('desktop-roots');
  const screen = $('desktop-screen');
  const ear = $('btn-ear');
  const apps = data.apps || {};
  const files = data.files || {};
  const eyes = data.screen || {};
  const schedule = data.reminders || {};
  if (sum) {
    const bits = [`${apps.known || 0} apps I can open`,
                  `files in ${(files.roots || []).length} folder${(files.roots || []).length === 1 ? '' : 's'}`,
                  `${schedule.count || 0} scheduled`];
    const health = data.win32 || {};
    if (!data.windows) bits.push('desktop keys need Windows');
    else if (health.ok === false) bits.push('Win32 gaps: ' + (health.missing || []).slice(0, 3).join(', '));
    sum.textContent = bits.join(' · ');
    const unhealthy = (apps.known || 0) === 0 || health.ok === false;
    if (health.message) sum.title = health.message;
    sum.className = 'mt-2 font-mono text-[9.5px] leading-snug '
      + (unhealthy ? 'text-amber-200/80' : 'text-cyan-100/55');
  }
  if (roots) {
    roots.textContent = `files: ${(files.roots || []).join(', ') || 'none yet'}`.slice(0, 220);
    roots.title = `Writes stay inside these folders; anything else needs a spoken “confirm”. Deletes go to the Recycle Bin (${files.delete_policy || 'recycle'}). Last ${files.journal || 0} file actions are journaled, so “undo that” works.`;
  }
  if (screen) {
    screen.textContent = eyes.ocr_ready ? (eyes.vision ? 'screen: ocr + ai' : 'screen: ocr') : 'screen: ai only';
    screen.title = eyes.detail || 'Windows OCR availability';
    screen.classList.toggle('chip-warn', !eyes.ocr_ready);
  }
  if (ear) {
    const on = !!(data.listening && data.listening.running);
    ear.textContent = on ? 'ear on' : 'ear off';
    ear.title = on
      ? `Say “${data.listening.wake_word || 'Jarvis'}” from any app. ${data.listening.detail || data.listening.reason || ''}`
      : (data.listening && (data.listening.detail || data.listening.reason)) || 'Wake-word ear off — click to start it';
    ear.classList.toggle('chip-live', on);
  }
}

async function refreshDesktop() {
  if (document.hidden) return;
  try {
    renderDesktop(await rest('/api/desktop'));
  } catch (error) {
    const sum = $('desktop-summary');
    if (sum) sum.textContent = `desktop check failed: ${String(error.message || error).slice(0, 80)}`;
  }
}

async function toggleEar() {
  const button = $('btn-ear');
  if (button) { button.disabled = true; button.textContent = '…'; }
  const running = Boolean(window.__jarvisEarOn);
  try {
    const data = await rest('/api/listening', { action: running ? 'stop' : 'start' });
    log(data.ok === false ? 'warn' : 'sys', `ear: ${data.message || data.detail || (data.running ? 'listening for the wake word' : 'stopped')}`);
    refreshDesktop();
  } catch (error) {
    log('err', `ear toggle failed: ${error.message}`);
  } finally {
    if (button) button.disabled = false;
  }
}

async function refreshProviders() {
  if (providersBusy || document.hidden) return;
  providersBusy = true;
  try {
    renderProviders(await rest('/api/llm'));
    providersWarned = false;
  } catch {
    if (!providersWarned) {           // the core is offline: say it once, then stay quiet
      providersWarned = true;
      const summary = $('llm-summary');
      if (summary) summary.textContent = 'core unreachable — provider list paused';
    }
  } finally {
    providersBusy = false;
  }
}

function applyTelemetry(data) {
  const cpu = num(data.cpu);
  setText('cpu-value', `${cpu.toFixed(0)}`, '%');
  setWidth('cpu-bar', pct(cpu));
  toggleClass('cpu-bar', 'hot', cpu > 88);

  const ram = num(data.ram);
  setText('ram-value', `${ram.toFixed(0)}`, '%');
  setWidth('ram-bar', pct(ram));
  toggleClass('ram-bar', 'hot', ram > 90);
  setText('ram-used', `${num(data.ram_used_gb).toFixed(1)} GB`);
  setText('ram-total', `${num(data.ram_total_gb).toFixed(1)} GB`);
  setText('swap-value', pct(data.swap));
  setText('disk-value', `${pct(data.disk)} · ${num(data.disk_free_gb).toFixed(0)}G free`);

  const gpu = data.gpu || {};
  const gpuValue = num(gpu.util_pct);
  if (gpu.available) {
    setText('gpu-value', `${gpuValue.toFixed(0)}`, '%');
    setWidth('gpu-bar', pct(gpuValue));
    const vramPct = gpu.mem_total_mb ? (gpu.mem_used_mb / gpu.mem_total_mb) * 100 : 0;
    setText('gpu-detail', `${gpu.name || 'GPU'} · ${gpu.mem_used_mb}/${gpu.mem_total_mb} MB (${vramPct.toFixed(0)}%) · ${gpu.temp_c ?? '—'}°C`);
    toggleClass('gpu-bar', 'hot', vramPct > 85);
  } else {
    setText('gpu-value', '—', '');
    setWidth('gpu-bar', '0%');
    setText('gpu-detail', 'nvidia-smi unavailable — telemetry is CPU/RAM only');
  }

  renderCores(data.per_core || [], num(data.cpu_count));
  pushHistory(cpu, ram, gpu.available ? gpuValue : 0);
  drawSpark('cpu-spark', state.history.cpu, '#22d3ee', [
    { series: state.history.ram, colour: '#fbbf24' },
    { series: state.history.gpu, colour: '#60a5fa' },
  ]);

  const procs = $('proc-list');
  if (procs) {
    const list = data.top_processes || [];
    if (list.length) {
      procs.replaceChildren(...list.map((proc) => {
        const item = document.createElement('li');
        item.className = 'flex items-center justify-between gap-2';
        const name = document.createElement('span');
        name.className = 'truncate';
        name.textContent = proc.name;
        const stat = document.createElement('span');
        stat.className = 'shrink-0 text-cyan-300/70 tabular-nums';
        stat.textContent = `${num(proc.cpu).toFixed(0)}% / ${num(proc.mem).toFixed(1)}%`;
        item.append(name, stat);
        return item;
      }));
    }
  }
  setText('net-down', `${num(data.net_recv_mb).toFixed(0)} MB`);
  setText('net-up', `${num(data.net_sent_mb).toFixed(0)} MB`);
  setText('boot-time', data.boot_time || '—');
  setText('uptime', humanDuration(num(data.uptime_s)));
  const info = $('track-info');
  if (info) info.title = `mode ${data.mode || '—'} · ${num(data.clients)} hud client(s) · ${num(data.process_count)} procs`;
}

function setText(id, value, suffix) {
  const el = $(id);
  if (!el) return;
  el.textContent = value;
  if (suffix !== undefined) {
    const tail = document.createElement('span');
    tail.className = 'text-sm';
    tail.textContent = suffix;
    el.appendChild(tail);
  }
}
function setWidth(id, value) { const el = $(id); if (el) el.style.width = value; }
function toggleClass(id, klass, on) { const el = $(id); if (el) el.classList.toggle(klass, !!on); }

function humanDuration(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  return d ? `${d}d ${h}h` : (h ? `${h}h ${m}m` : `${m}m ${s % 60}s`);
}

let coreCount = 0;
function renderCores(perCore, count) {
  const host = $('cpu-cores');
  if (!host) return;
  const wanted = clamp(num(count, perCore.length) || perCore.length || 8, 1, 32);
  if (wanted !== coreCount) {
    coreCount = wanted;
    setText('cpu-cores-count', String(wanted));
    host.style.gridTemplateColumns = `repeat(${Math.min(wanted, 16)}, minmax(0, 1fr))`;
    host.replaceChildren(...Array.from({ length: wanted }, () => document.createElement('i')));
  }
  Array.from(host.children).forEach((cell, index) => {
    const value = num(perCore[index], 0);
    cell.style.setProperty('--core', pct(value));
    cell.classList.toggle('hot', value > 90);
    cell.title = `core ${index} · ${value.toFixed(0)}%`;
  });
}

function pushHistory(cpu, ram, gpu) {
  const limit = 90;
  for (const [key, value] of [['cpu', cpu], ['ram', ram], ['gpu', gpu]]) {
    const list = state.history[key];
    list.push(clamp(value, 0, 100));
    if (list.length > limit) list.shift();
  }
}

function drawSpark(id, series, colour, extras = []) {
  const cnv = $(id);
  if (!cnv || !series.length) return;
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const width = cnv.clientWidth || 320;
  const height = cnv.clientHeight || 40;
  if (cnv.width !== Math.floor(width * ratio)) { cnv.width = Math.floor(width * ratio); cnv.height = Math.floor(height * ratio); }
  const ctx = cnv.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const max = Math.max(12, ...series);
  const step = width / Math.max(1, series.length - 1);
  const points = series.map((value, index) => [index * step, height - (value / max) * (height - 4) - 2]);

  ctx.beginPath();
  ctx.moveTo(points[0][0], height);
  points.forEach(([x, y]) => ctx.lineTo(x, y));
  ctx.lineTo(points[points.length - 1][0], height);
  ctx.closePath();
  const fill = ctx.createLinearGradient(0, 0, 0, height);
  fill.addColorStop(0, `${colour}55`);
  fill.addColorStop(1, 'rgba(4,7,13,0)');
  ctx.fillStyle = fill;
  ctx.fill();

  for (const extra of extras) {
    if (!extra.series.length) continue;
    const stepX = width / Math.max(1, extra.series.length - 1);
    ctx.beginPath();
    extra.series.forEach((value, index) => {
      const y = height - (value / max) * (height - 4) - 2;
      if (index === 0) ctx.moveTo(0, y); else ctx.lineTo(index * stepX, y);
    });
    ctx.strokeStyle = extra.colour;
    ctx.lineWidth = 0.9;
    ctx.globalAlpha = 0.75;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  ctx.beginPath();
  points.forEach(([x, y], index) => (index ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.strokeStyle = colour;
  ctx.lineWidth = 1.3;
  ctx.shadowColor = colour;
  ctx.shadowBlur = 8;
  ctx.stroke();
  ctx.shadowBlur = 0;
}

/* ==========================================================================
   5 · STRUCTURED CARDS (tool output)
   ========================================================================== */

const cards = $('cards');

function card(title, nodes) {
  if (!cards) return;
  const el = document.createElement('article');
  el.className = 'card pointer-events-auto';
  const head = document.createElement('h3');
  head.textContent = title;
  const body = document.createElement('div');
  body.className = 'card-body';
  body.append(...nodes);
  el.append(head, body);
  cards.prepend(el);
  while (cards.children.length > 6) cards.lastElementChild.remove();
  setTimeout(() => { if (el.isConnected && cards.children.length > 1) el.style.opacity = '0.92'; }, 6000);
}

const row = (main, sub, href) => {
  const wrap = document.createElement('div');
  wrap.className = 'card-row';
  const title = document.createElement('div');
  if (href) {
    const link = document.createElement('a');
    link.href = href;
    link.target = '_blank';
    link.rel = 'noreferrer noopener';
    link.textContent = main;
    title.appendChild(link);
  } else {
    title.textContent = main;
  }
  wrap.appendChild(title);
  if (sub) {
    const detail = document.createElement('div');
    detail.className = 'card-sub';
    detail.textContent = sub;
    wrap.appendChild(detail);
  }
  return wrap;
};

function renderCard(payload) {
  const kind = payload.type;
  if (kind === 'search') {
    const nodes = (payload.items || []).slice(0, 6).map((item) =>
      row(item.title || item.url, (item.snippet || '').slice(0, 150), item.url));
    if (nodes.length) card(`web search · ${payload.query || ''}`, nodes);
  } else if (kind === 'notes') {
    const nodes = (payload.items || []).map((note) => row(
      `${note.title || note.name}`,
      `${(note.excerpt || '').replace(/\s+/g, ' ').slice(0, 190)} · ${note.modified || ''}`));
    if (nodes.length) card(`notes · ${payload.topic || 'all'}`, nodes);
  } else if (kind === 'sports') {
    const nodes = [];
    (payload.recent || []).forEach((fixture) => nodes.push(row(
      `${fixture.home} ${fixture.score} ${fixture.away}`,
      `${fixture.date} · ${fixture.league || ''} · ${fixture.status || ''}${fixture.winner ? ` · ${fixture.winner}` : ''}`)));
    (payload.upcoming || []).slice(0, 3).forEach((fixture) => nodes.push(row(
      `next: ${fixture.home} vs ${fixture.away}`, `${fixture.date} · ${fixture.league || ''}`)));
    if (payload.standing) nodes.push(row(
      `standing: ${payload.standing.rank}th`, `${payload.standing.points} pts · ${payload.standing.league || ''}`));
    if (nodes.length) card(`football · ${payload.team || ''}`, nodes);
  } else if (kind === 'system') {
    const data = payload.items || {};
    const nodes = [row('system report', `cpu ${pct(data.cpu)} · ram ${pct(data.ram)} · disk ${pct(data.disk)}`)];
    if (data.gpu?.available) nodes.push(row('gpu', `${data.gpu.name} · ${data.gpu.mem_used_mb}/${data.gpu.mem_total_mb} MB`));
    (data.top_processes || []).forEach((proc) => nodes.push(row(proc.name, `pid ${proc.pid} · cpu ${proc.cpu}% · mem ${proc.mem}%`)));
    card('telemetry snapshot', nodes);
  } else if (Array.isArray(payload.items)) {
    card(payload.type || 'core payload', payload.items.map((item) => row(typeof item === 'string' ? item : JSON.stringify(item).slice(0, 180), '')));
  }
}

/* ==========================================================================
   6 · MICROPHONE — clip mode + live 16 kHz PCM stream
   ========================================================================== */

let audioCtx = null;
let analyser = null;
let micStream = null;
let micSource = null;
let recorder = null;
let chunks = [];
let tapMode = false;
let holdTimer = 0;
let holding = false;
let clipNode = null;        // ScriptProcessor capturing 16 kHz PCM for a mic clip
let clipChunks = null;      // Float32Array[] at 16 kHz, accumulated while the mic is held
let workletNode = null;
let scriptNode = null;
let streamSamples = null;   // float32 accumulation between flushes
let flushTimer = 0;

function pickMimeType() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
  ];
  for (const type of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) return type;
  }
  return '';
}

async function ensureAudioGraph() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  if (!micStream) {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
      video: false,
    });
    micSource = audioCtx.createMediaStreamSource(micStream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.72;
    micSource.connect(analyser);
  }
  if (audioCtx.state === 'suspended') await audioCtx.resume();
}

async function startMic(toggle = false) {
  const button = $('btn-mic');
  try {
    await ensureAudioGraph();
  } catch (error) {
    log('err', `microphone refused: ${error.message} — check Windows privacy ▸ Microphone`);
    setMode('idle');
    button?.classList.remove('armed', 'busy');
    return;
  }
  tapMode = toggle;
  button?.classList.add('armed');
  state.listening = true;
  setMode('listening', 'capturing');

  if (state.liveStream) { startLiveStream(); return; }

  // Preferred: capture raw 16 kHz PCM so the core can transcribe it with NO ffmpeg.
  // (ffmpeg is a separate manual install; the old MediaRecorder path silently produced
  // webm/opus clips that decode to nothing on a machine without it.)
  if (captureClip()) return;

  chunks = [];
  const mimeType = pickMimeType();
  try {
    recorder = new MediaRecorder(micStream, mimeType ? { mimeType } : undefined);
  } catch (error) {
    log('err', `MediaRecorder unavailable (${error.message})`);
    recorder = null;
    return;
  }
  recorder.ondataavailable = (event) => { if (event.data && event.data.size) chunks.push(event.data); };
  recorder.start(250);
}

function captureClip() {
  try {
    const inputRate = audioCtx.sampleRate;
    clipChunks = [];
    clipNode = audioCtx.createScriptProcessor(1024, 1, 1);
    clipNode.onaudioprocess = (event) => {
      clipChunks.push(downsample(new Float32Array(event.inputBuffer.getChannelData(0)), inputRate, 16000));
    };
    // ScriptProcessor must be pulled; route it into a silent gain so it never feeds back.
    const sink = audioCtx.createGain();
    sink.gain.value = 0;
    micSource.connect(clipNode);
    clipNode.connect(sink);
    sink.connect(audioCtx.destination);
    return true;
  } catch (error) {
    log('warn', `PCM capture unavailable (${error.message}) — falling back to compressed clip`);
    clipNode = null;
    clipChunks = null;
    return false;
  }
}

function stopClipCapture() {
  if (clipNode) {
    try { micSource.disconnect(clipNode); clipNode.disconnect(); } catch { /* already gone */ }
  }
  clipNode = null;
  if (!clipChunks || !clipChunks.length) { clipChunks = null; return null; }
  const total = clipChunks.reduce((sum, part) => sum + part.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const part of clipChunks) { merged.set(part, offset); offset += part.length; }
  clipChunks = null;
  const pcm = new Int16Array(merged.length);
  for (let i = 0; i < merged.length; i += 1) pcm[i] = clamp(merged[i], -1, 1) * 0x7fff | 0;
  return pcm;
}

function sendClip(bytes, format, size, button) {
  const kb = (size / 1024).toFixed(0);
  if (send({ type: 'mic', format, data: base64FromBytes(bytes), size })) {
    log('sys', `clip sent over ws · ${kb} kB · ${format}`);
    button?.classList.remove('busy');
  } else {
    fetch(`${backendOrigin()}/api/listen`, {
      method: 'POST',
      headers: { 'content-type': `audio/${format}` },
      body: bytes,
    }).then(() => {
      log('sys', `clip sent over http · ${kb} kB · ${format}`);
      button?.classList.remove('busy');
    }).catch((error) => {
      log('err', `mic upload failed: ${error.message}`);
      button?.classList.remove('busy');
      setMode('idle');
    });
  }
}

function stopMic() {
  const button = $('btn-mic');
  clearTimeout(holdTimer);
  holding = false;
  state.listening = false;
  button?.classList.remove('armed');
  button?.classList.add('busy');

  if (state.liveStream) { stopLiveStream(); button?.classList.remove('busy'); setMode('thinking', 'closing stream'); return; }

  if (clipNode) {
    const pcm = stopClipCapture();
    if (!pcm || pcm.length * 2 < 3200) {   // under ~0.1 s of 16 kHz mono
      log('warn', 'clip too short — hold the mic button while speaking');
      button?.classList.remove('busy');
      setMode('idle');
      return;
    }
    sendClip(pcm.buffer, 'pcm', pcm.length * 2, button);
    return;
  }

  if (!recorder || recorder.state === 'inactive') { button?.classList.remove('busy'); setMode('idle'); return; }

  recorder.onstop = async () => {
    const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
    chunks = [];
    if (blob.size < 2000) {
      log('warn', 'clip too short — hold the mic button while speaking');
      button?.classList.remove('busy');
      setMode('idle');
      return;
    }
    const format = (recorder.mimeType || 'audio/webm').includes('mp4') ? 'm4a'
      : (recorder.mimeType || '').includes('ogg') ? 'ogg' : 'webm';
    try {
      const bytes = await blob.arrayBuffer();
      const kb = (blob.size / 1024).toFixed(0);
      if (send({ type: 'mic', format, data: base64FromBytes(bytes), size: blob.size })) {
        log('sys', `clip sent over ws · ${kb} kB · ${format}`);
      } else {
        await fetch(`${backendOrigin()}/api/listen`, {
          method: 'POST',
          headers: { 'content-type': `audio/${format}` },
          body: bytes,
        });
        log('sys', `clip sent over http · ${kb} kB · ${format}`);
      }
      button?.classList.remove('busy');
    } catch (error) {
      log('err', `mic upload failed: ${error.message}`);
      button?.classList.remove('busy');
      setMode('idle');
    }
  };
  recorder.stop();
}

function base64FromBytes(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  const CHUNK = 0x8000;                      // stay under the argument-size limit
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

/* ---- live streaming: 16 kHz mono s16le frames over the same WebSocket --- */
function startLiveStream() {
  if (!send({ type: 'stream-start' })) { log('warn', 'live stream needs the core link — using clip mode'); state.liveStream = false; return; }
  const inputRate = audioCtx.sampleRate;
  streamSamples = [];
  const push = (float32) => {
    const down = downsample(float32, inputRate, 16000);
    streamSamples.push(down);
    let total = 0;
    for (const part of streamSamples) total += part.length;
    if (total >= 1600) flushStream();          // ~100 ms frames
  };

  scriptNode = audioCtx.createScriptProcessor(1024, 1, 1);
  scriptNode.onaudioprocess = (event) => push(new Float32Array(event.inputBuffer.getChannelData(0)));
  // ScriptProcessor needs a downstream connection to be pulled, but routing it
  // straight to the speakers would feed back the microphone: use a dead gain.
  const sink = audioCtx.createGain();
  sink.gain.value = 0;
  micSource.connect(scriptNode);
  scriptNode.connect(sink);
  sink.connect(audioCtx.destination);
  log('sys', 'live stream opened · 16 kHz mono pcm_s16le → core VAD');
}

function flushStream() {
  if (!streamSamples || !streamSamples.length) return;
  const total = streamSamples.reduce((sum, part) => sum + part.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const part of streamSamples) { merged.set(part, offset); offset += part.length; }
  streamSamples = [];
  const pcm = new Int16Array(merged.length);
  for (let i = 0; i < merged.length; i += 1) pcm[i] = clamp(merged[i], -1, 1) * 0x7fff | 0;
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(pcm.buffer);
}

function stopLiveStream() {
  flushStream();
  if (scriptNode) { try { micSource.disconnect(scriptNode); scriptNode.disconnect(); } catch { /* already gone */ } }
  scriptNode = null;
  streamSamples = null;
  send({ type: 'stream-stop' });
  setMode('thinking', 'flushing stream');
  setTimeout(() => { if (state.mode === 'thinking') setMode('idle'); }, 2600);
}

function downsample(input, fromRate, toRate) {
  if (fromRate === toRate) return new Float32Array(input);
  const ratio = fromRate / toRate;
  const length = Math.floor(input.length / ratio);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j += 1) sum += input[j];
    out[i] = sum / Math.max(1, end - start);
  }
  return out;
}

/* ---- playback analyser so the reactor reacts to JARVIS's own voice ------ */
let playbackAnalyser = null;
function attachPlaybackAnalyser() {
  try {
    if (!audioCtx) return;
    if (!playbackAnalyser) {
      playbackAnalyser = audioCtx.createMediaElementSource(audio);
      const gain = audioCtx.createGain();
      gain.gain.value = 1;
      playbackAnalyser.connect(gain);
      gain.connect(audioCtx.destination);
      if (analyser) playbackAnalyser.connect(analyser);   // tap own voice for the waveform
    }
  } catch { /* already routed */ }
}

/* ---- waveform strip ---------------------------------------------------- */
const waveCanvas = $('waveform');
const waveCtx = waveCanvas?.getContext('2d');
let timeData = null;

function drawWaveform() {
  if (!waveCtx || !waveCanvas) return;
  const width = waveCanvas.width;
  const height = waveCanvas.height;
  waveCtx.clearRect(0, 0, width, height);
  let level = 0;
  let points = null;

  if (analyser && (state.listening || state.mode === 'speaking')) {
    if (!timeData || timeData.length !== analyser.fftSize) timeData = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(timeData);
    points = timeData;
  }

  waveCtx.lineWidth = Math.max(1, height * 0.035);
  const gradient = waveCtx.createLinearGradient(0, 0, width, 0);
  const tone = { idle: '#155e75', listening: '#34d399', thinking: '#fbbf24', speaking: '#60a5fa', error: '#fb7185', boot: '#155e75' }[state.mode] || '#155e75';
  gradient.addColorStop(0, 'rgba(8,51,68,0.15)');
  gradient.addColorStop(0.5, tone);
  gradient.addColorStop(1, 'rgba(8,51,68,0.15)');
  waveCtx.strokeStyle = gradient;
  waveCtx.shadowColor = tone;
  waveCtx.shadowBlur = 10;

  waveCtx.beginPath();
  const steps = 128;
  for (let i = 0; i <= steps; i += 1) {
    let sample;
    if (points) {
      const index = Math.floor((i / steps) * (points.length - 1));
      sample = (points[index] - 128) / 128;
    } else {
      sample = Math.sin(i * 0.34 + performance.now() / 520) * 0.055 * (0.4 + state.energy);
    }
    level = Math.max(level, Math.abs(sample));
    const x = (i / steps) * width;
    const y = height / 2 + sample * height * 0.46;
    if (i === 0) waveCtx.moveTo(x, y); else waveCtx.lineTo(x, y);
  }
  waveCtx.stroke();
  waveCtx.shadowBlur = 0;
  state.audioLevel = state.audioLevel * 0.72 + level * 0.28;
  requestAnimationFrame(drawWaveform);
}
requestAnimationFrame(drawWaveform);

/* ==========================================================================
   7 · WIRE UP CONTROLS
   ========================================================================== */

$('cmd-input')?.addEventListener('keydown', (event) => {
  const input = event.target;
  if (event.key === 'Enter') {
    event.preventDefault();
    const text = input.value.trim();
    if (text) { input.value = ''; sendCommand(text); }
    return;
  }
  if (event.key === 'ArrowUp' && state.prompts.length) {
    event.preventDefault();
    state.promptIndex = state.promptIndex < 0 ? state.prompts.length - 1 : Math.max(0, state.promptIndex - 1);
    input.value = state.prompts[state.promptIndex] || '';
    input.setSelectionRange(input.value.length, input.value.length);
  }
  if (event.key === 'ArrowDown' && state.prompts.length) {
    event.preventDefault();
    state.promptIndex = Math.min(state.prompts.length, state.promptIndex + 1);
    input.value = state.promptIndex >= state.prompts.length ? '' : state.prompts[state.promptIndex];
  }
});

$('btn-send')?.addEventListener('click', () => {
  const input = $('cmd-input');
  if (input?.value.trim()) { const text = input.value; input.value = ''; sendCommand(text); }
});

$('btn-stop')?.addEventListener('click', stopEverything);
function stopEverything() {
  audio.pause();
  if (recorder && recorder.state === 'recording') stopMic();
  else if (state.liveStream && state.listening) stopLiveStream();
  send({ type: 'stop' });
  setMode('idle', 'interrupted');
  log('warn', 'user interrupted the assistant');
}

const micButton = $('btn-mic');
micButton?.addEventListener('pointerdown', (event) => {
  event.preventDefault();
  if (event.shiftKey) {
    state.liveStream = !state.liveStream;
    log('sys', `live stream mode ${state.liveStream ? 'ON (continuous VAD)' : 'off (single clip)'}`);
    return;
  }
  if (state.listening && tapMode) { stopMic(); return; }
  holding = false;
  clearTimeout(holdTimer);
  holdTimer = setTimeout(() => { holding = true; }, 420);
  startMic(true);
});
window.addEventListener('pointerup', () => {
  clearTimeout(holdTimer);
  if (holding && state.listening) stopMic();
  holding = false;
});
micButton?.addEventListener('contextmenu', (event) => event.preventDefault());

document.querySelectorAll('.chip[data-cmd]').forEach((chip) => {
  chip.addEventListener('click', () => sendCommand(chip.dataset.cmd));
});

$('btn-voice')?.addEventListener('click', (event) => {
  state.voiceOn = !state.voiceOn;
  event.currentTarget.classList.toggle('is-on', state.voiceOn);
  event.currentTarget.classList.toggle('is-off', !state.voiceOn);
  log('sys', `spoken replies ${state.voiceOn ? 'enabled' : 'muted (text only)'}`);
});

$('btn-mirror')?.addEventListener('click', async (event) => {
  const button = event.currentTarget;
  const turningOn = !button.classList.contains('is-on');
  try {
    const result = await rest('/api/discord/mirror', { on: turningOn });
    button.classList.toggle('is-on', !!result.mirror);
    button.classList.toggle('is-off', !result.mirror);
    log('sys', `discord relay ${result.mirror ? 'armed — replies are posted to the bound channel' : 'disarmed'}`);
  } catch (error) {
    log('err', `relay toggle failed: ${error.message}`);
  }
});

$('btn-autoscroll')?.addEventListener('click', (event) => {
  state.autoscroll = !state.autoscroll;
  event.currentTarget.classList.toggle('is-on', state.autoscroll);
});

$('btn-clear')?.addEventListener('click', async () => {
  if (terminal) terminal.replaceChildren();
  terminalLines = 0;
  if (cards) cards.replaceChildren();
  try { await rest('/api/history/clear', {}); } catch { /* core offline is fine */ }
  log('sys', 'terminal and agent context cleared');
});

$('btn-llm-reset')?.addEventListener('click', async () => {
  try {
    const data = await rest('/api/llm/reset', {});
    log('sys', `providers back in rotation: ${(data.cleared || []).join(', ') || 'nothing was cooling'}`);
    await refreshProviders();
  } catch (error) { log('err', `llm reset failed: ${error.message}`); }
});

$('btn-llm-probe')?.addEventListener('click', async () => {
  const button = $('btn-llm-probe');
  if (button) { button.disabled = true; button.textContent = 'probing…'; }
  try {
    const data = await rest('/api/llm/probe', {});
    // probe() answers {ok, providers: {groq: {...}, ...}} -> render one line each
    const rows = Object.entries(data.providers || {});
    rows.forEach(([key, r]) => log(r.ok ? 'sys' : 'warn',
      `probe ${key}: ${r.ok ? `${r.latency_ms ?? '?'}ms on ${r.model || ''} → ${r.reply || 'ok'}` : String(r.error || 'failed').slice(0, 120)}`));
    if (!rows.length) log('sys', 'probe: no provider keys configured yet — nothing to ping');
    await refreshProviders();
  } catch (error) {
    log('err', `probe failed: ${error.message}`);
  } finally {
    if (button) { button.disabled = false; button.textContent = 'probe'; }
  }
});

setInterval(refreshProviders, 15000);
setInterval(refreshDesktop, 25000);
$('btn-ear')?.addEventListener('click', toggleEar);
$('btn-bar')?.addEventListener('click', async () => {
  try {
    const data = await rest('/api/bar', { action: 'toggle' });
    log('sys', `bar: ${data.ok ? (data.visible ? 'shown over your other apps' : 'hidden') : data.message || 'unavailable'}`);
  } catch (error) { log('err', `bar toggle failed: ${error.message}`); }
});
document.addEventListener('visibilitychange', () => { if (!document.hidden) { refreshProviders(); refreshDesktop(); } });
refreshDesktop();

document.addEventListener('keydown', (event) => {
  const typing = /^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName || '');
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'm') {
    event.preventDefault();
    state.listening ? stopMic() : startMic(true);
  }
  if (event.key === 'Escape') { event.preventDefault(); stopEverything(); }
  if (event.key === 'l' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); $('cmd-input')?.focus(); }
  if (event.key === '/' && !typing) { event.preventDefault(); $('cmd-input')?.focus(); }
});

/* clock (HUD-side so it keeps ticking even when the core link is down) */
function tickClock() {
  const el = $('clock');
  if (el) el.textContent = new Date().toLocaleTimeString('en-GB', { hour12: false });
}
setInterval(tickClock, 1000);
tickClock();

/* keep the input focused like a real console, unless the user is elsewhere */
window.addEventListener('load', () => setTimeout(() => $('cmd-input')?.focus(), 400));

/* window controls when running inside pywebview */
if (window.JARVIS_BOOT?.frameless) {
  document.querySelectorAll('.hud-panel').forEach((panel) => {
    panel.addEventListener('dblclick', (event) => {
      if (event.target.closest('button, input, a')) return;
      window.pywebview?.api?.toggle_frameless?.();
    });
  });
}

/* ------------------------------------------------------------------ start */
setMode('boot', 'linking core');
log('sys', 'HUD online — connecting to the JARVIS core…');
connect();

/* One-shot status pull so panels are populated even before the first tick. */
(async () => {
  try {
    const status = await rest('/api/status');
    applyVoiceState(status.voice, status.brain);
    if (status.mode) setMode(status.mode);
  } catch { /* WebSocket hello will supply the same data */ }
})();

/* Debug surface: `JARVIS.say('hello')` from devtools. */
window.JARVIS = {
  send: sendCommand,
  setMode,
  log,
  get state() { return { ...state, history: undefined }; },
  get socket() { return socket; },
  async say(text) { return rest('/api/speak', { text, play: true }); },
  async command(text) { return rest('/api/command', { text, source: 'hud-api' }); },
  reactivate: connect,
  three: { scene, camera, renderer, reactor, torus, core, composer: () => composer },
};

"use strict";

/* Mortred Console — "Lumen" application.
 * Product-grade supervisor front end: KPI strip, GPU telemetry chart,
 * model fleet grid, per-model workbench with test bench and logs,
 * command palette, day/night themes. Zero dependencies, zero build. */

/* ---------------- icons (inline SVG, lucide-style strokes) ---------------- */

const ICONS = {
  grid: '<rect x="3.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="3.5" y="13.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.5"/>',
  cube: '<path d="M12 3 4 7.2v9.6L12 21l8-4.2V7.2L12 3z"/><path d="M4 7.2l8 4.2 8-4.2M12 11.4V21"/>',
  pulse: '<path d="M3 12h4l2.5-6.5 4 13L16 12h5"/>',
  cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="10" y="10" width="4" height="4" rx="1"/><path d="M9 2.5v3M15 2.5v3M9 18.5v3M15 18.5v3M2.5 9h3M2.5 15h3M18.5 9h3M18.5 15h3"/>',
  moon: '<path d="M20 14.5A8 8 0 1 1 9.5 4 6.6 6.6 0 0 0 20 14.5z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4 6 18M18 6l1.4-1.4"/>',
  command: '<path d="M9 9V6a3 3 0 1 0-3 3h3zm0 0v6m0-6h6m-6 6H6a3 3 0 1 0 3 3v-3zm6-6V6a3 3 0 1 1 3 3h-3zm0 0v6m0 0h3a3 3 0 1 1-3 3v-3z"/>',
  key: '<circle cx="7.5" cy="15.5" r="4"/><path d="m10.6 12.4 8.4-8.4M15 4.5 19.5 9M17.5 7l2.5 2.5"/>',
  "chevron-left": '<path d="m14 6-6 6 6 6"/>',
  "chevron-right": '<path d="m10 6 6 6-6 6"/>',
  "chevron-down": '<path d="m6 9 6 6 6-6"/>',
  play: '<path d="M8 5.2v13.6L19 12 8 5.2z"/>',
  square: '<rect x="6" y="6" width="12" height="12" rx="2.5"/>',
  rotate: '<path d="M21 12a9 9 0 1 1-2.64-6.36L21 8"/><path d="M21 3v5h-5"/>',
  upload: '<path d="M12 16V4m0 0 4 4m-4-4-4 4"/><path d="M4 20h16"/>',
  "upload-cloud": '<path d="M6.5 19a4.5 4.5 0 0 1-.4-9A6 6 0 0 1 17.7 9h.3a4.5 4.5 0 0 1 0 9h-2"/><path d="M12 12v7m0-7-2.5 2.5M12 12l2.5 2.5"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"/>',
  send: '<path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4 20-7z"/>',
  x: '<path d="m6 6 12 12M18 6 6 18"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2.5"/><circle cx="9" cy="10" r="1.6"/><path d="m5 19 5-5 3 3 3-3 3 3"/>',
  layers: '<path d="M12 3 2.5 8l9.5 5 9.5-5L12 3z"/><path d="m2.5 13.5 9.5 5 9.5-5"/>',
  terminal: '<path d="m6 8 3.5 3.5L6 15"/><path d="M12 15h6"/>',
  idcard: '<rect x="3" y="5" width="18" height="14" rx="2.5"/><circle cx="9" cy="11" r="2"/><path d="M6.2 16.2c.6-1.8 5-1.8 5.6 0"/><path d="M14.5 10h4M14.5 14h3"/>',
  flask: '<path d="M9.5 3h5"/><path d="M10 3v6l-5.4 8.8A2 2 0 0 0 6.3 21h11.4a2 2 0 0 0 1.7-3.2L14 9V3"/><path d="M8 15h8"/>',
  "check-circle": '<circle cx="12" cy="12" r="9"/><path d="m8.5 12.5 2.5 2.5 4.5-5"/>',
  "alert-circle": '<circle cx="12" cy="12" r="9"/><path d="M12 7.5V13"/><path d="M12 16.5h.01"/>',
  "info-circle": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.5 2"/>',
  tag: '<path d="M3 11V4a1 1 0 0 1 1-1h7l10 10-8 8L3 11z"/><circle cx="7.5" cy="7.5" r="1.2"/>',
  scan: '<path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2"/><rect x="8.5" y="8.5" width="7" height="7" rx="1" stroke-dasharray="2.6 2.4"/>',
  "scan-face": '<path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2"/><path d="M9 10h.01M15 10h.01"/><path d="M9.5 14.5a3.5 3.5 0 0 0 5 0"/>',
  type: '<path d="M4 7V5h16v2"/><path d="M12 5v14"/><path d="M9 19h6"/>',
  wand: '<path d="m15 4 5 5L8 21l-5-5L15 4z"/><path d="m13 6 5 5"/><path d="M8.5 2h.01M3.5 5h.01M2 9.5h.01"/>',
  sparkles: '<path d="M12 4l1.7 4.6L18.3 10.3l-4.6 1.7L12 16.6l-1.7-4.6L5.7 10.3l4.6-1.7L12 4z"/><path d="M18.5 15.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8.8-2.2z"/>',
  crosshair: '<circle cx="12" cy="12" r="7"/><path d="M12 2.5V6M12 18v3.5M2.5 12H6M18 12h3.5"/><circle cx="12" cy="12" r="1.2"/>',
  share: '<circle cx="6" cy="12" r="2.6"/><circle cx="18" cy="6" r="2.6"/><circle cx="18" cy="18" r="2.6"/><path d="m8.4 10.8 7.2-3.6M8.4 13.2l7.2 3.6"/>',
  mountain: '<path d="m3 19 6.5-11 4.5 7.5 2-3.2L21 19H3z"/>',
  shapes: '<circle cx="7.5" cy="7.5" r="3.6"/><rect x="13" y="13" width="7" height="7" rx="1.2"/><path d="M13.5 10.5 17 4.5l3.5 6h-7z"/>',
  aperture: '<circle cx="12" cy="12" r="9"/><path d="m8.5 4.2 7 14M4.6 16.5h14.2M15.5 4.2l-7 14"/>',
  video: '<rect x="3" y="6" width="13" height="12" rx="2.5"/><path d="m16 11 5-3v8l-5-3"/>',
};

function icon(name, size = 16) {
  const body = ICONS[name] || ICONS.cube;
  return '<svg class="ic" width="' + size + '" height="' + size + '" viewBox="0 0 24 24" fill="none"' +
    ' stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    body + "</svg>";
}
function mountStaticIcons() {
  document.querySelectorAll("[data-icon]").forEach((el) => {
    const size = el.classList.contains("be-ic") || el.classList.contains("dz-ic") ? 24 : 16;
    el.innerHTML = icon(el.dataset.icon, size);
  });
}

/* ---------------- category identity ---------------- */

const CAT_META = {
  classification: { color: "#7c3aed", icon: "tag", label: "Classification" },
  object_detection: { color: "#2563eb", icon: "scan", label: "Object detection" },
  face_detection: { color: "#db2777", icon: "scan-face", label: "Face detection" },
  scene_segmentation: { color: "#0891b2", icon: "layers", label: "Scene segmentation" },
  ocr: { color: "#ca8a04", icon: "type", label: "OCR" },
  matting: { color: "#c026d3", icon: "wand", label: "Matting" },
  enhancement: { color: "#ea580c", icon: "sparkles", label: "Enhancement" },
  feature_point: { color: "#0d9488", icon: "crosshair", label: "Feature points" },
  feature_embedding: { color: "#4f46e5", icon: "share", label: "Embedding" },
  mono_depth_estimation: { color: "#65a30d", icon: "mountain", label: "Monocular depth" },
  segment_anything: { color: "#e11d48", icon: "shapes", label: "Segment anything" },
  diffusion: { color: "#9333ea", icon: "aperture", label: "Diffusion" },
  mot: { color: "#0284c7", icon: "video", label: "Multi-object tracking" },
  other: { color: "#64748b", icon: "cube", label: "Model" },
};
function catMeta(c) { return CAT_META[c] || CAT_META.other; }
function catColor(c) { return catMeta(c).color; }

/* maps state -> the .st-dot CSS modifier class (running/starting/backoff/…) */
const STATE_SC = {
  running: "running", starting: "starting", backoff: "backoff",
  failed: "failed", stopped: "",
};
function stateVar(v) { return "var(--" + v + ")"; }
function stateColors() {
  return {
    running: [stateVar("ok"), stateVar("ok-soft"), stateVar("ok-line")],
    starting: [stateVar("warn"), stateVar("warn-soft"), stateVar("warn-line")],
    backoff: [stateVar("warn"), stateVar("warn-soft"), stateVar("warn-line")],
    failed: [stateVar("err"), stateVar("err-soft"), stateVar("err-line")],
    stopped: [stateVar("ink-3"), "var(--inset)", "var(--line)"],
  };
}

/* ---------------- utilities ---------------- */

const $ = (id) => document.getElementById(id);
const TOKEN_KEY = "mortred_supervisor_token";
const THEME_KEY = "mortred_theme";

function uid() { return Math.random().toString(36).slice(2, 10); }
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
const cssVarCache = {};
function cssVar(name) {
  if (!(name in cssVarCache)) {
    cssVarCache[name] = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  return cssVarCache[name];
}
function clearCssVarCache() { for (const k in cssVarCache) delete cssVarCache[k]; }

function fmtMib(m) { if (m == null || m < 0) return "--"; return m >= 1024 ? (m / 1024).toFixed(1) + " GiB" : m + " MiB"; }
function fmtGiB(m) { return (m / 1024).toFixed(1); }
/* window length never leaks floats like 4.8333333 min */
function fmtWindow(seconds) {
  if (seconds < 120) return seconds + " s";
  const min = seconds / 60;
  return (min >= 10 ? Math.round(min) : Math.round(min * 10) / 10) + " min";
}

function uptimeOf(s) {
  if (!s.started_at_ms || s.state !== "running") return null;
  const sec = Math.max(0, Math.floor((Date.now() - s.started_at_ms) / 1000));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60);
  if (h >= 48) return Math.floor(h / 24) + "d " + (h % 24) + "h";   // long-running: compact
  if (h > 0) return h + "h " + String(m).padStart(2, "0") + "m";
  return m + "m " + String(sec % 60).padStart(2, "0") + "s";
}

/* ---------------- theme ---------------- */

function applyTheme(name, animate) {
  if (animate) {
    document.body.classList.add("theme-fade");
    setTimeout(() => document.body.classList.remove("theme-fade"), 320);
  }
  document.documentElement.dataset.theme = name;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = name === "night" ? "#0b0f17" : "#f6f7f9";
  $("btn-theme").querySelector(".sb-ic").innerHTML = icon(name === "night" ? "sun" : "moon");
  clearCssVarCache();
  drawAllCharts();
}
function initTheme() {
  let t = null;
  try { t = localStorage.getItem(THEME_KEY); } catch (e) {}
  if (t !== "day" && t !== "night") {
    t = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "night" : "day";
  }
  applyTheme(t, false);
  $("btn-theme").onclick = () => {
    const next = document.documentElement.dataset.theme === "day" ? "night" : "day";
    try { localStorage.setItem(THEME_KEY, next); } catch (e) {}
    applyTheme(next, true);
  };
}

/* ---------------- token & api ---------------- */

let tokenDialogResolve = null;
function askToken(prefill) {
  return new Promise((resolve) => {
    tokenDialogResolve = resolve;
    $("token-input").value = prefill || "";
    $("token-dialog").classList.remove("hidden");
    $("token-input").focus();
  });
}
function closeTokenDialog(value) {
  $("token-dialog").classList.add("hidden");
  if (tokenDialogResolve) { tokenDialogResolve(value); tokenDialogResolve = null; }
}
function getToken() { try { return localStorage.getItem(TOKEN_KEY) || ""; } catch (e) { return ""; } }
function setToken(t) { try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); } catch (e) {} }

async function authorizedFetch(path, options) {
  options = options || {};
  options.headers = Object.assign({}, options.headers || {});
  const token = getToken();
  if (token) options.headers["Authorization"] = "Bearer " + token;
  let resp = await fetch(path, options);
  if (resp.status === 401) {
    const next = await askToken(token);
    if (next !== null && next.trim()) {
      setToken(next.trim());
      options.headers["Authorization"] = "Bearer " + next.trim();
      resp = await fetch(path, options);
    }
  }
  return resp;
}
async function api(path, options) {
  const resp = await authorizedFetch(path, options);
  let data = null;
  try { data = await resp.json(); } catch (e) {}
  return { ok: resp.ok, status: resp.status, data };
}

/* ---------------- state ---------------- */

const state = {
  servers: [], gateway: null, selectedId: null,
  batchAbort: null, logs: {}, logServerId: null,
  river: [], gpuHistory: [],
  inferHistory: [],
  fleetFilter: "all", fleetState: "all",
  bench: {}, inflight: 0,
  rpsHist: [],
  catalogReady: false,   // first successful catalog+status poll has landed
};

function benchOf(id) { return state.bench[id] || (state.bench[id] = { files: [], results: [] }); }
function serverById(id) { return state.servers.find((s) => s.id === id) || null; }

/* ---------------- toasts ---------------- */

function showToast(msg, type = "info") {
  const el = document.createElement("div");
  el.className = "toast " + type;
  const ic = type === "success" ? "check-circle" : type === "error" ? "alert-circle" : "info-circle";
  el.innerHTML = '<span class="t-ic">' + icon(ic, 16) + "</span><span></span>";
  el.lastElementChild.textContent = msg;
  $("toast-container").appendChild(el);
  setTimeout(() => { el.classList.add("leaving"); }, 2600);
  setTimeout(() => { el.remove(); }, 3000);
}

/* ---------------- activity river ---------------- */

function riverPush(kind, text, serverId) {
  const n = new Date();
  const t = String(n.getHours()).padStart(2, "0") + ":" + String(n.getMinutes()).padStart(2, "0") + ":" + String(n.getSeconds()).padStart(2, "0");
  state.river.push({ t, kind, text, serverId: serverId || null });
  if (state.river.length > 200) state.river.shift();
  const count = $("river-count"); if (count) count.innerHTML = state.river.length + " events";
  const nav = $("nav-activity-count"); if (nav) nav.textContent = state.river.length;
  renderRiver();
}
function renderRiver() {
  const box = $("river-content"); if (!box) return;
  box.innerHTML = state.river.slice(-60).map((e) => {
    const srv = e.serverId
      ? ' <span class="rv-srv" style="--cat:' + catColor((serverById(e.serverId) || {}).category) + '">' + escapeHtml(e.serverId) + "</span>"
      : "";
    return '<div class="river-line kind-' + escapeHtml(e.kind) + '"><span class="river-t">' + e.t + '</span><span class="rv-kind"></span><span class="rv-text">' + escapeHtml(e.text) + "</span>" + srv + "</div>";
  }).join("");
  box.scrollTop = box.scrollHeight;
}

/* ---------------- polling: status / catalog ---------------- */

let prevStates = {};

async function refresh() {
  let cat, st;
  try {
    [cat, st] = await Promise.all([api("/api/v1/catalog"), api("/api/v1/status")]);
  } catch (e) {
    cat = st = { ok: false };
  }
  if (!cat.ok || !st.ok) {
    setConn("err", "link down");
    $("link-overlay").classList.remove("hidden");
    return;
  }
  $("link-overlay").classList.add("hidden");
  setConn("ok", "connected");
  state.catalogReady = true;
  const byId = {}; for (const s of (st.data.servers || [])) byId[s.id] = s;
  state.gateway = st.data.gateway || null;
  state.servers = (cat.data.servers || []).map((s) => Object.assign({}, s, byId[s.id] || {}));
  for (const s of state.servers) {
    const prev = prevStates[s.id];
    if (prev && prev !== s.state) {
      riverPush(s.state === "running" ? "ok" : s.state === "failed" ? "err" : "info", s.id + " " + prev + " → " + s.state, s.id);
    }
    prevStates[s.id] = s.state;
  }
  const liveN = state.servers.filter((s) => ["running", "starting", "backoff"].includes(s.state)).length;
  document.title = (liveN > 0 ? "●" + liveN + " · " : "") + "Mortred Console";
  updateFavicon(liveN);
  renderGateway();
  if (!state._bootLogged) {
    state._bootLogged = true;
    riverPush("ok", "supervisor link established", null);
    riverPush("info", "catalog loaded · " + state.servers.length + " models · " + liveN + " live", null);
    if (state.gpuHistory.length) riverPush("info", "gpu telemetry online", null);
    else setTimeout(() => { if (state.gpuHistory.length) riverPush("info", "gpu telemetry online", null); }, 3000);
  }
  renderCurrentView();
}

function setConn(kind, label) {
  const pill = $("conn-pill");
  pill.className = "conn-pill " + (kind === "ok" ? "ok" : kind === "err" ? "err" : "unknown");
  $("conn-label").textContent = label;
}

function updateFavicon(liveN) {
  const c = document.createElement("canvas"); c.width = c.height = 32;
  const x = c.getContext("2d");
  const grad = x.createLinearGradient(0, 0, 32, 32);
  grad.addColorStop(0, "#6366f1"); grad.addColorStop(1, "#22d3ee");
  x.fillStyle = "#0b0f17";
  x.beginPath();
  if (x.roundRect) { x.roundRect(0, 0, 32, 32, 8); } else { x.rect(0, 0, 32, 32); }
  x.fill();
  const on = liveN > 0;
  const pulse = on ? 0.8 + 0.2 * Math.abs(Math.sin(Date.now() / 700)) : 1;
  x.save();
  x.translate(16, 16); x.scale(pulse, pulse);
  x.fillStyle = grad;
  x.beginPath(); x.arc(0, 0, 8, 0, Math.PI * 2); x.fill();
  x.fillStyle = "#ffffff";
  x.beginPath(); x.arc(0, 0, 2.8, 0, Math.PI * 2); x.fill();
  x.restore();
  let link = document.querySelector("link[rel='icon']");
  if (!link) { link = document.createElement("link"); link.rel = "icon"; document.head.appendChild(link); }
  link.href = c.toDataURL("image/png");
}

function renderGateway() {
  const g = state.gateway;
  const pill = $("gateway-pill");
  if (!g || !g.address) { pill.textContent = "gateway —"; return; }
  let host = g.address.host || "";
  if (host === "0.0.0.0" || host === "::" || host === "[::]") host = window.location.hostname || "127.0.0.1";
  const cls = g.state === "running" ? "gw-ok" : "gw-bad";
  pill.innerHTML = "gateway <span class=\"" + cls + "\">" + (g.state === "running" ? "●" : "○") + "</span> " + escapeHtml(host + ":" + g.address.port) + (g.state === "running" ? "" : " down");
}

/* ---------------- polling: gpu ---------------- */

async function pollGpu() {
  const r = await api("/api/v1/gpu");
  const panel = $("sec-gpu"); if (!panel) return;
  if (!r.ok || !r.data || !r.data.available) {
    panel.classList.add("gpu-na");
    $("gpu-name").textContent = "GPU offline";
    $("gpu-meta").textContent = "";
    $("gpu-chips").innerHTML = "";
    return;
  }
  panel.classList.remove("gpu-na");
  state.gpuHistory = r.data.samples || [];
  $("gpu-name").textContent = r.data.name || "GPU";
  const winS = Math.round(state.gpuHistory.length * 2);
  $("gpu-meta").textContent = state.gpuHistory.length
    ? "1 gpu · last " + fmtWindow(winS) + " · 2 s poll · hover to inspect"
    : "";
  const last = state.gpuHistory.length ? state.gpuHistory[state.gpuHistory.length - 1] : null;
  if (last) { updateGpuChips(last); updateKpis(); updateGpuGauge(last); }
  drawGpuHero();
  updateWorkbenchGpu();
}

function updateGpuGauge(last) {
  const arc = $("gauge-util"), val = $("gauge-val");
  if (!arc || !val) return;
  const C = 2 * Math.PI * 54;
  const util = (last.util != null && last.util >= 0) ? Math.max(0, Math.min(100, last.util)) : null;
  if (util == null) { val.textContent = "—"; return; }
  arc.style.strokeDasharray = C.toFixed(1);
  arc.style.strokeDashoffset = (C * (1 - util / 100)).toFixed(1);
  tweenKpi("gauge", util, (v) => { val.textContent = Math.round(v) + "%"; });
}

function updateGpuChips(last) {
  const chips = [
    { label: "VRAM", text: last.mem_total_mib > 0 ? fmtGiB(last.mem_used_mib) + " / " + fmtGiB(last.mem_total_mib) + " GiB" : "--",
      hot: last.mem_total_mib > 0 && last.mem_used_mib / last.mem_total_mib > 0.85 },
    { label: "Temp", text: last.temp_c >= 0 ? Math.round(last.temp_c) + "°C" : "--", hot: last.temp_c > 80 },
    { label: "Power", text: last.power_w >= 0 ? Math.round(last.power_w) + " W" : "--", hot: last.power_w > 300 },
    { label: "SM clock", text: last.clocks_sm_mhz >= 0 ? Math.round(last.clocks_sm_mhz) + " MHz" : "--", hot: false },
    { label: "Fan", text: last.fan_pct >= 0 ? Math.round(last.fan_pct) + "%" : "--", hot: false },
  ];
  $("gpu-chips").innerHTML = chips.map((c) =>
    '<span class="stat-chip' + (c.hot ? " hot" : "") + '"><i>' + c.label + "</i>" + escapeHtml(c.text) + "</span>").join("");
  const lastS = state.gpuHistory[state.gpuHistory.length - 1];
  if (lastS) {
    const lgU = $("lg-util"), lgM = $("lg-mem"), lgP = $("lg-pwr");
    if (lgU) lgU.textContent = lastS.util >= 0 ? Math.round(lastS.util) + "%" : "—";
    if (lgM) lgM.textContent = lastS.mem_total_mib > 0 ? Math.round(100 * lastS.mem_used_mib / lastS.mem_total_mib) + "%" : "—";
    if (lgP) lgP.textContent = lastS.power_w >= 0 ? Math.round(lastS.power_w) + " W" : "—";
  }
}

function updateWorkbenchGpu() {
  const box = $("wb-gpu"); if (!box) return;
  const s = state.gpuHistory;
  const last = s.length ? s[s.length - 1] : null;
  box.classList.toggle("hidden", !last);
  if (!last) return;
  const set = (id, txt, hot) => {
    const el = $(id); if (!el) return;
    el.textContent = txt;
    el.parentElement.classList.toggle("hot", !!hot);
  };
  set("wg-util", last.util >= 0 ? last.util + "%" : "--", last.util > 85);
  set("wg-vram", last.mem_total_mib > 0 ? Math.round(100 * last.mem_used_mib / last.mem_total_mib) + "%" : "--",
    last.mem_total_mib > 0 && last.mem_used_mib / last.mem_total_mib > 0.85);
  set("wg-temp", last.temp_c >= 0 ? Math.round(last.temp_c) + "°C" : "--", last.temp_c > 80);
  const cv = $("wg-spark");
  if (cv) {
    const data = s.slice(-40).map((x) => x.util < 0 ? 0 : x.util);
    drawSparkline(cv, data.length >= 2 ? data : null, cssVar("--brand"));
  }
}

/* ---------------- KPI strip ---------------- */

/* numeric tween: KPI values count toward their target so the page feels alive */
const kpiTween = {};
function tweenKpi(key, target, render) {
  const from = (kpiTween[key] != null && isFinite(kpiTween[key])) ? kpiTween[key] : target;
  kpiTween[key] = target;
  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced || from === target) { render(target); return; }
  const t0 = performance.now(), dur = 420;
  const step = (t) => {
    const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    render(from + (target - from) * e);
    if (k < 1) requestAnimationFrame(step); else render(target);
  };
  requestAnimationFrame(step);
}

function setKpi(name, html) {
  const el = document.querySelector("[data-kpi=\"" + name + "\"]");
  if (el) el.innerHTML = html;
}
function drawLiveRing(live, total) {
  const ring = document.querySelector(".live-ring .ring-value");
  if (!ring) return;
  const C = 2 * Math.PI * 19;
  const pct = total > 0 ? Math.max(0, Math.min(1, live / total)) : 0;
  ring.style.strokeDasharray = C.toFixed(1);
  ring.style.strokeDashoffset = (C * (1 - pct)).toFixed(1);
  ring.parentElement.parentElement.style.opacity = total > 0 ? "" : "0";
}

function updateKpis() {
  const live = state.servers.filter((s) => ["running", "starting", "backoff"].includes(s.state));
  const failed = state.servers.filter((s) => s.state === "failed").length;
  tweenKpi("live", live.length, (v) => setKpi("live", Math.round(v)));
  setKpi("live-sub", " of " + state.servers.length + (failed ? " · " + failed + " failed" : ""));
  drawLiveRing(live.length, state.servers.length);

  const g = state.gpuHistory;
  const last = g.length ? g[g.length - 1] : null;
  if (last && last.util != null && last.util >= 0) {
    const ref = g.length > 16 ? g[g.length - 17] : null;
    const d = ref && ref.util >= 0 ? last.util - ref.util : null;
    tweenKpi("util", last.util, (v) => setKpi("util", Math.round(v) + '<span class="kpi-u">%</span>' +
      (d == null ? "" : ' <span class="trend-pill ' + (d >= 0 ? "up" : "down") + '">' + (d >= 0 ? "▲" : "▼") + Math.abs(Math.round(d)) + "</span>")));
    const c = document.querySelector('canvas[data-kpi-spark="util"]');
    if (c) drawSparkline(c, g.slice(-40).map((x) => x.util < 0 ? 0 : x.util), cssVar("--brand"));
  } else { setKpi("util", "—"); }

  if (last && last.mem_total_mib > 0) {
    setKpi("vram", fmtGiB(last.mem_used_mib) + ' <span class="kpi-u">/ ' + fmtGiB(last.mem_total_mib) + " GiB</span>");
    $("kpi-vram-fill").style.width = Math.min(100, 100 * last.mem_used_mib / last.mem_total_mib).toFixed(1) + "%";
  } else { setKpi("vram", "—"); }

  const sub = $("ov-sub");
  if (sub) {
    const n = new Date();
    sub.textContent = state.servers.length + " models · " + live.length + " running · updated " +
      String(n.getHours()).padStart(2, "0") + ":" + String(n.getMinutes()).padStart(2, "0") + ":" + String(n.getSeconds()).padStart(2, "0");
  }
  const fc = $("nav-fleet-count"); if (fc) fc.textContent = state.servers.length;
}

function updateAggregateRps() {
  let sum = 0, live = 0;
  for (const s of state.servers) {
    if (s.state !== "running") continue;
    const q = fleetQps[s.id];
    if (q && q.qps != null) { sum += q.qps; live++; }
  }
  const running = state.servers.filter((s) => ["running", "starting", "backoff"].includes(s.state)).length;
  if (live) {
    state.rpsHist.push(sum);
    if (state.rpsHist.length > 60) state.rpsHist.shift();
    tweenKpi("rps", sum, (v) => setKpi("rps", v.toFixed(1)));
  } else {
    // warm-up window: models are up but the first sample has not landed yet —
    // show a truthful 0.0 rather than an empty tile
    setKpi("rps", running ? '<span style="color:var(--ink-3)">0.0</span>' : "—");
  }

  const rc = document.querySelector('canvas[data-kpi-spark="rps"]');
  if (rc && state.rpsHist.length >= 2) {
    drawSparkline(rc, state.rpsHist.slice(-40), cssVar("--brand"));
  } else if (rc && running) {
    drawSparkline(rc, null, cssVar("--brand"));
  }
  const chips = $("ov-chips");
  if (chips) {
    let html = '<span class="head-chip"><span class="dot"></span><b>' + running + "</b>&nbsp;live</span>";
    if (live) html += '<span class="head-chip">Σ <b>' + sum.toFixed(1) + "</b>&nbsp;req/s</span>";
    const failed = state.servers.filter((s) => s.state === "failed").length;
    if (failed) html += '<span class="head-chip err"><b>' + failed + "</b>&nbsp;failed</span>";
    chips.innerHTML = html;
  }
  const fl = $("fleet-live");
  if (fl) fl.innerHTML = "<b>" + running + "</b> live" + (live ? " · Σ " + sum.toFixed(1) + " req/s" : "");
}

/* ---------------- charts ---------------- */

const SANS_STACK = '-apple-system, "Segoe UI", system-ui, sans-serif';

function smoothPathThrough(ctx, pts) {
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) {
    const p0 = pts[i - 1], p1 = pts[i];
    const prev = pts[i - 2] || p0, next = pts[i + 1] || p1;
    const c1x = p0[0] + (p1[0] - prev[0]) / 6, c1y = p0[1] + (p1[1] - prev[1]) / 6;
    const c2x = p1[0] - (next[0] - p0[0]) / 6, c2y = p1[1] - (next[1] - p0[1]) / 6;
    ctx.bezierCurveTo(c1x, c1y, c2x, c2y, p1[0], p1[1]);
  }
}

function drawGpuHero() {
  const cv = $("gpu-canvas"); if (!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight; if (w === 0) return;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const s = state.gpuHistory; if (s.length < 2) return;
  const n = s.length, step = w / (n - 1), pad = 8;
  const norm = (v, lo, hi) => Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
  const yOf = (f) => h - pad - f * (h - pad * 2);

  // horizontal grid (dual-scale chart: gridlines only, values live at line ends)
  ctx.strokeStyle = cssVar("--chart-grid");
  ctx.lineWidth = 1;
  for (let g2 = 1; g2 < 4; g2++) {
    const gy = Math.round(h * g2 / 4) + 0.5;
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
  }

  // power (dim, dashed)
  {
    const pts = s.map((x, i) => [i * step, yOf(norm(x.power_w < 0 ? 0 : x.power_w, 0, 400))]);
    ctx.beginPath(); smoothPathThrough(ctx, pts);
    ctx.strokeStyle = cssVar("--chart-pwr");
    ctx.lineWidth = 1.4;
    ctx.setLineDash([5, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  // vram (violet)
  let vramLabel = null;
  {
    const pts = s.map((x, i) => [i * step, yOf(norm(x.mem_total_mib > 0 ? x.mem_used_mib / x.mem_total_mib : 0, 0, 1))]);
    ctx.beginPath(); smoothPathThrough(ctx, pts);
    ctx.strokeStyle = cssVar("--chart-mem");
    ctx.lineWidth = 1.6;
    ctx.stroke();
    const lastVram = s[s.length - 1];
    if (lastVram && lastVram.mem_total_mib > 0) {
      vramLabel = { text: Math.round(100 * lastVram.mem_used_mib / lastVram.mem_total_mib) + "%", color: cssVar("--chart-mem"), y: pts[pts.length - 1][1] };
    }
  }
  // utilization (brand gradient, filled)
  let utilLabel = null;
  {
    const pts = s.map((x, i) => [i * step, yOf(norm(x.util < 0 ? 0 : x.util, 0, 100))]);
    ctx.beginPath(); smoothPathThrough(ctx, pts);
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, cssVar("--grad-a"));
    grad.addColorStop(1, cssVar("--grad-b"));
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.4;
    ctx.lineJoin = "round";
    ctx.stroke();
    // gradient fill under the curve
    const fillPath = new Path2D();
    fillPath.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) {
      const p0 = pts[i - 1], p1 = pts[i];
      const prev = pts[i - 2] || p0, next = pts[i + 1] || p1;
      fillPath.bezierCurveTo(p0[0] + (p1[0] - prev[0]) / 6, p0[1] + (p1[1] - prev[1]) / 6,
        p1[0] - (next[0] - p0[0]) / 6, p1[1] - (next[1] - p0[1]) / 6, p1[0], p1[1]);
    }
    fillPath.lineTo(pts[pts.length - 1][0], h); fillPath.lineTo(pts[0][0], h); fillPath.closePath();
    const fg = ctx.createLinearGradient(0, 0, 0, h);
    fg.addColorStop(0, "rgba(99,102,241,0.20)");
    fg.addColorStop(1, "rgba(99,102,241,0)");
    ctx.fillStyle = fg;
    ctx.fill(fillPath);
    // end dot
    const e = pts[pts.length - 1];
    ctx.beginPath(); ctx.arc(e[0] - 1, e[1], 3.4, 0, Math.PI * 2);
    ctx.fillStyle = cssVar("--grad-b");
    ctx.fill();
    ctx.beginPath(); ctx.arc(e[0] - 1, e[1], 6.5, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(34,211,238,0.35)"; ctx.lineWidth = 2; ctx.stroke();
    const lastUtil = s[s.length - 1];
    if (lastUtil && lastUtil.util >= 0) {
      utilLabel = { text: Math.round(lastUtil.util) + "%", color: cssVar("--brand"), y: e[1] };
    }
  }
  // end-value labels: collision-aware — when the two series converge at the
  // right edge their labels would stack on top of each other; offset instead
  if (utilLabel && vramLabel && Math.abs(utilLabel.y - vramLabel.y) < 15) {
    // util keeps the spot nearer its line; push vram below (above if clipped)
    vramLabel.y = vramLabel.y + 15 <= h - 4 ? vramLabel.y + 15 : vramLabel.y - 15;
  }
  if (vramLabel) endValueLabel(ctx, w, vramLabel.y, vramLabel.text, vramLabel.color, h);
  if (utilLabel) endValueLabel(ctx, w, utilLabel.y, utilLabel.text, utilLabel.color, h);
}

function endValueLabel(ctx, w, y, text, color, maxBottom) {
  ctx.font = "650 10.5px " + SANS_STACK;
  ctx.textAlign = "right";
  ctx.fillStyle = color;
  const ty = Math.min(Math.max(11, y - 7), maxBottom - 4);
  ctx.fillText(text, w - 6, ty);
  ctx.textAlign = "left";
}

function wireGpuCrosshair() {
  const cv = $("gpu-canvas"); if (!cv) return;
  const hair = $("gpu-crosshair"), tip = $("gpu-tip");
  const fmt = (v, u) => v == null || v < 0 ? "--" : Math.round(v) + u;
  cv.addEventListener("mousemove", (ev) => {
    const s = state.gpuHistory; if (s.length < 2) return;
    const rect = cv.getBoundingClientRect();
    const x = ev.clientX - rect.left, w = rect.width;
    const idx = Math.max(0, Math.min(s.length - 1, Math.round((x / w) * (s.length - 1))));
    const d = s[idx];
    hair.classList.remove("hidden");
    hair.style.left = ((idx / (s.length - 1)) * w) + "px";
    tip.classList.remove("hidden");
    tip.innerHTML =
      '<div class="tip-hero">' + fmt(d.util, "%") + '<small>util</small></div>' +
      '<div class="row"><span class="k">vram</span><span>' + (d.mem_total_mib > 0 ? Math.round(100 * d.mem_used_mib / d.mem_total_mib) + "%" : "--") + "</span></div>" +
      '<div class="row"><span class="k">power</span><span>' + fmt(d.power_w, " W") + "</span></div>" +
      '<div class="row"><span class="k">temp</span><span>' + fmt(d.temp_c, "°C") + "</span></div>";
    const tipW = tip.offsetWidth;
    const left = Math.max(0, Math.min(w - tipW, (idx / (s.length - 1)) * w + 12));
    tip.style.left = left + "px";
  });
  cv.addEventListener("mouseleave", () => {
    hair.classList.add("hidden"); tip.classList.add("hidden");
  });
}

function drawSparkline(canvas, data, color) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight; if (!w || !h) return;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (!data || data.length < 2) {
    // designed idle baseline — an empty tile still reads as "alive, no traffic"
    ctx.strokeStyle = cssVar("--line-2");
    ctx.lineWidth = 1;
    ctx.setLineDash([2.5, 4]);
    ctx.beginPath();
    ctx.moveTo(2, h - 3.5);
    ctx.lineTo(w - 2, h - 3.5);
    ctx.stroke();
    ctx.setLineDash([]);
    return;
  }
  const max = Math.max(...data, 1);
  const step = w / (data.length - 1);
  const pts = data.map((v, i) => [i * step, h - 3 - (v / max) * (h - 7)]);
  ctx.beginPath(); smoothPathThrough(ctx, pts);
  ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.lineJoin = "round";
  ctx.stroke();
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  ctx.fillStyle = color + "1e";
  ctx.fill();
  const e = pts[pts.length - 1];
  ctx.beginPath(); ctx.arc(Math.min(e[0] - 3, w - 3), e[1], 2.2, 0, Math.PI * 2);
  ctx.fillStyle = color; ctx.fill();
}

function drawAllCharts() {
  drawGpuHero();
  drawFleetSparks();
  updateWorkbenchGpu();
  updateKpis();
}

/* ---------------- router ---------------- */

function currentRoute() {
  const h = location.hash.replace(/^#/, "");
  const m = h.match(/^\/model\/([A-Za-z0-9_-]+)/);
  if (m) return { view: "model", id: m[1] };
  return { view: "overview" };
}
function navigate(h) { if (location.hash !== h) location.hash = h; }
window.addEventListener("hashchange", () => renderCurrentView());

function renderCurrentView() {
  const r = currentRoute();
  if (r.view === "model" && serverById(r.id)) {
    state.selectedId = r.id; showView("workbench"); renderWorkbench();
  } else {
    // deep links land here for the first ~2s before the catalog arrives —
    // only report "not found" once we actually have a catalog to check against
    if (r.view === "model" && state.catalogReady) showToast("Model not found: " + r.id, "info");
    state.selectedId = null; showView("overview"); renderOverview();
  }
}
function showView(n) {
  const el = n === "overview" ? $("view-overview") : $("view-workbench");
  const other = n === "overview" ? $("view-workbench") : $("view-overview");
  const changed = el.classList.contains("hidden");
  other.classList.add("hidden");
  el.classList.remove("hidden");
  // only replay the entrance when the view actually switched — the 2s poll
  // re-renders constantly and must not re-trigger the animation
  if (changed) { el.classList.remove("enter"); void el.offsetWidth; el.classList.add("enter"); }
}

/* ---------------- sidebar nav ---------------- */

function wireNav() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.onclick = () => {
      setNavActive(btn.dataset.anchor); // instant feedback; scrollspy refines
      if (currentRoute().view !== "overview") { navigate("#/overview"); }
      requestAnimationFrame(() => {
        const sec = document.getElementById(btn.dataset.anchor);
        if (sec) sec.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    };
  });
  // the page scrolls on the window, not on the view element
  window.addEventListener("scroll", updateNavActive, { passive: true });
  window.addEventListener("resize", () => { drawAllCharts(); });
}
function setNavActive(anchor) {
  document.querySelectorAll(".nav-item").forEach((b) => {
    b.classList.toggle("active", b.dataset.anchor === anchor);
  });
}
function updateNavActive() {
  const view = $("view-overview");
  if (!view || view.classList.contains("hidden")) { setNavActive(null); return; }
  // scrollspy: the active section is the LAST one whose top crossed the line
  const ids = ["sec-kpi", "sec-fleet", "sec-activity"];
  let best = ids[0];
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el && el.getBoundingClientRect().top <= 160) best = id;
  }
  setNavActive(best);
}

/* ---------------- OVERVIEW ---------------- */

function renderOverview() {
  updateKpis();
  updateAggregateRps();
  renderFleetControls();
  const shown = state.servers.filter((s) =>
    (state.fleetFilter === "all" || s.category === state.fleetFilter) &&
    (state.fleetState === "all" || fleetStateOf(s) === state.fleetState));
  const grid = $("fleet-grid");
  let tiles = grid._tiles; if (!tiles) { tiles = grid._tiles = new Map(); }
  const want = new Set(shown.map((s) => s.id));
  for (const [id, el] of tiles) { if (!want.has(id)) { el.remove(); tiles.delete(id); } }
  for (const s of shown) { upsertModelCard(tiles, grid, s); }
  const orderKey = shown.map((s) => s.id).join(",");
  if (grid._order !== orderKey) {
    for (const s of shown) { const el = tiles.get(s.id); if (el) grid.appendChild(el); }
    grid._order = orderKey;
  }
  const emptyHint = grid.querySelector(".fleet-empty");
  const needEmpty = !shown.length;
  if (needEmpty && !emptyHint) {
    const e = document.createElement("div"); e.className = "fleet-empty";
    e.textContent = "No models match the current filters"; grid.appendChild(e);
  } else if (!needEmpty && emptyHint) { emptyHint.remove(); }
  drawFleetSparks();
  updateNavActive();
}

function fleetStateOf(s) {
  if (s.state === "failed") return "failed";
  if (s.state === "stopped") return "stopped";
  if (s.state === "running") return "running";
  return "starting";
}

function upsertModelCard(tiles, grid, s) {
  const meta = catMeta(s.category);
  const isRun = s.state === "running";
  const sig = [s.state, s.ready ? 1 : 0, s.restart_count, s.port, s.name, s.type].join("|");
  let tile = tiles.get(s.id);
  if (!tile) {
    tile = document.createElement("div");
    tile.className = "model-card";
    tile.setAttribute("role", "button");
    tile.setAttribute("tabindex", "0");
    tile.innerHTML =
      '<div class="mc-row"><span class="cat-icon"></span><div class="mc-titles" style="flex:1;min-width:0"><span class="mc-name"></span><span class="mc-id"></span></div></div>' +
      '<div class="mc-state"></div>' +
      '<div class="mc-spark"><canvas class="mc-spark-cv" data-id="' + escapeHtml(s.id) + '"></canvas></div>' +
      '<div class="spark-tag"></div>';
    const open = () => navigate("#/model/" + s.id);
    tile.onclick = open;
    tile.onkeydown = (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); open(); } };
    tiles.set(s.id, tile);
    grid.appendChild(tile);
    tile._sig = "";
  }
  tile.setAttribute("aria-label", s.id + ", " + s.state + ", port " + s.port);
  tile.className = "model-card" + (isRun ? " live" : "") + (s.state === "failed" ? " dead" : "");
  tile.style.setProperty("--cat", meta.color);
  if (tile._sig === sig) return;
  tile._sig = sig;
  tile.querySelector(".cat-icon").innerHTML = icon(meta.icon, 18);
  tile.querySelector(".mc-name").textContent = s.name || s.id;
  tile.querySelector(".mc-id").textContent = s.id + (s.type ? " · " + String(s.type).toUpperCase() : "");
  const dotCls = STATE_SC[s.state] || "";
  tile.querySelector(".mc-state").innerHTML =
    '<span class="st-dot ' + dotCls + '"></span><span class="st-label">' +
    (isRun ? 'running · <span class="uptime" data-id="' + escapeHtml(s.id) + '">' + (uptimeOf(s) || "booting") + "</span>"
      : escapeHtml(s.state)) + "</span>" +
    (s.restart_count > 0 ? '<span class="badge restarts" title="' + s.restart_count + ' restarts">↻ ' + s.restart_count + "</span>" : "") +
    '<span class="mc-port">:' + s.port + "</span>";
}

function renderFleetControls() {
  const statesBox = $("fleet-states"), catsBox = $("fleet-filters");
  if (!statesBox || !catsBox) return;
  const nBy = { running: 0, starting: 0, stopped: 0, failed: 0 };
  for (const s of state.servers) nBy[fleetStateOf(s)]++;
  const counts = {};
  for (const s of state.servers) counts[s.category] = (counts[s.category] || 0) + 1;
  const sig = JSON.stringify([state.fleetState, state.fleetFilter, nBy, counts, state.servers.length]);
  if (statesBox._sig === sig) return;
  statesBox._sig = sig;
  const mkChip = (label, n, on, onclick, extra) => {
    const c = document.createElement("button");
    c.type = "button";
    c.className = "chip" + (extra || "") + (on ? " on" : "");
    c.innerHTML = escapeHtml(label) + '<span class="n">' + n + "</span>";
    c.onclick = onclick;
    return c;
  };
  statesBox.innerHTML = "";
  const cap1 = document.createElement("span"); cap1.className = "fleet-cap"; cap1.textContent = "Status";
  statesBox.appendChild(cap1);
  statesBox.appendChild(mkChip("All", state.servers.length, state.fleetState === "all",
    () => { state.fleetState = "all"; renderOverview(); }));
  for (const [key, v] of [["running", "ok"], ["starting", "warn"], ["failed", "err"], ["stopped", "ink"]]) {
    const c = mkChip(key, nBy[key], state.fleetState === key,
      () => { state.fleetState = key; renderOverview(); }, " st-chip");
    c.style.setProperty("--sc", "var(--" + v + ")");
    c.style.setProperty("--sc-soft", "var(--" + v + "-soft)");
    c.style.setProperty("--sc-line", "var(--" + v + "-line)");
    if (v === "ink") { c.style.setProperty("--sc", "var(--ink-3)"); c.style.setProperty("--sc-soft", "var(--inset)"); c.style.setProperty("--sc-line", "var(--line-2)"); }
    statesBox.appendChild(c);
  }
  catsBox.innerHTML = "";
  const cap2 = document.createElement("span"); cap2.className = "fleet-cap"; cap2.textContent = "Category";
  catsBox.appendChild(cap2);
  catsBox.appendChild(mkChip("All", state.servers.length, state.fleetFilter === "all",
    () => { state.fleetFilter = "all"; renderOverview(); }));
  for (const cat of Object.keys(counts).sort()) {
    const c = mkChip(catMeta(cat).label, counts[cat], state.fleetFilter === cat,
      () => { state.fleetFilter = cat; renderOverview(); });
    c.style.setProperty("--cat-glow", catColor(cat));
    if (state.fleetFilter === cat) {
      c.style.color = catColor(cat);
      c.style.borderColor = catColor(cat) + "66";
      c.style.background = catColor(cat) + "14";
    }
    catsBox.appendChild(c);
  }
}

function drawFleetSparks() {
  for (const s of state.servers) {
    if (state.fleetFilter !== "all" && s.category !== state.fleetFilter) continue;
    if (state.fleetState !== "all" && fleetStateOf(s) !== state.fleetState) continue;
    const cv = document.querySelector("canvas.mc-spark-cv[data-id=\"" + s.id + "\"]");
    if (!cv) continue;
    const tag = cv.closest(".model-card") && cv.closest(".model-card").querySelector(".spark-tag");
    const live = fleetQps[s.id];
    if (live && live.hist && live.hist.length >= 2) {
      drawSparkline(cv, live.hist.slice(-30), catColor(s.category));
      if (tag) {
        tag.className = "spark-tag live";
        tag.innerHTML = '<span class="tag-dot"></span>live · ' + (live.qps != null ? live.qps.toFixed(1) + " req/s" : "…");
      }
      continue;
    }
    const data = state.inferHistory.filter((x) => x.serverId === s.id).slice(-30).map((x) => x.ms);
    drawSparkline(cv, data.length >= 2 ? data : null, catColor(s.category));
    if (tag) {
      tag.className = "spark-tag";
      tag.innerHTML = data.length >= 2
        ? "session · " + data.length + " calls"
        : '<span class="idle-tag">no traffic</span>';
    }
  }
}

/* ---------------- WORKBENCH ---------------- */

function renderWorkbench() {
  const s = serverById(state.selectedId); if (!s) return;
  const sig = JSON.stringify([s.id, s.state, s.ready, s.restart_count, s.pid, s.port, s.uri, s.category, s.name, s.type]);
  if (state._wbSig !== sig) {
    state._wbSig = sig;
    const meta = catMeta(s.category);
    const sc = stateColors()[s.state] || stateColors().stopped;
    const catIcon = $("wb-cat-icon");
    catIcon.style.setProperty("--cat", meta.color);
    catIcon.innerHTML = icon(meta.icon, 22);
    $("wb-title").textContent = s.name || s.id;
    $("wb-id").textContent = s.id + " · " + (s.type ? String(s.type).toUpperCase() : meta.label) + " · :" + s.port;
    const pill = $("wb-pill");
    pill.textContent = s.state + (s.ready ? " · ready" : "");
    pill.style.setProperty("--sc", sc[0]);
    pill.style.setProperty("--sc-soft", sc[1]);
    pill.style.setProperty("--sc-line", sc[2]);
    $("wb-identity").innerHTML =
      idRow("Category", '<span class="cat-v" style="--cat:' + meta.color + '">' + escapeHtml(meta.label) + "</span>") +
      idRow("Gateway URI", '<span class="id-v">' + escapeHtml(s.uri || "") + "</span>") +
      idRow("Port", '<span class="id-v">:' + s.port + "</span>") +
      idRow("Restarts", '<span class="id-v">' + (s.restart_count || 0) + "</span>") +
      idRow("Uptime", '<span class="id-v uptime" data-id="' + s.id + '">' + (uptimeOf(s) || "--") + "</span>") +
      idRow("PID", '<span class="id-v">' + (s.pid > 0 ? s.pid : "—") + "</span>");
    const running = ["running", "starting", "backoff"].includes(s.state);
    $("wb-start").disabled = running;
    $("wb-restart").disabled = !running;
    $("wb-stop").disabled = !running;
    $("wb-start").onclick = () => controlServer(s.id, "start");
    $("wb-restart").onclick = () => controlServer(s.id, "restart");
    $("wb-stop").onclick = () => controlServer(s.id, "stop");
    $("image-input-area").classList.remove("hidden");
    $("bench-empty").classList.add("hidden");
    const note = $("bench-note");
    if (note) note.textContent = "requests go through the gateway";
    renderFileList();
    renderResultsList(s.id);
    updateSessionStats(s.id);
  }
  state.logServerId = s.id;
  if (!state.logs[s.id]) { resetLogState(s.id); }
  syncLogSelector();
  pollProcessInfo(s.id);
  pollServerMetrics(s.id);
  updateWorkbenchGpu();
  // keep an open switcher in sync with the 2s poll (states/current marker)
  if (!$("wb-switch-menu").classList.contains("hidden")) renderModelSwitcher();
}

function idRow(k, vHtml) {
  return '<div class="id-row"><span class="id-k">' + k + '</span><span style="min-width:0;flex:1">' + vHtml + "</span></div>";
}

/* ---------------- model switcher (workbench) ---------------- */

/* jump between models without returning to the fleet grid */
function openModelSwitcher() {
  renderModelSwitcher();
  $("wb-switch-menu").classList.remove("hidden");
  $("wb-switch-btn").classList.add("on");
}
function closeModelSwitcher() {
  $("wb-switch-menu").classList.add("hidden");
  $("wb-switch-btn").classList.remove("on");
}
function renderModelSwitcher() {
  const menu = $("wb-switch-menu");
  const order = { running: 0, starting: 1, backoff: 1, stopped: 2, failed: 3 };
  const list = [...state.servers].sort((a, b) =>
    ((order[a.state] != null ? order[a.state] : 2) - (order[b.state] != null ? order[b.state] : 2)) ||
    String(a.id).localeCompare(String(b.id)));
  menu.innerHTML = list.map((s) => {
    const meta = catMeta(s.category);
    const dot = STATE_SC[s.state] || "";
    return '<button class="sw-row' + (s.id === state.selectedId ? " cur" : "") + '" data-id="' + escapeHtml(s.id) + '">' +
      '<span class="cat-icon xs" style="--cat:' + meta.color + '">' + icon(meta.icon, 14) + "</span>" +
      '<span class="sw-name">' + escapeHtml(s.name || s.id) + "</span>" +
      '<span class="sw-id">' + escapeHtml(s.id) + "</span>" +
      '<span class="st-dot ' + dot + '"></span>' +
      "</button>";
  }).join("");
  menu.querySelectorAll(".sw-row").forEach((row) => {
    row.onclick = () => {
      closeModelSwitcher();
      navigate("#/model/" + row.dataset.id);
    };
  });
}
function cycleModel(dir) {
  const r = currentRoute();
  if (r.view !== "model" || !state.servers.length) return;
  const ids = state.servers.map((s) => s.id);
  const i = ids.indexOf(r.id);
  const next = ids[(i + dir + ids.length) % ids.length];
  if (next && next !== r.id) navigate("#/model/" + next);
}

async function pollProcessInfo(id) {
  const r = await api("/api/v1/servers/" + encodeURIComponent(id) + "/process");
  const box = $("wb-process"); if (!box) return;
  if (!r.ok || !r.data || !r.data.process) { if (box._sig !== "none") { box.innerHTML = "<div class=\"empty-inline\">process not running</div>"; box._sig = "none"; } return; }
  const p = r.data.process;
  const sig = p.rss_kb + "|" + p.threads;
  if (box._sig === sig) return;
  box._sig = sig;
  box.innerHTML =
    idRow("Memory", '<span class="id-v">' + (p.rss_kb > 0 ? (p.rss_kb / 1024 / 1024).toFixed(2) + " GiB" : "--") + "</span>") +
    idRow("Threads", '<span class="id-v">' + (p.threads > 0 ? p.threads : "--") + "</span>");
}

/* ---------------- server telemetry ---------------- */

function promSum(text, name) {
  let sum = 0;
  const re = new RegExp("^" + name + "(?:\\{[^}]*\\})?\\s+([0-9.eE+-]+)", "gm");
  let m;
  while ((m = re.exec(text)) !== null) sum += parseFloat(m[1]);
  return sum;
}
function promGauge(text, name) {
  const m = new RegExp("^" + name + "(?:\\{[^}]*\\})?\\s+([0-9.eE+-]+)", "m").exec(text);
  return m ? parseFloat(m[1]) : null;
}
function promQuantile(text, histName, q) {
  const re = new RegExp("^" + histName + "_bucket\\{[^}]*le=\"([0-9.eE+]+)\"\\}\\s+([0-9.eE+-]+)", "gm");
  const buckets = [];
  let m;
  while ((m = re.exec(text)) !== null) buckets.push([parseFloat(m[1]), parseFloat(m[2])]);
  if (buckets.length < 2) return null;
  buckets.sort((a, b) => a[0] - b[0]);
  const total = buckets[buckets.length - 1][1];
  if (total <= 0) return null;
  const rank = q * total;
  for (let i = 0; i < buckets.length; i++) {
    if (buckets[i][1] >= rank) {
      const prev = i === 0 ? [0, 0] : buckets[i - 1];
      const span = buckets[i][1] - prev[1];
      if (span <= 0) return buckets[i][0];
      const frac = (rank - prev[1]) / span;
      return Math.round(prev[0] + frac * (buckets[i][0] - prev[0]));
    }
  }
  return null;
}

const metricsPrev = {};
async function pollServerMetrics(id) {
  const box = $("wb-servermetrics");
  if (!box || state.selectedId !== id) return;
  const now0 = Date.now();
  if (metricsPrev[id] && now0 - metricsPrev[id].t < 4000) return;
  let text = null, ok = false;
  try {
    const resp = await authorizedFetch("/api/v1/servers/" + encodeURIComponent(id) + "/metrics");
    if (resp.ok) { text = await resp.text(); ok = typeof text === "string" && text.length > 0; }
  } catch (e) { ok = false; }
  if (!ok) {
    if (box._sig !== "na") { box.innerHTML = "<div class=\"empty-inline\">metrics unavailable</div>"; box._sig = "na"; }
    return;
  }
  const now = Date.now();
  const inferTotal = promSum(text, "mortred_inference_requests_total");
  const prev = metricsPrev[id];
  let qps = null;
  if (prev && now > prev.t && inferTotal >= prev.total) {
    qps = (inferTotal - prev.total) / ((now - prev.t) / 1000);
  }
  metricsPrev[id] = { t: now, total: inferTotal };
  const rows = [
    ["Throughput", qps != null ? qps.toFixed(2) + " req/s" : "—"],
    ["p50 latency", (v => v != null ? v + " ms" : "—")(promQuantile(text, "mortred_inference_duration_ms", 0.5))],
    ["p95 latency", (v => v != null ? v + " ms" : "—")(promQuantile(text, "mortred_inference_duration_ms", 0.95))],
    ["Workers busy", (v => v != null ? v : "—")(promGauge(text, "mortred_workers_busy"))],
    ["Workers idle", (v => v != null ? v : "—")(promGauge(text, "mortred_workers_available"))],
    ["Queue depth", (v => v != null ? v : "—")(promGauge(text, "mortred_queue_depth"))],
    ["Waiting jobs", (v => v != null ? v : "—")(promGauge(text, "mortred_waiting_jobs"))],
    ["Finished jobs", (v => v != null ? Math.round(v).toLocaleString() : "—")(promGauge(text, "mortred_finished_jobs_total"))],
  ];
  const sig = JSON.stringify(rows);
  if (box._sig === sig) return;
  box._sig = sig;
  box.innerHTML = rows.map(([k, v]) => idRow(k, '<span class="id-v">' + escapeHtml(String(v)) + "</span>")).join("");
}

/* ---------------- fleet-wide live qps ---------------- */

const fleetQps = {};
async function pollFleetMetrics() {
  if (document.hidden) return;
  for (const s of state.servers) {
    if (s.state !== "running") { delete fleetQps[s.id]; continue; }
    try {
      const resp = await authorizedFetch("/api/v1/servers/" + encodeURIComponent(s.id) + "/metrics");
      if (!resp.ok) { continue; }
      const text = await resp.text();
      const total = promSum(text, "mortred_inference_requests_total");
      const now = Date.now();
      const p = fleetQps[s.id] || { hist: [] };
      if (p.t && now > p.t && total >= p.total) {
        p.qps = (total - p.total) / ((now - p.t) / 1000);
        p.hist.push(p.qps);
        if (p.hist.length > 40) p.hist.shift();
      }
      p.t = now; p.total = total;
      fleetQps[s.id] = p;
    } catch (e) { /* keep last sample */ }
    await new Promise((r) => setTimeout(r, 250));
  }
  updateAggregateRps();
}

setInterval(() => {
  document.querySelectorAll(".uptime").forEach((el) => {
    const s = serverById(el.dataset.id);
    if (s) el.textContent = uptimeOf(s) || "--";
  });
}, 1000);
setInterval(() => { if (state.selectedId) pollProcessInfo(state.selectedId); }, 5000);
setInterval(() => { if (state.selectedId) pollServerMetrics(state.selectedId); }, 5000);

/* ---------------- gateway & control ---------------- */

function gatewayBaseUrl() {
  const g = state.gateway; if (!g || !g.address) return "";
  let host = g.address.host || "";
  if (host === "0.0.0.0" || host === "::" || host === "[::]") host = window.location.hostname || "127.0.0.1";
  return "http://" + host + ":" + g.address.port;
}
async function controlServer(id, action) {
  if (action === "stop") {
    const okToStop = await askConfirm("Stop " + id + "?",
      "In-flight requests through the gateway will fail while this model is down. You can start it again at any time.",
      "Stop model");
    if (!okToStop) return;
  }
  const r = await api("/api/v1/servers/" + encodeURIComponent(id) + "/" + action, { method: "POST" });
  if (!r.ok) { showToast(action + " failed: HTTP " + r.status, "error"); riverPush("err", id + " " + action + " HTTP " + r.status, id); }
  else { showToast(action + " " + id, "success"); riverPush("ok", id + " " + action + " ok", id); }
  refresh();
}

/* styled confirm dialog for destructive actions (resolves false on cancel) */
let confirmResolve = null;
function askConfirm(title, body, okLabel) {
  return new Promise((resolve) => {
    confirmResolve = resolve;
    $("confirm-title").textContent = title;
    $("confirm-body").textContent = body;
    $("confirm-ok").textContent = okLabel || "Confirm";
    $("confirm-dialog").classList.remove("hidden");
    $("confirm-ok").focus();
  });
}
function closeConfirm(value) {
  $("confirm-dialog").classList.add("hidden");
  if (confirmResolve) { confirmResolve(value); confirmResolve = null; }
}

/* ---------------- test bench ---------------- */

function base64ToSrc(b) { return "data:image/png;base64," + b.replace(/^data:[^,]+,/, ""); }
function loadImageAsBase64(f) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => { resolve(r.result.slice(r.result.indexOf(",") + 1)); };
    r.onerror = reject; r.readAsDataURL(f);
  });
}
/* folder picks (webkitdirectory) and some Linux/WSL drag paths hand us files
 * with an EMPTY File.type — accept by extension too or the whole folder
 * except one lucky MIME-carrying file gets silently dropped */
const IMAGE_EXT = /\.(png|jpe?g|bmp|gif|webp|ppm|pgm|pbm|tif|tiff|avif)$/i;
function isImageFile(f) {
  if (f.type && f.type.startsWith("image/")) return true;
  return IMAGE_EXT.test(f.name || "");
}
async function addFiles(fl) {
  const bench = benchOf(state.selectedId);
  if (!bench) return;
  const files = [...fl].filter(isImageFile);
  for (const f of files) {
    const b64 = await loadImageAsBase64(f);
    bench.files.push({ name: f.name, url: base64ToSrc(b64), base64: b64, status: "ready" });
  }
  renderFileList();
  if (files.length) showToast("Added " + files.length + (files.length === 1 ? " image" : " images") + " to the bench", "success");
  else showToast("No images found in the selection", "info");
}
/* a whole folder can queue hundreds of chips — cap the wall, keep the queue */
const FILE_CHIP_CAP = 12;
function renderFileList() {
  const box = $("file-list"); if (!box) return;
  const bench = benchOf(state.selectedId);
  box.innerHTML = "";
  const shown = bench.files.slice(0, FILE_CHIP_CAP);
  for (const f of shown) {
    const chip = document.createElement("div");
    chip.className = "file-chip" + (f.status ? " " + f.status : "");
    chip.innerHTML = '<img src="' + f.url + '" alt=""><span class="fc-name">' + escapeHtml(f.name) + "</span>" +
      (f.status === "sending" ? '<span class="chip-st">sending…</span>' :
        f.status === "failed" ? '<span class="chip-st" title="send failed — stays queued for retry">failed</span>' : "");
    const rm = document.createElement("span");
    rm.textContent = "✕"; rm.className = "chip-rm";
    rm.onclick = () => { bench.files = bench.files.filter((x) => x !== f); renderFileList(); };
    chip.appendChild(rm); box.appendChild(chip);
  }
  const rest = bench.files.length - shown.length;
  if (rest > 0) {
    const more = document.createElement("div");
    more.className = "fc-more";
    more.textContent = "+ " + rest + " more queued";
    box.appendChild(more);
  }
}
function unifiedPayload(d) {
  if (d && Array.isArray(d.results) && d.results.length) return d.results[0].data;
  return d && d.data !== undefined ? d.data : null;
}
function topScore(p) {
  if (!p) return null;
  if (Array.isArray(p) && p.length && typeof p[0].score === "number") return p[0].score;
  return null;
}
function percentile(sorted, q) {
  if (!sorted.length) return null;
  const i = Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)));
  return sorted[i];
}
function updateSessionStats(id) {
  const hist = state.inferHistory.filter((x) => x.serverId === id);
  const ok = hist.filter((x) => x.ok).length, fail = hist.length - ok;
  const lat = hist.filter((x) => x.ok).map((x) => x.ms).sort((a, b) => a - b);
  const p50 = percentile(lat, 0.5), p95 = percentile(lat, 0.95);
  $("ws-sent").textContent = hist.length;
  $("ws-ok").textContent = ok;
  $("ws-fail").textContent = fail;
  $("ws-fail").style.color = fail ? "var(--err)" : "";
  $("ws-p50").textContent = p50 != null ? p50 + " ms" : "—";
  $("ws-p95").textContent = p95 != null ? p95 + " ms" : "—";
  $("ws-inflight").textContent = state.inflight;
  $("ws-inflight").style.color = state.inflight ? "var(--warn)" : "";
}

async function sendBatch() {
  const s = serverById(state.selectedId); if (!s) return;
  const bench = benchOf(s.id);
  const queue = bench.files.filter((f) => f.status !== "sending");
  if (!queue.length) { showToast("Nothing queued — upload images first", "info"); return; }
  const base = gatewayBaseUrl();
  if (!base) { showToast("Gateway address unknown", "error"); return; }
  const inferUrl = base + "/v1/models/" + encodeURIComponent(s.id) + "/infer";
  try {
    const pre = await fetch(base + "/healthz", { method: "GET", signal: AbortSignal.timeout(3000) });
    if (!pre.ok) throw new Error("gateway healthz returned " + pre.status);
  } catch (e) {
    const detail = e && e.name === "TimeoutError" ? "timeout after 3s" : (e && e.message) || String(e);
    showToast("Gateway unreachable: " + detail + "\n→ " + base + "\nCheck the tunnel (8080) and the gateway process", "error");
    riverPush("err", "pre-flight fail: " + base + " (" + detail + ")", s.id);
    return;
  }
  $("btn-cancel").classList.remove("hidden"); setBatchProgress(0, queue.length);
  let aborted = false; state.batchAbort = () => { aborted = true; };
  let done = 0;
  for (const f of queue) {
    if (aborted) break;
    f.status = "sending"; renderFileList();
    const reqId = uid(), t0 = performance.now();
    state.inflight++; updateSessionStats(s.id);
    try {
      const resp = await authorizedFetch(inferUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ req_id: reqId, images: [f.base64] }),
      });
      const ms = Math.round(performance.now() - t0);
      const body = await resp.json().catch(() => null);
      state.inferHistory.push({ t: Date.now(), serverId: s.id, ms, ok: resp.ok });
      if (state.inferHistory.length > 600) state.inferHistory.shift();
      if (resp.ok) {
        f.status = "done";
        bench.files = bench.files.filter((x) => x !== f);
        addResult(s, f, body, reqId, ms);
        riverPush("ok", s.id + " 200 " + ms + "ms", s.id);
      } else {
        f.status = "failed";
        showToast("HTTP " + resp.status, "error");
        riverPush("err", s.id + " HTTP " + resp.status, s.id);
      }
    } catch (e) {
      f.status = "failed";
      state.inferHistory.push({ t: Date.now(), serverId: s.id, ms: Math.round(performance.now() - t0), ok: false });
      if (state.inferHistory.length > 600) state.inferHistory.shift();
      const detail = e && e.message ? e.message : String(e);
      showToast("Inference failed: " + detail + "\n→ " + inferUrl, "error");
      riverPush("err", s.id + " transport: " + detail, s.id);
    }
    state.inflight--; updateSessionStats(s.id);
    done++; setBatchProgress(done, queue.length);
    renderFileList();
  }
  state.batchAbort = null; $("btn-cancel").classList.add("hidden");
  drawFleetSparks();
}

function setBatchProgress(d, t) {
  $("batch-progress").classList.toggle("hidden", t === 0);
  $("batch-progress-fill").style.width = t ? ((d / t) * 100).toFixed(1) + "%" : "0";
  $("batch-progress-text").textContent = d + " / " + t;
}

function addResult(server, input, result, reqId, elapsed) {
  const bench = benchOf(server.id);
  bench.results.unshift({ input, result, reqId, elapsed });
  if (bench.results.length > 20) bench.results.length = 20;
  if (state.selectedId === server.id) {
    const card = buildResultCard(server, input, result, reqId, elapsed);
    $("results-list").prepend(card);
  }
}
function renderResultsList(id) {
  const box = $("results-list"); if (!box) return;
  const server = serverById(id); if (!server) return;
  box.innerHTML = "";
  if (!benchOf(id).results.length) {
    const e = document.createElement("div");
    e.className = "empty-inline";
    e.textContent = "No results yet — send an image to see model output here.";
    box.appendChild(e);
    return;
  }
  for (const r of benchOf(id).results) {
    box.appendChild(buildResultCard(server, r.input, r.result, r.reqId, r.elapsed));
  }
}
function buildResultCard(server, input, result, reqId, elapsed) {
  const payload = unifiedPayload(result);
  const card = document.createElement("div"); card.className = "result-card";
  const score = topScore(payload);
  const detCount = Array.isArray(payload) ? payload.length : 0;
  const kind = resultKind(payload);
  card.innerHTML =
    '<div class="head"><div class="head-main">' +
    '<span class="req-name">' + escapeHtml(input.name) + "</span>" +
    '<span class="head-chips">' +
    '<span class="chiplet">' + elapsed + " ms</span>" +
    (detCount ? '<span class="chiplet">' + detCount + " " + kind + "</span>" : "") +
    (score != null ? '<span class="chiplet acc">top ' + score.toFixed(3) + "</span>" : "") +
    "</span></div>" +
    '<span class="req-meta">req ' + escapeHtml(reqId) + "</span></div>";
  const vizWrap = document.createElement("div"); vizWrap.className = "viz"; card.appendChild(vizWrap);
  const raw = document.createElement("details"); raw.className = "raw-json"; raw.innerHTML = "<summary>raw response</summary>";
  const pre = document.createElement("pre"); pre.textContent = JSON.stringify(result, null, 1); raw.appendChild(pre);
  card.appendChild(raw);
  visualize(server, input, payload, vizWrap).catch(() => {});
  return card;
}
function resultKind(payload) {
  if (Array.isArray(payload) && payload.length) {
    if (payload[0].bbox) return "hits";
    if (payload[0].location) return "points";
    if (payload[0].polygon) return "regions";
  }
  return "";
}

async function visualize(server, input, payload, vizWrap) {
  const img = new Image(); img.src = input.url;
  await img.decode().catch(() => {});
  if (payload && typeof payload === "object" && !Array.isArray(payload) && payload.category !== undefined && Array.isArray(payload.scores)) {
    vizWrap.appendChild(img);
    const box = document.createElement("div"); box.className = "cls-result";
    const scores = (payload.scores || []).slice(0, 5);
    const max = scores.length ? scores[0] : 1;
    box.innerHTML = '<span class="cls-cat">' + escapeHtml(payload.category) + "</span>" +
      scores.map((s) => '<span class="cls-score"><i style="width:' + Math.max(4, Math.round(100 * s / (max || 1))) + '%"></i><b>' + (+s).toFixed(3) + "</b></span>").join("");
    vizWrap.appendChild(box);
    return;
  }
  if (payload && typeof payload === "object" && !Array.isArray(payload) && Array.isArray(payload.embedding)) {
    vizWrap.appendChild(img);
    let norm = 0; const v = payload.embedding;
    for (let i = 0; i < v.length; i += 1) norm += v[i] * v[i];
    const box = document.createElement("div"); box.className = "cls-result";
    box.innerHTML = '<span class="cls-cat">embedding</span><span class="cls-dim">dim=' + (payload.dim != null ? payload.dim : v.length) + " · ‖v‖=" + Math.sqrt(norm).toFixed(2) + "</span>";
    vizWrap.appendChild(box);
    return;
  }
  if (Array.isArray(payload) && payload.length && payload[0].segmentation !== undefined && payload[0].predicted_iou !== undefined) {
    const cv = document.createElement("canvas"); cv.className = "overlay"; vizWrap.appendChild(cv);
    drawDetection(cv, img.src, payload.map((m) => ({ bbox: m.bbox, score: m.predicted_iou, category: "mask", stability: m.stability_score, area: m.area })), true);
    appendLegend(vizWrap, [{ hue: 172, label: payload.length + " masks · iou top " + (payload[0].predicted_iou != null ? (+payload[0].predicted_iou).toFixed(2) : "--") }]);
    return;
  }
  if (Array.isArray(payload) && payload.length && typeof payload[0].bbox === "object") {
    const cv = document.createElement("canvas"); cv.className = "overlay"; vizWrap.appendChild(cv);
    drawDetection(cv, img.src, payload);
    const seen = new Set(); const legend = [];
    for (const b of payload) {
      if (seen.has(b.class_id)) continue;
      seen.add(b.class_id);
      legend.push({ hue: (b.class_id * 47) % 360, label: (b.category || b.class_id) + " " + (b.score != null ? b.score.toFixed(2) : "") });
    }
    appendLegend(vizWrap, legend);
    return;
  }
  if (Array.isArray(payload) && payload.length && Array.isArray(payload[0].location)) {
    const cv = document.createElement("canvas"); cv.className = "overlay"; vizWrap.appendChild(cv);
    drawFeaturePoints(cv, img.src, payload);
    appendLegend(vizWrap, [{ hue: 150, label: payload.length + " points · top " + (payload[0].score != null ? payload[0].score.toFixed(2) : "--") }]);
    return;
  }
  if (Array.isArray(payload) && payload.length && Array.isArray(payload[0].polygon)) {
    const cv = document.createElement("canvas"); cv.className = "overlay"; vizWrap.appendChild(cv);
    drawOcr(cv, img.src, payload);
    return;
  }
  if (payload && Array.isArray(payload.regions)) {
    const cv = document.createElement("canvas"); cv.className = "overlay"; vizWrap.appendChild(cv);
    drawOcr(cv, img.src, payload.regions);
    return;
  }
  if (payload && (payload.image || payload.colorized_mask || payload.alpha || payload.matting_image)) {
    const out = payload.image || payload.colorized_mask || payload.alpha || payload.matting_image;
    const im = new Image(); im.src = base64ToSrc(out); vizWrap.appendChild(im);
    return;
  }
  if (payload && Array.isArray(payload.keypoints)) {
    const cv = document.createElement("canvas"); cv.className = "overlay"; vizWrap.appendChild(cv);
    drawKeypoints(cv, img.src, payload);
    return;
  }
  vizWrap.appendChild(img);
}

function appendLegend(vizWrap, items) {
  const legend = document.createElement("div"); legend.className = "det-legend";
  legend.innerHTML = items.map((it) =>
    '<span class="det-chip" style="--h:' + it.hue + '"><span class="det-dot"></span>' + escapeHtml(it.label) + "</span>").join("");
  vizWrap.appendChild(legend);
}

function drawFeaturePoints(canvas, imgUrl, points) {
  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const base = Math.max(1.2, img.naturalWidth / 420);
    for (const p of points) {
      const [x, y] = p.location;
      const score = p.score != null ? p.score : 0.5;
      const r = base * (0.9 + score * 1.8);
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = "#4f46e5"; ctx.fill();
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = base * 0.6; ctx.stroke();
    }
  };
  img.src = imgUrl;
}
function drawDetection(canvas, imgUrl, boxes, isMasks) {
  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    ctx.lineWidth = Math.max(2, Math.round(img.naturalWidth / 300));
    for (const b of boxes) {
      const [x1, y1, x2, y2] = b.bbox;
      const bw = Math.max(1, x2 - x1), bh = Math.max(1, y2 - y1);
      const fpx = Math.max(11, Math.min(Math.round(img.naturalWidth / 40), Math.round(bw / 3.5), Math.round(bh / 2)));
      ctx.font = fpx + "px ui-monospace, monospace";
      const hue = isMasks ? 172 : ((b.class_id * 47) % 360);
      const stroke = isMasks ? "#0d9488" : "hsl(" + hue + " 85% 45%)";
      ctx.strokeStyle = stroke;
      ctx.strokeRect(x1, y1, bw, bh);
      let label;
      if (isMasks) {
        label = "mask " + (b.score != null ? (+b.score).toFixed(2) : "") + (b.area ? " · " + b.area + "px" : "");
      } else {
        label = (b.category || b.class_id) + " " + (b.score != null ? b.score.toFixed(2) : "");
      }
      const tw = ctx.measureText(label).width + fpx * 0.7;
      const plateH = Math.round(fpx * 1.4);
      const outside = y1 - plateH - 2 >= 0 && bh >= plateH * 1.8;
      const py = outside ? y1 - plateH - 2 : Math.min(y1 + 2, img.naturalHeight - plateH);
      ctx.fillStyle = stroke;
      ctx.fillRect(x1, py, tw, plateH);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x1 + fpx * 0.35, py + plateH - fpx * 0.32);
      if (Array.isArray(b.landmarks) && b.landmarks.length) {
        ctx.fillStyle = "#f59e0b";
        for (const p of b.landmarks) {
          ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(2, img.naturalWidth / 240), 0, Math.PI * 2); ctx.fill();
        }
      }
    }
  };
  img.src = imgUrl;
}
function drawOcr(canvas, imgUrl, regions) {
  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    ctx.lineWidth = 2; ctx.strokeStyle = "#d97706";
    for (const r of regions) {
      const pts = r.polygon || r.points || r.bbox_points || [];
      if (pts.length >= 3) {
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.closePath(); ctx.stroke();
      }
      const label = r.text || (r.score != null ? r.score.toFixed(2) : "");
      if (label) {
        ctx.fillStyle = "#b45309"; ctx.font = "600 14px ui-monospace, monospace";
        ctx.fillText(label, pts[0] ? pts[0][0] : 4, pts[0] ? pts[0][1] - 4 : 14);
      }
    }
  };
  img.src = imgUrl;
}
function drawKeypoints(canvas, imgUrl, payload) {
  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    ctx.fillStyle = "#4f46e5";
    for (const p of payload.keypoints) {
      ctx.beginPath(); ctx.arc(p.x, p.y, 3.4, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke();
    }
  };
  img.src = imgUrl;
}

/* ---------------- logs ---------------- */

function resetLogState(id) {
  state.logs[id] = { offset: 0, filter: "", paused: false, follow: true, lines: [] };
  const el = $("log-content"); if (el) el.textContent = "";
}
function syncLogSelector() {
  const sel = $("log-server"); if (!sel) return;
  const ids = state.servers.map((s) => s.id);
  const sig = ids.join(",") + "|" + state.logServerId;
  if (sel._sig === sig) return;
  sel._sig = sig;
  sel.innerHTML = "";
  for (const id of ids) {
    const opt = document.createElement("option"); opt.value = id; opt.textContent = id; sel.appendChild(opt);
  }
  if (state.logServerId) sel.value = state.logServerId;
}
function renderLogContent() {
  const el = $("log-content"); if (!el || !state.logServerId) return;
  const st = state.logs[state.logServerId]; if (!st) return;
  const lines = st.filter ? st.lines.filter((l) => l.toLowerCase().includes(st.filter)) : st.lines;
  el.innerHTML = lines.slice(-400).map((l) => highlightLine(l, st.filter)).join("\n");
  if (st.follow) el.scrollTop = el.scrollHeight;
}
function highlightLine(line, filter) {
  const esc = escapeHtml(line);
  if (/\bERROR\b|FATAL/.test(line)) return '<span class="log-err">' + esc + "</span>";
  if (!filter) return esc;
  const idx = esc.toLowerCase().indexOf(filter);
  if (idx < 0) return esc;
  return esc.slice(0, idx) + "<mark>" + esc.slice(idx, idx + filter.length) + "</mark>" + esc.slice(idx + filter.length);
}
async function pollLogs() {
  if (!state.logServerId) return;
  const st = state.logs[state.logServerId]; if (!st || st.paused) return;
  const r = await api("/api/v1/servers/" + encodeURIComponent(state.logServerId) + "/logs?offset=" + st.offset + "&limit=100");
  if (!r.ok || !r.data) return;
  st.lines.push(...(r.data.lines || []));
  st.offset = r.data.offset != null ? r.data.offset + (r.data.lines || []).length : st.lines.length;
  $("log-meta").textContent = st.lines.length + " lines";
  renderLogContent();
  for (const l of (r.data.lines || [])) {
    if (/\bERROR\b|FATAL/.test(l)) riverPush("err", l.slice(0, 120), state.logServerId);
  }
}

/* ---------------- command palette ---------------- */

const PALETTE_RECENT_KEY = "mortred_palette_recent";
function paletteRecent() {
  try { return JSON.parse(localStorage.getItem(PALETTE_RECENT_KEY) || "[]").slice(0, 3); } catch (e) { return []; }
}
function paletteRemember(label) {
  const rec = paletteRecent().filter((x) => x !== label);
  rec.unshift(label);
  try { localStorage.setItem(PALETTE_RECENT_KEY, JSON.stringify(rec.slice(0, 3))); } catch (e) {}
}
function buildPaletteItems() {
  const nav = [], ctrl = [];
  for (const s of state.servers) {
    nav.push({ label: "open " + s.id, hint: "nav", ic: "chevron-right", act: () => navigate("#/model/" + s.id) });
    const running = ["running", "starting", "backoff"].includes(s.state);
    if (!running) ctrl.push({ label: "start " + s.id, hint: "ctrl", ic: "play", act: () => controlServer(s.id, "start") });
    if (running) ctrl.push({ label: "stop " + s.id, hint: "ctrl", ic: "square", act: () => controlServer(s.id, "stop") });
    ctrl.push({ label: "restart " + s.id, hint: "ctrl", ic: "rotate", act: () => controlServer(s.id, "restart") });
  }
  nav.push({ label: "overview", hint: "nav", ic: "grid", act: () => navigate("#/overview") });
  const set = [
    { label: "toggle theme", hint: "set", ic: "moon", act: () => $("btn-theme").click() },
    { label: "api token", hint: "set", ic: "key", act: async () => {
      const t = await askToken(getToken());
      if (t !== null && t.trim()) { setToken(t.trim()); location.reload(); }
    } },
  ];
  return nav.concat(ctrl, set);
}
function paletteItems() {
  const all = buildPaletteItems();
  const items = paletteRecent()
    .map((label) => ({ found: all.find((it) => it.label === label), label }))
    .filter((x) => x.found)
    .map((x) => ({ label: x.label, hint: "recent", ic: "clock", act: x.found.act }));
  return items.concat(all);
}
function fuzzyMatch(q, l) {
  let li = 0, sc = 0, streak = 0;
  const Q = q.toLowerCase(), L = l.toLowerCase();
  for (const ch of Q) {
    const f = L.indexOf(ch, li); if (f < 0) return -1;
    streak = f === li ? streak + 1 : 0;
    sc += 1 + streak; li = f + 1;
  }
  return sc;
}
let paletteStagger = false;
function openPalette() {
  const p = $("palette");
  p.classList.remove("hidden");
  paletteStagger = true;
  const input = $("palette-input");
  input.value = ""; renderPalette("");
  input.focus();
  paletteStagger = false;
}
function closePalette() { $("palette").classList.add("hidden"); }
function renderPalette(query) {
  const list = $("palette-list");
  const items = paletteItems()
    .map((it) => ({ it, sc: query ? fuzzyMatch(query, it.label) : 0 }))
    .filter((x) => x.sc >= 0).sort((a, b) => b.sc - a.sc).slice(0, 10);
  let sel = 0, idx = 0;
  list.innerHTML = "";
  const GROUP_LABEL = { recent: "Recent", nav: "Navigate", ctrl: "Control", set: "System" };
  let lastGroup = null;
  for (const { it } of items) {
    if (!query && it.hint !== lastGroup) {
      lastGroup = it.hint;
      const g = document.createElement("div"); g.className = "palette-group";
      g.textContent = GROUP_LABEL[it.hint] || it.hint;
      list.appendChild(g);
    }
    const row = document.createElement("div"); row.className = "palette-row";
    if (paletteStagger) { row.classList.add("stagger"); row.style.animationDelay = (idx++ * 16) + "ms"; }
    row.innerHTML = '<span class="pl-main"><span class="pl-ic">' + icon(it.ic || "chevron-right", 15) + '</span><span class="pl-label">' + escapeHtml(it.label) + "</span></span>" +
      '<span class="palette-hint">' + escapeHtml(it.hint) + "</span>";
    row.onclick = () => { closePalette(); paletteRemember(it.label); it.act(); };
    list.appendChild(row);
  }
  if (!items.length) {
    const none = document.createElement("div"); none.className = "palette-empty";
    none.textContent = "No matching commands"; list.appendChild(none);
  }
  const firstRow = list.querySelector(".palette-row");
  if (firstRow) firstRow.classList.add("sel");
  $("palette-input").onkeydown = (ev) => {
    const rows = [...list.querySelectorAll(".palette-row")];
    if (ev.key === "Escape") closePalette();
    else if (ev.key === "Enter") { if (rows[sel]) { closePalette(); items[sel].it.act(); } }
    else if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      ev.preventDefault();
      if (rows[sel]) rows[sel].classList.remove("sel");
      sel = ev.key === "ArrowDown" ? Math.min(rows.length - 1, sel + 1) : Math.max(0, sel - 1);
      if (rows[sel]) rows[sel].classList.add("sel");
    } else setTimeout(() => renderPalette($("palette-input").value), 0);
  };
}

/* ---------------- keyboard ---------------- */

document.addEventListener("keydown", (ev) => {
  const inInput = document.activeElement && /input|textarea|select/i.test(document.activeElement.tagName);
  // palette first: ⌘K/Ctrl+K must never be swallowed by single-key nav (the
  // 'k' in the modifier combo used to fall into the j/k branch and return)
  if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
    ev.preventDefault();
    $("palette").classList.contains("hidden") ? openPalette() : closePalette();
    return;
  }
  if (ev.key === "Escape") {
    if (!$("wb-switch-menu").classList.contains("hidden")) closeModelSwitcher();
    if (!$("palette").classList.contains("hidden")) closePalette();
    if (!$("token-dialog").classList.contains("hidden")) closeTokenDialog(null);
    if (!$("confirm-dialog").classList.contains("hidden")) closeConfirm(false);
    return;
  }
  if (inInput) return;
  const noMod = !ev.metaKey && !ev.ctrlKey && !ev.altKey;
  if (noMod && (ev.key === "j" || ev.key === "k")) {
    const view = $("view-overview");
    if (!view || view.classList.contains("hidden")) return;
    const cards = [...document.querySelectorAll(".model-card")];
    if (!cards.length) return;
    ev.preventDefault();
    const cur = cards.indexOf(document.activeElement);
    const next = ev.key === "j"
      ? cards[Math.min(cards.length - 1, cur < 0 ? 0 : cur + 1)]
      : cards[Math.max(0, cur < 0 ? 0 : cur - 1)];
    if (next) next.focus();
    return;
  }
  // [ / ] cycle through the fleet without going back to the grid
  if (noMod && (ev.key === "[" || ev.key === "]")) {
    cycleModel(ev.key === "]" ? 1 : -1);
  }
});

/* ---------------- wiring ---------------- */

function wire() {
  mountStaticIcons();
  initTheme();
  wireNav();
  wireGpuCrosshair();

  $("wb-back").onclick = () => navigate("#/overview");
  $("wb-switch-btn").onclick = (ev) => {
    ev.stopPropagation();
    $("wb-switch-menu").classList.contains("hidden") ? openModelSwitcher() : closeModelSwitcher();
  };
  // close the switcher when clicking anywhere else
  document.addEventListener("click", (ev) => {
    if (!$("wb-switch-menu").classList.contains("hidden") &&
        !ev.target.closest(".wb-switch")) closeModelSwitcher();
  });
  $("btn-palette").onclick = () => { $("palette").classList.contains("hidden") ? openPalette() : closePalette(); };
  $("btn-token").onclick = async () => {
    const t = await askToken(getToken());
    if (t !== null) { setToken(t.trim()); showToast("Token saved", "success"); refresh(); }
  };
  $("token-save").onclick = () => closeTokenDialog($("token-input").value);
  $("token-cancel").onclick = () => closeTokenDialog(null);
  $("token-backdrop").onclick = () => closeTokenDialog(null);
  $("confirm-ok").onclick = () => closeConfirm(true);
  $("confirm-cancel").onclick = () => closeConfirm(false);
  $("confirm-backdrop").onclick = () => closeConfirm(false);
  $("token-input").onkeydown = (ev) => {
    if (ev.key === "Enter") closeTokenDialog(ev.target.value);
    if (ev.key === "Escape") closeTokenDialog(null);
  };
  const plat = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || "";
  $("palette-kbd").textContent = /Mac|iPhone|iPad/.test(plat) ? "⌘K" : "Ctrl K";

  $("btn-pick-file").onclick = () => $("file-input").click();
  $("btn-pick-folder").onclick = () => $("folder-input").click();
  $("file-input").onchange = (ev) => { addFiles(ev.target.files); ev.target.value = ""; };
  $("folder-input").onchange = (ev) => { addFiles(ev.target.files); ev.target.value = ""; };
  const dz = $("drop-zone");
  dz.ondragover = (ev) => { ev.preventDefault(); dz.classList.add("dragover"); };
  dz.ondragleave = () => dz.classList.remove("dragover");
  dz.ondrop = (ev) => { ev.preventDefault(); dz.classList.remove("dragover"); addFiles(ev.dataTransfer.files); };
  // whole bench card accepts drops
  const benchCard = document.querySelector(".bench-card");
  benchCard.ondragover = (ev) => { ev.preventDefault(); };
  benchCard.ondrop = (ev) => { ev.preventDefault(); addFiles(ev.dataTransfer.files); };
  $("btn-send").onclick = () => sendBatch();
  $("btn-cancel").onclick = () => { if (state.batchAbort) state.batchAbort(); };

  $("btn-log-pause").onclick = () => {
    const st = state.logs[state.logServerId]; if (!st) return;
    st.paused = !st.paused;
    $("btn-log-pause").textContent = st.paused ? "Resume" : "Pause";
  };
  $("log-follow").onchange = (ev) => {
    const st = state.logs[state.logServerId];
    if (st) st.follow = ev.target.checked;
  };
  $("log-server").onchange = (ev) => { state.logServerId = ev.target.value; resetLogState(state.logServerId); };
  $("log-filter").onkeydown = (ev) => {
    if (ev.key !== "Enter") return;
    const st = state.logs[state.logServerId];
    if (st) { st.filter = ev.target.value.toLowerCase(); renderLogContent(); }
  };
  $("btn-log-clear").onclick = () => resetLogState(state.logServerId);
  $("palette-backdrop").onclick = closePalette;
}

/* ---------------- boot ---------------- */

/* version badge: public endpoint, safe to fetch before any token exists */
(async function fetchVersion() {
  try {
    const r = await fetch("/api/v1/version");
    if (!r.ok) return;
    const d = await r.json();
    if (d && d.version) $("ver-pill").textContent = (d.component || "supervisor") + " · v" + d.version;
  } catch (e) { /* badge stays generic */ }
})();

wire();
if (!sessionStorage.getItem("booted")) {
  sessionStorage.setItem("booted", "1");
  document.body.classList.add("boot");
  setTimeout(() => document.body.classList.remove("boot"), 900);
}
if (!location.hash) location.hash = "#/overview";
renderCurrentView();
refresh(); pollGpu(); pollFleetMetrics();
setInterval(refresh, 2000);
setInterval(pollFleetMetrics, 6000);
setInterval(pollGpu, 2000);
setInterval(pollLogs, 1000);

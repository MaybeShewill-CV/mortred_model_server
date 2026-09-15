"use strict";

/* Mortred Supervisor UI — Mission Control.
 *
 * Architecture: one polling store → hash router → three views.
 *   #/overview       fleet grid + gpu heartbeat + activity river   (default)
 *   #/model/<ID>     per-model workbench: identity / test bench / log
 *   (⌘K)             command palette overlay
 *
 * All network/business logic (authorizedFetch, api, refresh polling,
 * upload/inference, log paging, control actions) is preserved verbatim
 * from the previous single-screen UI; only the presentation layer was
 * rebuilt around a store that fans data out to whichever view is active.
 * Zero deps, zero build chain. */

/* ---------------- state / store ---------------- */

const state = {
  servers: [],
  gateway: null,
  selectedId: null,          // workbench model (null = overview)
  files: [],                 // [{name, url, base64}]
  batchAbort: null,
  logs: {},                  // id -> {offset, filter, paused, follow, lines}
  logServerId: null,
  river: [],                  // activity events [{t, kind, text, serverId}]
  gpuHistory: [],             // [{t, util, mem_used_mib, ...}]
};

const TOKEN_KEY = "mortred_supervisor_token";

const $ = (id) => document.getElementById(id);

function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

function setToken(token) {
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
}

async function authorizedFetch(path, options) {
  options = options || {};
  options.headers = Object.assign({}, options.headers || {});
  const token = getToken();
  if (token) {
    options.headers["Authorization"] = "Bearer " + token;
  }
  let resp = await fetch(path, options);
  if (resp.status === 401) {
    const nextToken = prompt("访问被拒绝（401），请输入 Supervisor API 令牌：", token);
    if (nextToken !== null && nextToken.trim()) {
      setToken(nextToken.trim());
      options.headers["Authorization"] = "Bearer " + nextToken.trim();
      resp = await fetch(path, options);
    }
  }
  return resp;
}

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

async function api(path, options) {
  const resp = await authorizedFetch(path, options);
  let data = null;
  try { data = await resp.json(); } catch (e) { data = null; }
  return { ok: resp.ok, status: resp.status, data };
}

/* ---------------- status glyphs ---------------- */

function dotClassOf(s) {
  if (s.state === "running") return s.ready ? "running" : "starting";
  if (s.state === "starting") return "starting";
  if (s.state === "backoff") return "backoff";
  if (s.state === "failed") return "failed";
  return "stopped";
}

const ST_GLYPH = { running: "●", starting: "◐", backoff: "◑", failed: "✕", stopped: "·" };

const CAT_COLOR = {
  classification: "#00ff9c", object_detection: "#5fd3f0", face_detection: "#5fd3f0",
  scene_segmentation: "#c792ea", ocr: "#ffd166", matting: "#c792ea",
  enhancement: "#ffd166", feature_point: "#5fd3f0", feature_embedding: "#c792ea",
  mono_depth_estimation: "#ffd166", segment_anything: "#c792ea",
  diffusion: "#ff9e64", mot: "#5fd3f0", other: "#6fae85",
};

function catColor(cat) { return CAT_COLOR[cat] || CAT_COLOR.other; }

function uptimeOf(s) {
  if (!s.started_at_ms || s.state !== "running") return null;
  const sec = Math.max(0, Math.floor((Date.now() - s.started_at_ms) / 1000));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), ss = sec % 60;
  return (h > 0 ? h + ":" : "") + String(m).padStart(2, "0") + ":" + String(ss).padStart(2, "0");
}

/* ---------------- activity river ---------------- */

function riverPush(kind, text, serverId) {
  const now = new Date();
  const t = String(now.getHours()).padStart(2, "0") + ":" +
    String(now.getMinutes()).padStart(2, "0") + ":" + String(now.getSeconds()).padStart(2, "0");
  state.river.push({ t, kind, text, serverId: serverId || null });
  if (state.river.length > 200) state.river.shift();
  renderRiver();
}

function renderRiver() {
  const box = $("river-content");
  if (!box) return;
  box.innerHTML = state.river.slice(-80).map((e) => {
    const c = e.kind === "err" ? "var(--err)" : e.kind === "ok" ? "var(--ok)" : "var(--mid)";
    const srv = e.serverId ? ` <span style="color:${catColor((serverById(e.serverId) || {}).category)}">${escapeHtml(e.serverId)}</span>` : "";
    return `<div class="river-line"><span class="river-t">${e.t}</span> <span style="color:${c}">${escapeHtml(e.text)}</span>${srv}</div>`;
  }).join("");
  box.scrollTop = box.scrollHeight;
}

/* ---------------- polling store ---------------- */

let prevStates = {};

async function refresh() {
  const [cat, st] = await Promise.all([api("/api/v1/catalog"), api("/api/v1/status")]);
  if (!cat.ok || !st.ok) {
    $("conn-status").textContent = "LINK DOWN";
    $("conn-status").className = "conn-err";
    document.body.classList.add("link-down");
    return;
  }
  document.body.classList.remove("link-down");
  $("conn-status").textContent = "LINK OK";
  $("conn-status").className = "conn-ok";

  const statusById = {};
  for (const s of (st.data.servers || [])) statusById[s.id] = s;
  state.gateway = st.data.gateway || null;
  state.servers = (cat.data.servers || []).map((s) => Object.assign({}, s, statusById[s.id] || {}));

  // state transitions feed the river
  for (const s of state.servers) {
    const prev = prevStates[s.id];
    if (prev && prev !== s.state) {
      riverPush(s.state === "running" ? "ok" : s.state === "failed" ? "err" : "info",
        `${s.id} ${prev} → ${s.state}`, s.id);
    }
    prevStates[s.id] = s.state;
  }

  renderCurrentView();
}

async function pollGpu() {
  const r = await api("/api/v1/gpu");
  const box = $("gpu-panel");
  if (!box) return;
  if (!r.ok || !r.data || !r.data.available) {
    box.classList.add("gpu-na");
    $("gpu-readout").textContent = "gpu n/a";
    return;
  }
  box.classList.remove("gpu-na");
  const s = r.data.samples || [];
  state.gpuHistory = s;
  const last = s.length ? s[s.length - 1] : null;
  if (last) {
    $("gpu-readout").textContent =
      `util ${last.util < 0 ? "--" : last.util + "%"}   mem ${fmtMib(last.mem_used_mib)}/${fmtMib(last.mem_total_mib)}`
      + (last.temp >= 0 ? `   ${last.temp}°C` : "");
  }
  drawGpuChart();
}

function fmtMib(mib) {
  if (mib == null || mib < 0) return "--";
  return mib >= 1024 ? (mib / 1024).toFixed(1) + "G" : mib + "M";
}

/* ---------------- gpu heartbeat chart ---------------- */

function drawGpuChart() {
  const canvas = $("gpu-canvas");
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (w === 0) return;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const s = state.gpuHistory;
  if (s.length < 2) return;

  const util = (x) => Math.max(0, Math.min(100, x.util));
  const memPct = (x) => x.mem_total_mib > 0 ? Math.max(0, Math.min(100, 100 * x.mem_used_mib / x.mem_total_mib)) : 0;
  const n = s.length, step = w / (n - 1);

  // grid
  ctx.strokeStyle = "rgba(0,255,156,0.08)";
  ctx.setLineDash([2, 4]);
  for (let g = 1; g < 3; g++) {
    ctx.beginPath(); ctx.moveTo(0, h * g / 3); ctx.lineTo(w, h * g / 3); ctx.stroke();
  }
  ctx.setLineDash([]);

  // mem line (amber, area)
  ctx.beginPath();
  s.forEach((x, i) => { const y = h - (memPct(x) / 100) * (h - 6) - 3; i ? ctx.lineTo(i * step, y) : ctx.moveTo(0, y); });
  ctx.strokeStyle = "#ffd166"; ctx.lineWidth = 1;
  ctx.stroke();
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  ctx.fillStyle = "rgba(255,209,102,0.08)"; ctx.fill();

  // util line (green, glow area)
  ctx.beginPath();
  s.forEach((x, i) => { const y = h - (util(x) / 100) * (h - 6) - 3; i ? ctx.lineTo(i * step, y) : ctx.moveTo(0, y); });
  ctx.strokeStyle = "#00ff9c"; ctx.lineWidth = 1.5;
  ctx.shadowColor = "rgba(0,255,156,0.5)"; ctx.shadowBlur = 6;
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(0,255,156,0.18)"); grad.addColorStop(1, "rgba(0,255,156,0)");
  ctx.fillStyle = grad; ctx.fill();
}

/* ---------------- hash router ---------------- */

function currentRoute() {
  const h = location.hash.replace(/^#/, "");
  const m = h.match(/^\/model\/([A-Za-z0-9_-]+)/);
  if (m) return { view: "model", id: m[1] };
  return { view: "overview" };
}

function navigate(hash) {
  if (location.hash === hash) return;
  location.hash = hash;
}

window.addEventListener("hashchange", () => renderCurrentView());

function renderCurrentView() {
  const r = currentRoute();
  if (r.view === "model" && serverById(r.id)) {
    state.selectedId = r.id;
    showView("workbench");
    renderWorkbench();
  } else {
    state.selectedId = null;
    showView("overview");
    renderOverview();
  }
}

function showView(name) {
  $("view-overview").classList.toggle("hidden", name !== "overview");
  $("view-workbench").classList.toggle("hidden", name !== "workbench");
}

/* ---------------- OVERVIEW view ---------------- */

function renderOverview() {
  const grid = $("fleet-grid");
  $("fleet-count").textContent = state.servers.length;
  const frag = document.createDocumentFragment();
  const groups = {};
  for (const s of state.servers) (groups[s.category] = groups[s.category] || []).push(s);
  for (const cat of Object.keys(groups).sort()) {
    const head = document.createElement("div");
    head.className = "fleet-cat";
    head.textContent = cat;
    head.style.color = catColor(cat);
    frag.appendChild(head);
    for (const s of groups[cat]) {
      const st = dotClassOf(s);
      const tile = document.createElement("div");
      tile.className = "cartridge" + (s.state === "running" ? " live" : "") +
        (s.state === "failed" ? " dead" : "");
      tile.style.borderColor = s.state === "running" ? catColor(cat) : "";
      tile.innerHTML =
        `<div class="cartridge-row">
           <span class="st ${st}" title="${escapeHtml(s.state)}">${ST_GLYPH[st]}</span>
           <span class="cartridge-name">${escapeHtml(s.id.toLowerCase())}</span>
           ${s.restart_count > 0 ? `<span class="badge restarts">↻${s.restart_count}</span>` : ""}
         </div>
         <div class="cartridge-sub">${s.state === "running" ? `<span class="uptime" data-id="${s.id}">${uptimeOf(s) || ""}</span>` : escapeHtml(s.state)}</div>`;
      tile.onclick = () => navigate("#/model/" + s.id);
      frag.appendChild(tile);
    }
  }
  grid.innerHTML = "";
  grid.appendChild(frag);
  renderGatewayBar();
}

/* ---------------- WORKBENCH view ---------------- */

function renderWorkbench() {
  const s = serverById(state.selectedId);
  if (!s) return;
  const st = dotClassOf(s);
  $("wb-breadcrumb").textContent = "‹ fleet / " + s.id.toLowerCase();
  $("wb-breadcrumb").onclick = () => navigate("#/overview");
  $("wb-title").innerHTML =
    `<span class="st ${st}">${ST_GLYPH[st]}</span> ${escapeHtml(s.id.toLowerCase())}` +
    ` <span class="wb-state">${escapeHtml(s.state)}${s.state === "running" && !s.ready ? " · probing" : s.ready ? " · ready" : ""}</span>`;
  $("wb-identity").innerHTML =
    `<div class="id-row"><span class="id-k">port</span><span>:${s.port}</span></div>
     <div class="id-row"><span class="id-k">uri</span><span>${escapeHtml(s.uri || "")}</span></div>
     <div class="id-row"><span class="id-k">cat</span><span style="color:${catColor(s.category)}">${escapeHtml(s.category)}</span></div>
     <div class="id-row"><span class="id-k">↻</span><span>${s.restart_count || 0}</span></div>
     <div class="id-row"><span class="id-k">up</span><span class="uptime" data-id="${s.id}">${uptimeOf(s) || "--"}</span></div>`;

  const running = s.state === "running" || s.state === "starting" || s.state === "backoff";
  $("wb-start").disabled = running;
  $("wb-restart").disabled = !running;
  $("wb-stop").disabled = !running;
  $("wb-start").onclick = () => controlServer(s.id, "start");
  $("wb-restart").onclick = () => controlServer(s.id, "restart");
  $("wb-stop").onclick = () => controlServer(s.id, "stop");

  $("image-input-area").classList.remove("hidden");
  const hint = $("empty-hint");
  if (hint) hint.classList.add("hidden");
  state.logServerId = s.id;
  resetLogState(s.id);
  syncLogSelector();
  renderFileList();
}

function serverById(id) {
  return state.servers.find((s) => s.id === id) || null;
}

/* uptime tickers: update in place every second without re-render */
setInterval(() => {
  document.querySelectorAll(".uptime").forEach((el) => {
    const s = serverById(el.dataset.id);
    if (s) el.textContent = uptimeOf(s) || "--";
  });
}, 1000);

/* ---------------- gateway / control ---------------- */

function gatewayBaseUrl() {
  const g = state.gateway;
  if (!g || !g.address) return "";
  let host = g.address.host || "";
  if (host === "0.0.0.0" || host === "::" || host === "[::]") {
    host = window.location.hostname || "127.0.0.1";
  }
  return `http://${host}:${g.address.port}`;
}

function renderGatewayBar() {
  const g = state.gateway;
  const bar = $("gateway-status");
  if (!g) { bar.textContent = "gw ?"; return; }
  const addr = g.address ? `${g.address.host}:${g.address.port}` : "";
  const cls = g.state === "running" ? "gw-ok" : "gw-bad";
  bar.innerHTML = `gw <span class="${cls}">${g.state === "running" ? "●" : "○"}</span>` +
    (addr ? ` ${escapeHtml(addr)}` : "") + (g.state === "running" ? "" : " (infer down)");
}

async function controlServer(id, action) {
  const zh = { start: "启动", restart: "重启", stop: "停止" }[action];
  const r = await api(`/api/v1/servers/${encodeURIComponent(id)}/${action}`, { method: "POST" });
  if (!r.ok) {
    showToast(`${zh}失败：${r.status}`, "error");
    riverPush("err", `${id} ${action} → HTTP ${r.status}`, id);
  } else {
    showToast(`已${zh} ${id}`, "success");
    riverPush("ok", `${id} ${action} ok`, id);
  }
  refresh();
}

/* ---------------- toast ---------------- */

function showToast(msg, type = "info") {
  const el = document.createElement("div");
  el.className = "toast " + type;
  el.textContent = msg;
  $("toast-container").appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; }, 2400);
  setTimeout(() => { el.remove(); }, 2800);
}

/* ---------------- test bench (upload / infer / visualize) ---------------- */

function base64ToSrc(b64) {
  return "data:image/png;base64," + b64.replace(/^data:[^,]+,/, "");
}

function loadImageAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result;
      const idx = dataUrl.indexOf(",");
      resolve(dataUrl.slice(idx + 1));
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function addFiles(fileList) {
  for (const f of fileList) {
    if (!f.type.startsWith("image/")) continue;
    const b64 = await loadImageAsBase64(f);
    state.files.push({ name: f.name, url: base64ToSrc(b64), base64: b64 });
  }
  renderFileList();
}

function renderFileList() {
  const box = $("file-list");
  if (!box) return;
  box.innerHTML = "";
  for (const f of state.files) {
    const chip = document.createElement("div");
    chip.className = "file-chip";
    chip.innerHTML = `<img src="${f.url}"><span>${escapeHtml(f.name)}</span>`;
    const rm = document.createElement("span");
    rm.textContent = "✕";
    rm.className = "chip-rm";
    rm.onclick = () => {
      state.files = state.files.filter((x) => x !== f);
      renderFileList();
    };
    chip.appendChild(rm);
    box.appendChild(chip);
  }
}

function unifiedPayload(data) {
  if (data && typeof data.results === "object" && Array.isArray(data.results) && data.results.length) {
    return data.results[0].data;
  }
  return data && typeof data.data !== "undefined" ? data.data : null;
}

function topScore(payload) {
  if (!payload) return null;
  if (Array.isArray(payload) && payload.length && typeof payload[0].score === "number") return payload[0].score;
  if (typeof payload.top1 === "object" && payload.top1) return payload.top1.score;
  return null;
}

async function sendBatch() {
  const s = serverById(state.selectedId);
  if (!s || !state.files.length) return;
  const base = gatewayBaseUrl();
  if (!base) { showToast("gateway 地址未知", "error"); return; }
  $("btn-cancel").classList.remove("hidden");
  setBatchProgress(0, state.files.length);
  let aborted = false;
  state.batchAbort = () => { aborted = true; };
  let done = 0;
  for (const f of state.files) {
    if (aborted) break;
    const reqId = uid();
    const t0 = performance.now();
    try {
      const resp = await authorizedFetch(`${base}/v1/models/${encodeURIComponent(s.id)}/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ req_id: reqId, images: [f.base64] }),
      });
      const elapsed = Math.round(performance.now() - t0);
      const body = await resp.json().catch(() => null);
      if (resp.ok) {
        addResultCard(s, f, body, reqId, elapsed);
        riverPush("ok", `${s.id} 200 OK ${elapsed}ms`, s.id);
      } else {
        showToast(`HTTP ${resp.status}`, "error");
        riverPush("err", `${s.id} HTTP ${resp.status}`, s.id);
      }
    } catch (e) {
      showToast("推理请求失败：" + e, "error");
      riverPush("err", `${s.id} transport fail`, s.id);
    }
    done++;
    setBatchProgress(done, state.files.length);
  }
  state.batchAbort = null;
  $("btn-cancel").classList.add("hidden");
}

function setBatchProgress(done, total) {
  $("batch-progress").classList.toggle("hidden", total === 0);
  $("batch-progress-fill").style.width = total ? ((done / total) * 100).toFixed(1) + "%" : "0";
  $("batch-progress-text").textContent = `${done}/${total}`;
}

function addResultCard(server, input, result, reqId, elapsed) {
  const payload = unifiedPayload(result);
  const card = document.createElement("div");
  card.className = "result-card";
  const score = topScore(payload);
  card.innerHTML =
    `<div class="head">
       <span class="req-meta">${escapeHtml(input.name)} · ${elapsed}ms · ${escapeHtml(reqId)}</span>
       <span class="req-meta">${score != null ? "top " + score.toFixed(3) : ""}</span>
     </div>`;
  const vizWrap = document.createElement("div");
  vizWrap.className = "viz";
  card.appendChild(vizWrap);
  const raw = document.createElement("details");
  raw.className = "raw-json";
  raw.innerHTML = `<summary>raw</summary>`;
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(result, null, 1);
  raw.appendChild(pre);
  card.appendChild(raw);
  $("results-list").prepend(card);
  visualize(server, input, payload, vizWrap).catch(() => {});
}

/* visualization: renders bbox/ocr/keypoint overlays into vizWrap */
async function visualize(server, input, payload, vizWrap) {
  const img = new Image();
  img.src = input.url;
  await img.decode().catch(() => {});
  if (Array.isArray(payload) && payload.length && typeof payload[0].bbox === "object") {
    const canvas = document.createElement("canvas");
    canvas.className = "overlay";
    vizWrap.appendChild(canvas);
    drawDetection(canvas, img.src, payload);
    return;
  }
  if (payload && Array.isArray(payload.regions)) {
    const canvas = document.createElement("canvas");
    canvas.className = "overlay";
    vizWrap.appendChild(canvas);
    drawOcr(canvas, img.src, payload.regions);
    return;
  }
  if (payload && (payload.image || payload.colorized_mask || payload.alpha || payload.matting_image)) {
    const out = payload.image || payload.colorized_mask || payload.alpha || payload.matting_image;
    const src = base64ToSrc(out);
    const im = new Image();
    im.src = src;
    vizWrap.appendChild(im);
    return;
  }
  if (payload && Array.isArray(payload.keypoints)) {
    const canvas = document.createElement("canvas");
    canvas.className = "overlay";
    vizWrap.appendChild(canvas);
    drawKeypoints(canvas, img.src, payload);
    return;
  }
  vizWrap.appendChild(img);
}

function drawDetection(canvas, imgUrl, boxes) {
  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    ctx.lineWidth = Math.max(2, Math.round(img.naturalWidth / 300));
    ctx.font = Math.max(14, Math.round(img.naturalWidth / 40)) + "px monospace";
    for (const b of boxes) {
      const [x1, y1, x2, y2] = b.bbox;
      const hue = (b.class_id * 47) % 360;
      ctx.strokeStyle = `hsl(${hue} 90% 60%)`;
      ctx.shadowColor = ctx.strokeStyle;
      ctx.shadowBlur = 6;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.shadowBlur = 0;
      const label = `${b.category || b.class_id} ${b.score != null ? b.score.toFixed(2) : ""}`;
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = `hsl(${hue} 90% 60%)`;
      ctx.fillRect(x1, Math.max(0, y1 - 20), tw, 18);
      ctx.fillStyle = "#000";
      ctx.fillText(label, x1 + 4, Math.max(14, y1 - 6));
    }
  };
  img.src = imgUrl;
}

function drawOcr(canvas, imgUrl, regions) {
  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffd166";
    for (const r of regions) {
      const pts = r.points || r.bbox_points || [];
      if (pts.length === 4) {
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < 4; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.closePath();
        ctx.stroke();
      }
      if (r.text) {
        ctx.fillStyle = "#ffd166";
        ctx.font = "14px monospace";
        ctx.fillText(r.text, pts[0] ? pts[0][0] : 4, pts[0] ? pts[0][1] - 4 : 14);
      }
    }
  };
  img.src = imgUrl;
}

function drawKeypoints(canvas, imgUrl, payload) {
  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    ctx.fillStyle = "#00ff9c";
    ctx.shadowColor = "#00ff9c";
    ctx.shadowBlur = 4;
    for (const p of payload.keypoints) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.shadowBlur = 0;
  };
  img.src = imgUrl;
}

/* ---------------- logs (contextual: workbench filters to its model) -------- */

function resetLogState(id) {
  state.logs[id] = { offset: 0, filter: "", paused: false, follow: true, lines: [] };
  const el = $("log-content");
  if (el) el.textContent = "";
}

function syncLogSelector() {
  const sel = $("log-server");
  if (!sel) return;
  sel.innerHTML = "";
  for (const s of state.servers) {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = s.id;
    sel.appendChild(opt);
  }
  if (state.logServerId) sel.value = state.logServerId;
}

function renderLogContent() {
  const el = $("log-content");
  if (!el || !state.logServerId) return;
  const st = state.logs[state.logServerId];
  if (!st) return;
  const lines = st.filter ? st.lines.filter((l) => l.toLowerCase().includes(st.filter)) : st.lines;
  el.innerHTML = lines.slice(-400).map((l) => highlightLine(l, st.filter)).join("\n");
  if (st.follow) el.scrollTop = el.scrollHeight;
}

function highlightLine(line, filter) {
  const esc = escapeHtml(line);
  if (!filter) return esc;
  const idx = esc.toLowerCase().indexOf(filter);
  if (idx < 0) return esc;
  return esc.slice(0, idx) + "<mark>" + esc.slice(idx, idx + filter.length) + "</mark>" + esc.slice(idx + filter.length);
}

async function pollLogs() {
  if (!state.logServerId) return;
  const st = state.logs[state.logServerId];
  if (!st || st.paused) return;
  const r = await api(`/api/v1/servers/${encodeURIComponent(state.logServerId)}/logs?offset=${st.offset}&limit=100`);
  if (!r.ok || !r.data) return;
  st.lines.push(...(r.data.lines || []));
  st.offset = r.data.offset != null ? r.data.offset + (r.data.lines || []).length : st.lines.length;
  $("log-meta").textContent = `${st.lines.length} lines`;
  renderLogContent();
  for (const l of (r.data.lines || [])) {
    if (/\bERROR\b|FATAL/.test(l)) riverPush("err", l.slice(0, 120), state.logServerId);
  }
}

/* ---------------- command palette ---------------- */

function paletteItems() {
  const items = [];
  for (const s of state.servers) {
    items.push({ label: `open ${s.id.toLowerCase()}`, hint: "navigate", act: () => navigate("#/model/" + s.id) });
    const running = ["running", "starting", "backoff"].includes(s.state);
    if (!running) items.push({ label: `start ${s.id.toLowerCase()}`, hint: "control", act: () => controlServer(s.id, "start") });
    if (running) items.push({ label: `stop ${s.id.toLowerCase()}`, hint: "control", act: () => controlServer(s.id, "stop") });
    items.push({ label: `restart ${s.id.toLowerCase()}`, hint: "control", act: () => controlServer(s.id, "restart") });
  }
  items.push({ label: "overview", hint: "navigate", act: () => navigate("#/overview") });
  items.push({ label: "token", hint: "settings", act: () => {
    const t = prompt("Supervisor API 令牌：", getToken());
    if (t !== null) { setToken(t.trim()); location.reload(); }
  }});
  return items;
}

function fuzzyMatch(query, label) {
  let li = 0, score = 0, streak = 0;
  const q = query.toLowerCase(), l = label.toLowerCase();
  for (const ch of q) {
    const found = l.indexOf(ch, li);
    if (found < 0) return -1;
    streak = found === li ? streak + 1 : 0;
    score += 1 + streak;
    li = found + 1;
  }
  return score;
}

function openPalette() {
  const p = $("palette");
  p.classList.remove("hidden");
  const input = $("palette-input");
  input.value = "";
  renderPalette("");
  input.focus();
}

function closePalette() {
  $("palette").classList.add("hidden");
}

function renderPalette(query) {
  const list = $("palette-list");
  const items = paletteItems()
    .map((it) => ({ it, score: query ? fuzzyMatch(query, it.label) : 0 }))
    .filter((x) => x.score >= 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);
  let sel = 0;
  list.innerHTML = "";
  for (const { it } of items) {
    const row = document.createElement("div");
    row.className = "palette-row";
    row.innerHTML = `<span class="palette-label">${escapeHtml(it.label)}</span><span class="palette-hint">${it.hint}</span>`;
    row.onclick = () => { closePalette(); it.act(); };
    list.appendChild(row);
  }
  if (items.length) list.firstChild.classList.add("sel");
  $("palette-input").onkeydown = (ev) => {
    const rows = [...list.children];
    if (ev.key === "Escape") { closePalette(); }
    else if (ev.key === "Enter") { if (rows[sel]) { closePalette(); items[sel].it.act(); } }
    else if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      ev.preventDefault();
      if (rows[sel]) rows[sel].classList.remove("sel");
      sel = ev.key === "ArrowDown" ? Math.min(rows.length - 1, sel + 1) : Math.max(0, sel - 1);
      if (rows[sel]) rows[sel].classList.add("sel");
    } else {
      setTimeout(() => renderPalette($("palette-input").value), 0);
    }
  };
}

document.addEventListener("keydown", (ev) => {
  if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
    ev.preventDefault();
    $("palette").classList.contains("hidden") ? openPalette() : closePalette();
  }
  if (ev.key === "Escape" && !$("palette").classList.contains("hidden")) closePalette();
});

/* ---------------- wiring ---------------- */

function wire() {
  $("btn-pick-file").onclick = () => $("file-input").click();
  $("btn-pick-folder").onclick = () => $("folder-input").click();
  $("file-input").onchange = (ev) => { addFiles(ev.target.files); ev.target.value = ""; };
  $("folder-input").onchange = (ev) => { addFiles(ev.target.files); ev.target.value = ""; };
  const dropZone = $("drop-zone");
  dropZone.ondragover = (ev) => { ev.preventDefault(); dropZone.classList.add("dragover"); };
  dropZone.ondragleave = () => dropZone.classList.remove("dragover");
  dropZone.ondrop = (ev) => {
    ev.preventDefault();
    dropZone.classList.remove("dragover");
    addFiles(ev.dataTransfer.files);
  };
  $("btn-send").onclick = () => sendBatch();
  $("btn-cancel").onclick = () => { if (state.batchAbort) state.batchAbort(); };
  $("btn-token").onclick = () => {
    const t = prompt("Supervisor API 令牌（Bearer Token）：", getToken());
    if (t !== null) { setToken(t.trim()); showToast("令牌已保存", "success"); refresh(); }
  };
  $("btn-log-pause").onclick = () => {
    const st = state.logs[state.logServerId];
    if (!st) return;
    st.paused = !st.paused;
    $("btn-log-pause").textContent = st.paused ? "resume" : "pause";
  };
  $("log-follow").onchange = (ev) => {
    const st = state.logs[state.logServerId];
    if (st) st.follow = ev.target.checked;
  };
  $("log-server").onchange = (ev) => {
    state.logServerId = ev.target.value;
    resetLogState(state.logServerId);
  };
  $("log-filter").onkeydown = (ev) => {
    if (ev.key !== "Enter") return;
    const st = state.logs[state.logServerId];
    if (st) { st.filter = ev.target.value.toLowerCase(); renderLogContent(); }
  };
  $("btn-log-clear").onclick = () => { resetLogState(state.logServerId); };
  $("palette-backdrop").onclick = closePalette;
  window.addEventListener("resize", drawGpuChart);
}

/* ---------------- boot ---------------- */

wire();
if (!sessionStorage.getItem("booted")) {
  sessionStorage.setItem("booted", "1");
  document.body.classList.add("boot");
  setTimeout(() => document.body.classList.remove("boot"), 600);
}
if (!location.hash) location.hash = "#/overview";
renderCurrentView();
refresh();
pollGpu();
setInterval(refresh, 2000);
setInterval(pollGpu, 2000);
setInterval(pollLogs, 1000);

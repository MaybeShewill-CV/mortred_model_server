"use strict";

/* ---------------- state ---------------- */
const state = {
  servers: [],
  selectedId: null,
  files: [],            // [{name, url, base64}]
  chatImage: null,      // {name, base64}
  batchAbort: null,
  logs: {},             // id -> {offset, filter, paused, follow}
  logServerId: null,
};

const $ = (id) => document.getElementById(id);

/* ---------------- helpers ---------------- */
const TOKEN_KEY = "mortred_web_token";

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

/***
 * 带 Bearer Token 的 fetch 封装：自动附带 token，遇到 401 时让用户重新输入并重试一次。
 */
async function authorizedFetch(path, options) {
  options = options || {};
  options.headers = Object.assign({}, options.headers || {});
  const token = getToken();
  if (token) {
    options.headers["Authorization"] = "Bearer " + token;
  }
  let resp = await fetch(path, options);
  if (resp.status === 401) {
    const nextToken = prompt("访问被拒绝（401），请输入 Web Console 访问令牌：", token);
    if (nextToken) {
      setToken(nextToken.trim());
      options.headers["Authorization"] = "Bearer " + nextToken.trim();
      resp = await fetch(path, options);
    }
  }
  return resp;
}

$("btn-token").onclick = () => {
  const token = prompt("请输入 Web Console 访问令牌（Bearer Token）：", getToken());
  if (token !== null) {
    setToken(token.trim());
    showToast("令牌已保存", "success");
    loadCatalog();
  }
};

function uid() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return "req-" + Date.now() + "-" + Math.random().toString(36).slice(2, 10);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

async function api(path, options) {
  const resp = await authorizedFetch(path, options);
  const text = await resp.text();
  let data;
  try { data = JSON.parse(text); } catch (e) { data = text; }
  return { ok: resp.ok, status: resp.status, data };
}

function serverById(id) {
  return state.servers.find((s) => s.id === id) || null;
}

function base64ToSrc(b64) {
  if (!b64) return "";
  return "data:image/png;base64," + b64;
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

/* ---------------- catalog & server list ---------------- */
async function loadCatalog() {
  const { ok, data } = await api("/api/catalog");
  if (!ok) {
    $("conn-status").textContent = "后端连接失败";
    $("conn-status").className = "conn-err";
    return;
  }
  $("conn-status").textContent = "已连接";
  $("conn-status").className = "conn-ok";
  state.servers = data.servers || [];
  $("server-count").textContent = state.servers.length;
  if (!state.selectedId && state.servers.length) {
    selectServer(state.servers[0].id);
  }
  renderServerList();
  syncLogSelector();
}

function renderServerList() {
  const box = $("server-list");
  box.innerHTML = "";
  const groups = {};
  for (const s of state.servers) {
    (groups[s.category] = groups[s.category] || []).push(s);
  }
  for (const cat of Object.keys(groups).sort()) {
    const g = document.createElement("div");
    g.className = "cat-group";
    const title = document.createElement("div");
    title.className = "cat-name";
    title.textContent = cat;
    g.appendChild(title);
    for (const s of groups[cat]) {
      const item = document.createElement("div");
      item.className = "server-item" + (s.id === state.selectedId ? " selected" : "");
      item.onclick = () => selectServer(s.id);
      const dotClass = s.running ? (s.ready ? "running" : "starting") : "stopped";
      item.innerHTML =
        `<div class="row1">
           <span class="dot ${dotClass}"></span>
           <span class="server-name" title="${escapeHtml(s.name)}">${escapeHtml(s.name)}</span>
           <span class="badge ${s.type}">${s.type}</span>
         </div>
         <div class="row2">
           <span class="port-text">:${s.port} ${escapeHtml(s.uri)}</span>
           <span class="actions">
             <button class="btn small" data-action="start" ${s.running ? "disabled" : ""}>启动</button>
             <button class="btn small" data-action="stop" ${s.running ? "" : "disabled"}>停止</button>
           </span>
         </div>`;
      item.querySelector('[data-action="start"]').onclick = (ev) => {
        ev.stopPropagation();
        controlServer(s.id, "start");
      };
      item.querySelector('[data-action="stop"]').onclick = (ev) => {
        ev.stopPropagation();
        controlServer(s.id, "stop");
      };
      g.appendChild(item);
    }
    box.appendChild(g);
  }
}

async function controlServer(id, action) {
  const { data } = await api(`/api/servers/${encodeURIComponent(id)}/${action}`, { method: "POST" });
  const s = serverById(id);
  const label = s ? s.name : id;
  if (data && data.ok === false) {
    showToast(`${action === "start" ? "启动" : "停止"}失败：${data.error || "未知错误"}`, "error");
  } else {
    showToast(`${action === "start" ? "已启动" : "已停止"} ${label}`, "success");
  }
  loadCatalog();
}

function selectServer(id) {
  if (id !== state.selectedId) {
    clearResults();
    $("chat-history").innerHTML = "";
    // log panel follows the sidebar selection
    if (state.logServerId !== id) {
      state.logServerId = id;
      resetLogState(id);
      $("log-server").value = id;
    }
  }
  state.selectedId = id;
  const s = serverById(id);
  renderServerList();
  if (!s) return;
  $("selected-server-info").textContent = `${s.name}  :${s.port}${s.uri}${s.running && !s.ready ? "（启动中…）" : ""}`;
  if (s.type === "chat") {
    $("image-input-area").classList.add("hidden");
    $("chat-input-area").classList.remove("hidden");
  } else {
    $("chat-input-area").classList.add("hidden");
    $("image-input-area").classList.remove("hidden");
  }
}

function clearResults() {
  $("results-list").innerHTML = "";
}

function showToast(msg, type = "info") {
  const box = $("toast-container");
  const el = document.createElement("div");
  el.className = "toast " + type;
  el.textContent = msg;
  box.appendChild(el);
  setTimeout(() => el.remove(), 2600);
}

/* ---------------- image input ---------------- */
$("btn-pick-file").onclick = () => $("file-input").click();
$("btn-pick-folder").onclick = () => $("folder-input").click();

$("file-input").onchange = (ev) => { addFiles([...ev.target.files]); ev.target.value = ""; };
$("folder-input").onchange = (ev) => { addFiles([...ev.target.files]); ev.target.value = ""; };

const dropZone = $("drop-zone");
dropZone.ondragover = (ev) => { ev.preventDefault(); dropZone.classList.add("dragover"); };
dropZone.ondragleave = () => dropZone.classList.remove("dragover");
dropZone.ondrop = (ev) => {
  ev.preventDefault();
  dropZone.classList.remove("dragover");
  addFiles([...ev.dataTransfer.files]);
};

async function addFiles(fileList) {
  const imgs = fileList.filter((f) => f.type.startsWith("image/"));
  for (const f of imgs) {
    if (state.files.some((x) => x.name === f.name && x.size === f.size)) continue;
    const b64 = await loadImageAsBase64(f);
    state.files.push({ name: f.name, url: URL.createObjectURL(f), base64: b64, size: f.size });
  }
  renderFileList();
}

function renderFileList() {
  const box = $("file-list");
  box.innerHTML = "";
  for (let i = 0; i < state.files.length; ++i) {
    const f = state.files[i];
    const chip = document.createElement("div");
    chip.className = "file-chip";
    chip.innerHTML = `<img src="${f.url}"><span>${escapeHtml(f.name)}</span>
      <button class="btn small" data-i="${i}">✕</button>`;
    chip.querySelector("button").onclick = () => {
      URL.revokeObjectURL(f.url);
      state.files.splice(i, 1);
      renderFileList();
    };
    box.appendChild(chip);
  }
}

/* ---------------- send images ---------------- */
$("btn-send").onclick = () => sendBatch();
$("btn-cancel").onclick = () => {
  if (state.batchAbort) state.batchAbort.abort();
};

async function sendBatch() {
  const s = serverById(state.selectedId);
  if (!s || s.type !== "image") return;
  if (!s.running) { alert("请先启动该 server"); return; }
  if (!s.ready) { showToast(`${s.name} 尚未就绪（模型加载中），请稍候再试`, "error"); return; }
  if (!state.files.length) { alert("请先选择图片"); return; }
  if (state.batchAbort) { showToast("上一批仍在发送中，请稍候", "error"); return; }

  // keep only the latest input/returns on the page
  clearResults();
  state.batchAbort = new AbortController();
  const total = state.files.length;
  $("batch-progress").classList.remove("hidden");
  $("btn-cancel").classList.remove("hidden");
  $("btn-send").disabled = true;

  for (let i = 0; i < total; ++i) {
    if (state.batchAbort.signal.aborted) break;
    const f = state.files[i];
    const reqId = uid();
    const body = JSON.stringify({ server_id: s.id, req_id: reqId, img_data: f.base64 });
    const t0 = performance.now();
    let result;
    try {
      const resp = await authorizedFetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
        signal: state.batchAbort.signal,
      });
      const text = await resp.text();
      let parsed = text;
      try { parsed = JSON.parse(text); } catch (e) { /* keep raw text */ }
      result = { ok: resp.ok, status: resp.status, data: parsed, raw: text };
    } catch (e) {
      result = { ok: false, status: 0, data: null, raw: String(e) };
    }
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    addResultCard(s, { name: f.name, url: f.url, base64: f.base64 }, result, reqId, elapsed);
    setBatchProgress(i + 1, total);
  }

  const aborted = state.batchAbort.signal.aborted;
  $("batch-progress").classList.add("hidden");
  $("btn-cancel").classList.add("hidden");
  $("btn-send").disabled = false;
  state.batchAbort = null;
  if (!aborted) {
    // clear the consumed input files so a second click never re-sends the same images
    const sent = state.files.length;
    state.files.forEach((f) => URL.revokeObjectURL(f.url));
    state.files = [];
    renderFileList();
    showToast(`已发送 ${sent} 张图片`, "success");
  } else {
    showToast("已取消发送", "info");
  }
}

function setBatchProgress(done, total) {
  $("batch-progress-fill").style.width = (total ? (done / total) * 100 : 0) + "%";
  $("batch-progress-text").textContent = `${done} / ${total}`;
}

/* ---------------- chat ---------------- */
$("btn-chat-image").onclick = () => $("chat-image-input").click();
$("chat-image-input").onchange = async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  state.chatImage = { name: f.name, base64: await loadImageAsBase64(f) };
  $("chat-image-name").textContent = "已附加：" + f.name;
};

$("btn-chat-send").onclick = () => sendChat();
$("chat-text").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); sendChat(); }
});

async function sendChat() {
  const s = serverById(state.selectedId);
  if (!s || s.type !== "chat") return;
  if (!s.running) { alert("请先启动该 server"); return; }
  if (!s.ready) { showToast(`${s.name} 尚未就绪（模型加载中），请稍候再试`, "error"); return; }
  const text = $("chat-text").value.trim();
  if (!text) return;

  // keep only the latest chat turn's returns on the page
  clearResults();
  const isVLM = s.id.indexOf("qwen2_vl") !== -1;
  const data = [{ role: "system", content: "You are a smart AI assistant from Mortred Company." }];
  if (isVLM) {
    const content = [{ type: "text", text }];
    if (state.chatImage) content.push({ type: "image", image: state.chatImage.base64 });
    data.push({ role: "user", content });
  } else {
    data.push({ role: "user", content: text });
  }

  const reqId = uid();
  const body = JSON.stringify({ server_id: s.id, req_id: reqId, data });
  appendChatBubble("user", text + (state.chatImage ? "\n[图片] " + state.chatImage.name : ""));
  $("chat-text").value = "";
  state.chatImage = null;
  $("chat-image-name").textContent = "";

  const t0 = performance.now();
  const resp = await authorizedFetch("/api/infer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
  const raw = await resp.text();
  let parsed = raw;
  try { parsed = JSON.parse(raw); } catch (e) { /* keep raw */ }
  const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
  const answer = parsed && parsed.data && parsed.data.response ? parsed.data.response : raw;
  appendChatBubble("assistant", answer);
  addResultCard(s, { name: reqId, url: null, base64: null }, { ok: resp.ok, status: resp.status, data: parsed, raw }, reqId, elapsed);
}

function appendChatBubble(role, text) {
  const box = $("chat-history");
  const div = document.createElement("div");
  div.className = "chat-msg " + role;
  div.textContent = text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

/* ---------------- results ---------------- */
function addResultCard(server, input, result, reqId, elapsed) {
  const box = $("results-list");
  const card = document.createElement("div");
  card.className = "result-card";
  const statusText = result.ok ? `HTTP ${result.status}` : `失败 (${result.status})`;
  card.innerHTML = `
    <div class="head">
      <div>
        <b>${escapeHtml(input.name || reqId)}</b>
        <span class="req-meta"> · ${escapeHtml(server.name)} · req_id=${escapeHtml(reqId)} · ${elapsed}s</span>
      </div>
      <span class="badge image">${statusText}</span>
    </div>
    <div class="viz"></div>
    <details class="raw-json">
      <summary>原始返回</summary>
      <pre></pre>
      <button class="btn small" data-copy="1">复制</button>
      <button class="btn small" data-download="1">下载</button>
    </details>`;
  card.querySelector(".viz").appendChild(visualize(server, input, result));
  const pre = card.querySelector(".raw-json pre");
  pre.textContent = typeof result.raw === "string" ? result.raw : JSON.stringify(result.data, null, 2);
  card.querySelector('[data-copy="1"]').onclick = () => {
    navigator.clipboard.writeText(pre.textContent);
  };
  card.querySelector('[data-download="1"]').onclick = () => {
    const blob = new Blob([pre.textContent], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${server.id}_${reqId}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  box.prepend(card);
}

function visualize(server, input, result) {
  const wrap = document.createElement("div");
  const data = result.data;
  const payload = data && typeof data === "object" ? data.data : null;

  if (!result.ok || !data || typeof data !== "object") {
    wrap.textContent = "请求失败或响应非 JSON：";
    return wrap;
  }

  const cat = server.category;
  const base = document.createElement("div");
  const imgUrl = input.url;

  if (cat === "classification") {
    base.innerHTML = `
      <div class="compare">
        <figure>${imgUrl ? `<img src="${imgUrl}">` : ""}<figcaption>输入</figcaption></figure>
        <figure><div class="top1-card">
          <div class="label">预测类别</div>
          <div style="font-size:18px">${escapeHtml(payload && payload.category || "-")}</div>
          <div class="label">class_id: ${payload ? payload.class_id : "-"}</div>
          <div class="label">score: ${payload ? payload.scores : "-"}</div>
          <div class="score-bar"><div class="score-bar-fill" style="width:${Math.round((payload && payload.scores || 0) * 100)}%"></div></div>
        </div></figure>
      </div>`;
  } else if (cat === "object_detection") {
    const canvas = document.createElement("canvas");
    canvas.className = "overlay";
    const boxes = Array.isArray(payload) ? payload : [];
    drawDetection(canvas, imgUrl, boxes);
    base.appendChild(canvas);
  } else if (cat === "scene_segmentation") {
    base.innerHTML = `
      <div class="compare">
        <figure>${imgUrl ? `<img src="${imgUrl}">` : ""}<figcaption>输入</figcaption></figure>
        <figure><img src="${base64ToSrc(payload && payload.colorized_seg_mask)}"><figcaption>分割结果</figcaption></figure>
      </div>`;
  } else if (cat === "matting") {
    const seg = base64ToSrc(payload && payload.segment_result);
    base.innerHTML = `
      <div class="compare">
        <figure>${imgUrl ? `<img src="${imgUrl}">` : ""}<figcaption>输入</figcaption></figure>
        <figure><img src="${seg}"><figcaption>抠图结果</figcaption></figure>
      </div>`;
    if (imgUrl && seg) base.appendChild(makeComposite(imgUrl, seg));
  } else if (cat === "enhancement") {
    base.innerHTML = `
      <div class="compare">
        <figure>${imgUrl ? `<img src="${imgUrl}">` : ""}<figcaption>输入</figcaption></figure>
        <figure><img src="${base64ToSrc(payload && payload.enhance_result)}"><figcaption>增强结果</figcaption></figure>
      </div>`;
  } else if (cat === "mono_depth_estimation") {
    base.innerHTML = `
      <div class="compare">
        <figure>${imgUrl ? `<img src="${imgUrl}">` : ""}<figcaption>输入</figcaption></figure>
        <figure><img src="${base64ToSrc(payload && payload.estimate_result)}"><figcaption>深度估计</figcaption></figure>
      </div>`;
  } else if (cat === "ocr") {
    const canvas = document.createElement("canvas");
    canvas.className = "overlay";
    const regions = Array.isArray(payload) ? payload : [];
    drawOcr(canvas, imgUrl, regions);
    base.appendChild(canvas);
    if (regions.length) {
      const table = document.createElement("table");
      table.className = "ocr-table";
      table.innerHTML = "<tr><th>#</th><th>score</th><th>bbox</th></tr>" +
        regions.map((r, i) => `<tr><td>${i + 1}</td><td>${r.score}</td><td>${JSON.stringify(r.bbox)}</td></tr>`).join("");
      base.appendChild(table);
    }
  } else if (cat === "feature_point") {
    const canvas = document.createElement("canvas");
    canvas.className = "overlay";
    drawKeypoints(canvas, imgUrl, payload);
    base.appendChild(canvas);
  } else if (cat === "llm") {
    const answer = payload && payload.response ? payload.response : JSON.stringify(data);
    const div = document.createElement("div");
    div.className = "chat-msg assistant";
    div.textContent = answer;
    base.appendChild(div);
  } else {
    base.textContent = "（该类别暂无专门可视化，见原始返回）";
  }

  wrap.appendChild(base);
  return wrap;
}

function drawDetection(canvas, imgUrl, boxes) {
  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const colors = ["#f87171", "#60a5fa", "#4ade80", "#facc15", "#c084fc", "#fb923c"];
    boxes.forEach((b, i) => {
      const pts = b.points || [];
      const color = colors[i % colors.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      if (pts.length >= 2 && Array.isArray(pts[0])) {
        const x = pts[0][0], y = pts[0][1];
        const x2 = pts[1][0], y2 = pts[1][1];
        ctx.strokeRect(x, y, x2 - x, y2 - y);
        ctx.fillStyle = color;
        ctx.font = "bold 16px sans-serif";
        ctx.fillText(`${b.category || ""} ${(b.score || "")}`, x, y > 18 ? y - 6 : y + 18);
      }
      if (Array.isArray(b.landmark)) {
        ctx.fillStyle = color;
        b.landmark.forEach((p) => {
          if (Array.isArray(p)) { ctx.beginPath(); ctx.arc(p[0], p[1], 3, 0, Math.PI * 2); ctx.fill(); }
        });
      }
    });
  };
  if (imgUrl) img.src = imgUrl;
}

function drawOcr(canvas, imgUrl, regions) {
  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    regions.forEach((r) => {
      ctx.strokeStyle = "#22c55e";
      ctx.lineWidth = 2;
      if (Array.isArray(r.bbox) && r.bbox.length >= 2) {
        const [p1, p2] = r.bbox;
        ctx.strokeRect(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1]);
      }
      if (Array.isArray(r.polygon) && r.polygon.length > 2) {
        ctx.strokeStyle = "#60a5fa";
        ctx.beginPath();
        r.polygon.forEach((p, i) => { i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]); });
        ctx.closePath();
        ctx.stroke();
      }
    });
  };
  if (imgUrl) img.src = imgUrl;
}

function drawKeypoints(canvas, imgUrl, payload) {
  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const loc = payload && payload.location;
    if (!loc) return;
    ctx.fillStyle = "#f87171";
    const arr = Array.isArray(loc[0]) ? loc.flat() : loc;
    for (let i = 0; i + 1 < arr.length; i += 2) {
      ctx.beginPath();
      ctx.arc(arr[i], arr[i + 1], 3, 0, Math.PI * 2);
      ctx.fill();
    }
  };
  if (imgUrl) img.src = imgUrl;
}

function makeComposite(origSrc, maskSrc) {
  const canvas = document.createElement("canvas");
  canvas.className = "overlay";
  const orig = new Image();
  const mask = new Image();
  let loaded = 0;
  const tryDraw = () => {
    if (loaded < 2) return;
    canvas.width = orig.naturalWidth;
    canvas.height = orig.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(orig, 0, 0, canvas.width, canvas.height);
    ctx.globalCompositeOperation = "destination-in";
    ctx.drawImage(mask, 0, 0, canvas.width, canvas.height);
  };
  orig.onload = () => { loaded++; tryDraw(); };
  mask.onload = () => { loaded++; tryDraw(); };
  orig.src = origSrc;
  mask.src = maskSrc;
  const fig = document.createElement("figure");
  fig.innerHTML = "<figcaption>合成预览</figcaption>";
  fig.insertBefore(canvas, fig.firstChild);
  return fig;
}

/* ---------------- log panel ---------------- */
function resetLogState(id) {
  state.logs[id] = { offset: 0, follow: true, paused: false, filter: "", lines: [] };
  $("log-content").textContent = "";
  $("log-meta").textContent = "";
  const s = serverById(id);
  $("log-current-server").textContent = s ? s.name : id;
}

function syncLogSelector() {
  const sel = $("log-server");
  const prev = sel.value;
  sel.innerHTML = "";
  for (const s of state.servers) {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = s.name;
    sel.appendChild(opt);
  }
  if (state.logServerId && serverById(state.logServerId)) {
    sel.value = state.logServerId;
  } else if (prev) {
    sel.value = prev;
  }
  if (sel.value) state.logServerId = sel.value;
}

$("log-server").onchange = () => {
  const id = $("log-server").value;
  state.logServerId = id;
  resetLogState(id);
};

$("log-follow").onchange = () => {
  const id = state.logServerId;
  if (id) state.logs[id] = state.logs[id] || { offset: 0, follow: true, paused: false, filter: "" };
  if (id) state.logs[id].follow = $("log-follow").checked;
};

$("btn-log-pause").onclick = () => {
  const id = state.logServerId;
  if (!id) return;
  const st = state.logs[id] = state.logs[id] || { offset: 0, follow: true, paused: false, filter: "" };
  st.paused = !st.paused;
  $("btn-log-pause").textContent = st.paused ? "继续" : "暂停";
};

$("btn-log-clear").onclick = () => {
  const id = state.logServerId;
  if (!id) return;
  const st = state.logs[id] = state.logs[id] || { offset: 0, follow: true, paused: false, filter: "" };
  st.offset = 0;
  $("log-content").textContent = "";
  $("log-meta").textContent = "";
};

$("btn-log-copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText($("log-content").textContent);
  } catch (e) { /* ignore */ }
};

$("log-filter").oninput = () => {
  const id = state.logServerId;
  if (id && state.logs[id]) state.logs[id].filter = $("log-filter").value;
  renderLogContent();
};

function renderLogContent() {
  const id = state.logServerId;
  const st = state.logs[id] || { filter: "", lines: [] };
  const filter = (st.filter || "").toLowerCase();
  const box = $("log-content");
  const lines = st.lines || [];
  box.innerHTML = "";
  for (const line of lines) {
    if (filter && line.toLowerCase().indexOf(filter) === -1) continue;
    const div = document.createElement("div");
    if (filter) {
      div.innerHTML = highlightLine(line, filter);
    } else {
      div.textContent = line;
    }
    box.appendChild(div);
  }
  box.scrollTop = box.scrollHeight;
}

function highlightLine(line, filter) {
  const lower = line.toLowerCase();
  let pos = 0;
  let html = "";
  while (true) {
    const idx = lower.indexOf(filter, pos);
    if (idx === -1) {
      html += escapeHtml(line.slice(pos));
      break;
    }
    html += escapeHtml(line.slice(pos, idx));
    html += "<mark>" + escapeHtml(line.slice(idx, idx + filter.length)) + "</mark>";
    pos = idx + filter.length;
  }
  return html;
}

async function pollLogs() {
  const id = state.logServerId;
  if (!id) return;
  const st = state.logs[id] = state.logs[id] || { offset: 0, follow: true, paused: false, filter: "" };
  if (!st.follow || st.paused) return;
  const { ok, data } = await api(`/api/logs/${encodeURIComponent(id)}?offset=${st.offset}&limit=500`);
  if (!ok || !data || !Array.isArray(data.lines)) return;
  if (data.lines.length) {
    const lines = st.lines = st.lines || [];
    lines.push(...data.lines);
    if (lines.length > 5000) lines.splice(0, lines.length - 5000);
    st.offset += data.lines.length;
    renderLogContent();
  }
  $("log-meta").textContent = `total=${data.total} offset=${st.offset}`;
}

/* ---------------- init ---------------- */
loadCatalog();
setInterval(loadCatalog, 3000);
setInterval(pollLogs, 1000);

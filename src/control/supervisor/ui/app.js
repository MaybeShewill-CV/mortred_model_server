"use strict";

/* Mortred Supervisor UI — Mission Control HUD.
 * Sci-fi command-deck aesthetic: full-field GPU heartbeat, glowing fleet
 * cartridges with live sparklines, telemetry band with req-rate, and a
 * per-model workbench with identity gauges. Zero deps, zero build. */

const state = {
  servers: [], gateway: null, selectedId: null, files: [],
  batchAbort: null, logs: {}, logServerId: null,
  river: [], gpuHistory: [],
  inferHistory: [],     // [{t, ok, ms}] from sendBatch
};

const TOKEN_KEY = "mortred_supervisor_token";
const $ = (id) => document.getElementById(id);

function getToken() { return localStorage.getItem(TOKEN_KEY) || ""; }
function setToken(t) { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); }

async function authorizedFetch(path, options) {
  options = options || {};
  options.headers = Object.assign({}, options.headers || {});
  const token = getToken();
  if (token) options.headers["Authorization"] = "Bearer " + token;
  let resp = await fetch(path, options);
  if (resp.status === 401) {
    const next = prompt("401 — Supervisor API token:", token);
    if (next !== null && next.trim()) {
      setToken(next.trim());
      options.headers["Authorization"] = "Bearer " + next.trim();
      resp = await fetch(path, options);
    }
  }
  return resp;
}

function uid() { return Math.random().toString(36).slice(2, 10); }
function escapeHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
async function api(path, options) {
  const resp = await authorizedFetch(path, options);
  let data = null;
  try { data = await resp.json(); } catch (e) {}
  return { ok: resp.ok, status: resp.status, data };
}

/* ---------------- glyphs & colors ---------------- */

const ST_GLYPH = { running:"●", starting:"◐", backoff:"◑", failed:"✕", stopped:"·" };
function dotClassOf(s) {
  if (s.state==="running") return s.ready?"running":"starting";
  if (s.state==="starting") return "starting";
  if (s.state==="backoff") return "backoff";
  if (s.state==="failed") return "failed";
  return "stopped";
}
const CAT_COLOR = {
  classification:"#00ff9c", object_detection:"#5fd3f0", face_detection:"#5fd3f0",
  scene_segmentation:"#c792ea", ocr:"#ffd166", matting:"#c792ea",
  enhancement:"#ffd166", feature_point:"#5fd3f0", feature_embedding:"#c792ea",
  mono_depth_estimation:"#ffd166", segment_anything:"#c792ea",
  diffusion:"#ff9e64", mot:"#5fd3f0", other:"#6fae85",
};
function catColor(c) { return CAT_COLOR[c] || CAT_COLOR.other; }

function uptimeOf(s) {
  if (!s.started_at_ms || s.state!=="running") return null;
  const sec = Math.max(0, Math.floor((Date.now()-s.started_at_ms)/1000));
  const h=Math.floor(sec/3600), m=Math.floor((sec%3600)/60), ss=sec%60;
  return (h>0?h+":":"")+String(m).padStart(2,"0")+":"+String(ss).padStart(2,"0");
}

/* ---------------- activity river ---------------- */

function riverPush(kind, text, serverId) {
  const n = new Date();
  const t = String(n.getHours()).padStart(2,"0")+":"+String(n.getMinutes()).padStart(2,"0")+":"+String(n.getSeconds()).padStart(2,"0");
  state.river.push({t, kind, text, serverId: serverId||null});
  if (state.river.length>200) state.river.shift();
  renderRiver();
}
function renderRiver() {
  const box=$("river-content"); if(!box) return;
  box.innerHTML = state.river.slice(-60).map(e=>{
    const c = e.kind==="err"?"var(--err)":e.kind==="ok"?"var(--ok)":"var(--mid)";
    const srv = e.serverId?` <span style="color:${catColor((serverById(e.serverId)||{}).category)}">${escapeHtml(e.serverId)}</span>`:"";
    return `<div class="river-line"><span class="river-t">${e.t}</span><span style="color:${c}">${escapeHtml(e.text)}</span>${srv}</div>`;
  }).join("");
  box.scrollTop = box.scrollHeight;
}

/* ---------------- polling store ---------------- */

let prevStates={};

async function refresh() {
  const [cat,st] = await Promise.all([api("/api/v1/catalog"), api("/api/v1/status")]);
  if (!cat.ok||!st.ok) {
    $("conn-status").textContent="LINK DOWN"; $("conn-status").className="conn-err";
    document.body.classList.add("link-down"); return;
  }
  document.body.classList.remove("link-down");
  $("conn-status").textContent="LINK OK"; $("conn-status").className="conn-ok";
  const byId={}; for(const s of(st.data.servers||[])) byId[s.id]=s;
  state.gateway = st.data.gateway||null;
  state.servers = (cat.data.servers||[]).map(s=>Object.assign({},s,byId[s.id]||{}));
  for(const s of state.servers){
    const prev=prevStates[s.id];
    if(prev&&prev!==s.state) riverPush(s.state==="running"?"ok":s.state==="failed"?"err":"info",`${s.id} ${prev}→${s.state}`,s.id);
    prevStates[s.id]=s.state;
  }
  renderCurrentView();
}

async function pollGpu() {
  const r = await api("/api/v1/gpu");
  const panel=$("gpu-panel"); if(!panel) return;
  if(!r.ok||!r.data||!r.data.available){
    panel.classList.add("gpu-na");
    $("gpu-name").textContent="gpu offline";
    $("gpu-metrics").innerHTML="— — —";
    return;
  }
  panel.classList.remove("gpu-na");
  state.gpuHistory = r.data.samples||[];
  $("gpu-name").textContent = r.data.name || "gpu";
  const last = state.gpuHistory.length ? state.gpuHistory[state.gpuHistory.length-1] : null;
  if(last){
    const cells = [
      {label:"UTIL", val:last.util<0?"--":last.util+"%", hot:last.util>85},
      {label:"VRAM", val:last.mem_total_mib>0?fmtMib(last.mem_used_mib)+"/"+fmtMib(last.mem_total_mib):"--", hot:last.mem_total_mib>0&&last.mem_used_mib/last.mem_total_mib>0.85},
      {label:"TEMP", val:last.temp_c<0?"--":last.temp_c+"°C", hot:last.temp_c>80},
      {label:"PWR", val:last.power_w<0?"--":last.power_w.toFixed(0)+"W", hot:last.power_w>300},
      {label:"SM CLK", val:last.clocks_sm_mhz<0?"--":last.clocks_sm_mhz+"MHz", hot:false},
      {label:"FAN", val:last.fan_pct<0?"--":last.fan_pct+"%", hot:false},
    ];
    $("gpu-metrics").innerHTML = cells.map(c=>
      `<div class="hud-cell${c.hot?" hot":""}"><div class="hud-val">${c.val}</div><div class="hud-k">${c.label}</div></div>`).join("");
  }
  drawGpuHero();
}

function fmtMib(m){ if(m==null||m<0)return"--"; return m>=1024?(m/1024).toFixed(1)+"G":m+"M"; }

/* ---------------- GPU hero: triple-line canvas + sparklines ---------------- */

function drawGpuHero(){
  const cv=$("gpu-canvas"); if(!cv) return;
  const dpr=window.devicePixelRatio||1;
  const w=cv.clientWidth,h=cv.clientHeight; if(w===0)return;
  cv.width=w*dpr; cv.height=h*dpr;
  const ctx=cv.getContext("2d"); ctx.scale(dpr,dpr); ctx.clearRect(0,0,w,h);
  const s=state.gpuHistory; if(s.length<2)return;
  const n=s.length, step=w/(n-1), pad=4;

  const norm=(v,lo,hi)=>Math.max(0,Math.min(1,(v-lo)/(hi-lo)));
  const yOf=(f)=>h-pad-(f*(h-pad*2));

  // grid
  ctx.strokeStyle="rgba(0,255,156,0.06)"; ctx.setLineDash([1,6]);
  for(let g=1;g<4;g++){ctx.beginPath();ctx.moveTo(0,h*g/4);ctx.lineTo(w,h*g/4);ctx.stroke();}
  ctx.setLineDash([]);

  function drawSeries(get,color,fill,glow,dash){
    ctx.beginPath();
    s.forEach((x,i)=>{const y=yOf(get(x)); i?ctx.lineTo(i*step,y):ctx.moveTo(0,y);});
    ctx.strokeStyle=color; ctx.lineWidth=1.2;
    if(dash){ctx.setLineDash([4,3]);}else{ctx.setLineDash([]);}
    if(glow){ctx.shadowColor=color;ctx.shadowBlur=8;}
    ctx.stroke(); ctx.shadowBlur=0; ctx.setLineDash([]);
    if(fill){
      ctx.lineTo(w,h);ctx.lineTo(0,h);ctx.closePath();
      const g2=ctx.createLinearGradient(0,0,0,h);
      g2.addColorStop(0,fill); g2.addColorStop(1,"rgba(0,0,0,0)");
      ctx.fillStyle=g2; ctx.fill();
    }
  }

  drawSeries(x=>norm(x.power_w<0?0:x.power_w,0,400),"rgba(255,158,100,0.5)",null,false,true);  // power (dim orange dashed)
  drawSeries(x=>norm(x.mem_total_mib>0?x.mem_used_mib/x.mem_total_mib:0,0,1),"#ffd166","rgba(255,209,102,0.06)",false,false); // mem (amber)
  drawSeries(x=>norm(x.util<0?0:x.util,0,100),"#00ff9c","rgba(0,255,156,0.12)",true,false);     // util (green glow)
}

/* draw sparkline in a small canvas */
function drawSparkline(canvas,data,color){
  if(!canvas||!data||data.length<2)return;
  const dpr=window.devicePixelRatio||1;
  const w=canvas.clientWidth,h=canvas.clientHeight; if(w===0)return;
  canvas.width=w*dpr; canvas.height=h*dpr;
  const ctx=canvas.getContext("2d"); ctx.scale(dpr,dpr); ctx.clearRect(0,0,w,h);
  const max=Math.max(...data,1), step=w/(data.length-1);
  ctx.beginPath();
  data.forEach((v,i)=>{const y=h-2-(v/max)*(h-4); i?ctx.lineTo(i*step,y):ctx.moveTo(0,y);});
  ctx.strokeStyle=color; ctx.lineWidth=1;
  ctx.shadowColor=color; ctx.shadowBlur=3;
  ctx.stroke(); ctx.shadowBlur=0;
  ctx.lineTo(w,h);ctx.lineTo(0,h);ctx.closePath();
  ctx.fillStyle=color.replace(")",",0.1)").replace("#","rgba(").replace(/rgba\((\w+)\)/, (m,c)=>{
    // hex to rgba
    if(c.length===6){const r=parseInt(c.slice(0,2),16),g=parseInt(c.slice(2,4),16),b=parseInt(c.slice(4,6),16);return`rgba(${r},${g},${b},0.08)`;}
    return m;
  });
  ctx.fill();
}

/* ---------------- router ---------------- */

function currentRoute(){
  const h=location.hash.replace(/^#/,"");
  const m=h.match(/^\/model\/([A-Za-z0-9_-]+)/);
  if(m)return{view:"model",id:m[1]};
  return{view:"overview"};
}
function navigate(h){if(location.hash!==h)location.hash=h;}
window.addEventListener("hashchange",()=>renderCurrentView());

function renderCurrentView(){
  const r=currentRoute();
  if(r.view==="model"&&serverById(r.id)){
    state.selectedId=r.id; showView("workbench"); renderWorkbench();
  }else{
    state.selectedId=null; showView("overview"); renderOverview();
  }
}
function showView(n){
  $("view-overview").classList.toggle("hidden",n!=="overview");
  $("view-workbench").classList.toggle("hidden",n!=="workbench");
}

/* ---------------- OVERVIEW: HUD ---------------- */

function renderOverview(){
  const grid=$("fleet-grid");
  $("fleet-count").textContent=state.servers.length;
  const running=state.servers.filter(s=>["running","starting","backoff"].includes(s.state)).length;
  $("fleet-live").textContent=running+" live";
  const frag=document.createDocumentFragment();
  const groups={};
  for(const s of state.servers)(groups[s.category]=groups[s.category]||[]).push(s);
  for(const cat of Object.keys(groups).sort()){
    const head=document.createElement("div");head.className="fleet-cat";head.textContent=cat;
    head.style.color=catColor(cat);head.style.borderColor=catColor(cat)+"33";
    frag.appendChild(head);
    for(const s of groups[cat]){
      const st=dotClassOf(s);
      const isRun=s.state==="running";
      const tile=document.createElement("div");
      tile.className="cartridge"+(isRun?" live":"")+(s.state==="failed"?" dead":"");
      if(isRun){tile.style.setProperty("--cat-glow",catColor(cat));tile.style.borderColor=catColor(cat)+"66";}
      tile.innerHTML=
        `${isRun?'<div class="accent-top"></div>':''}
         <div class="cartridge-row">
           <span class="st ${st}">${ST_GLYPH[st]}</span>
           <span class="cartridge-name">${escapeHtml(s.id.toLowerCase())}</span>
           ${s.restart_count>0?`<span class="badge restarts">↻${s.restart_count}</span>`:""}
         </div>
         <div class="cartridge-sub">${isRun?`<span class="uptime" data-id="${s.id}">${uptimeOf(s)||""}</span>`:escapeHtml(s.state)}</div>
         <div class="cartridge-port">${s.port}</div>
         <canvas class="cartridge-spark" data-id="${s.id}" width="120" height="22"></canvas>
         ${isRun?`<div class="resource-bar"><div class="resource-fill" data-id="${s.id}"></div></div>`:''}`;
      tile.onclick=()=>navigate("#/model/"+s.id);
      frag.appendChild(tile);
    }
  }
  grid.innerHTML="";grid.appendChild(frag);
  renderGatewayBar();
  // draw sparklines for running models (infer rate from state.river)
  drawFleetSparks();
}

function drawFleetSparks(){
  for(const s of state.servers){
    if(s.state!=="running")continue;
    const cv=document.querySelector(`canvas.cartridge-spark[data-id="${s.id}"]`);
    if(!cv)continue;
    // use inferHistory entries for this server
    const data=state.inferHistory.filter(x=>x.serverId===s.id).slice(-30).map(x=>x.ms);
    if(data.length>=2)drawSparkline(cv,data,catColor(s.category));
  }
}

/* ---------------- WORKBENCH ---------------- */

function renderWorkbench(){
  const s=serverById(state.selectedId); if(!s)return;
  const st=dotClassOf(s);
  $("wb-breadcrumb").textContent="‹ fleet / "+s.id.toLowerCase();
  $("wb-breadcrumb").onclick=()=>navigate("#/overview");
  $("wb-title").innerHTML=`<span class="st ${st}">${ST_GLYPH[st]}</span> ${escapeHtml(s.id.toLowerCase())}<span class="wb-state">${escapeHtml(s.state)}${s.ready?" · ready":""}</span>`;
  $("wb-identity").innerHTML=
    `<div class="id-row"><span class="id-k">port</span><span>:${s.port}</span></div>
     <div class="id-row"><span class="id-k">uri</span><span>${escapeHtml(s.uri||"")}</span></div>
     <div class="id-row"><span class="id-k">cat</span><span style="color:${catColor(s.category)}">${escapeHtml(s.category)}</span></div>
     <div class="id-row"><span class="id-k">↻</span><span>${s.restart_count||0}</span></div>
     <div class="id-row"><span class="id-k">up</span><span class="uptime" data-id="${s.id}">${uptimeOf(s)||"--"}</span></div>
     <div class="id-row"><span class="id-k">pid</span><span>${s.pid>0?s.pid:"—"}</span></div>`;
  const running=["running","starting","backoff"].includes(s.state);
  $("wb-start").disabled=running;
  $("wb-restart").disabled=!running;
  $("wb-stop").disabled=!running;
  $("wb-start").onclick=()=>controlServer(s.id,"start");
  $("wb-restart").onclick=()=>controlServer(s.id,"restart");
  $("wb-stop").onclick=()=>controlServer(s.id,"stop");
  $("image-input-area").classList.remove("hidden");
  const hint=$("empty-hint"); if(hint)hint.classList.add("hidden");
  state.logServerId=s.id; resetLogState(s.id); syncLogSelector(); renderFileList();
  // fetch process info for gauges
  pollProcessInfo(s.id);
}

async function pollProcessInfo(id){
  const r=await api(`/api/v1/servers/${encodeURIComponent(id)}/process`);
  const box=$("wb-process"); if(!box) return;
  if(!r.ok||!r.data||!r.data.process){box.innerHTML="";return;}
  const p=r.data.process;
  const rss=p.rss_kb>0?(p.rss_kb/1024).toFixed(1)+"M":"--";
  box.innerHTML=
    `<div class="id-row"><span class="id-k">rss</span><span>${rss}</span></div>
     <div class="id-row"><span class="id-k">thr</span><span>${p.threads>0?p.threads:"--"}</span></div>`;
}

function serverById(id){return state.servers.find(s=>s.id===id)||null;}

setInterval(()=>{document.querySelectorAll(".uptime").forEach(el=>{
  const s=serverById(el.dataset.id);if(s)el.textContent=uptimeOf(s)||"--";});},1000);
setInterval(()=>{if(state.selectedId)pollProcessInfo(state.selectedId);},5000);

/* ---------------- gateway / control ---------------- */

function gatewayBaseUrl(){
  const g=state.gateway;if(!g||!g.address)return"";
  let host=g.address.host||"";
  if(host==="0.0.0.0"||host==="::"||host==="[::]")host=window.location.hostname||"127.0.0.1";
  return`http://${host}:${g.address.port}`;
}
function renderGatewayBar(){
  const g=state.gateway;const bar=$("gateway-status");
  if(!g){bar.textContent="gw ?";return;}
  const addr=g.address?`${g.address.host}:${g.address.port}`:"";
  const cls=g.state==="running"?"gw-ok":"gw-bad";
  bar.innerHTML=`gw <span class="${cls}">${g.state==="running"?"●":"○"}</span>${addr?` ${escapeHtml(addr)}`:""}${g.state==="running"?"":" down"}`;
}
async function controlServer(id,action){
  const r=await api(`/api/v1/servers/${encodeURIComponent(id)}/${action}`,{method:"POST"});
  if(!r.ok){showToast(`${action} fail: ${r.status}`,"error");riverPush("err",`${id} ${action} HTTP ${r.status}`,id);}
  else{showToast(`${action} ${id}`,"success");riverPush("ok",`${id} ${action} ok`,id);}
  refresh();
}

/* ---------------- toast ---------------- */
function showToast(msg,type="info"){
  const el=document.createElement("div");el.className="toast "+type;el.textContent=msg;
  $("toast-container").appendChild(el);
  setTimeout(()=>{el.style.opacity="0";},2400);
  setTimeout(()=>{el.remove();},2800);
}

/* ---------------- test bench ---------------- */

function base64ToSrc(b){return"data:image/png;base64,"+b.replace(/^data:[^,]+,/,"");}
function loadImageAsBase64(f){
  return new Promise((resolve,reject)=>{
    const r=new FileReader();
    r.onload=()=>{resolve(r.result.slice(r.result.indexOf(",")+1));};
    r.onerror=reject;r.readAsDataURL(f);
  });
}
async function addFiles(fl){
  for(const f of fl){
    if(!f.type.startsWith("image/"))continue;
    const b64=await loadImageAsBase64(f);
    state.files.push({name:f.name,url:base64ToSrc(b64),base64:b64});
  }
  renderFileList();
}
function renderFileList(){
  const box=$("file-list");if(!box)return;box.innerHTML="";
  for(const f of state.files){
    const chip=document.createElement("div");chip.className="file-chip";
    chip.innerHTML=`<img src="${f.url}"><span>${escapeHtml(f.name)}</span>`;
    const rm=document.createElement("span");rm.textContent="✕";rm.className="chip-rm";
    rm.onclick=()=>{state.files=state.files.filter(x=>x!==f);renderFileList();};
    chip.appendChild(rm);box.appendChild(chip);
  }
}
function unifiedPayload(d){
  if(d&&Array.isArray(d.results)&&d.results.length)return d.results[0].data;
  return d&&d.data!==undefined?d.data:null;
}
function topScore(p){
  if(!p)return null;
  if(Array.isArray(p)&&p.length&&typeof p[0].score==="number")return p[0].score;
  return null;
}

async function sendBatch(){
  const s=serverById(state.selectedId);if(!s||!state.files.length)return;
  const base=gatewayBaseUrl();if(!base){showToast("gateway unknown","error");return;}
  // pre-flight: tell the user exactly which URL will be hit and if it's reachable
  const inferUrl=`${base}/v1/models/${encodeURIComponent(s.id)}/infer`;
  try{
    const pre=await fetch(`${base}/healthz`,{method:"GET",signal:AbortSignal.timeout(3000)});
    if(!pre.ok)throw new Error(`gateway healthz returned ${pre.status}`);
  }catch(e){
    const detail=e&&e.name==="TimeoutError"?"timeout 3s":(e&&e.message)||String(e);
    showToast(`gateway unreachable: ${detail}\n→ ${base}\n检查隧道(8080)和 WSL gateway 进程`, "error");
    riverPush("err",`pre-flight fail: ${base} (${detail})`,s.id);
    return;
  }
  $("btn-cancel").classList.remove("hidden");setBatchProgress(0,state.files.length);
  let aborted=false;state.batchAbort=()=>{aborted=true;};
  let done=0;
  for(const f of state.files){
    if(aborted)break;
    const reqId=uid(),t0=performance.now();
    try{
      const resp=await authorizedFetch(`${base}/v1/models/${encodeURIComponent(s.id)}/infer`,{
        method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({req_id:reqId,images:[f.base64]}),
      });
      const ms=Math.round(performance.now()-t0);
      const body=await resp.json().catch(()=>null);
      state.inferHistory.push({t:Date.now(),serverId:s.id,ms,ok:resp.ok});
      if(state.inferHistory.length>600)state.inferHistory.shift();
      if(resp.ok){addResultCard(s,f,body,reqId,ms);riverPush("ok",`${s.id} 200 ${ms}ms`,s.id);}
      else{showToast(`HTTP ${resp.status}`,"error");riverPush("err",`${s.id} HTTP ${resp.status}`,s.id);}
    }catch(e){
      const url = `${base}/v1/models/${encodeURIComponent(s.id)}/infer`;
      const detail = e && e.message ? e.message : String(e);
      showToast(`infer fail: ${detail}\n→ ${url}`, "error");
      riverPush("err", `${s.id} transport: ${detail} (${url})`, s.id);
    }
    done++;setBatchProgress(done,state.files.length);
  }
  state.batchAbort=null;$("btn-cancel").classList.add("hidden");
  drawFleetSparks();
}

function setBatchProgress(d,t){
  $("batch-progress").classList.toggle("hidden",t===0);
  $("batch-progress-fill").style.width=t?((d/t)*100).toFixed(1)+"%":"0";
  $("batch-progress-text").textContent=`${d}/${t}`;
}

function addResultCard(server,input,result,reqId,elapsed){
  const payload=unifiedPayload(result);
  const card=document.createElement("div");card.className="result-card";
  const score=topScore(payload);
  const detCount=Array.isArray(payload)?payload.length:0;
  card.innerHTML=
    `<div class="head">
       <span class="req-meta">${escapeHtml(input.name)} · ${elapsed}ms · ${detCount?detCount+" hits · ":""}${escapeHtml(reqId)}</span>
       <span class="req-meta">${score!=null?"top "+score.toFixed(3):""}</span>
     </div>`;
  const vizWrap=document.createElement("div");vizWrap.className="viz";card.appendChild(vizWrap);
  const raw=document.createElement("details");raw.className="raw-json";raw.innerHTML="<summary>raw</summary>";
  const pre=document.createElement("pre");pre.textContent=JSON.stringify(result,null,1);raw.appendChild(pre);
  card.appendChild(raw);
  $("results-list").prepend(card);
  visualize(server,input,payload,vizWrap).catch(()=>{});
}

async function visualize(server,input,payload,vizWrap){
  const img=new Image();img.src=input.url;await img.decode().catch(()=>{});
  if(Array.isArray(payload)&&payload.length&&typeof payload[0].bbox==="object"){
    const cv=document.createElement("canvas");cv.className="overlay";vizWrap.appendChild(cv);
    drawDetection(cv,img.src,payload);
    // class legend
    const legend=document.createElement("div");legend.className="det-legend";
    const seen=new Set();
    for(const b of payload){
      if(seen.has(b.class_id))continue;seen.add(b.class_id);
      const hue=(b.class_id*47)%360;
      legend.innerHTML+=`<span class="det-chip" style="--h:${hue}"><span class="det-dot"></span>${escapeHtml(b.category||b.class_id)} ${b.score!=null?b.score.toFixed(2):""}</span>`;
    }
    vizWrap.appendChild(legend);
    return;
  }
  if(payload&&Array.isArray(payload.regions)){
    const cv=document.createElement("canvas");cv.className="overlay";vizWrap.appendChild(cv);
    drawOcr(cv,img.src,payload.regions);return;
  }
  if(payload&&(payload.image||payload.colorized_mask||payload.alpha||payload.matting_image)){
    const out=payload.image||payload.colorized_mask||payload.alpha||payload.matting_image;
    const im=new Image();im.src=base64ToSrc(out);vizWrap.appendChild(im);return;
  }
  if(payload&&Array.isArray(payload.keypoints)){
    const cv=document.createElement("canvas");cv.className="overlay";vizWrap.appendChild(cv);
    drawKeypoints(cv,img.src,payload);return;
  }
  vizWrap.appendChild(img);
}

function drawDetection(canvas,imgUrl,boxes){
  const img=new Image();
  img.onload=()=>{
    canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;
    const ctx=canvas.getContext("2d");
    ctx.drawImage(img,0,0);
    ctx.lineWidth=Math.max(2,Math.round(img.naturalWidth/300));
    ctx.font=Math.max(14,Math.round(img.naturalWidth/40))+"px monospace";
    for(const b of boxes){
      const[x1,y1,x2,y2]=b.bbox;
      const hue=(b.class_id*47)%360;
      ctx.strokeStyle=`hsl(${hue} 90% 60%)`;
      ctx.shadowColor=ctx.strokeStyle;ctx.shadowBlur=8;
      ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.shadowBlur=0;
      const label=`${b.category||b.class_id} ${b.score!=null?b.score.toFixed(2):""}`;
      const tw=ctx.measureText(label).width+10;
      ctx.fillStyle=`hsl(${hue} 90% 60%)`;
      ctx.fillRect(x1,Math.max(0,y1-22),tw,20);
      ctx.fillStyle="#000";
      ctx.fillText(label,x1+5,Math.max(15,y1-7));
    }
  };
  img.src=imgUrl;
}
function drawOcr(canvas,imgUrl,regions){
  const img=new Image();
  img.onload=()=>{
    canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;
    const ctx=canvas.getContext("2d");ctx.drawImage(img,0,0);
    ctx.lineWidth=2;ctx.strokeStyle="#ffd166";
    for(const r of regions){
      const pts=r.points||r.bbox_points||[];
      if(pts.length===4){ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);
        for(let i=1;i<4;i++)ctx.lineTo(pts[i][0],pts[i][1]);ctx.closePath();ctx.stroke();}
      if(r.text){ctx.fillStyle="#ffd166";ctx.font="14px monospace";
        ctx.fillText(r.text,pts[0]?pts[0][0]:4,pts[0]?pts[0][1]-4:14);}
    }
  };
  img.src=imgUrl;
}
function drawKeypoints(canvas,imgUrl,payload){
  const img=new Image();
  img.onload=()=>{
    canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;
    const ctx=canvas.getContext("2d");ctx.drawImage(img,0,0);
    ctx.fillStyle="#00ff9c";ctx.shadowColor="#00ff9c";ctx.shadowBlur=5;
    for(const p of payload.keypoints){ctx.beginPath();ctx.arc(p.x,p.y,3,0,Math.PI*2);ctx.fill();}
    ctx.shadowBlur=0;
  };
  img.src=imgUrl;
}

/* ---------------- logs ---------------- */

function resetLogState(id){
  state.logs[id]={offset:0,filter:"",paused:false,follow:true,lines:[]};
  const el=$("log-content");if(el)el.textContent="";
}
function syncLogSelector(){
  const sel=$("log-server");if(!sel)return;sel.innerHTML="";
  for(const s of state.servers){
    const opt=document.createElement("option");opt.value=s.id;opt.textContent=s.id;sel.appendChild(opt);
  }
  if(state.logServerId)sel.value=state.logServerId;
}
function renderLogContent(){
  const el=$("log-content");if(!el||!state.logServerId)return;
  const st=state.logs[state.logServerId];if(!st)return;
  const lines=st.filter?st.lines.filter(l=>l.toLowerCase().includes(st.filter)):st.lines;
  el.innerHTML=lines.slice(-400).map(l=>highlightLine(l,st.filter)).join("\n");
  if(st.follow)el.scrollTop=el.scrollHeight;
}
function highlightLine(line,filter){
  const esc=escapeHtml(line);
  // colorize ERROR lines
  if(/\bERROR\b|FATAL/.test(line))return`<span class="log-err">${esc}</span>`;
  if(!filter)return esc;
  const idx=esc.toLowerCase().indexOf(filter);
  if(idx<0)return esc;
  return esc.slice(0,idx)+"<mark>"+esc.slice(idx,idx+filter.length)+"</mark>"+esc.slice(idx+filter.length);
}
async function pollLogs(){
  if(!state.logServerId)return;
  const st=state.logs[state.logServerId];if(!st||st.paused)return;
  const r=await api(`/api/v1/servers/${encodeURIComponent(state.logServerId)}/logs?offset=${st.offset}&limit=100`);
  if(!r.ok||!r.data)return;
  st.lines.push(...(r.data.lines||[]));
  st.offset=r.data.offset!=null?r.data.offset+(r.data.lines||[]).length:st.lines.length;
  $("log-meta").textContent=`${st.lines.length} lines`;
  renderLogContent();
  for(const l of(r.data.lines||[])){
    if(/\bERROR\b|FATAL/.test(l))riverPush("err",l.slice(0,120),state.logServerId);
  }
}

/* ---------------- command palette ---------------- */

function paletteItems(){
  const items=[];
  for(const s of state.servers){
    items.push({label:`open ${s.id.toLowerCase()}`,hint:"nav",act:()=>navigate("#/model/"+s.id)});
    const running=["running","starting","backoff"].includes(s.state);
    if(!running)items.push({label:`start ${s.id.toLowerCase()}`,hint:"ctrl",act:()=>controlServer(s.id,"start")});
    if(running)items.push({label:`stop ${s.id.toLowerCase()}`,hint:"ctrl",act:()=>controlServer(s.id,"stop")});
    items.push({label:`restart ${s.id.toLowerCase()}`,hint:"ctrl",act:()=>controlServer(s.id,"restart")});
  }
  items.push({label:"overview",hint:"nav",act:()=>navigate("#/overview")});
  items.push({label:"token",hint:"set",act:()=>{
    const t=prompt("Supervisor API token:",getToken());
    if(t!==null){setToken(t.trim());location.reload();}
  }});
  return items;
}
function fuzzyMatch(q,l){
  let li=0,sc=0,streak=0;const Q=q.toLowerCase(),L=l.toLowerCase();
  for(const ch of Q){
    const f=L.indexOf(ch,li);if(f<0)return-1;
    streak=f===li?streak+1:0;sc+=1+streak;li=f+1;
  }
  return sc;
}
function openPalette(){
  const p=$("palette");p.classList.remove("hidden");
  const input=$("palette-input");input.value="";renderPalette("");input.focus();
}
function closePalette(){$("palette").classList.add("hidden");}
function renderPalette(query){
  const list=$("palette-list");
  const items=paletteItems()
    .map(it=>({it,sc:query?fuzzyMatch(query,it.label):0}))
    .filter(x=>x.sc>=0).sort((a,b)=>b.sc-a.sc).slice(0,10);
  let sel=0;list.innerHTML="";
  for(const{it}of items){
    const row=document.createElement("div");row.className="palette-row";
    row.innerHTML=`<span>${escapeHtml(it.label)}</span><span class="palette-hint">${it.hint}</span>`;
    row.onclick=()=>{closePalette();it.act();};
    list.appendChild(row);
  }
  if(items.length)list.firstChild.classList.add("sel");
  $("palette-input").onkeydown=(ev)=>{
    const rows=[...list.children];
    if(ev.key==="Escape")closePalette();
    else if(ev.key==="Enter"){if(rows[sel]){closePalette();items[sel].it.act();}}
    else if(ev.key==="ArrowDown"||ev.key==="ArrowUp"){
      ev.preventDefault();
      if(rows[sel])rows[sel].classList.remove("sel");
      sel=ev.key==="ArrowDown"?Math.min(rows.length-1,sel+1):Math.max(0,sel-1);
      if(rows[sel])rows[sel].classList.add("sel");
    }else setTimeout(()=>renderPalette($("palette-input").value),0);
  };
}
document.addEventListener("keydown",(ev)=>{
  if((ev.metaKey||ev.ctrlKey)&&ev.key.toLowerCase()==="k"){
    ev.preventDefault();
    $("palette").classList.contains("hidden")?openPalette():closePalette();
  }
  if(ev.key==="Escape"&&!$("palette").classList.contains("hidden"))closePalette();
});

/* ---------------- wiring ---------------- */

function wire(){
  $("btn-pick-file").onclick=()=>$("file-input").click();
  $("btn-pick-folder").onclick=()=>$("folder-input").click();
  $("file-input").onchange=(ev)=>{addFiles(ev.target.files);ev.target.value="";};
  $("folder-input").onchange=(ev)=>{addFiles(ev.target.files);ev.target.value="";};
  const dz=$("drop-zone");
  dz.ondragover=(ev)=>{ev.preventDefault();dz.classList.add("dragover");};
  dz.ondragleave=()=>dz.classList.remove("dragover");
  dz.ondrop=(ev)=>{ev.preventDefault();dz.classList.remove("dragover");addFiles(ev.dataTransfer.files);};
  $("btn-send").onclick=()=>sendBatch();
  $("btn-cancel").onclick=()=>{if(state.batchAbort)state.batchAbort();};
  $("btn-token").onclick=()=>{
    const t=prompt("Supervisor API token:",getToken());
    if(t!==null){setToken(t.trim());showToast("token saved","success");refresh();}
  };
  $("btn-log-pause").onclick=()=>{
    const st=state.logs[state.logServerId];if(!st)return;
    st.paused=!st.paused;$("btn-log-pause").textContent=st.paused?"resume":"pause";
  };
  $("log-follow").onchange=(ev)=>{
    const st=state.logs[state.logServerId];if(st)st.follow=ev.target.checked;
  };
  $("log-server").onchange=(ev)=>{state.logServerId=ev.target.value;resetLogState(state.logServerId);};
  $("log-filter").onkeydown=(ev)=>{
    if(ev.key!=="Enter")return;
    const st=state.logs[state.logServerId];
    if(st){st.filter=ev.target.value.toLowerCase();renderLogContent();}
  };
  $("btn-log-clear").onclick=()=>resetLogState(state.logServerId);
  $("palette-backdrop").onclick=closePalette;
  window.addEventListener("resize",()=>{drawGpuHero();drawFleetSparks();});
}

/* ---------------- boot ---------------- */

wire();
if(!sessionStorage.getItem("booted")){
  sessionStorage.setItem("booted","1");document.body.classList.add("boot");
  setTimeout(()=>document.body.classList.remove("boot"),600);
}
if(!location.hash)location.hash="#/overview";
renderCurrentView();
refresh();pollGpu();
setInterval(refresh,2000);
setInterval(pollGpu,2000);
setInterval(pollLogs,1000);

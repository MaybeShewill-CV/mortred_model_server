"use strict";

/* Mortred Supervisor UI — Mission Control HUD.
 * Sci-fi command-deck aesthetic: full-field GPU heartbeat, glowing fleet
 * cartridges with live sparklines, telemetry band with req-rate, and a
 * per-model workbench with identity gauges. Zero deps, zero build. */

const state = {
  servers: [], gateway: null, selectedId: null,
  batchAbort: null, logs: {}, logServerId: null,
  river: [], gpuHistory: [],
  inferHistory: [],     // [{t, ok, ms, serverId}] from sendBatch
  fleetFilter: "all",   // "all" | category id
  fleetState: "all",    // "all" | running | stopped | failed | starting
  bench: {},            // per model id: { files: [{name,url,base64,status}], results: [] }
  inflight: 0,          // bench requests currently in flight from this tab
};
function benchOf(id){ return state.bench[id] || (state.bench[id]={files:[],results:[]}); }

const TOKEN_KEY = "mortred_supervisor_token";
const $ = (id) => document.getElementById(id);

function getToken() { return localStorage.getItem(TOKEN_KEY) || ""; }
function setToken(t) { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); }

/* styled token prompt (replaces native prompt(); resolves null on cancel) */
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

function uid() { return Math.random().toString(36).slice(2, 10); }
function escapeHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
/* charts follow the CSS token system — one source of truth for color */
const cssVarCache = {};
function cssVar(name) {
  if (!(name in cssVarCache)) {
    cssVarCache[name] = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  return cssVarCache[name];
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
/* identity palette: one hue family per task category, uniform perceptual
 * weight (same lightness/chroma band) so no category shouts over statuses */
const CAT_COLOR = {
  classification:"#2dd4bf", object_detection:"#38bdf8", face_detection:"#60a5fa",
  scene_segmentation:"#a78bfa", ocr:"#fbbf24", matting:"#e879f9",
  enhancement:"#fb923c", feature_point:"#fb7185", feature_embedding:"#a78bfa",
  mono_depth_estimation:"#fb923c", segment_anything:"#e879f9",
  diffusion:"#f472b6", mot:"#60a5fa", other:"#94a3b8",
};
function catColor(c) { return CAT_COLOR[c] || CAT_COLOR.other; }

const STATE_COLOR = {
  running:"var(--acc)", starting:"var(--warn)", backoff:"var(--warn)",
  failed:"var(--err)", stopped:"var(--txt-dim)",
};

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
  const count=$("river-count"); if(count) count.textContent=state.river.length+" events";
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
  let cat, st;
  try {
    [cat,st] = await Promise.all([api("/api/v1/catalog"), api("/api/v1/status")]);
  } catch (e) {
    cat = st = { ok: false };
  }
  if (!cat.ok||!st.ok) {
    $("conn-status").textContent="LINK DOWN"; $("conn-status").className="conn-err";
    document.body.classList.add("link-down");
    $("link-overlay").classList.remove("hidden");
    return;
  }
  document.body.classList.remove("link-down");
  $("link-overlay").classList.add("hidden");
  $("conn-status").textContent="LINK OK"; $("conn-status").className="conn-ok";
  const byId={}; for(const s of(st.data.servers||[])) byId[s.id]=s;
  state.gateway = st.data.gateway||null;
  state.servers = (cat.data.servers||[]).map(s=>Object.assign({},s,byId[s.id]||{}));
  for(const s of state.servers){
    const prev=prevStates[s.id];
    if(prev&&prev!==s.state) riverPush(s.state==="running"?"ok":s.state==="failed"?"err":"info",`${s.id} ${prev}→${s.state}`,s.id);
    prevStates[s.id]=s.state;
  }
  const liveN = state.servers.filter(s=>["running","starting","backoff"].includes(s.state)).length;
  document.title = (liveN>0?`●${liveN} live · `:"")+"Mortred Supervisor";
  updateFavicon(liveN);
  bootSequence(liveN);
  renderCurrentView();
}

/* pulsing favicon: the console's heartbeat lives in the browser chrome */
function updateFavicon(liveN){
  const c=document.createElement("canvas");c.width=c.height=32;
  const x=c.getContext("2d");
  x.fillStyle="#07090b";x.fillRect(0,0,32,32);
  const on=liveN>0;
  const pulse=on?0.75+0.25*Math.abs(Math.sin(Date.now()/700)):1;
  x.fillStyle=on?"#00e08c":"#6f857b";
  x.shadowColor=x.fillStyle;x.shadowBlur=on?10:4;
  x.beginPath();x.arc(16,16,(on?8.5:6)*pulse,0,Math.PI*2);x.fill();
  if(on){x.shadowBlur=0;x.fillStyle="#04120b";x.beginPath();x.arc(16,16,3.4,0,Math.PI*2);x.fill();}
  let link=document.querySelector("link[rel='icon']");
  if(!link){link=document.createElement("link");link.rel="icon";document.head.appendChild(link);}
  link.href=c.toDataURL("image/png");
}

/* one-shot typed boot line — the console comes alive on first link */
let bootDone = false;
function bootSequence(liveN){
  if (bootDone) return;
  bootDone = true;
  setTimeout(()=>{
    const gw=state.gateway&&state.gateway.state==="running"?"gateway up":"gateway --";
    const gpuTxt=state.gpuHistory.length?"gpu online":"gpu --";
    typeBootLines([
      "// mortred supervisor · initiating link…",
      `// catalog ${state.servers.length} models · ${gw} · ${gpuTxt}`,
      `// link established · ${liveN} live · console ready`,
    ]);
  }, state.gpuHistory.length?0:700);
}
function typeBootLines(lines){
  const el=$("boot-line");
  const reduced=window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const hide=()=>{ el.classList.add("fading"); setTimeout(()=>el.classList.add("hidden"),550); };
  if(reduced){ el.textContent=lines.join("\n"); setTimeout(hide,3200); return; }
  let li=0, ci=0;
  el.textContent="";
  const typer=setInterval(()=>{
    if(li>=lines.length){ clearInterval(typer); setTimeout(hide,1700); return; }
    ci++;
    el.textContent=lines.slice(0,li).join("\n")+(li?"\n":"")+lines[li].slice(0,ci);
    if(ci>=lines[li].length){ li++; ci=0; }
  },13);
}

async function pollGpu() {
  const r = await api("/api/v1/gpu");
  const panel=$("gpu-panel"); if(!panel) return;
  if(!r.ok||!r.data||!r.data.available){
    panel.classList.add("gpu-na");
    $("gpu-name").textContent="gpu offline";
    $("gpu-meta").textContent="";
    $("gpu-metrics").innerHTML="— — —";
    return;
  }
  panel.classList.remove("gpu-na");
  state.gpuHistory = r.data.samples||[];
  $("gpu-name").textContent = r.data.name || "gpu";
  const winS = Math.round(state.gpuHistory.length*2);
  $("gpu-meta").textContent = state.gpuHistory.length ? `${winS>=120?(winS/60)+"m":winS+"s"} window · 2s poll` : "";
  const last = state.gpuHistory.length ? state.gpuHistory[state.gpuHistory.length-1] : null;
  if(last){
    updateHudCells(last);
  }
  drawGpuHero();
  updateWorkbenchGpu();
}

/* compact GPU readout on the workbench — you watch the GPU where you send */
function updateWorkbenchGpu(){
  const box=$("wb-gpu");if(!box)return;
  const s=state.gpuHistory;
  const last=s.length?s[s.length-1]:null;
  box.classList.toggle("hidden",!last);
  if(!last)return;
  const set=(id,txt,hot)=>{
    const el=$(id);if(!el)return;el.textContent=txt;
    el.parentElement.classList.toggle("hot",!!hot);
  };
  set("wg-util",last.util>=0?last.util+"%":"--",last.util>85);
  set("wg-vram",last.mem_total_mib>0?Math.round(100*last.mem_used_mib/last.mem_total_mib)+"%":"--",
      last.mem_total_mib>0&&last.mem_used_mib/last.mem_total_mib>0.85);
  set("wg-temp",last.temp_c>=0?last.temp_c+"°C":"--",last.temp_c>80);
  const cv=$("wg-spark");
  if(cv){
    const data=s.slice(-40).map(x=>x.util<0?0:x.util);
    drawSparkline(cv,data.length>=2?data:null,cssVar("--acc"));
  }
}

/* numeric tween so the HUD counts toward its new value */
const hudTween = {};
function tweenValue(key, target, fmt, render){
  const from = (hudTween[key]!=null && isFinite(hudTween[key])) ? hudTween[key] : target;
  hudTween[key] = target;
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches || from===target){
    render(fmt(target)); return;
  }
  const t0=performance.now(), dur=420;
  const step=(t)=>{
    const k=Math.min(1,(t-t0)/dur), e=1-Math.pow(1-k,3);
    render(fmt(from+(target-from)*e));
    if(k<1) requestAnimationFrame(step); else render(fmt(target));
  };
  requestAnimationFrame(step);
}

function updateHudCells(last){
  const s = state.gpuHistory;
  // trend vs ~32s ago for the hero readout
  const ref = s.length > 16 ? s[s.length - 17] : null;
  const dUtil = ref && last.util >= 0 && ref.util >= 0 ? last.util - ref.util : null;
  const deltaHtml = dUtil == null ? "" :
    `<div class="hud-delta ${dUtil >= 0 ? "up" : "down"}">${dUtil >= 0 ? "▲" : "▼"} ${Math.abs(Math.round(dUtil))}% / 32s</div>`;
  const hero = last.util==null||last.util<0
    ? `<div class="hud-cell hero-cell"><div class="hud-val">--</div><div class="hud-k">util</div></div>`
    : `<div class="hud-cell hero-cell${last.util>85?" hot":""}"><div class="hud-val" data-tw="util">--</div><div class="hud-k">util</div>${deltaHtml}</div>`;
  const cells = [
    {label:"VRAM", key:null, plain:last.mem_total_mib>0?fmtMib(last.mem_used_mib)+"/"+fmtMib(last.mem_total_mib):"--",
      hot:last.mem_total_mib>0&&last.mem_used_mib/last.mem_total_mib>0.85},
    {label:"TEMP",  key:"temp",   val:last.temp_c, fmt:v=>Math.round(v)+"°C", hot:last.temp_c>80},
    {label:"PWR",   key:"pwr",    val:last.power_w,fmt:v=>Math.round(v)+"W",  hot:last.power_w>300},
    {label:"SM CLK",key:"clk",    val:last.clocks_sm_mhz, fmt:v=>Math.round(v)+"MHz", hot:false},
    {label:"FAN",   key:"fan",    val:last.fan_pct,fmt:v=>Math.round(v)+"%",  hot:false},
  ];
  let html = hero + cells.map(c=>{
    if(c.plain!==undefined){
      return `<div class="hud-cell${c.hot?" hot":""}"><div class="hud-val">${c.plain}</div><div class="hud-k">${c.label}</div></div>`;
    }
    if(c.val==null||c.val<0){
      return `<div class="hud-cell"><div class="hud-val">--</div><div class="hud-k">${c.label}</div></div>`;
    }
    return `<div class="hud-cell${c.hot?" hot":""}"><div class="hud-val" data-tw="${c.key}">--</div><div class="hud-k">${c.label}</div></div>`;
  }).join("");
  $("gpu-metrics").innerHTML=html;
  if(last.util!=null&&last.util>=0){
    const el=document.querySelector('.hud-val[data-tw="util"]');
    if(el)tweenValue("util",last.util,v=>Math.round(v)+"%",(str)=>el.textContent=str);
  }
  for(const c of cells){
    if(c.val==null||c.val<0)continue;
    const el=document.querySelector(`.hud-val[data-tw="${c.key}"]`);
    if(el)tweenValue(c.key,c.val,c.fmt,(s)=>el.textContent=s);
  }
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
  ctx.strokeStyle="rgba(255,255,255,0.05)"; ctx.setLineDash([1,6]);
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

  drawSeries(x=>norm(x.power_w<0?0:x.power_w,0,400),"rgba(255,158,100,0.45)",null,false,true);  // power (dim orange dashed)
  drawSeries(x=>norm(x.mem_total_mib>0?x.mem_used_mib/x.mem_total_mib:0,0,1),cssVar("--warn"),"rgba(255,199,87,0.06)",false,false); // mem (amber)
  // util trace with phosphor afterglow: older segments decay like a scope trace
  {
    const get=x=>norm(x.util<0?0:x.util,0,100);
    const acc=cssVar("--acc");
    for(let i=1;i<s.length;i++){
      const a=0.18+0.82*(i/s.length);
      ctx.beginPath();
      ctx.moveTo((i-1)*step,yOf(get(s[i-1])));
      ctx.lineTo(i*step,yOf(get(s[i])));
      ctx.strokeStyle=acc;ctx.globalAlpha=a;ctx.lineWidth=1.2;
      ctx.shadowColor=acc;ctx.shadowBlur=i>s.length-6?8:0;
      ctx.stroke();
    }
    ctx.globalAlpha=1;ctx.shadowBlur=0;
    // fill under the newest portion only
    ctx.beginPath();
    const from=Math.max(0,s.length-15);
    for(let i=from;i<s.length;i++){const y=yOf(get(s[i])); i===from?ctx.moveTo(i*step,y):ctx.lineTo(i*step,y);}
    ctx.lineTo((s.length-1)*step,h);ctx.lineTo(from*step,h);ctx.closePath();
    const g2=ctx.createLinearGradient(0,0,0,h);
    g2.addColorStop(0,"rgba(0,224,140,0.12)");g2.addColorStop(1,"rgba(0,0,0,0)");
    ctx.fillStyle=g2;ctx.fill();
  }

  // util end-point marker
  const lastS=s[s.length-1];
  if(lastS){
    const ex=w, ey=yOf(norm(lastS.util<0?0:lastS.util,0,100));
    ctx.beginPath(); ctx.arc(ex-3,ey,2.5,0,Math.PI*2);
    ctx.fillStyle=cssVar("--acc"); ctx.shadowColor=cssVar("--acc"); ctx.shadowBlur=6;
    ctx.fill(); ctx.shadowBlur=0;
  }
}

/* GPU chart crosshair: sample inspector on hover */
function wireGpuCrosshair(){
  const cv=$("gpu-canvas"); if(!cv)return;
  const hair=$("gpu-crosshair"), tip=$("gpu-tip");
  const fmt=(v,u)=>v==null||v<0?"--":v+u;
  cv.addEventListener("mousemove",(ev)=>{
    const s=state.gpuHistory; if(s.length<2)return;
    const rect=cv.getBoundingClientRect();
    const x=ev.clientX-rect.left, w=rect.width;
    const idx=Math.max(0,Math.min(s.length-1,Math.round((x/w)*(s.length-1))));
    const d=s[idx];
    hair.classList.remove("hidden");
    hair.style.left=((idx/(s.length-1))*w)+"px";
    tip.classList.remove("hidden");
    tip.innerHTML=
      `<div class="row"><span class="k">util</span><span style="color:var(--acc)">${fmt(d.util,"%")}</span></div>
       <div class="row"><span class="k">vram</span><span style="color:var(--warn)">${d.mem_total_mib>0?Math.round(100*d.mem_used_mib/d.mem_total_mib)+"%":"--"}</span></div>
       <div class="row"><span class="k">pwr</span><span style="color:#ff9e64">${fmt(d.power_w,"W")}</span></div>
       <div class="row"><span class="k">temp</span><span>${fmt(d.temp_c,"°C")}</span></div>`;
    const tipW=tip.offsetWidth;
    const left=Math.max(0,Math.min(w-tipW,((idx/(s.length-1))*w)+10));
    tip.style.left=left+"px";
  });
  cv.addEventListener("mouseleave",()=>{
    hair.classList.add("hidden"); tip.classList.add("hidden");
  });
}

/* draw sparkline in a small canvas; null data = designed idle baseline */
function drawSparkline(canvas,data,color){
  if(!canvas)return;
  const dpr=window.devicePixelRatio||1;
  const w=canvas.clientWidth,h=canvas.clientHeight; if(w===0)return;
  canvas.width=w*dpr; canvas.height=h*dpr;
  const ctx=canvas.getContext("2d"); ctx.scale(dpr,dpr); ctx.clearRect(0,0,w,h);
  if(!data||data.length<2){
    ctx.strokeStyle="rgba(255,255,255,0.10)"; ctx.lineWidth=1;
    ctx.setLineDash([2,4]); ctx.beginPath();
    ctx.moveTo(2,h-2.5); ctx.lineTo(w-2,h-2.5); ctx.stroke();
    ctx.setLineDash([]);
    return;
  }
  const max=Math.max(...data,1), step=w/(data.length-1);
  ctx.beginPath();
  data.forEach((v,i)=>{const y=h-2-(v/max)*(h-6); i?ctx.lineTo(i*step,y):ctx.moveTo(0,y);});
  ctx.strokeStyle=color; ctx.lineWidth=1.2;
  ctx.shadowColor=color; ctx.shadowBlur=4;
  ctx.stroke(); ctx.shadowBlur=0;
  ctx.lineTo(w,h);ctx.lineTo(0,h);ctx.closePath();
  ctx.fillStyle=color+"14";
  ctx.fill();
  // last-value end dot ties the line to "now"
  const lx=w-1.5, ly=h-2-(data[data.length-1]/max)*(h-6);
  ctx.beginPath(); ctx.arc(lx,ly,1.8,0,Math.PI*2);
  ctx.fillStyle=color; ctx.shadowColor=color; ctx.shadowBlur=5;
  ctx.fill(); ctx.shadowBlur=0;
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
  const el=n==="overview"?$("view-overview"):$("view-workbench");
  const other=n==="overview"?$("view-workbench"):$("view-overview");
  other.classList.add("hidden");
  el.classList.remove("hidden");
  el.classList.remove("enter"); void el.offsetWidth; el.classList.add("enter");
}

/* ---------------- OVERVIEW: HUD ---------------- */

function renderOverview(){
  const grid=$("fleet-grid");
  $("fleet-count").textContent=state.servers.length;
  const running=state.servers.filter(s=>["running","starting","backoff"].includes(s.state)).length;
  $("fleet-live").textContent=running+" live";
  renderFleetControls();
  const shown=state.servers.filter(s=>
    (state.fleetFilter==="all"||s.category===state.fleetFilter)&&
    (state.fleetState==="all"||fleetStateOf(s)===state.fleetState));
  /* Keyed reconciliation: the 2s poll updates tiles in place instead of
   * rebuilding the grid — no flicker, no hover/animation resets. */
  const grid2=$("fleet-grid");
  let tiles=grid2._tiles; if(!tiles){tiles=grid2._tiles=new Map();}
  const want=new Set(shown.map(s=>s.id));
  for(const [id,el] of tiles){ if(!want.has(id)){ el.remove(); tiles.delete(id);} }
  for(const s of shown){ upsertCartridge(tiles,grid2,s); }
  const orderKey=shown.map(s=>s.id).join(",");
  if(grid2._order!==orderKey){
    for(const s of shown){ const el=tiles.get(s.id); if(el)grid2.appendChild(el); }
    grid2._order=orderKey;
  }
  const emptyHint=grid2.querySelector(".fleet-empty");
  const needEmpty=!shown.length;
  if(needEmpty&&!emptyHint){
    const e=document.createElement("div");e.className="empty-hint fleet-empty";
    e.textContent="// no models match the current filters";grid2.appendChild(e);
  }else if(!needEmpty&&emptyHint){emptyHint.remove();}
  renderGatewayBar();
  drawFleetSparks();
  updateAggregateRps();
}
/* fleet-wide aggregate throughput: sum of live per-model qps */
function updateAggregateRps(){
  let sum=0, live=0;
  for(const s of state.servers){
    if(s.state!=="running")continue;
    const q=fleetQps[s.id];
    if(q&&q.qps!=null){sum+=q.qps;live++;}
  }
  const txt=live?`Σ ${sum.toFixed(1)} r/s`:"Σ -- r/s";
  const head=$("fleet-live");
  const running=state.servers.filter(s=>["running","starting","backoff"].includes(s.state)).length;
  if(head)head.textContent=`${running} live${live?` · ${txt}`:""}`;
  const top=$("topbar-rps");
  if(top){
    top.classList.toggle("hidden",!live);
    if(live)top.textContent=txt;
  }
}

function upsertCartridge(tiles,grid,s){
  const st=dotClassOf(s);
  const isRun=s.state==="running";
  const sig=`${s.state}|${s.ready?1:0}|${s.restart_count}|${s.port}`;
  let tile=tiles.get(s.id);
  if(!tile){
    tile=document.createElement("div");
    tile.className="cartridge";
    tile.setAttribute("role","button");
    tile.setAttribute("tabindex","0");
    tile.innerHTML=
      `<div class="cartridge-body"></div>
       <i class="c-tl"></i><i class="c-br"></i>
       <canvas class="cartridge-spark" data-id="${escapeHtml(s.id)}"></canvas>
       <div class="spark-tag"></div>`;
    const open=()=>navigate("#/model/"+s.id);
    tile.onclick=open;
    tile.onkeydown=(ev)=>{if(ev.key==="Enter"||ev.key===" "){ev.preventDefault();open();}};
    tiles.set(s.id,tile);
    grid.appendChild(tile);
    tile._sig="";
  }
  tile.setAttribute("aria-label",`${s.id}, ${s.state}, port ${s.port}`);
  tile.className="cartridge"+(isRun?" live":"")+(s.state==="failed"?" dead":"");
  if(isRun){tile.style.setProperty("--cat-glow",catColor(s.category));}
  else{tile.style.removeProperty("--cat-glow");}
  if(tile._sig===sig)return;
  tile._sig=sig;
  tile.querySelector(".cartridge-body").innerHTML=
    `${isRun?'<div class="accent-top"></div>':''}
     <div class="cartridge-row">
       <span class="st ${st}">${ST_GLYPH[st]}</span>
       <span class="cartridge-name">${escapeHtml(s.id.toLowerCase())}</span>
       ${s.restart_count>0?`<span class="badge restarts" title="restarts: ${s.restart_count}">↻${s.restart_count}</span>`:""}
     </div>
     <div class="cartridge-sub">${isRun?`<span class="uptime" data-id="${escapeHtml(s.id)}">${uptimeOf(s)||"booting"}</span>`:escapeHtml(s.state)}<span class="cartridge-port">:${s.port}</span></div>`;
}

/* coarse state group used by the summary chips */
function fleetStateOf(s){
  if(s.state==="failed")return "failed";
  if(s.state==="stopped")return "stopped";
  if(s.state==="running")return "running";
  return "starting"; // starting + backoff
}

function renderFleetControls(){
  const statesBox=$("fleet-states"), catsBox=$("fleet-filters");
  if(!statesBox||!catsBox)return;
  const nBy={running:0,starting:0,stopped:0,failed:0};
  for(const s of state.servers)nBy[fleetStateOf(s)]++;
  const counts={};
  for(const s of state.servers)counts[s.category]=(counts[s.category]||0)+1;
  // rebuild the chip rows only when their content actually changed
  const sig=JSON.stringify([state.fleetState,state.fleetFilter,nBy,counts,state.servers.length]);
  if(statesBox._sig===sig)return;
  statesBox._sig=sig;
  const mkChip=(label,n,on,onclick,extra)=>{
    const c=document.createElement("button");
    c.type="button";
    c.className="chip"+(extra||"")+(on?" on":"");
    c.innerHTML=`${escapeHtml(label)}<span class="n">${n}</span>`;
    c.onclick=onclick;
    return c;
  };
  statesBox.innerHTML="";
  statesBox.appendChild(mkChip("all",state.servers.length,state.fleetState==="all",
    ()=>{state.fleetState="all";renderOverview();}));
  for(const [key,color] of [["running","var(--acc)"],["starting","var(--warn)"],["failed","var(--err)"],["stopped","var(--txt-dim)"]]){
    const c=mkChip(key,nBy[key],state.fleetState===key,
      ()=>{state.fleetState=key;renderOverview();}," st-chip");
    c.style.setProperty("--sc",color);
    statesBox.appendChild(c);
  }
  catsBox.innerHTML="";
  catsBox.appendChild(mkChip("all",state.servers.length,state.fleetFilter==="all",
    ()=>{state.fleetFilter="all";renderOverview();}));
  for(const cat of Object.keys(counts).sort()){
    const c=mkChip(cat,counts[cat],state.fleetFilter===cat,
      ()=>{state.fleetFilter=cat;renderOverview();});
    c.style.setProperty("--cat-glow",catColor(cat));
    if(state.fleetFilter===cat)c.style.color=catColor(cat);
    catsBox.appendChild(c);
  }
}

function drawFleetSparks(){
  for(const s of state.servers){
    if(state.fleetFilter!=="all"&&s.category!==state.fleetFilter)continue;
    if(state.fleetState!=="all"&&fleetStateOf(s)!==state.fleetState)continue;
    const cv=document.querySelector(`canvas.cartridge-spark[data-id="${s.id}"]`);
    if(!cv)continue;
    const tag=cv.parentElement&&cv.parentElement.querySelector(".spark-tag");
    // 1st choice: live server qps; 2nd: this tab's bench samples; else idle baseline
    const live=fleetQps[s.id];
    if(live&&live.hist&&live.hist.length>=2){
      drawSparkline(cv,live.hist.slice(-30),catColor(s.category));
      if(tag){tag.textContent=`live · ${live.qps!=null?live.qps.toFixed(1)+" r/s":"…"}`;
        tag.className="spark-tag has-data live";}
      continue;
    }
    const data=state.inferHistory.filter(x=>x.serverId===s.id).slice(-30).map(x=>x.ms);
    drawSparkline(cv,data.length>=2?data:null,catColor(s.category));
    if(tag){
      tag.textContent=data.length>=2?`session · ${data.length}`:"idle";
      tag.className="spark-tag"+(data.length>=2?" has-data":"");
    }
  }
}

/* ---------------- WORKBENCH ---------------- */

function renderWorkbench(){
  const s=serverById(state.selectedId); if(!s)return;
  /* signature guard: the 2s poll re-renders constantly — only touch the DOM
   * when this model's fields actually changed (kills the flicker) */
  const sig=JSON.stringify([s.id,s.state,s.ready,s.restart_count,s.pid,s.port,s.uri,s.category]);
  if(state._wbSig!==sig){
    state._wbSig=sig;
    const st=dotClassOf(s);
    $("wb-breadcrumb").textContent="‹ fleet / "+s.id.toLowerCase();
    $("wb-breadcrumb").onclick=()=>navigate("#/overview");
    const pillColor=s.state==="running"?"var(--acc)":s.state==="failed"?"var(--err)":
      (s.state==="stopped"?"var(--txt-dim)":"var(--warn)");
    $("wb-title").innerHTML=`<span class="st ${st}">${ST_GLYPH[st]}</span> ${escapeHtml(s.id.toLowerCase())} <span class="state-pill" style="--sc:${pillColor}">${escapeHtml(s.state)}${s.ready?" · ready":""}</span>`;
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
    renderFileList();
    renderResultsList(s.id);
    updateSessionStats(s.id);
  }
  state.logServerId=s.id;
  // reset log state only when switching models — refresh() re-renders every 2s
  // and must not wipe the live log buffer each tick
  if(!state.logs[s.id]){resetLogState(s.id);}
  syncLogSelector();
  // fetch process info + server telemetry for the dossier
  pollProcessInfo(s.id);
  pollServerMetrics(s.id);
  updateWorkbenchGpu();
}

async function pollProcessInfo(id){
  const r=await api(`/api/v1/servers/${encodeURIComponent(id)}/process`);
  const box=$("wb-process"); if(!box) return;
  if(!r.ok||!r.data||!r.data.process){if(box._sig!=="none"){box.innerHTML="";box._sig="none";}return;}
  const p=r.data.process;
  const sig=`${p.rss_kb}|${p.threads}`;
  if(box._sig===sig)return;
  box._sig=sig;
  const rss=p.rss_kb>0?(p.rss_kb/1024).toFixed(1)+"M":"--";
  box.innerHTML=
    `<div class="id-row"><span class="id-k">rss</span><span>${rss}</span></div>
     <div class="id-row"><span class="id-k">thr</span><span>${p.threads>0?p.threads:"--"}</span></div>`;
}

/* ---------------- server telemetry (real /metrics via supervisor proxy) ---------------- */

function promSum(text, name) {
  // sum all samples of a counter regardless of labels
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
  // histogram_bucket{...,le="N"} cumulative counts -> linear-interpolated quantile
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
  // throttle: called from both renderWorkbench (2s) and the 5s interval —
  // back-to-back samples would make the qps delta spike
  const now0 = Date.now();
  if (metricsPrev[id] && now0 - metricsPrev[id].t < 4000) return;
  let text = null, ok = false;
  try {
    const resp = await authorizedFetch(`/api/v1/servers/${encodeURIComponent(id)}/metrics`);
    if (resp.ok) { text = await resp.text(); ok = typeof text === "string" && text.length > 0; }
  } catch (e) { ok = false; }
  if (!ok) {
    if (box._sig !== "na") { box.innerHTML = '<div class="empty-hint">metrics n/a</div>'; box._sig = "na"; }
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
    ["qps", qps != null ? qps.toFixed(2) : "—"],
    ["p50", (v => v != null ? v + "ms" : "—")(promQuantile(text, "mortred_inference_duration_ms", 0.5))],
    ["p95", (v => v != null ? v + "ms" : "—")(promQuantile(text, "mortred_inference_duration_ms", 0.95))],
    ["busy", (v => v != null ? v : "—")(promGauge(text, "mortred_workers_busy"))],
    ["idle", (v => v != null ? v : "—")(promGauge(text, "mortred_workers_available"))],
    ["queue", (v => v != null ? v : "—")(promGauge(text, "mortred_queue_depth"))],
    ["waiting", (v => v != null ? v : "—")(promGauge(text, "mortred_waiting_jobs"))],
    ["done", (v => v != null ? Math.round(v).toString() : "—")(promGauge(text, "mortred_finished_jobs_total"))],
  ];
  const sig = JSON.stringify(rows);
  if (box._sig === sig) return;
  box._sig = sig;
  box.innerHTML = rows.map(([k, v]) =>
    `<div class="id-row"><span class="id-k">${k}</span><span>${escapeHtml(String(v))}</span></div>`).join("");
}

/* fleet-wide live qps: staggered per-running-model metrics polls (6s cycle) */
const fleetQps = {};   // id -> {t, total, qps, hist[]}
async function pollFleetMetrics() {
  if (document.hidden) return;
  for (const s of state.servers) {
    if (s.state !== "running") { delete fleetQps[s.id]; continue; }
    try {
      const resp = await authorizedFetch(`/api/v1/servers/${encodeURIComponent(s.id)}/metrics`);
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
    await new Promise(r => setTimeout(r, 250));
  }
}

function serverById(id){return state.servers.find(s=>s.id===id)||null;}

setInterval(()=>{document.querySelectorAll(".uptime").forEach(el=>{
  const s=serverById(el.dataset.id);if(s)el.textContent=uptimeOf(s)||"--";});},1000);
setInterval(()=>{if(state.selectedId)pollProcessInfo(state.selectedId);},5000);
setInterval(()=>{if(state.selectedId)pollServerMetrics(state.selectedId);},5000);

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
  // show a reachable host, not the raw bind address
  let host=g.address?g.address.host:"";
  if(host==="0.0.0.0"||host==="::"||host==="[::]")host=window.location.hostname||"127.0.0.1";
  const addr=g.address?`${host}:${g.address.port}`:"";
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
  setTimeout(()=>{el.classList.add("leaving");},2400);
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
  const bench=benchOf(state.selectedId);
  for(const f of fl){
    if(!f.type.startsWith("image/"))continue;
    const b64=await loadImageAsBase64(f);
    bench.files.push({name:f.name,url:base64ToSrc(b64),base64:b64,status:"ready"});
  }
  renderFileList();
}
function renderFileList(){
  const box=$("file-list");if(!box)return;
  const bench=benchOf(state.selectedId);
  box.innerHTML="";
  for(const f of bench.files){
    const chip=document.createElement("div");
    chip.className="file-chip"+(f.status?" "+f.status:"");
    chip.innerHTML=`<img src="${f.url}"><span>${escapeHtml(f.name)}</span>`+
      (f.status==="sending"?'<span class="chip-st">⋯</span>':
       f.status==="failed"?'<span class="chip-st" title="send failed — stays queued for retry">✗</span>':"");
    const rm=document.createElement("span");rm.textContent="✕";rm.className="chip-rm";
    rm.onclick=()=>{bench.files=bench.files.filter(x=>x!==f);renderFileList();};
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
function percentile(sorted,q){
  if(!sorted.length)return null;
  const i=Math.min(sorted.length-1,Math.floor(q*(sorted.length-1)));
  return sorted[i];
}
/* session stats: what this browser session has done against the model */
function updateSessionStats(id){
  const box=$("wb-stats");if(!box)return;
  const hist=state.inferHistory.filter(x=>x.serverId===id);
  const ok=hist.filter(x=>x.ok).length, fail=hist.length-ok;
  const lat=hist.filter(x=>x.ok).map(x=>x.ms).sort((a,b)=>a-b);
  const p50=percentile(lat,0.5),p95=percentile(lat,0.95);
  const parts=[
    `<span class="ws-k">sent</span><b>${hist.length}</b>`,
    `<span class="ws-k">ok</span><b style="color:var(--acc)">${ok}</b>`,
    `<span class="ws-k">fail</span><b style="color:${fail?"var(--err)":"var(--txt-dim)"}">${fail}</b>`,
  ];
  if(p50!=null)parts.push(`<span class="ws-k">p50</span><b>${p50}ms</b>`);
  if(p95!=null)parts.push(`<span class="ws-k">p95</span><b>${p95}ms</b>`);
  parts.push(`<span class="ws-k">in-flight</span><b style="color:${state.inflight?"var(--warn)":"var(--txt-dim)"}">${state.inflight}</b>`);
  box.innerHTML=parts.join('<span class="ws-sep">·</span>');
}

async function sendBatch(){
  const s=serverById(state.selectedId);if(!s)return;
  const bench=benchOf(s.id);
  const queue=bench.files.filter(f=>f.status!=="sending");
  if(!queue.length){showToast("nothing queued — upload images first","info");return;}
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
  /* send semantics: each click sends the CURRENT queue; a file that succeeds
   * leaves the queue (upload b after sending a sends b only), a file that
   * fails stays queued for retry */
  $("btn-cancel").classList.remove("hidden");setBatchProgress(0,queue.length);
  let aborted=false;state.batchAbort=()=>{aborted=true;};
  let done=0;
  for(const f of queue){
    if(aborted)break;
    f.status="sending";renderFileList();
    const reqId=uid(),t0=performance.now();
    state.inflight++;updateSessionStats(s.id);
    try{
      const resp=await authorizedFetch(`${base}/v1/models/${encodeURIComponent(s.id)}/infer`,{
        method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({req_id:reqId,images:[f.base64]}),
      });
      const ms=Math.round(performance.now()-t0);
      const body=await resp.json().catch(()=>null);
      state.inferHistory.push({t:Date.now(),serverId:s.id,ms,ok:resp.ok});
      if(state.inferHistory.length>600)state.inferHistory.shift();
      if(resp.ok){
        f.status="done";
        bench.files=bench.files.filter(x=>x!==f); // success leaves the queue
        addResult(s,f,body,reqId,ms);
        riverPush("ok",`${s.id} 200 ${ms}ms`,s.id);
      }else{
        f.status="failed";
        showToast(`HTTP ${resp.status}`,"error");riverPush("err",`${s.id} HTTP ${resp.status}`,s.id);
      }
    }catch(e){
      f.status="failed";
      // transport-level failures count in the session stats too (ok:false)
      state.inferHistory.push({t:Date.now(),serverId:s.id,ms:Math.round(performance.now()-t0),ok:false});
      if(state.inferHistory.length>600)state.inferHistory.shift();
      const detail = e && e.message ? e.message : String(e);
      showToast(`infer fail: ${detail}\n→ ${inferUrl}`, "error");
      riverPush("err", `${s.id} transport: ${detail}`, s.id);
    }
    state.inflight--;updateSessionStats(s.id);
    done++;setBatchProgress(done,queue.length);
    renderFileList();
  }
  state.batchAbort=null;$("btn-cancel").classList.add("hidden");
  drawFleetSparks();
}

function setBatchProgress(d,t){
  $("batch-progress").classList.toggle("hidden",t===0);
  $("batch-progress-fill").style.width=t?((d/t)*100).toFixed(1)+"%":"0";
  $("batch-progress-text").textContent=`${d}/${t}`;
}

/* results are stored per model and rebuilt on switch — each model owns its bench */
function addResult(server,input,result,reqId,elapsed){
  const bench=benchOf(server.id);
  bench.results.unshift({input,result,reqId,elapsed});
  if(bench.results.length>20)bench.results.length=20;
  if(state.selectedId===server.id){
    const card=buildResultCard(server,input,result,reqId,elapsed);
    $("results-list").prepend(card);
  }
}
function renderResultsList(id){
  const box=$("results-list");if(!box)return;
  const server=serverById(id);if(!server)return;
  box.innerHTML="";
  if(!benchOf(id).results.length){
    const e=document.createElement("div");e.className="empty-hint";
    e.textContent="// no results yet — upload & send";
    box.appendChild(e);
    return;
  }
  for(const r of benchOf(id).results){
    box.appendChild(buildResultCard(server,r.input,r.result,r.reqId,r.elapsed));
  }
}
function buildResultCard(server,input,result,reqId,elapsed){
  const payload=unifiedPayload(result);
  const card=document.createElement("div");card.className="result-card";
  const score=topScore(payload);
  const detCount=Array.isArray(payload)?payload.length:0;
  const kind=resultKind(payload);
  /* two-line head: name + metric chiplets / req id — no more middot soup */
  card.innerHTML=
    `<div class="head">
       <div class="head-main">
         <span class="req-name">${escapeHtml(input.name)}</span>
         <span class="head-chips">
           <span class="chiplet">${elapsed}ms</span>
           ${detCount?`<span class="chiplet">${detCount} ${kind}</span>`:""}
           ${score!=null?`<span class="chiplet acc">top ${score.toFixed(3)}</span>`:""}
         </span>
       </div>
       <span class="req-meta">req ${escapeHtml(reqId)}</span>
     </div>`;
  const vizWrap=document.createElement("div");vizWrap.className="viz";card.appendChild(vizWrap);
  const raw=document.createElement("details");raw.className="raw-json";raw.innerHTML="<summary>raw</summary>";
  const pre=document.createElement("pre");pre.textContent=JSON.stringify(result,null,1);raw.appendChild(pre);
  card.appendChild(raw);
  visualize(server,input,payload,vizWrap).catch(()=>{});
  return card;
}
function resultKind(payload){
  if(Array.isArray(payload)&&payload.length){
    if(payload[0].bbox)return "hits";
    if(payload[0].location)return "points";
    if(payload[0].polygon)return "regions";
  }
  return "";
}

async function visualize(server,input,payload,vizWrap){
  const img=new Image();img.src=input.url;await img.decode().catch(()=>{});
  // classification contract: {class_id, category, scores[]} — an OBJECT, not an array
  if(payload&&typeof payload==="object"&&!Array.isArray(payload)&&payload.category!==undefined&&Array.isArray(payload.scores)){
    vizWrap.appendChild(img);
    const box=document.createElement("div");box.className="cls-result";
    const scores=(payload.scores||[]).slice(0,5);
    const max=scores.length?scores[0]:1;
    box.innerHTML=`<span class="cls-cat">${escapeHtml(payload.category)}</span>`+
      scores.map(s=>`<span class="cls-score"><i style="width:${Math.max(4,Math.round(100*s/(max||1)))}%"></i><b>${(+s).toFixed(3)}</b></span>`).join("");
    vizWrap.appendChild(box);
    return;
  }
  // feature embedding contract: {dim, embedding[]} — show dim + norm, vector has no pixels
  if(payload&&typeof payload==="object"&&!Array.isArray(payload)&&Array.isArray(payload.embedding)){
    vizWrap.appendChild(img);
    let norm=0;const v=payload.embedding;
    for(let i=0;i<v.length;i+=1)norm+=v[i]*v[i];
    const box=document.createElement("div");box.className="cls-result";
    box.innerHTML=`<span class="cls-cat">embedding</span><span class="cls-dim">dim=${payload.dim!=null?payload.dim:v.length} · ‖v‖=${Math.sqrt(norm).toFixed(2)}</span>`;
    vizWrap.appendChild(box);
    return;
  }
  // SAM AMG contract: [{segmentation png, area, bbox?, predicted_iou, stability_score}]
  // must be checked BEFORE the generic detection branch (items also carry bbox)
  if(Array.isArray(payload)&&payload.length&&payload[0].segmentation!==undefined&&payload[0].predicted_iou!==undefined){
    const cv=document.createElement("canvas");cv.className="overlay";vizWrap.appendChild(cv);
    drawDetection(cv,img.src,payload.map(m=>({bbox:m.bbox,score:m.predicted_iou,category:"mask",stability:m.stability_score,area:m.area})),true);
    const legend=document.createElement("div");legend.className="det-legend";
    legend.innerHTML=`<span class="det-chip" style="--h:170"><span class="det-dot"></span>${payload.length} masks · iou top ${(payload[0].predicted_iou!=null?(+payload[0].predicted_iou).toFixed(2):"--")} · pngs in raw</span>`;
    vizWrap.appendChild(legend);
    return;
  }
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
  // feature points (superpoint): [{score, location:[x,y], descriptor[]}]
  if(Array.isArray(payload)&&payload.length&&Array.isArray(payload[0].location)){
    const cv=document.createElement("canvas");cv.className="overlay";vizWrap.appendChild(cv);
    drawFeaturePoints(cv,img.src,payload);
    const legend=document.createElement("div");legend.className="det-legend";
    legend.innerHTML=`<span class="det-chip" style="--h:150"><span class="det-dot"></span>${payload.length} points · top ${(payload[0].score!=null?payload[0].score.toFixed(2):"--")}</span>`;
    vizWrap.appendChild(legend);
    return;
  }
  // ocr text regions: [{score, bbox, polygon:[[x,y]×4]}]
  if(Array.isArray(payload)&&payload.length&&Array.isArray(payload[0].polygon)){
    const cv=document.createElement("canvas");cv.className="overlay";vizWrap.appendChild(cv);
    drawOcr(cv,img.src,payload);
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

/* feature points (superpoint contract): glow dots sized by score */
function drawFeaturePoints(canvas,imgUrl,points){
  const img=new Image();
  img.onload=()=>{
    canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;
    const ctx=canvas.getContext("2d");
    ctx.drawImage(img,0,0);
    const base=Math.max(1.2,img.naturalWidth/420);
    for(const p of points){
      const [x,y]=p.location;
      const score=p.score!=null?p.score:0.5;
      const r=base*(0.9+score*1.8);
      ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);
      ctx.fillStyle="#00e08c";ctx.shadowColor="#00e08c";ctx.shadowBlur=6;
      ctx.fill();ctx.shadowBlur=0;
    }
  };
  img.src=imgUrl;
}
function drawDetection(canvas,imgUrl,boxes,isMasks){
  const img=new Image();
  img.onload=()=>{
    canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;
    const ctx=canvas.getContext("2d");
    ctx.drawImage(img,0,0);
    ctx.lineWidth=Math.max(2,Math.round(img.naturalWidth/300));
    for(const b of boxes){
      const[x1,y1,x2,y2]=b.bbox;
      const bw=Math.max(1,x2-x1), bh=Math.max(1,y2-y1);
      // label font adapts to box size — small boxes get smaller, never clipped labels
      const fpx=Math.max(11,Math.min(Math.round(img.naturalWidth/40),Math.round(bw/3.5),Math.round(bh/2)));
      ctx.font=fpx+"px monospace";
      const hue=isMasks?170:((b.class_id*47)%360);
      ctx.strokeStyle=isMasks?"#2dd4bf":`hsl(${hue} 90% 60%)`;
      ctx.shadowColor=ctx.strokeStyle;ctx.shadowBlur=8;
      ctx.strokeRect(x1,y1,bw,bh);ctx.shadowBlur=0;
      let label;
      if(isMasks){
        label=`mask ${b.score!=null?(+b.score).toFixed(2):""}${b.area?` · ${b.area}px`:""}`;
      }else{
        label=`${b.category||b.class_id} ${b.score!=null?b.score.toFixed(2):""}`;
      }
      const tw=ctx.measureText(label).width+fpx*0.7;
      const plateH=Math.round(fpx*1.4);
      // place above when it fits in-canvas and the box is tall enough; else inside the top
      const outside=y1-plateH-2>=0&&bh>=plateH*1.8;
      const py=outside?y1-plateH-2:Math.min(y1+2,img.naturalHeight-plateH);
      ctx.fillStyle=isMasks?"#2dd4bf":`hsl(${hue} 90% 60%)`;
      ctx.fillRect(x1,py,tw,plateH);
      ctx.fillStyle="#000";
      ctx.fillText(label,x1+fpx*0.35,py+plateH-fpx*0.32);
      // face landmarks contract: landmarks [[x,y],…] drawn as amber dots
      if(Array.isArray(b.landmarks)&&b.landmarks.length){
        ctx.fillStyle="#ffd166";ctx.shadowColor="#ffd166";ctx.shadowBlur=4;
        for(const p of b.landmarks){
          ctx.beginPath();ctx.arc(p[0],p[1],Math.max(2,img.naturalWidth/240),0,Math.PI*2);
          ctx.fill();
        }
        ctx.shadowBlur=0;
      }
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
      // real contract: {polygon: [[x,y]×4]}; legacy: {points|bbox_points}
      const pts=r.polygon||r.points||r.bbox_points||[];
      if(pts.length>=3){
        ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);
        for(let i=1;i<pts.length;i++)ctx.lineTo(pts[i][0],pts[i][1]);
        ctx.closePath();ctx.stroke();
      }
      const label=r.text||(r.score!=null?r.score.toFixed(2):"");
      if(label){ctx.fillStyle="#ffd166";ctx.font="14px monospace";
        ctx.fillText(label,pts[0]?pts[0][0]:4,pts[0]?pts[0][1]-4:14);}
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
  const sel=$("log-server");if(!sel)return;
  const ids=state.servers.map(s=>s.id);
  const sig=ids.join(",")+"|"+state.logServerId;
  if(sel._sig===sig)return;
  sel._sig=sig;
  sel.innerHTML="";
  for(const id of ids){
    const opt=document.createElement("option");opt.value=id;opt.textContent=id;sel.appendChild(opt);
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

const PALETTE_RECENT_KEY="mortred_palette_recent";
function paletteRecent(){
  try{ return JSON.parse(localStorage.getItem(PALETTE_RECENT_KEY)||"[]").slice(0,3); }catch(e){ return []; }
}
function paletteRemember(label){
  const rec=paletteRecent().filter(x=>x!==label);
  rec.unshift(label);
  try{ localStorage.setItem(PALETTE_RECENT_KEY,JSON.stringify(rec.slice(0,3))); }catch(e){}
}
function paletteItems(){
  const items=paletteRecent()
    .map(label=>({found:buildPaletteItems().find(it=>it.label===label),label}))
    .filter(x=>x.found)
    .map(x=>({label:x.label,hint:"recent",act:x.found.act}));
  return items.concat(buildPaletteItems());
}
function buildPaletteItems(){
  const nav=[],ctrl=[];
  for(const s of state.servers){
    nav.push({label:`open ${s.id.toLowerCase()}`,hint:"nav",act:()=>navigate("#/model/"+s.id)});
    const running=["running","starting","backoff"].includes(s.state);
    if(!running)ctrl.push({label:`start ${s.id.toLowerCase()}`,hint:"ctrl",act:()=>controlServer(s.id,"start")});
    if(running)ctrl.push({label:`stop ${s.id.toLowerCase()}`,hint:"ctrl",act:()=>controlServer(s.id,"stop")});
    ctrl.push({label:`restart ${s.id.toLowerCase()}`,hint:"ctrl",act:()=>controlServer(s.id,"restart")});
  }
  nav.push({label:"overview",hint:"nav",act:()=>navigate("#/overview")});
  const set=[{label:"token",hint:"set",act:async()=>{
    const t=await askToken(getToken());
    if(t!==null&&t.trim()){setToken(t.trim());location.reload();}
  }}];
  return nav.concat(ctrl,set);
}
function fuzzyMatch(q,l){
  let li=0,sc=0,streak=0;const Q=q.toLowerCase(),L=l.toLowerCase();
  for(const ch of Q){
    const f=L.indexOf(ch,li);if(f<0)return-1;
    streak=f===li?streak+1:0;sc+=1+streak;li=f+1;
  }
  return sc;
}
let paletteStagger=false;
function openPalette(){
  const p=$("palette");p.classList.remove("hidden");
  paletteStagger=true;
  const input=$("palette-input");input.value="";renderPalette("");input.focus();
  paletteStagger=false;
}
function closePalette(){$("palette").classList.add("hidden");}
function renderPalette(query){
  const list=$("palette-list");
  const HINT_ICON={nav:"▸",ctrl:"⟳",set:"◈",recent:"↺"};
  const items=paletteItems()
    .map(it=>({it,sc:query?fuzzyMatch(query,it.label):0}))
    .filter(x=>x.sc>=0).sort((a,b)=>b.sc-a.sc).slice(0,10);
  let sel=0, idx=0; list.innerHTML="";
  const GROUP_LABEL={recent:"recent",nav:"navigate",ctrl:"control",set:"system"};
  let lastGroup=null;
  for(const{it}of items){
    if(!query && it.hint!==lastGroup){
      lastGroup=it.hint;
      const g=document.createElement("div");g.className="palette-group";
      g.textContent=GROUP_LABEL[it.hint]||it.hint;
      list.appendChild(g);
    }
    const row=document.createElement("div");row.className="palette-row";
    if(paletteStagger){ row.classList.add("stagger"); row.style.animationDelay=(idx++*14)+"ms"; }
    const icon = it.label.indexOf("stop ")===0 ? "■" : (HINT_ICON[it.hint]||"·");
    row.innerHTML=`<span><span class="pi">${icon}</span>${escapeHtml(it.label)}</span><span class="palette-hint">${it.hint}</span>`;
    row.onclick=()=>{closePalette();paletteRemember(it.label);it.act();};
    list.appendChild(row);
  }
  if(!items.length){
    const none=document.createElement("div");none.className="palette-empty";
    none.textContent="// no matching commands";list.appendChild(none);
  }
  const firstRow=list.querySelector(".palette-row");
  if(firstRow)firstRow.classList.add("sel");
  $("palette-input").onkeydown=(ev)=>{
    const rows=[...list.querySelectorAll(".palette-row")];
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
/* j/k walks the fleet — vim-style, works with Enter from R1's card a11y */
document.addEventListener("keydown",(ev)=>{
  const inInput = document.activeElement && /input|textarea|select/i.test(document.activeElement.tagName);
  if(!inInput && (ev.key==="j"||ev.key==="k")){
    const view=$("view-overview");
    if(!view||view.classList.contains("hidden"))return;
    const cards=[...document.querySelectorAll(".cartridge")];
    if(!cards.length)return;
    ev.preventDefault();
    const cur=cards.indexOf(document.activeElement);
    const next=ev.key==="j"
      ? cards[Math.min(cards.length-1,cur<0?0:cur+1)]
      : cards[Math.max(0,cur<0?0:cur-1)];
    if(next)next.focus();
    return;
  }
  if((ev.metaKey||ev.ctrlKey)&&ev.key.toLowerCase()==="k"){
    ev.preventDefault();
    $("palette").classList.contains("hidden")?openPalette():closePalette();
  }
  if(ev.key==="Escape"){
    if(!$("palette").classList.contains("hidden"))closePalette();
    if(!$("token-dialog").classList.contains("hidden"))closeTokenDialog(null);
  }
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
  $("btn-token").onclick=async()=>{
    const t=await askToken(getToken());
    if(t!==null){setToken(t.trim());showToast("token saved","success");refresh();}
  };
  $("token-save").onclick=()=>closeTokenDialog($("token-input").value);
  $("token-cancel").onclick=()=>closeTokenDialog(null);
  $("token-backdrop").onclick=()=>closeTokenDialog(null);
  $("token-input").onkeydown=(ev)=>{
    if(ev.key==="Enter")closeTokenDialog(ev.target.value);
    if(ev.key==="Escape")closeTokenDialog(null);
  };
  // cross-platform palette shortcut label
  const plat=(navigator.userAgentData&&navigator.userAgentData.platform)||navigator.platform||"";
  $("palette-hint").innerHTML=`<kbd>${/Mac|iPhone|iPad/.test(plat)?"⌘":"Ctrl"} K</kbd>`;
  $("palette-hint").onclick=()=>openPalette();
  // CRT scanline overlay toggle ([S]), persisted
  const CRT_KEY="mortred_crt";
  const applyCrt=(on)=>{ $("crt-overlay").classList.toggle("hidden",!on); $("btn-crt").classList.toggle("on",on); };
  applyCrt(localStorage.getItem(CRT_KEY)==="1");
  $("btn-crt").onclick=()=>{
    const on=localStorage.getItem(CRT_KEY)!=="1";
    localStorage.setItem(CRT_KEY,on?"1":"0");
    applyCrt(on);
    showToast(on?"crt scanlines on":"crt scanlines off","info");
  };
  wireGpuCrosshair();
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
refresh();pollGpu();pollFleetMetrics();
setInterval(refresh,2000);
setInterval(pollFleetMetrics,6000);
setInterval(pollGpu,2000);
setInterval(pollLogs,1000);

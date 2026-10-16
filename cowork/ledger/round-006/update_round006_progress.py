#!/usr/bin/env python3
"""Regenerate round-006 progress_data.json + progress.html + issue.html from ledger."""
from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROUND = Path(__file__).resolve().parent
ISSUES = ROUND / "issues"
OUT_JSON = ROUND / "progress_data.json"
OUT_HTML = ROUND / "progress.html"
OUT_ISSUE = ROUND / "issue.html"

TITLES = {
    "p1-1": ("P1-1", "Rate-limit authority (gateway sole GCRA)"),
    "p1-2": ("P1-2", "Multi-tenant quotas at gateway"),
    "p1-3": ("P1-3", "Pluggable GpuProbe (NVML + smi)"),
    "p1-4": ("P1-4", "Release/CI reproducibility evidence"),
}

MERGE_SHA_RE = re.compile(r"^\s*-\s*merge commit:\s*([0-9a-f]{7,40})", re.I | re.M)
BRANCH_SHA_RE = re.compile(r"^\s*-\s*问题分支提交:\s*([0-9a-f]{7,40})", re.I | re.M)


def first_lines(path: Path, n: int = 12) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[:n])


def infer_step(item: dict) -> str:
    if item["status"] == "done":
        return "已合入 · merge " + (item.get("merge_sha") or "?")
    if item["has_qa"]:
        return "QA 完成 · 待合入"
    if item["has_rd"]:
        return "实现中/已提交 · 待 QA"
    if item["has_design"]:
        return "方案已锁 · 待实现"
    return "待开始 · Step 3 方案"


def read_issue(issue_id: str) -> dict:
    label, title = TITLES[issue_id]
    base = ISSUES / issue_id
    design = base / "03-design-final.md"
    rd = base / "04-rd.md"
    qa = base / "05-qa.md"
    merge = base / "05-merge.md"
    item = {
        "id": label,
        "issue_id": issue_id,
        "title": title,
        "priority": "P1",
        "status": "todo",
        "merge_sha": None,
        "branch_sha": None,
        "has_design": design.is_file(),
        "has_rd": rd.is_file(),
        "has_qa": qa.is_file(),
        "has_merge": merge.is_file(),
        "paths": {
            "design": f"cowork/ledger/round-006/issues/{issue_id}/03-design-final.md",
            "rd": f"cowork/ledger/round-006/issues/{issue_id}/04-rd.md",
            "qa": f"cowork/ledger/round-006/issues/{issue_id}/05-qa.md",
            "merge": f"cowork/ledger/round-006/issues/{issue_id}/05-merge.md",
        },
        "snippets": {
            "design": first_lines(design),
            "rd": first_lines(rd),
            "qa": first_lines(qa),
            "merge": first_lines(merge),
        },
    }
    if merge.is_file():
        text = merge.read_text(encoding="utf-8", errors="replace")
        m = MERGE_SHA_RE.search(text)
        b = BRANCH_SHA_RE.search(text)
        if m:
            item["merge_sha"] = m.group(1)
            item["status"] = "done"
        else:
            item["status"] = "running"
        if b:
            item["branch_sha"] = b.group(1)
    elif item["has_qa"] or item["has_rd"]:
        item["status"] = "running"
    elif item["has_design"]:
        item["status"] = "designed"
    item["step_text"] = infer_step(item)
    return item


def esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def render_html(data: dict) -> str:
    done = data["done"]
    total = data["total"]
    pct = data["percent"]
    open_p1 = data["open_p1"]
    status = data["round_status"]
    gen = data["generated_at"]
    current = data.get("current") or "—"
    refresh_meta = (
        '<meta http-equiv="refresh" content="0"/>'
        if status == "done"
        else '<meta http-equiv="refresh" content="60"/>'
    )
    refresh_note = "已停止刷新 · 队列清空" if status == "done" else "meta refresh 60s"
    cards = []
    detail_panels = []
    for it in data["issues"]:
        cls = "p1 done" if it["status"] == "done" else (
            "p1 run" if it["status"] in ("running", "designed") else "p1"
        )
        sha = it["merge_sha"] or it["branch_sha"] or "—"
        state = {
            "done": "已合入",
            "running": "进行中",
            "designed": "方案已锁",
            "todo": "待开始",
        }.get(it["status"], it["status"])
        iid = it["issue_id"]
        cards.append(
            f"""    <article class="card {cls}" role="button" tabindex="0"
      data-id="{esc(iid)}" onclick="location.hash='#issue/{esc(iid)}'"
      onkeydown="if(event.key==='Enter')location.hash='#issue/{esc(iid)}'">
      <div class="top"><span class="badge">{esc(it['id'])}</span><span class="state">{esc(state)}</span></div>
      <div class="title">{esc(it['title'])}</div>
      <div class="step">{esc(it.get('step_text') or '')}</div>
      <div class="sha">{esc(sha)}</div>
      <div class="hint">点击查看详情 →</div>
    </article>"""
        )
        flags = (
            f"design={'Y' if it['has_design'] else '.'} "
            f"rd={'Y' if it['has_rd'] else '.'} "
            f"qa={'Y' if it['has_qa'] else '.'} "
            f"merge={'Y' if it['has_merge'] else '.'}"
        )
        snip = it.get("snippets") or {}
        paths = it.get("paths") or {}
        detail_panels.append(
            f"""    <section class="detail" id="panel-{esc(iid)}" hidden>
      <div class="detail-head">
        <button type="button" class="back" onclick="location.hash=''">← 返回</button>
        <h3>{esc(it['id'])} · {esc(it['title'])}</h3>
        <span class="state">{esc(state)}</span>
      </div>
      <div class="meta-grid">
        <div><div class="t">当前步骤</div><div class="v">{esc(it.get('step_text') or '')}</div></div>
        <div><div class="t">产物状态</div><div class="v">{esc(flags)}</div></div>
        <div><div class="t">merge SHA</div><div class="v mono">{esc(it.get('merge_sha') or '—')}</div></div>
        <div><div class="t">branch SHA</div><div class="v mono">{esc(it.get('branch_sha') or '—')}</div></div>
      </div>
      <h4>产物路径</h4>
      <ul class="paths">
        <li><a href="issue.html?id={esc(iid)}">issue.html?id={esc(iid)}</a></li>
        <li><code>{esc(paths.get('design',''))}</code></li>
        <li><code>{esc(paths.get('rd',''))}</code></li>
        <li><code>{esc(paths.get('qa',''))}</code></li>
        <li><code>{esc(paths.get('merge',''))}</code></li>
      </ul>
      <h4>Ledger 摘录</h4>
      <pre class="snip">{esc(snip.get('merge') or snip.get('qa') or snip.get('rd') or snip.get('design') or '（尚无产物）')}</pre>
    </section>"""
        )
    cards_html = "\n".join(cards)
    details_html = "\n".join(detail_panels)
    merged = ", ".join(it["id"] for it in data["issues"] if it["status"] == "done") or "无"
    dot = "ok" if status == "done" else "run"
    status_label = "done · 队列清空" if status == "done" else f"running · 当前 {current}"
    done_display = "block" if status == "done" else "none"
    run_class = " run" if status != "done" else ""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
{refresh_meta}
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Mortred · Round 6 进度</title>
<style>
  :root {{
    --bg:#0b1220; --panel:#121a2b; --line:#243049; --text:#e8eefc; --muted:#93a0b8;
    --p1:#f5a524; --ok:#3dd68c; --run:#5b8cff; --glow:rgba(91,140,255,.25);
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; font-family: ui-sans-serif, system-ui, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background:
      radial-gradient(1200px 600px at 10% -10%, #1a2a4a 0%, transparent 55%),
      radial-gradient(900px 500px at 90% 0%, #1b3a32 0%, transparent 50%),
      var(--bg);
    color:var(--text); min-height:100vh;
  }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:28px 22px 48px; }}
  header {{ display:flex; flex-wrap:wrap; gap:18px; justify-content:space-between; align-items:flex-end; margin-bottom:22px; }}
  h1 {{ margin:0; font-size:28px; }}
  .sub {{ color:var(--muted); margin-top:6px; font-size:14px; }}
  .pill {{
    background:var(--panel); border:1px solid var(--line); border-radius:999px;
    padding:10px 14px; display:flex; gap:14px; align-items:center; font-size:13px;
  }}
  .dot {{ width:8px; height:8px; border-radius:50%; background:var(--run); box-shadow:0 0 0 4px var(--glow); }}
  .dot.ok {{ background:var(--ok); box-shadow:0 0 0 4px rgba(61,214,140,.25); }}
  .dot.run {{ animation:pulse 1.6s infinite; }}
  @keyframes pulse {{ 50% {{ opacity:.45; }} }}
  .hero {{ display:grid; grid-template-columns: 1.2fr .8fr; gap:16px; margin-bottom:18px; }}
  @media (max-width:860px) {{ .hero {{ grid-template-columns:1fr; }} }}
  .panel {{
    background:linear-gradient(180deg, #152036, var(--panel));
    border:1px solid var(--line); border-radius:18px; padding:18px;
  }}
  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }}
  @media (max-width:860px) {{ .kpis {{ grid-template-columns:repeat(2,1fr); }} }}
  .kpi {{ background:#0e1626; border:1px solid var(--line); border-radius:14px; padding:12px; }}
  .kpi .v {{ font-size:26px; font-weight:700; margin-top:4px; }}
  .kpi .t {{ color:var(--muted); font-size:12px; }}
  .bar {{ height:12px; background:#0a1220; border-radius:999px; overflow:hidden; border:1px solid var(--line); margin-top:14px; }}
  .bar > i {{ display:block; height:100%; width:{pct}%; background:linear-gradient(90deg,#3dd68c,#5b8cff); }}
  .bar-meta {{ display:flex; justify-content:space-between; color:var(--muted); font-size:12px; margin-top:8px; }}
  h2 {{ font-size:15px; letter-spacing:.4px; text-transform:uppercase; color:var(--muted); margin:22px 0 10px; }}
  .grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }}
  @media (max-width:640px) {{ .grid {{ grid-template-columns:1fr; }} }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:14px; position:relative; overflow:hidden; cursor:pointer; transition: transform .12s ease, border-color .12s; }}
  .card:hover {{ transform: translateY(-2px); border-color:#3d5fbf; }}
  .card::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--p1); }}
  .card.done {{ border-color:#2f6b4f; }}
  .card.done::after {{ content:"✓"; position:absolute; right:12px; top:10px; color:var(--ok); font-weight:700; }}
  .card.run {{ border-color:#3d5fbf; box-shadow:0 0 0 1px rgba(91,140,255,.25) inset; }}
  .top {{ display:flex; gap:8px; align-items:center; margin-bottom:8px; }}
  .badge {{ font-size:11px; font-weight:700; padding:3px 8px; border-radius:999px; background:#2a2110; color:#ffd48a; border:1px solid #5a4518; }}
  .state {{ font-size:12px; color:var(--muted); }}
  .title {{ margin:4px 0 0; font-weight:600; }}
  .step {{ margin-top:8px; font-size:12px; color:#b7c4de; }}
  .sha {{ margin-top:10px; font-family:ui-monospace, Menlo, Consolas, monospace; font-size:12px; color:#9db7ff; word-break:break-all; }}
  .hint {{ margin-top:8px; font-size:11px; color:var(--muted); }}
  .detail {{ background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:18px; margin-top:8px; }}
  .detail-head {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-bottom:14px; }}
  .detail-head h3 {{ margin:0; flex:1; font-size:18px; }}
  .back {{ background:#0e1626; color:var(--text); border:1px solid var(--line); border-radius:999px; padding:6px 12px; cursor:pointer; }}
  .meta-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:10px; margin-bottom:14px; }}
  .meta-grid .t {{ color:var(--muted); font-size:11px; }}
  .meta-grid .v {{ margin-top:4px; font-size:13px; }}
  .mono {{ font-family:ui-monospace, Menlo, Consolas, monospace; word-break:break-all; }}
  .paths {{ padding-left:18px; color:var(--muted); font-size:12px; line-height:1.7; }}
  .paths a {{ color:#9db7ff; }}
  .snip {{ background:#0e1626; border:1px solid var(--line); border-radius:12px; padding:12px; overflow:auto; font-size:12px; color:#c9d4ea; white-space:pre-wrap; }}
  footer {{ margin-top:22px; color:var(--muted); font-size:12px; line-height:1.5; }}
  code {{ background:#0e1626; padding:1px 6px; border-radius:6px; border:1px solid var(--line); }}
  #board.hidden {{ display:none; }}
  .done-banner {{ display:{done_display}; background:#123226; border:1px solid #2f6b4f; color:#b7f5d3; border-radius:14px; padding:12px 14px; margin-bottom:14px; }}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>Mortred · Round 6</h1>
        <div class="sub">cv-cowork · 基线 <code>sub-agent-co-work</code> · 长期根本修复 · {esc(refresh_note)}</div>
      </div>
      <div class="pill"><span class="dot {dot}{run_class}"></span><span>状态 {esc(status_label)}</span></div>
    </header>
    <div class="done-banner">全部 issue 已合入并关闭。监视器应已退出；本页不再自动刷新。</div>
    <div id="board">
      <div class="hero">
        <section class="panel">
          <div class="kpis">
            <div class="kpi"><div class="t">完成</div><div class="v">{done}/{total}</div></div>
            <div class="kpi"><div class="t">进度</div><div class="v">{pct}%</div></div>
            <div class="kpi"><div class="t">P0 未关</div><div class="v">0</div></div>
            <div class="kpi"><div class="t">P1 未关</div><div class="v">{open_p1}</div></div>
          </div>
          <div class="bar"><i></i></div>
          <div class="bar-meta"><span>已合入 {esc(merged)}</span><span>生成 {esc(gen)}</span></div>
        </section>
        <section class="panel">
          <div style="color:var(--muted);font-size:12px;margin-bottom:8px;">数据源</div>
          <div style="font-size:13px;line-height:1.55;color:var(--muted)">
            <code>issues/*/03..05-*.md</code> → <code>progress_data.json</code> → 本页<br/>
            监视器 <code>progress_watch.sh</code> 每 60s；100% 后退出并删 pid
          </div>
        </section>
      </div>
      <h2>Issues（点击卡片打开详情）</h2>
      <div class="grid">
{cards_html}
      </div>
    </div>
    <div id="details">
{details_html}
    </div>
    <footer>
      路径: <code>cowork/ledger/round-006/progress.html</code> ·
      详情: <code>issue.html?id=p1-1</code> 或 hash <code>#issue/p1-1</code><br/>
      JSON: <code>cowork/ledger/round-006/progress_data.json</code>
    </footer>
  </div>
<script>
function showFromHash() {{
  const m = (location.hash || '').match(/^#issue\\/(p1-[1-4])$/);
  const board = document.getElementById('board');
  document.querySelectorAll('.detail').forEach(el => el.hidden = true);
  if (!m) {{
    board.classList.remove('hidden');
    return;
  }}
  board.classList.add('hidden');
  const panel = document.getElementById('panel-' + m[1]);
  if (panel) panel.hidden = false;
}}
window.addEventListener('hashchange', showFromHash);
showFromHash();
</script>
</body>
</html>
"""


def render_issue_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    refresh = (
        '<meta http-equiv="refresh" content="0"/>'
        if data["round_status"] == "done"
        else '<meta http-equiv="refresh" content="60"/>'
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
{refresh}
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Round 6 · Issue detail</title>
<style>
  body {{ margin:0; font-family:system-ui,sans-serif; background:#0b1220; color:#e8eefc; }}
  .wrap {{ max-width:880px; margin:0 auto; padding:24px; }}
  a {{ color:#9db7ff; }}
  .card {{ background:#121a2b; border:1px solid #243049; border-radius:16px; padding:18px; }}
  .muted {{ color:#93a0b8; font-size:13px; }}
  pre {{ background:#0e1626; border:1px solid #243049; border-radius:12px; padding:12px; white-space:pre-wrap; }}
  code {{ background:#0e1626; padding:1px 6px; border-radius:6px; }}
  .badge {{ display:inline-block; padding:3px 8px; border-radius:999px; background:#2a2110; color:#ffd48a; border:1px solid #5a4518; font-size:12px; }}
</style>
</head>
<body>
<div class="wrap">
  <p><a href="progress.html">← Round 6 board</a></p>
  <div id="root" class="card"><p class="muted">loading…</p></div>
</div>
<script type="application/json" id="data">{payload}</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const id = new URLSearchParams(location.search).get('id') || 'p1-1';
const it = (data.issues || []).find(x => x.issue_id === id) || data.issues[0];
const root = document.getElementById('root');
function esc(s) {{
  return String(s || '').replace(/[&<>]/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}})[ch]);
}}
if (!it) {{ root.innerHTML = '<p>issue not found</p>'; }}
else {{
  const snip = it.snippets || {{}};
  const paths = it.paths || {{}};
  const excerpt = snip.merge || snip.qa || snip.rd || snip.design || '(empty)';
  root.innerHTML = `
    <div><span class="badge">${{esc(it.id)}}</span> <strong>${{esc(it.title)}}</strong></div>
    <p class="muted">status=${{esc(it.status)}} · ${{esc(it.step_text || '')}}</p>
    <p>merge: <code>${{esc(it.merge_sha || '—')}}</code><br/>branch: <code>${{esc(it.branch_sha || '—')}}</code></p>
    <p class="muted">design=${{it.has_design}} rd=${{it.has_rd}} qa=${{it.has_qa}} merge=${{it.has_merge}}</p>
    <h3>Paths</h3>
    <ul>
      <li><code>${{esc(paths.design || '')}}</code></li>
      <li><code>${{esc(paths.rd || '')}}</code></li>
      <li><code>${{esc(paths.qa || '')}}</code></li>
      <li><code>${{esc(paths.merge || '')}}</code></li>
    </ul>
    <h3>Ledger excerpt</h3>
    <pre>${{esc(excerpt)}}</pre>
  `;
  document.title = 'Round 6 · ' + it.id;
}}
</script>
</body>
</html>
"""


def main() -> int:
    items = [read_issue(i) for i in TITLES]
    done = sum(1 for i in items if i["status"] == "done")
    total = len(items)
    pct = int(round(100.0 * done / total)) if total else 0
    open_p1 = total - done
    current = None
    for i in items:
        if i["status"] != "done":
            current = i["id"]
            break
    round_status = "done" if open_p1 == 0 else "running"
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S CST")
    data = {
        "round": 6,
        "generated_at": now,
        "round_status": round_status,
        "current": current,
        "done": done,
        "total": total,
        "percent": pct,
        "open_p0": 0,
        "open_p1": open_p1,
        "issues": items,
    }
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_HTML.write_text(render_html(data), encoding="utf-8")
    OUT_ISSUE.write_text(render_issue_html(data), encoding="utf-8")
    print(f"updated progress: {done}/{total} ({pct}%) status={round_status} @ {now}")
    if round_status == "done":
        print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

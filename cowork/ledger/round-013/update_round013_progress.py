#!/usr/bin/env python3
"""Regenerate round-013 progress_data.json + progress.html + issue.html from ledger."""
from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROUND = Path(__file__).resolve().parent
ISSUES_DIR = ROUND / "issues"
OUT_JSON = ROUND / "progress_data.json"
OUT_HTML = ROUND / "progress.html"
OUT_ISSUE = ROUND / "issue.html"

TITLES = {
    "p0-2": ("P0-2", "TRT StagingSlotGuard/D2HSlotGuard error-path release"),
    "p1-5": ("P1-5", "superpoint sort strict weak ordering"),
    "p0-3": ("P0-3", "http_infer_rps.py default warmup workable"),
    "p1-1": ("P1-1", "positive max_queue_depth + 429 Retry-After"),
    "p1-4": ("P1-4", "relative MORTRED_CONTROL_CONFIG via project_root"),
    "p1-6": ("P1-6", "header probe before full imdecode"),
    "p1-7": ("P1-7", "systemd KillMode=control-group + orphan exit"),
    "p0-4": ("P0-4", "capacity: lower handler_threads + sheet"),
    "p1-2": ("P1-2", "unify YOLO unmap↔NMS order"),
    "p1-3": ("P1-3", "MORTRED_METRICS_TOKEN on model /metrics"),
    "p1-8": ("P1-8", "golden skip exit 77; presets/labels SSOT"),
}
ISSUE_ORDER = list(TITLES.keys())

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
    return "待开始"


def read_issue(issue_id: str) -> dict:
    label, title = TITLES[issue_id]
    base = ISSUES_DIR / issue_id
    design = base / "03-design-final.md"
    rd = base / "04-rd.md"
    qa = base / "05-qa.md"
    merge = base / "05-merge.md"
    item = {
        "id": label,
        "issue_id": issue_id,
        "title": title,
        "priority": "P0" if issue_id.startswith("p0") else "P1",
        "status": "todo",
        "merge_sha": None,
        "branch_sha": None,
        "has_design": design.is_file(),
        "has_rd": rd.is_file(),
        "has_qa": qa.is_file(),
        "has_merge": merge.is_file(),
        "paths": {
            "design": f"cowork/ledger/round-013/issues/{issue_id}/03-design-final.md",
            "rd": f"cowork/ledger/round-013/issues/{issue_id}/04-rd.md",
            "qa": f"cowork/ledger/round-013/issues/{issue_id}/05-qa.md",
            "merge": f"cowork/ledger/round-013/issues/{issue_id}/05-merge.md",
        },
        "snippets": {
            "design": first_lines(design),
            "rd": first_lines(rd),
            "qa": first_lines(qa),
            "merge": first_lines(merge),
        },
        "step_text": "",
    }
    if merge.is_file():
        text = merge.read_text(encoding="utf-8", errors="replace")
        m = MERGE_SHA_RE.search(text)
        if m:
            item["merge_sha"] = m.group(1)[:8]
            item["status"] = "done"
        b = BRANCH_SHA_RE.search(text)
        if b:
            item["branch_sha"] = b.group(1)[:8]
    if item["status"] != "done" and rd.is_file():
        text = rd.read_text(encoding="utf-8", errors="replace")
        b = BRANCH_SHA_RE.search(text)
        if b:
            item["branch_sha"] = b.group(1)[:8]
        if item["has_qa"]:
            item["status"] = "qa"
        else:
            item["status"] = "run"
    elif item["status"] != "done" and item["has_design"]:
        item["status"] = "design"
    item["step_text"] = infer_step(item)
    return item


def build() -> dict:
    issues = [read_issue(i) for i in ISSUE_ORDER]
    done = sum(1 for i in issues if i["status"] == "done")
    total = len(issues)
    percent = int(round(100.0 * done / total)) if total else 0
    current = None
    for i in issues:
        if i["status"] != "done":
            current = i["id"]
            if i["status"] == "todo":
                i["status"] = "run"
            break
    open_p0 = sum(1 for i in issues if i["priority"] == "P0" and i["status"] != "done")
    open_p1 = sum(1 for i in issues if i["priority"] == "P1" and i["status"] != "done")
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S CST")
    round_status = "done" if percent == 100 else "running"
    return {
        "round": 13,
        "generated_at": now,
        "round_status": round_status,
        "current": current,
        "done": done,
        "total": total,
        "percent": percent,
        "open_p0": open_p0,
        "open_p1": open_p1,
        "issues": issues,
    }


def render_html(data: dict) -> str:
    issues = data["issues"]
    pct = data["percent"]
    done_banner = ""
    if data["round_status"] == "done":
        done_banner = '<div class="done-banner">全部 issue 已合入并关闭。监视器应已退出；本页不再自动刷新。</div>'
    cards = []
    for it in issues:
        st = it["status"]
        cls = "card"
        if st == "done":
            cls += " done"
        elif st in ("run", "qa", "design"):
            cls += " run"
        pri = it["priority"]
        badge_bg = "#2a2110" if pri == "P1" else "#3a1520"
        badge_fg = "#ffd48a" if pri == "P1" else "#ff9aa8"
        sha = it.get("merge_sha") or it.get("branch_sha") or ""
        cards.append(
            f'<div class="{cls}" data-id="{html.escape(it["issue_id"])}" onclick="location.hash=\'{html.escape(it["issue_id"])}\'">'
            f'<div class="top"><span class="badge" style="background:{badge_bg};color:{badge_fg};border-color:#5a4518">{html.escape(it["id"])}</span>'
            f'<span class="state">{html.escape(st)}</span></div>'
            f'<div class="title">{html.escape(it["title"])}</div>'
            f'<div class="step">{html.escape(it.get("step_text") or "")}</div>'
            f'<div class="sha">{html.escape(sha)}</div>'
            f'<div class="hint">点击查看 design / RD / QA / merge</div></div>'
        )
    cards_html = "\n".join(cards)
    dot = "ok" if data["round_status"] == "done" else "run"
    status_label = "done · 队列清空" if data["round_status"] == "done" else f"running · 当前 {data.get('current') or '-'}"
    refresh = "" if data["round_status"] == "done" else '<meta http-equiv="refresh" content="60"/>'
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
{refresh}
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Mortred · Round 13 进度</title>
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
  .badge {{ font-size:11px; font-weight:700; padding:3px 8px; border-radius:999px; border:1px solid #5a4518; }}
  .state {{ font-size:12px; color:var(--muted); }}
  .title {{ margin:4px 0 0; font-weight:600; }}
  .step {{ margin-top:8px; font-size:12px; color:#b7c4de; }}
  .sha {{ margin-top:10px; font-family:ui-monospace, Menlo, Consolas, monospace; font-size:12px; color:#9db7ff; word-break:break-all; }}
  .hint {{ margin-top:8px; font-size:11px; color:var(--muted); }}
  .done-banner {{ display:block; background:#123226; border:1px solid #2f6b4f; color:#b7f5d3; border-radius:14px; padding:12px 14px; margin-bottom:14px; }}
  footer {{ margin-top:22px; color:var(--muted); font-size:12px; line-height:1.5; }}
  code {{ background:#0e1626; padding:1px 6px; border-radius:6px; border:1px solid var(--line); }}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>Mortred · Round 13</h1>
        <div class="sub">cv-cowork · 基线 <code>sub-agent-co-work @ fb982277</code> · triage P0-2→P1-8 · 60s 刷新</div>
      </div>
      <div class="pill"><span class="dot {dot}"></span><span>状态 {html.escape(status_label)}</span></div>
    </header>
    {done_banner}
    <div id="board">
      <div class="hero">
        <section class="panel">
          <div class="kpis">
            <div class="kpi"><div class="t">完成</div><div class="v">{data['done']}/{data['total']}</div></div>
            <div class="kpi"><div class="t">进度</div><div class="v">{pct}%</div></div>
            <div class="kpi"><div class="t">P0 未关</div><div class="v">{data['open_p0']}</div></div>
            <div class="kpi"><div class="t">P1 未关</div><div class="v">{data['open_p1']}</div></div>
          </div>
          <div class="bar"><i></i></div>
          <div class="bar-meta"><span>生成于 {html.escape(data['generated_at'])}</span><span>当前 {html.escape(str(data.get('current') or '—'))}</span></div>
        </section>
        <section class="panel">
          <div class="t" style="color:var(--muted);font-size:12px">队列（锁定）</div>
          <ol style="margin:10px 0 0;padding-left:18px;color:#c9d4ea;font-size:13px;line-height:1.55">
            <li>P0-2 TRT slot guard</li>
            <li>P1-5 superpoint sort</li>
            <li>P0-3 http_infer_rps warmup</li>
            <li>P1-1 max_queue_depth</li>
            <li>P1-4 CONTROL_CONFIG path</li>
            <li>P1-6 decode size probe</li>
            <li>P1-7 KillMode / orphan</li>
            <li>P0-4 capacity / threads</li>
            <li>P1-2 YOLO NMS order</li>
            <li>P1-3 metrics token</li>
            <li>P1-8 golden skip</li>
          </ol>
        </section>
      </div>
      <h2>Cards</h2>
      <div class="grid">
        {cards_html}
      </div>
    </div>
    <footer>Board: <code>cowork/ledger/round-013/progress.html</code> · data <code>progress_data.json</code> · click card → issue detail via hash (see issue.html).</footer>
  </div>
  <script>
    // navigate to issue.html?id= when hash set
    function go() {{
      const h = (location.hash || '').replace(/^#/, '');
      if (h) location.href = 'issue.html?id=' + encodeURIComponent(h);
    }}
    window.addEventListener('hashchange', go);
    if (location.hash) go();
  </script>
</body>
</html>
"""


def render_issue(data: dict) -> str:
    # simple issue detail page driven by ?id=
    blocks = []
    for it in data["issues"]:
        blocks.append(
            f'<section class="detail" id="{html.escape(it["issue_id"])}" hidden>'
            f'<div class="detail-head"><button class="back" onclick="location.href=\'progress.html\'">← 看板</button>'
            f'<h3>{html.escape(it["id"])} · {html.escape(it["title"])}</h3></div>'
            f'<div class="meta-grid">'
            f'<div><div class="t">status</div><div class="v">{html.escape(it["status"])}</div></div>'
            f'<div><div class="t">merge</div><div class="v mono">{html.escape(it.get("merge_sha") or "—")}</div></div>'
            f'<div><div class="t">branch</div><div class="v mono">{html.escape(it.get("branch_sha") or "—")}</div></div>'
            f'<div><div class="t">step</div><div class="v">{html.escape(it.get("step_text") or "")}</div></div>'
            f'</div>'
            f'<h4>design</h4><pre class="snip">{html.escape(it["snippets"].get("design") or "(missing)")}</pre>'
            f'<h4>RD</h4><pre class="snip">{html.escape(it["snippets"].get("rd") or "(missing)")}</pre>'
            f'<h4>QA</h4><pre class="snip">{html.escape(it["snippets"].get("qa") or "(missing)")}</pre>'
            f'<h4>merge</h4><pre class="snip">{html.escape(it["snippets"].get("merge") or "(missing)")}</pre>'
            f'</section>'
        )
    body = "\n".join(blocks)
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/><title>Round 13 issue</title>
<style>
  :root {{ --bg:#0b1220; --panel:#121a2b; --line:#243049; --text:#e8eefc; --muted:#93a0b8; }}
  body {{ margin:0; background:var(--bg); color:var(--text); font-family:system-ui,sans-serif; }}
  .wrap {{ max-width:900px; margin:0 auto; padding:24px; }}
  .detail {{ background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:18px; }}
  .detail-head {{ display:flex; gap:12px; align-items:center; margin-bottom:14px; }}
  .back {{ background:#0e1626; color:var(--text); border:1px solid var(--line); border-radius:999px; padding:6px 12px; cursor:pointer; }}
  .meta-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:10px; margin-bottom:14px; }}
  .t {{ color:var(--muted); font-size:11px; }} .v {{ margin-top:4px; font-size:13px; }}
  .mono {{ font-family:ui-monospace,Menlo,Consolas,monospace; word-break:break-all; }}
  .snip {{ background:#0e1626; border:1px solid var(--line); border-radius:12px; padding:12px; overflow:auto; font-size:12px; color:#c9d4ea; white-space:pre-wrap; }}
</style></head><body><div class="wrap">{body}</div>
<script>
  const id = new URLSearchParams(location.search).get('id');
  document.querySelectorAll('.detail').forEach(el => {{
    el.hidden = el.id !== id;
  }});
  if (!id || !document.getElementById(id)) {{
    document.querySelector('.wrap').innerHTML = '<p>missing id — <a href="progress.html" style="color:#9db7ff">back</a></p>';
  }}
</script></body></html>
"""


def main() -> None:
    data = build()
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_HTML.write_text(render_html(data), encoding="utf-8")
    OUT_ISSUE.write_text(render_issue(data), encoding="utf-8")
    print(f"round-013 {data['done']}/{data['total']} {data['percent']}% status={data['round_status']} current={data['current']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Regenerate round-014 progress_data.json + progress.html + issue.html from ledger."""
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
    "p0-1": ("P0-1", "TRT buffer.ensure fail path set abandon"),
    "p1-1": ("P1-1", "Trust chain: TSan SSOT + golden skip 77"),
    "p0-2": ("P0-2", "Capacity RSS/thread/fd + occupancy RAM"),
    "p1-2": ("P1-2", "Pre-decode size gate non-JPEG/PNG"),
    "p1-3": ("P1-3", "Same-family detection/matting/seg tail"),
    "p1-4": ("P1-4", "Supervisor child env allowlist"),
}
ISSUE_ORDER = ["p0-1", "p1-1", "p0-2", "p1-2", "p1-3", "p1-4"]
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
            "design": f"cowork/ledger/round-014/issues/{issue_id}/03-design-final.md",
            "rd": f"cowork/ledger/round-014/issues/{issue_id}/04-rd.md",
            "qa": f"cowork/ledger/round-014/issues/{issue_id}/05-qa.md",
            "merge": f"cowork/ledger/round-014/issues/{issue_id}/05-merge.md",
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
    status = data["round_status"]
    gen = data["generated_at"]
    current = data.get("current") or "—"
    refresh_meta = "" if status == "done" else '<meta http-equiv="refresh" content="60"/>'
    refresh_note = "已停止刷新 · 队列清空" if status == "done" else "meta refresh 60s · 点卡片看详情"
    cards = []
    detail_panels = []
    for it in data["issues"]:
        pri = "p0" if it["priority"] == "P0" else "p1"
        cls = f"{pri} done" if it["status"] == "done" else (
            f"{pri} run" if it["status"] in ("running", "designed") else pri
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
      <h2>{esc(it['id'])} · {esc(it['title'])}</h2>
      <p class="meta">{esc(flags)} · sha {esc(sha)}</p>
      <h3>design</h3><pre>{esc(snip.get('design') or '(none)')}</pre>
      <p class="path">{esc(paths.get('design') or '')}</p>
      <h3>rd</h3><pre>{esc(snip.get('rd') or '(none)')}</pre>
      <h3>qa</h3><pre>{esc(snip.get('qa') or '(none)')}</pre>
      <h3>merge</h3><pre>{esc(snip.get('merge') or '(none)')}</pre>
      <p><a href="#" onclick="location.hash='';return false">← 返回看板</a></p>
    </section>"""
        )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  {refresh_meta}
  <title>Round 14 · {done}/{total} ({pct}%)</title>
  <style>
    :root {{ --bg:#0f1419; --card:#1a2332; --p0:#e74c3c; --p1:#f39c12; --ok:#27ae60; --run:#3498db; --fg:#ecf0f1; --muted:#95a5a6; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font:15px/1.45 system-ui,sans-serif; background:var(--bg); color:var(--fg); }}
    header {{ padding:1.25rem 1.5rem; border-bottom:1px solid #243044; }}
    h1 {{ margin:0 0 .35rem; font-size:1.35rem; }}
    .bar {{ height:10px; background:#243044; border-radius:6px; overflow:hidden; margin:.6rem 0; max-width:420px; }}
    .bar > i {{ display:block; height:100%; background:linear-gradient(90deg,var(--ok),#2ecc71); width:{pct}%; }}
    .meta {{ color:var(--muted); font-size:.9rem; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:1rem; padding:1.25rem 1.5rem; }}
    .card {{ background:var(--card); border-radius:10px; padding:1rem 1.1rem; border:1px solid #2a3a50; cursor:pointer; transition:border-color .15s, transform .1s; }}
    .card:hover, .card:focus {{ border-color:#4a90d9; outline:none; transform:translateY(-1px); }}
    .card.done {{ border-color:var(--ok); }}
    .card.run {{ border-color:var(--run); }}
    .card.p0 .badge {{ background:var(--p0); }}
    .card.p1 .badge {{ background:var(--p1); }}
    .top {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:.4rem; }}
    .badge {{ font-size:.75rem; font-weight:700; padding:.15rem .45rem; border-radius:4px; color:#fff; }}
    .state {{ font-size:.8rem; color:var(--muted); }}
    .title {{ font-weight:600; margin-bottom:.35rem; }}
    .step,.sha,.hint {{ font-size:.8rem; color:var(--muted); }}
    .hint {{ margin-top:.5rem; color:#5dade2; }}
    .detail {{ padding:1.25rem 1.5rem; max-width:900px; }}
    .detail pre {{ background:#121a24; padding:.75rem; border-radius:6px; overflow:auto; white-space:pre-wrap; font-size:.82rem; }}
    .detail[hidden] {{ display:none; }}
    #board[hidden] {{ display:none; }}
  </style>
</head>
<body>
  <header>
    <h1>Round 14 residuals · {done}/{total} · {pct}%</h1>
    <div class="bar"><i></i></div>
    <p class="meta">current: {esc(current)} · {esc(refresh_note)} · generated {esc(gen)} CST</p>
  </header>
  <div id="board" class="grid">
{chr(10).join(cards)}
  </div>
{chr(10).join(detail_panels)}
  <script>
    function show() {{
      const m = location.hash.match(/^#issue\\/([\\w-]+)$/);
      const board = document.getElementById('board');
      document.querySelectorAll('.detail').forEach(el => el.hidden = true);
      if (m) {{
        const p = document.getElementById('panel-' + m[1]);
        if (p) {{ board.hidden = true; p.hidden = false; }}
        else {{ board.hidden = false; }}
      }} else {{ board.hidden = false; }}
    }}
    window.addEventListener('hashchange', show);
    show();
  </script>
</body>
</html>
"""


def main() -> None:
    issues = [read_issue(i) for i in ISSUE_ORDER]
    done = sum(1 for i in issues if i["status"] == "done")
    total = len(issues)
    pct = int(round(100.0 * done / total)) if total else 0
    status = "done" if done == total else "running"
    current = next((i["id"] for i in issues if i["status"] != "done"), None)
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "round": 14,
        "done": done,
        "total": total,
        "percent": pct,
        "open_p0": sum(1 for i in issues if i["priority"] == "P0" and i["status"] != "done"),
        "open_p1": sum(1 for i in issues if i["priority"] == "P1" and i["status"] != "done"),
        "round_status": status,
        "current": current,
        "generated_at": now,
        "issues": issues,
    }
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_HTML.write_text(render_html(data), encoding="utf-8")
    # thin issue.html redirect helper
    OUT_ISSUE.write_text(
        '<!DOCTYPE html><meta http-equiv="refresh" content="0;url=progress.html">\n',
        encoding="utf-8",
    )
    print(f"round-014 board {done}/{total} {pct}% {status}")


if __name__ == "__main__":
    main()

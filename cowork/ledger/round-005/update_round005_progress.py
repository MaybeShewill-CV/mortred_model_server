#!/usr/bin/env python3
"""Regenerate round-005 progress_data.json + progress.html from ledger merges."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROUND = Path(__file__).resolve().parent
ISSUES = ROUND / "issues"
OUT_JSON = ROUND / "progress_data.json"
OUT_HTML = ROUND / "progress.html"

TITLES = {
    "p1-1": ("P1-1", "FixedWindow vs GCRA dual semantics"),
    "p1-2": ("P1-2", "gpu_sampler nvidia-smi waitpid harden"),
    "p1-3": ("P1-3", "max_inflight gateway wiring"),
    "p1-4": ("P1-4", "VERSION honesty 1.0.0"),
    "p1-5": ("P1-5", "Occupancy test drift"),
}

MERGE_SHA_RE = re.compile(r"^\s*-\s*merge commit:\s*([0-9a-f]{7,40})", re.I | re.M)
BRANCH_SHA_RE = re.compile(r"^\s*-\s*问题分支提交:\s*([0-9a-f]{7,40})", re.I | re.M)


def read_merge(issue_id: str) -> dict:
    path = ISSUES / issue_id / "05-merge.md"
    label, title = TITLES[issue_id]
    item = {
        "id": label,
        "issue_id": issue_id,
        "title": title,
        "priority": "P1",
        "status": "todo",
        "merge_sha": None,
        "branch_sha": None,
        "has_design": (ISSUES / issue_id / "03-design-final.md").is_file(),
        "has_rd": (ISSUES / issue_id / "04-rd.md").is_file(),
        "has_qa": (ISSUES / issue_id / "05-qa.md").is_file(),
        "has_merge": path.is_file(),
    }
    if not path.is_file():
        # infer in-progress from artifacts
        if item["has_qa"] or item["has_rd"]:
            item["status"] = "running"
        elif item["has_design"]:
            item["status"] = "designed"
        return item
    text = path.read_text(encoding="utf-8", errors="replace")
    m = MERGE_SHA_RE.search(text)
    b = BRANCH_SHA_RE.search(text)
    if m:
        item["merge_sha"] = m.group(1)
        item["status"] = "done"
    else:
        item["status"] = "running"
    if b:
        item["branch_sha"] = b.group(1)
    return item


def render_html(data: dict) -> str:
    done = data["done"]
    total = data["total"]
    pct = data["percent"]
    open_p1 = data["open_p1"]
    status = data["round_status"]
    gen = data["generated_at"]
    current = data.get("current") or "—"
    cards = []
    for it in data["issues"]:
        cls = "p1 done" if it["status"] == "done" else ("p1 run" if it["status"] in ("running", "designed") else "p1")
        sha = it["merge_sha"] or it["branch_sha"] or "—"
        state = {
            "done": "已合入",
            "running": "进行中",
            "designed": "方案已锁",
            "todo": "待开始",
        }.get(it["status"], it["status"])
        cards.append(
            f"""    <article class="card {cls}">
      <div class="top"><span class="badge">{it['id']}</span><span class="state">{state}</span></div>
      <div class="title">{it['title']}</div>
      <div class="sha">{sha}</div>
    </article>"""
        )
    cards_html = "\n".join(cards)
    merged = ", ".join(it["id"] for it in data["issues"] if it["status"] == "done") or "无"
    dot = "ok" if status == "done" else "run"
    status_label = "done · 队列清空" if status == "done" else f"running · 当前 {current}"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta http-equiv="refresh" content="60"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Mortred · Round 5 进度</title>
<style>
  :root {{
    --bg:#0b1220; --panel:#121a2b; --line:#243049; --text:#e8eefc; --muted:#93a0b8;
    --p0:#ff5d5d; --p1:#f5a524; --ok:#3dd68c; --run:#5b8cff;
    --glow:rgba(91,140,255,.25);
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
  .grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
  @media (max-width:980px) {{ .grid {{ grid-template-columns:repeat(2,1fr); }} }}
  @media (max-width:640px) {{ .grid {{ grid-template-columns:1fr; }} }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:14px; position:relative; overflow:hidden; }}
  .card::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--p1); }}
  .card.done {{ border-color:#2f6b4f; }}
  .card.done::after {{ content:"✓"; position:absolute; right:12px; top:10px; color:var(--ok); font-weight:700; }}
  .card.run {{ border-color:#3d5fbf; box-shadow:0 0 0 1px rgba(91,140,255,.25) inset; }}
  .top {{ display:flex; gap:8px; align-items:center; margin-bottom:8px; }}
  .badge {{ font-size:11px; font-weight:700; padding:3px 8px; border-radius:999px; background:#2a2110; color:#ffd48a; border:1px solid #5a4518; }}
  .state {{ font-size:12px; color:var(--muted); }}
  .title {{ margin:4px 0 0; font-weight:600; }}
  .sha {{ margin-top:10px; font-family:ui-monospace, Menlo, Consolas, monospace; font-size:12px; color:#9db7ff; word-break:break-all; }}
  footer {{ margin-top:22px; color:var(--muted); font-size:12px; line-height:1.5; }}
  code {{ background:#0e1626; padding:1px 6px; border-radius:6px; border:1px solid var(--line); }}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>Mortred · Round 5</h1>
        <div class="sub">cv-cowork · 基线 <code>sub-agent-co-work</code> · DESKTOP-B8MJFUS / WSL · meta refresh 60s</div>
      </div>
      <div class="pill"><span class="dot {dot}{' run' if status != 'done' else ''}"></span><span>状态 {status_label}</span></div>
    </header>
    <div class="hero">
      <section class="panel">
        <div class="kpis">
          <div class="kpi"><div class="t">完成</div><div class="v">{done}/{total}</div></div>
          <div class="kpi"><div class="t">进度</div><div class="v">{pct}%</div></div>
          <div class="kpi"><div class="t">P0 未关</div><div class="v">0</div></div>
          <div class="kpi"><div class="t">P1 未关</div><div class="v">{open_p1}</div></div>
        </div>
        <div class="bar"><i></i></div>
        <div class="bar-meta"><span>已合入 {merged}</span><span>生成 {gen} · 刷新 60s</span></div>
      </section>
      <section class="panel">
        <div style="color:var(--muted);font-size:12px;margin-bottom:8px;">数据源</div>
        <div style="font-size:13px;line-height:1.55;color:var(--muted)">
          <code>issues/*/05-merge.md</code> → <code>progress_data.json</code> → 本页<br/>
          监视器 <code>progress_watch.sh</code> 每 60s 调用 <code>update_round005_progress.py</code>
        </div>
      </section>
    </div>
    <h2>Issues</h2>
    <div class="grid">
{cards_html}
    </div>
    <footer>
      路径: <code>cowork/ledger/round-005/progress.html</code><br/>
      JSON: <code>cowork/ledger/round-005/progress_data.json</code>
    </footer>
  </div>
</body>
</html>
"""


def main() -> None:
    items = [read_merge(i) for i in TITLES]
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
        "round": 5,
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
    print(f"updated {OUT_JSON.name} + {OUT_HTML.name}: {done}/{total} ({pct}%) @ {now}")


if __name__ == "__main__":
    main()

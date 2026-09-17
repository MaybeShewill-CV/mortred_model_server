#!/usr/bin/env python3
"""mortredctl ps / down — quick liveness probe + shutdown for the mortred
control plane (supervisor, gateway, model servers).

  mortredctl ps                  probe everything, print a detail table
  mortredctl down                stop ALL running model servers (asks first)
  mortredctl down --id a,b       stop specific model servers
  mortredctl down --all          servers + gateway + supervisor (local host only)

Standalone-friendly: runs anywhere python3 does (e.g. from your laptop over
an ssh tunnel); zero deps beyond the stdlib. Env: MORTREDCTL_ADDR,
MORTREDCTL_GATEWAY_ADDR, MORTREDCTL_TOKEN (or MORTRED_API_TOKEN).
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

SUPERVISOR_ADDR = os.environ.get("MORTREDCTL_ADDR", "http://127.0.0.1:8787")
GATEWAY_ADDR = os.environ.get("MORTREDCTL_GATEWAY_ADDR", "http://127.0.0.1:8080")
TOKEN = os.environ.get("MORTREDCTL_TOKEN", "") or os.environ.get("MORTRED_API_TOKEN", "")

TTY = sys.stdout.isatty()
GREEN, RED, AMBER, DIM, BOLD, RESET = (
    ("\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m") if TTY else ("",) * 6
)


def c(text, color):
    return f"{color}{text}{RESET}"


def http_json(base, path, token="", timeout=2.5, method="GET"):
    """Returns (status, parsed_json_or_None, raw_body). status -1 = transport error."""
    req = urllib.request.Request(base.rstrip("/") + path, method=method)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(raw), raw
            except ValueError:
                return resp.status, None, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw), raw
        except ValueError:
            return e.code, None, raw
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return -1, None, str(e)


def http_text(base, path, timeout=2.5):
    req = urllib.request.Request(base.rstrip("/") + path)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace").strip()
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, TimeoutError, OSError):
        return -1, ""


def reachable_host(base):
    """0.0.0.0/[::] bind addresses are not dialable — swap for loopback."""
    for wildcard in ("http://0.0.0.0:", "http://[::]:", "http://::"):
        if base.startswith(wildcard):
            return base.replace(wildcard, "http://127.0.0.1:", 1)
    return base


def fmt_uptime(started_at_ms):
    if not started_at_ms or started_at_ms <= 0:
        return "--"
    sec = max(0, int(time.time() * 1000 - started_at_ms) / 1000)
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    d, h = divmod(h, 24)
    if d:
        return f"{d}d{h}h"
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def state_colored(state):
    if state in ("running", "up", "ok"):
        return c("● " + state, GREEN)
    if state in ("failed", "down"):
        return c("● " + state, RED)
    if state in ("starting", "backoff", "stopping"):
        return c("● " + state, AMBER)
    return c("· " + state, DIM)


def probe(args):
    print(c("mortred ps", BOLD) + c(f" — {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", DIM))
    sup_addr = reachable_host(args.addr)
    gw_addr = reachable_host(args.gateway or GATEWAY_ADDR)

    # ---- supervisor ----
    sup_status, _, _ = http_json(sup_addr, "/api/v1/health", timeout=2)
    if sup_status != 200:
        print(f"  SUPERVISOR  {state_colored('down')}   {args.addr}")
        print(c("    hint: ssh -L 8787:<host>:8787 … / systemctl status mortred-supervisor", DIM))
        # still try the gateway directly — it may live while the supervisor is dead
    else:
        print(f"  SUPERVISOR  {state_colored('up')}   {args.addr}")

    # ---- gateway: truth = direct healthz probe; context = supervisor's view ----
    gw_hz, _ = http_text(gw_addr, "/healthz", timeout=2)
    gw_alive = gw_hz == 200
    print(f"  GATEWAY     {state_colored('up' if gw_alive else 'down')}   {gw_addr}  healthz={'ok' if gw_alive else 'no response'}")

    # ---- supervisor's view (needs token) ----
    st_status, status, _ = http_json(sup_addr, "/api/v1/status", token=args.token or TOKEN, timeout=3)
    cat_status, catalog, _ = http_json(sup_addr, "/api/v1/catalog", token=args.token or TOKEN, timeout=3)
    gpu_status, gpu, _ = http_json(sup_addr, "/api/v1/gpu", token=args.token or TOKEN, timeout=3)

    if st_status == 401 or cat_status == 401:
        print(c("    ⚠ management API needs a token: --token T / MORTREDCTL_TOKEN (server detail hidden)", AMBER))
    elif st_status != 200:
        if sup_status == 200:
            print(c(f"    ⚠ /api/v1/status unreachable (HTTP {st_status})", AMBER))

    if status and st_status == 200:
        g = status.get("gateway") or {}
        gaddr = (g.get("address") or {})
        line = f"    supervisor-view: state={g.get('state', '?')}"
        if g.get("pid", 0) > 0:
            line += f" pid={g['pid']}"
        line += f" restarts={g.get('restart_count', 0)}"
        if gaddr.get("port"):
            line += f" bind={gaddr.get('host')}:{gaddr.get('port')}"
        print(c(line, DIM))

    # ---- gpu ----
    if gpu_status == 200 and gpu and gpu.get("available"):
        s = (gpu.get("samples") or [])
        last = s[-1] if s else {}
        if last:
            mem_t = last.get("mem_total_mib", 0)
            mem_u = last.get("mem_used_mib", 0)
            mem = f"{mem_u / 1024:.1f}G/{mem_t / 1024:.1f}G ({mem_u * 100 // mem_t}%)" if mem_t > 0 else "--"
            print(
                f"  GPU         {c('●', GREEN)} {gpu.get('name', '?')}  "
                f"util {last.get('util', '--')}%  vram {mem}  "
                f"temp {last.get('temp_c', '--')}°C  pwr {last.get('power_w', '--')}W"
            )

    # ---- servers table (catalog ⨯ status merge) ----
    if not (catalog and status and st_status == 200):
        return 0 if (sup_status == 200 and gw_alive) else 1

    by_id = {s.get("id"): s for s in (status.get("servers") or [])}
    cat_by_id = {s.get("id"): s for s in (catalog.get("servers") or [])}
    ids = [s.get("id") for s in (catalog.get("servers") or [])] or list(by_id)
    running = sum(1 for s in by_id.values() if s.get("state") in ("running", "starting", "backoff"))
    total = len(set(ids) | set(by_id))
    print(f"  SERVERS     {running} live / {total} total")
    header = f"    {'id':<18}{'state':<12}{'ready':<7}{'category':<20}{'type':<6}{'port':<7}{'pid':<8}{'uptime':<10}{'↻':<4}info"
    print(c(header, DIM))
    for sid in ids:
        st = by_id.get(sid) or {}
        ct = cat_by_id.get(sid) or {}
        state_str = str(st.get("state", "stopped"))
        sc = (GREEN if state_str == "running" else
              RED if state_str == "failed" else
              AMBER if state_str in ("starting", "backoff") else DIM)
        ready = "yes" if st.get("ready") else "no"
        info_parts = []
        if st.get("error"):
            info_parts.append(c("err: " + str(st["error"])[:60], RED))
        if st.get("last_exit_status") is not None and state_str != "running":
            info_parts.append(c(f"last_exit={st['last_exit_status']}", DIM))
        if not st:
            info_parts.append(c("not in supervisor status (never started)", DIM))
        # pad the plain text first, then wrap in color — keeps column alignment
        state_cell = c(f"{state_str:<12}", sc)
        print(
            f"    {str(sid):<18}{state_cell}"
            f"{ready:<7}{str(ct.get('category', '--')):<20}{str(ct.get('type', '--')):<6}"
            f"{str(ct.get('port', st.get('port', '--'))):<7}{str(st.get('pid', '--') if st.get('pid') is not None else '--'):<8}"
            f"{fmt_uptime(st.get('started_at_ms')):<10}{str(st.get('restart_count', 0)):<4}{' '.join(info_parts)}"
        )
    rc_sup = sup_status == 200
    return 0 if (rc_sup and gw_alive) else 1


def confirm(question):
    if os.environ.get("MORTREDCTL_YES") == "1":
        return True
    if not TTY:
        print(c("refusing interactive confirm without a tty (use --yes)", AMBER))
        return False
    try:
        return input(f"{question} [y/N] ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def stop_servers(args, ids=None):
    sup_addr = reachable_host(args.addr)
    token = args.token or TOKEN
    st_status, status, _ = http_json(sup_addr, "/api/v1/status", token=token, timeout=3)
    if st_status != 200 or not status:
        print(c(f"cannot list servers (supervisor {args.addr} HTTP {st_status}); nothing stopped", RED))
        return 1
    running = [s["id"] for s in (status.get("servers") or [])
               if s.get("state") in ("running", "starting", "backoff")]
    targets = ids or running
    if ids:
        missing = [i for i in ids if i not in {s["id"] for s in status.get("servers", [])}]
        if missing:
            print(c(f"unknown server ids: {', '.join(missing)}", RED))
            return 2
    if not targets:
        print("no running model servers — nothing to stop")
        return 0
    label = ", ".join(targets) if len(targets) <= 8 else f"{len(targets)} servers"
    if not args.yes and not confirm(f"stop {label}?"):
        print("aborted")
        return 1
    rc = 0
    for sid in targets:
        code, body, _ = http_json(sup_addr, f"/api/v1/servers/{sid}/stop", token=token, method="POST", timeout=10)
        ok = 200 <= code < 300
        print(f"  {sid:<18} {'✓ stopped' if ok else c(f'✗ HTTP {code} {body}', RED)}")
        rc |= 0 if ok else 1
    return rc


def stop_control_plane(args):
    """Full shutdown: model servers via API, then gateway+supervisor locally."""
    rc = stop_servers(args)
    print("\nstopping gateway + supervisor (local host only)…")
    killed = []
    if shutil.which("systemctl"):
        unit = subprocess.run(["systemctl", "list-unit-files"], capture_output=True, text=True).stdout
        if "mortred-supervisor" in unit:
            r = subprocess.run(["systemctl", "stop", "mortred-supervisor"], capture_output=True, text=True)
            if r.returncode == 0:
                killed.append("systemctl stop mortred-supervisor (kills the whole tree)")
            else:
                print(c(f"systemctl stop failed: {r.stderr.strip()}", RED))
    if not killed:
        for pattern in ("mortred[-]supervisor", "mortred[-]gateway"):
            r = subprocess.run(["pkill", "-TERM", "-f", pattern], capture_output=True)
            if r.returncode == 0:
                killed.append(f"pkill -TERM -f {pattern.replace('[-]', '-')}")
    if killed:
        for k in killed:
            print(f"  ✓ {k}")
    else:
        print(c("  no local mortred processes found — supervisor/gateway run elsewhere?", AMBER))
        print(c("    (API has no gateway-stop endpoint; on the host: systemctl stop mortred-supervisor)", DIM))
    return rc


def main():
    ap = argparse.ArgumentParser(prog="mortredctl ps", description="probe/stop the mortred control plane")
    ap.add_argument("--addr", default=SUPERVISOR_ADDR, help="supervisor base url")
    ap.add_argument("--gateway", default=GATEWAY_ADDR, help="gateway base url (probe only)")
    ap.add_argument("--token", default=TOKEN, help="supervisor api token")
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("down", help="stop model servers / whole control plane")
    d.add_argument("--id", help="comma-separated server ids to stop (default: all running)")
    d.add_argument("--all", action="store_true", help="also stop gateway + supervisor (local host only)")
    d.add_argument("--yes", action="store_true", help="skip confirmation")
    args = ap.parse_args()

    if args.cmd is None or args.cmd == "ps":
        sys.exit(probe(args))
    ids = [x.strip() for x in args.id.split(",") if x.strip()] if args.id else None
    if args.all:
        if not (args.yes or confirm("stop ALL servers + gateway + supervisor?")):
            print("aborted")
            sys.exit(1)
        args.yes = True
        sys.exit(stop_control_plane(args))
    sys.exit(stop_servers(args, ids))


if __name__ == "__main__":
    main()

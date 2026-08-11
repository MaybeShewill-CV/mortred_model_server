#!/usr/bin/env bash
# kill app_server and any spawned model servers (match full cmdline, since
# comm is truncated to 15 chars without the ".out" suffix)
pkill -9 -x app_server 2>/dev/null
pkill -9 -f "server\.out" 2>/dev/null
pkill -9 -f "chatbot_server" 2>/dev/null
sleep 1
ss -tlnp 2>/dev/null | grep 8787 || echo PORT_FREE

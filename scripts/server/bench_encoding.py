#!/usr/bin/env python3
"""Benchmark the two envelope encodings against a live model server.

Sends the same image N times as JSON+base64 and as a raw body, then reports
p50/p99/mean latency, throughput and payload-size comparison. The response
envelope is identical by contract; this measures what the wire saves.

Usage:
  python scripts/server/bench_encoding.py --url http://localhost:9056/... \
      --image demo_data/model_test_input/dog.jpg -n 200 --token secret
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import statistics
import time
import urllib.parse
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def run_encoding(url: str, image_bytes: bytes, headers: dict, body_builder, n: int) -> dict:
    parsed = urllib.parse.urlparse(url)
    conn_factory = http.client.HTTPConnection if parsed.scheme == "http" else http.client.HTTPSConnection
    latencies: list[float] = []
    payload_bytes = 0
    errors = 0
    for i in range(n):
        body = body_builder(i)
        payload_bytes = len(body)
        started = time.perf_counter()
        try:
            conn = conn_factory(parsed.netloc, timeout=30)
            conn.request("POST", parsed.path + ("?" + parsed.query if parsed.query else ""),
                         body=body, headers=headers)
            resp = conn.getresponse()
            resp.read()
            if resp.status != 200:
                errors += 1
            conn.close()
        except Exception:
            errors += 1
        latencies.append((time.perf_counter() - started) * 1000.0)
    total = sum(latencies) / 1000.0
    return {
        "p50": percentile(latencies, 0.50),
        "p99": percentile(latencies, 0.99),
        "mean": statistics.fmean(latencies),
        "throughput": n / total if total > 0 else 0.0,
        "payload_bytes": payload_bytes,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("-n", type=int, default=200)
    parser.add_argument("--token", default="")
    parser.add_argument("--out", default="", help="optional markdown output path")
    args = parser.parse_args()

    image_bytes = Path(args.image).read_bytes()
    b64 = base64.b64encode(image_bytes).decode()
    base_headers = {"Authorization": "Bearer %s" % args.token} if args.token else {}

    json_result = run_encoding(
        args.url,
        image_bytes,
        {**base_headers, "Content-Type": "application/json"},
        lambda i: json.dumps({"images": [b64], "req_id": "bench-%d" % i}),
        args.n,
    )
    raw_result = run_encoding(
        args.url,
        image_bytes,
        {**base_headers, "Content-Type": "application/octet-stream", "X-Request-ID": "bench"},
        lambda i: image_bytes,
        args.n,
    )

    lines = [
        "# Envelope encoding benchmark",
        "",
        "- url: `%s`" % args.url,
        "- image: `%s` (%d bytes raw, %d bytes base64)" % (
            args.image,
            len(image_bytes),
            len(b64.encode()),
        ),
        "- requests per encoding: %d" % args.n,
        "",
        "| encoding | payload | p50 (ms) | p99 (ms) | mean (ms) | rps | errors |",
        "|---|---|---|---|---|---|---|",
        "| json+base64 | %d B | %.1f | %.1f | %.1f | %.1f | %d |" % (
            json_result["payload_bytes"], json_result["p50"], json_result["p99"],
            json_result["mean"], json_result["throughput"], json_result["errors"],
        ),
        "| raw body | %d B | %.1f | %.1f | %.1f | %.1f | %d |" % (
            raw_result["payload_bytes"], raw_result["p50"], raw_result["p99"],
            raw_result["mean"], raw_result["throughput"], raw_result["errors"],
        ),
        "",
        "payload saved: %.1f%%" % (
            100.0 * (1 - raw_result["payload_bytes"] / max(1, json_result["payload_bytes"]))
        ),
    ]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0 if json_result["errors"] == 0 and raw_result["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

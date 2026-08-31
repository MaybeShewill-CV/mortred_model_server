# Envelope encoding benchmark

- url: `http://localhost:9056/mortred_ai_server_v1/obj_detection/yolov8`
- image: `/mnt/g/Codex/mortred_model_server/demo_data/model_test_input/object_detection/bus.jpg` (487438 bytes raw, 649920 bytes base64)
- requests per encoding: 100

| encoding | payload | p50 (ms) | p99 (ms) | mean (ms) | rps | errors |
|---|---|---|---|---|---|---|
| json+base64 | 649958 B | 26.4 | 37.0 | 30.0 | 33.3 | 0 |
| raw body | 487438 B | 23.7 | 34.1 | 24.4 | 41.1 | 0 |

payload saved: 25.0%

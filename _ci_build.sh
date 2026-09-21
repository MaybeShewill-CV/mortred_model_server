cd /mnt/g/Codex/mortred_model_server
set -a && . conf/local/trust.env && set +a
pkill -f mortred-supervisor.out 2>/dev/null; pkill -f mortred-model-server.out 2>/dev/null; sleep 2
cd _bin
export MORTRED_PACK=/mnt/g/Codex/mortred_model_server/conf/local/local_server.toml
nohup ./mortred-supervisor.out > /tmp/mortred_supervisor_console.log 2>&1 &
disown
sleep 4
./mortredctl.out start YOLOV8 | tail -1
sleep 12
cd ..
# idle reference detection
curl -s -X POST -H "Authorization: Bearer $MORTRED_INTERNAL_TOKEN" -H "Content-Type: image/jpeg" \
  --data-binary @demo_data/model_test_input/object_detection/bus.jpg \
  http://127.0.0.1:9056/mortred_ai_server_v1/obj_detection/yolov8 > /tmp/idle_result.json
# load + mid-load detection probes
(python3 scripts/server/http_infer_rps.py --url http://127.0.0.1:9056/mortred_ai_server_v1/obj_detection/yolov8 --image demo_data/model_test_input/object_detection/bus.jpg -c 8 -d 12s --raw --warmup 2s --token "$MORTRED_INTERNAL_TOKEN" --quiet >/dev/null 2>&1) &
LOAD=$!
sleep 4
curl -s -X POST -H "Authorization: Bearer $MORTRED_INTERNAL_TOKEN" -H "Content-Type: image/jpeg" \
  --data-binary @demo_data/model_test_input/object_detection/bus.jpg \
  http://127.0.0.1:9056/mortred_ai_server_v1/obj_detection/yolov8 > /tmp/load_result.json
sleep 2
curl -s -X POST -H "Authorization: Bearer $MORTRED_INTERNAL_TOKEN" -H "Content-Type: image/jpeg" \
  --data-binary @demo_data/model_test_input/object_detection/bus.jpg \
  http://127.0.0.1:9056/mortred_ai_server_v1/obj_detection/yolov8 > /tmp/load_result2.json
wait $LOAD
python3 - <<'PYEOF'
import json
def summarize(p):
    d = json.load(open(p))
    r = d.get('results', [])
    boxes = r[0] if r else []
    return len(boxes), sorted(round(b.get('bbox', b.get('score',0)),2) if isinstance(b, dict) else str(b) for b in boxes)[:3]
# just compare full structure equality of detection payloads
a = json.load(open('/tmp/idle_result.json'))
b = json.load(open('/tmp/load_result.json'))
c = json.load(open('/tmp/load_result2.json'))
def norm(x):
    return json.dumps(x.get('results'), sort_keys=True)
print("idle detections:", len(a.get('results',[])), "load:", len(b.get('results',[])), "load2:", len(c.get('results',[])))
print("idle == load1:", norm(a)==norm(b), " idle == load2:", norm(a)==norm(c))
PYEOF

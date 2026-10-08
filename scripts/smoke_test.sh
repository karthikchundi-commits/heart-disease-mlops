#!/usr/bin/env bash
# Smoke-test a running Heart Disease API.   usage: scripts/smoke_test.sh [BASE_URL]
set -euo pipefail
BASE="${1:-http://localhost:8000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
pass=0; fail=0

check() {  # name expected_status method path [json_file|json_string]
  local name=$1 want=$2 method=$3 path=$4 body=${5:-}
  local args=(-s -o /tmp/heart_smoke_body -w "%{http_code}" -X "$method" "$BASE$path")
  if [[ -n "$body" ]]; then
    if [[ -f "$body" ]]; then args+=(-H "Content-Type: application/json" --data @"$body")
    else args+=(-H "Content-Type: application/json" --data "$body"); fi
  fi
  local got; got=$(curl "${args[@]}" || echo 000)
  if [[ "$got" == "$want" ]]; then
    pass=$((pass+1)); printf "  PASS  %-34s %s  %s\n" "$name" "$got" "$(head -c 160 /tmp/heart_smoke_body | tr -d '\n')"
  else
    fail=$((fail+1)); printf "  FAIL  %-34s got %s want %s  %s\n" "$name" "$got" "$want" "$(head -c 300 /tmp/heart_smoke_body | tr -d '\n')"
  fi
}

echo "Smoke testing $BASE"
for _ in $(seq 1 30); do curl -sf "$BASE/health" >/dev/null && break; sleep 2; done

check "GET /health"                 200 GET  /health
check "GET /ready"                  200 GET  /ready
check "GET /model-info"             200 GET  /model-info
check "POST /predict high-risk"     200 POST /predict "$ROOT/samples/patient_high_risk.json"
check "POST /predict low-risk"      200 POST /predict "$ROOT/samples/patient_low_risk.json"
check "POST /predict partial input" 200 POST /predict "$ROOT/samples/patient_partial.json"
check "POST /predict/batch"         200 POST /predict/batch "$ROOT/samples/batch.json"
check "POST /predict invalid -> 422" 422 POST /predict '{"age": 58, "cp": 9}'
check "GET /metrics"                200 GET  /metrics

echo "passed=$pass failed=$fail"
[[ $fail -eq 0 ]]

#!/usr/bin/env bash
# =============================================================================
#  Heart Disease MLOps - end-to-end lab runner (Linux / macOS / WSL / Git-Bash)
#
#  bash scripts/lab_run.sh all          # every stage below, in order
#  bash scripts/lab_run.sh <stage> ...  # individual stages
#
#  stages:
#    setup    create .venv and install requirements.txt (clean, reproducible env)
#    data     download UCI files + clean          -> data/processed/heart_clean.csv
#    eda      EDA figures                          -> reports/figures/eda/
#    train    tune 3 models, log to MLflow         -> mlflow.db, mlruns/, models/
#    test     ruff lint + pytest with coverage
#    nb       execute the notebooks in notebooks/
#    mlflow   start the MLflow UI on :5000 (background)
#    docker   build image, run container, smoke test it
#    compose  API + Prometheus + Grafana via docker compose
#    k8s      Minikube: start, load image, deploy, verify, port-forward
#    traffic  send sample traffic so the dashboards have data
#    clean    stop containers / port-forwards (minikube left running)
#
#  Every stage tees its output to logs/<stage>.log for the report screenshots.
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
IMAGE=heart-disease-api:1.0.0
PY="${PYTHON:-python3}"

detect_vpy() {
  if [[ -x .venv/bin/python ]]; then VPY=.venv/bin/python
  elif [[ -x .venv/Scripts/python.exe ]]; then VPY=.venv/Scripts/python.exe
  else VPY=""; fi
}

banner() { printf "\n\033[1;36m==================== %s ====================\033[0m\n" "$*"; }
need()   { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found. $2"; exit 1; }; }

stage_setup() {
  banner "SETUP: virtualenv + requirements"
  need "$PY" "Install Python 3.10-3.12 or set PYTHON=/path/to/python"
  "$PY" --version
  "$PY" -m venv .venv
  detect_vpy
  "$VPY" -m pip install --upgrade pip -q
  "$VPY" -m pip install -r requirements.txt
  "$VPY" -m pip freeze > logs/pip-freeze.txt
}

py() { [[ -n "$VPY" ]] || { echo "run 'setup' first"; exit 1; }; "$VPY" "$@"; }

stage_data()  { banner "DATA: download + clean";  py -m heart.data; head -5 data/processed/heart_clean.csv; }
stage_eda()   { banner "EDA: figures";            py -m heart.eda; ls -1 reports/figures/eda; }
stage_train() { banner "TRAIN: tuning + MLflow";  py -m heart.train; cat models/model_meta.json; }

stage_test() {
  banner "TEST: lint + unit tests"
  py -m ruff check .
  py -m pytest -v --cov=heart --cov-report=term-missing --junitxml=logs/junit.xml
}

stage_nb() {
  banner "NOTEBOOKS: execute"
  for nb in notebooks/*.ipynb; do
    py -m jupyter nbconvert --to notebook --execute --inplace "$nb"
    echo "executed $nb"
  done
}

stage_mlflow() {
  banner "MLFLOW UI -> http://localhost:5000"
  nohup "$VPY" -m mlflow ui --backend-store-uri "sqlite:///$ROOT/mlflow.db" \
      --host 0.0.0.0 --port 5000 > logs/mlflow-ui.out 2>&1 &
  echo "MLflow UI pid $! - open http://localhost:5000 (experiment: heart-disease-classification)"
}

stage_docker() {
  banner "DOCKER: build + run + smoke test"
  need docker "Install Docker Engine / Docker Desktop"
  [[ -f models/heart_model.joblib ]] || { echo "no model - run 'train' first"; exit 1; }
  docker build -t "$IMAGE" .
  docker image ls "$IMAGE"
  docker rm -f heart-api-test >/dev/null 2>&1 || true
  docker run -d --name heart-api-test -p 8000:8000 "$IMAGE"
  bash scripts/smoke_test.sh http://localhost:8000
  echo "--- container logs (structured JSON request logs) ---"
  docker logs --tail 15 heart-api-test
  docker inspect --format 'health={{.State.Health.Status}}' heart-api-test || true
  docker rm -f heart-api-test
}

stage_compose() {
  banner "COMPOSE: API + Prometheus + Grafana"
  need docker "Install Docker"
  docker compose up -d --build
  docker compose ps
  bash scripts/smoke_test.sh http://localhost:8000
  echo "API docs   : http://localhost:8000/docs"
  echo "Prometheus : http://localhost:9090/targets"
  echo "Grafana    : http://localhost:3000  (admin/admin) -> Dashboards -> Heart MLOps"
}

install_k8s_tools() {
  local bin="$HOME/.local/bin"; mkdir -p "$bin"; export PATH="$bin:$PATH"
  local arch; arch=$(uname -m); [[ "$arch" == "x86_64" ]] && arch=amd64; [[ "$arch" == "aarch64" ]] && arch=arm64
  if ! command -v kubectl >/dev/null; then
    echo "installing kubectl -> $bin"
    curl -sSLo "$bin/kubectl" "https://dl.k8s.io/release/$(curl -sSL https://dl.k8s.io/release/stable.txt)/bin/linux/$arch/kubectl"
    chmod +x "$bin/kubectl"
  fi
  if ! command -v minikube >/dev/null; then
    echo "installing minikube -> $bin"
    curl -sSLo "$bin/minikube" "https://storage.googleapis.com/minikube/releases/latest/minikube-linux-$arch"
    chmod +x "$bin/minikube"
  fi
}

stage_k8s() {
  banner "KUBERNETES: Minikube deployment"
  need docker "Minikube uses the docker driver"
  install_k8s_tools
  minikube status >/dev/null 2>&1 || minikube start --driver=docker --cpus=2 --memory=4096
  minikube addons enable ingress >/dev/null || true
  minikube addons enable metrics-server >/dev/null || true
  docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" .
  echo "loading $IMAGE into minikube"
  minikube image load "$IMAGE"
  bash scripts/deploy_k8s.sh

  pkill -f "heart-mlops port-forward" 2>/dev/null || true
  nohup kubectl -n heart-mlops port-forward svc/heart-api 8080:80 > logs/pf-api.out 2>&1 &
  nohup kubectl -n heart-mlops port-forward svc/prometheus 9090:9090 > logs/pf-prom.out 2>&1 &
  nohup kubectl -n heart-mlops port-forward svc/grafana 3000:3000 > logs/pf-grafana.out 2>&1 &
  sleep 4
  bash scripts/smoke_test.sh http://localhost:8080

  echo
  echo "Service URL (NodePort/LoadBalancer via minikube):"
  minikube service heart-api -n heart-mlops --url || true
  echo "Ingress: add '$(minikube ip) heart.local' to /etc/hosts, then: curl http://heart.local/health"
  echo "For a LoadBalancer EXTERNAL-IP run 'minikube tunnel' in another terminal, then: kubectl -n heart-mlops get svc heart-api"
  echo "API (port-forward) http://localhost:8080/docs | Prometheus http://localhost:9090 | Grafana http://localhost:3000"
  echo "--- pod logs ---"
  kubectl -n heart-mlops logs deploy/heart-api --tail=10
}

stage_traffic() {
  banner "TRAFFIC: sample requests for monitoring"
  local base="${BASE_URL:-http://localhost:8080}"
  curl -sf "$base/health" >/dev/null || base=http://localhost:8000
  py scripts/generate_traffic.py --base-url "$base" --requests "${N:-300}"
}

stage_clean() {
  banner "CLEAN"
  pkill -f "heart-mlops port-forward" 2>/dev/null || true
  pkill -f "mlflow ui" 2>/dev/null || true
  docker compose down 2>/dev/null || true
  docker rm -f heart-api-test 2>/dev/null || true
  echo "k8s resources kept - remove with: bash scripts/deploy_k8s.sh --delete ; minikube stop"
}

run_stage() {
  local s=$1
  declare -F "stage_$s" >/dev/null || { echo "unknown stage: $s"; exit 2; }
  detect_vpy
  "stage_$s" 2>&1 | tee "logs/$s.log"
  return "${PIPESTATUS[0]}"
}

[[ $# -gt 0 ]] || { sed -n '2,24p' "$0"; exit 0; }
if [[ "$1" == "all" ]]; then
  set -- setup data eda train test nb mlflow docker k8s traffic
fi
for s in "$@"; do run_stage "$s"; done
banner "DONE: $*"

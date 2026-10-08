#!/usr/bin/env bash
# Deploy the API + Prometheus + Grafana to the current kubectl context.
# Works on Minikube, Docker Desktop Kubernetes, kind, or a cloud cluster.
#
#   scripts/deploy_k8s.sh            # deploy (expects image heart-disease-api:1.0.0 available to the cluster)
#   scripts/deploy_k8s.sh --delete   # tear everything down
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NS=heart-mlops

if [[ "${1:-}" == "--delete" ]]; then
  kubectl delete -f "$ROOT/k8s/monitoring/" --ignore-not-found
  kubectl delete -k "$ROOT/k8s" --ignore-not-found
  exit 0
fi

echo "== context: $(kubectl config current-context)"
kubectl apply -k "$ROOT/k8s"

kubectl -n "$NS" create configmap grafana-dashboards \
  --from-file="$ROOT/monitoring/grafana/dashboards/heart_api.json" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "$ROOT/k8s/monitoring/"

echo "== waiting for rollouts"
kubectl -n "$NS" rollout status deployment/heart-api --timeout=180s
kubectl -n "$NS" rollout status deployment/prometheus --timeout=180s
kubectl -n "$NS" rollout status deployment/grafana --timeout=180s

kubectl -n "$NS" get deploy,pods,svc,ingress,hpa -o wide

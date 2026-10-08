# Lab run checklist

Everything that can run without Docker/Kubernetes has already been run and is committed
(data, EDA, models, MLflow screenshots, tests, notebooks). In the lab you only need to
produce the container / cluster / CI evidence and drop the PNGs into `screenshots/`.

## 0. Get the code onto the lab machine

```bash
git clone <repo-url> heart-disease-mlops && cd heart-disease-mlops
# (or copy the folder and run: find scripts -name '*.sh' -exec sed -i 's/\r$//' {} +)
```

## 1. Full run (about 10-15 min; most of it is image pulls)

```bash
bash scripts/lab_run.sh all
```

If something fails, rerun only that stage, e.g. `bash scripts/lab_run.sh docker`.
Every stage's output is saved in `logs/<stage>.log`.

## 2. Screenshots to capture (exact file names, the report picks them up automatically)

| File | What to show | How |
|---|---|---|
| `screenshots/40_docker_build_run.png` | end of `docker build`, `docker image ls`, smoke test `passed=9 failed=0`, container JSON logs | terminal after `bash scripts/lab_run.sh docker` |
| `screenshots/41_k8s_resources.png` | pods `Running 1/1`, services (LoadBalancer), ingress, HPA | `kubectl -n heart-mlops get all,ingress,hpa -o wide` |
| `screenshots/42_k8s_endpoint_verification.png` | smoke test through the Service plus `minikube service ... --url` / `curl http://heart.local/health` | terminal after `bash scripts/lab_run.sh k8s` |
| `screenshots/43_prometheus_targets.png` | both API pods `UP` | browser `http://localhost:9090/targets` |
| `screenshots/44_grafana_dashboard.png` | *Heart MLOps -> Heart Disease API* dashboard with data | run `bash scripts/lab_run.sh traffic` first, then `http://localhost:3000` (admin/admin) |
| `screenshots/50_github_actions_run.png` | workflow run, all 5 jobs green | GitHub -> Actions -> ci-cd run |
| `screenshots/51_github_actions_artifacts.png` | run summary (test table, model comparison) + Artifacts list | bottom of the same run page |

Optional extras: `minikube tunnel` + `kubectl -n heart-mlops get svc heart-api` showing an EXTERNAL-IP,
and `kubectl -n heart-mlops logs -l app=heart-api --tail=20`.

## 3. Rebuild the report with the new screenshots

```bash
npm install docx image-size@1       # once
STUDENT_NAME="C V R Karthik" STUDENT_ID="2025AE05851" \
REPO_URL="https://github.com/<you>/heart-disease-mlops" VIDEO_URL="<drive link>" \
node scripts/build_report.js
```

Open `reports/MLOps_Assignment1_Report.docx` in Word, right-click the table of contents and choose
**Update field**, then *File -> Save As -> PDF* if you need a PDF.

## 4. Video (3-5 min) - suggested flow

1. Repo tour: `src/heart`, `tests/`, `Dockerfile`, `k8s/`, `.github/workflows/ci.yml`.
2. `bash scripts/lab_run.sh train` -> MLflow UI (runs, nested candidates, registry `@champion`).
3. `bash scripts/lab_run.sh test` -> green tests.
4. `bash scripts/lab_run.sh docker` -> build + smoke test.
5. `bash scripts/lab_run.sh k8s` -> `kubectl get all`, curl `/predict` through the Service.
6. `bash scripts/lab_run.sh traffic` -> Prometheus targets + Grafana dashboard updating.
7. GitHub Actions run page with the five green jobs and artifacts.

## 5. Teardown

```bash
bash scripts/lab_run.sh clean
bash scripts/deploy_k8s.sh --delete
minikube stop
```

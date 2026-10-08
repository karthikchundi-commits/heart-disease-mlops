/*
 * Build the assignment report (reports/MLOps_Assignment1_Report.docx).
 *
 *   npm install docx image-size@1      (once, anywhere on NODE_PATH)
 *   node scripts/build_report.js
 *
 * Screenshots are pulled from screenshots/. Lab-only screenshots (Docker, Kubernetes,
 * Grafana, GitHub Actions) render as a labelled placeholder until the PNG exists -
 * capture it in the lab with the listed file name and re-run this script.
 */
const fs = require("fs");
const path = require("path");
const sizeOf = require("image-size");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, LevelFormat, PageBreak, Footer, Header, PageNumber,
  TableOfContents, ExternalHyperlink,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const SHOTS = path.join(ROOT, "screenshots");
const FIG = path.join(ROOT, "reports", "figures");
const OUT = path.join(ROOT, "reports", "MLOps_Assignment1_Report.docx");

const STUDENT = process.env.STUDENT_NAME || "C V R Karthik";
const STUDENT_ID = process.env.STUDENT_ID || "2025AE05851";
const REPO_URL = process.env.REPO_URL || "https://github.com/<your-account>/heart-disease-mlops";
const VIDEO_URL = process.env.VIDEO_URL || "<link to pipeline video recording>";

const CONTENT_W = 9638; // A4 11906 - 2 x 1134 (2 cm) margins
const MAX_PX = 560;
const FONT = "Calibri";
const ACCENT = "1F4E79";

// ------------------------------------------------------------------ helpers
function runs(text, base = {}) {
  const out = [];
  for (const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`)/)) {
    if (!part) continue;
    if (part.startsWith("**")) out.push(new TextRun({ ...base, text: part.slice(2, -2), bold: true }));
    else if (part.startsWith("`")) out.push(new TextRun({ ...base, text: part.slice(1, -1), font: "Consolas", size: 19 }));
    else out.push(new TextRun({ ...base, text: part }));
  }
  return out;
}
const p = (t, o = {}) => new Paragraph({ spacing: { after: 110 }, ...o, children: runs(t) });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: false, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 50 }, children: runs(t) });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });
const code = (lines) => lines.map((l, i) => new Paragraph({
  shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
  spacing: { after: i === lines.length - 1 ? 140 : 0 },
  indent: { left: 120 },
  children: [new TextRun({ text: l || " ", font: "Consolas", size: 17 })],
}));

let figNo = 0;
function figure(file, caption, maxPx = MAX_PX) {
  figNo += 1;
  const cap = new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 180 },
    children: [new TextRun({ text: `Figure ${figNo}. `, bold: true, size: 18, color: "404040" }),
      new TextRun({ text: caption, italics: true, size: 18, color: "404040" })],
  });
  if (!fs.existsSync(file)) {
    const name = path.relative(ROOT, file).replace(/\\/g, "/");
    return [new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 },
      shading: { type: ShadingType.CLEAR, fill: "FFF4E5", color: "auto" },
      border: { top: { style: BorderStyle.DASHED, size: 6, color: "E67E22" },
        bottom: { style: BorderStyle.DASHED, size: 6, color: "E67E22" },
        left: { style: BorderStyle.DASHED, size: 6, color: "E67E22" },
        right: { style: BorderStyle.DASHED, size: 6, color: "E67E22" } },
      children: [new TextRun({ text: `LAB SCREENSHOT - save as ${name} and re-run node scripts/build_report.js`,
        bold: true, color: "B9770E", size: 18 })],
    }), cap];
  }
  const buf = fs.readFileSync(file);
  const { width, height } = sizeOf(buf);
  const w = Math.min(width, maxPx);
  const h = Math.round(height * (w / width));
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: w, height: h } })] }), cap];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
function table(header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders: { top: border, bottom: border, left: border, right: border },
    shading: head ? { type: ShadingType.CLEAR, fill: ACCENT, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ children: runs(String(t), head ? { color: "FFFFFF", bold: true, size: 18 } : { size: 18 }) })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((t, i) => cell(t, i, false)) }))],
  });
}
const gap = () => new Paragraph({ spacing: { after: 100 }, children: [] });

// ------------------------------------------------------------------ data
const meta = JSON.parse(fs.readFileSync(path.join(ROOT, "models", "model_meta.json"), "utf8"));
const cmp = fs.readFileSync(path.join(ROOT, "reports", "model_comparison.csv"), "utf8").trim().split(/\r?\n/);
const cols = cmp[0].split(",");
const models = cmp.slice(1).map((l) => Object.fromEntries(l.split(",").map((v, i) => [cols[i], v])));
const f3 = (x) => Number(x).toFixed(3);
const pretty = (n) => n.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
const mm = meta.metrics;

// ------------------------------------------------------------------ content
const children = [];
const add = (...xs) => xs.flat().forEach((x) => children.push(x));

// Cover
add(
  new Paragraph({ spacing: { before: 1800, after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Machine Learning Operations (AIMLCZG523)", size: 28, color: "595959" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
    children: [new TextRun({ text: "Assignment 01", size: 28, color: "595959" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "Heart Disease Risk Prediction", bold: true, size: 52, color: ACCENT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1200 },
    children: [new TextRun({ text: "End-to-end ML model development, CI/CD and production deployment", size: 28 })] }),
  table(["", ""], [
    ["Student", STUDENT],
    ["BITS ID", STUDENT_ID],
    ["Code repository", REPO_URL],
    ["Pipeline video", VIDEO_URL],
    ["Champion model", `${pretty(meta.model_name)} - CV ROC-AUC ${f3(mm.cv_roc_auc)}, hold-out ROC-AUC ${f3(mm.test_roc_auc)}`],
    ["Stack", "scikit-learn, MLflow, FastAPI, Docker, Kubernetes (Minikube), GitHub Actions, Prometheus, Grafana"],
  ], [2600, 7038]),
  pageBreak(),
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Contents")] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  pageBreak(),
);

// 1 Introduction
add(
  h1("1. Introduction and architecture"),
  p("The goal is a **reproducible, monitored, cloud-ready API** that estimates the risk of heart disease from routine clinical measurements. The work follows the full MLOps lifecycle: data acquisition and EDA, feature engineering and model tuning, experiment tracking, packaging, automated testing and CI/CD, containerisation, Kubernetes deployment and runtime monitoring."),
  p("Figure 1 shows the system. Everything is driven from one Python package (`src/heart`), so the same code runs in notebooks, in the CLI, in CI, and inside the serving container."),
  figure(path.join(ROOT, "docs", "architecture.png"), "End-to-end architecture: training lane, CI/CD lane and serving/monitoring lane."),
  h2("Key design decisions"),
  bullet("**All four UCI sites (920 patients)** are used rather than the usual 303-row Cleveland subset. This gives three times more data and makes the pipeline handle real missing values."),
  bullet("**Preprocessing is part of the persisted model** (one sklearn `Pipeline`), so training and serving can never diverge."),
  bullet("**Model selection uses cross-validation only**; the 20 % hold-out set is used just for reporting."),
  bullet("**Missingness is modelled explicitly** (indicator flags plus a 'missing' category level), and the API accepts patients with unmeasured fields."),
  bullet("**Observability is built in**: structured JSON request logs, Prometheus metrics (including a prediction-mix drift signal), a provisioned Grafana dashboard and alert rules."),
);

// 2 Setup
add(
  h1("2. Setup and installation"),
  p("Prerequisites: Python 3.10-3.12, Docker, Git and internet access. `kubectl` and `minikube` are installed automatically into `~/.local/bin` if missing. The whole pipeline runs with one command:"),
  ...code([
    `git clone ${REPO_URL}`,
    "cd heart-disease-mlops",
    "bash scripts/lab_run.sh all        # setup data eda train test nb mlflow docker k8s traffic",
  ]),
  p("Each stage can also be run on its own, and each writes its console output to `logs/<stage>.log`:"),
  table(["Stage", "Command / effect"], [
    ["setup", "`python -m venv .venv` and `pip install -r requirements.txt` (pinned versions)"],
    ["data / eda / train", "`python -m heart.data`, `python -m heart.eda`, `python -m heart.train`"],
    ["test", "`ruff check .` and `pytest --cov=heart`"],
    ["mlflow", "`mlflow ui --backend-store-uri sqlite:///mlflow.db` on port 5000"],
    ["docker", "build `heart-disease-api:1.0.0`, run it, 9-check smoke test, show container logs"],
    ["k8s", "minikube start, `minikube image load`, `scripts/deploy_k8s.sh`, smoke test through the Service"],
    ["traffic / compose", "sample load for dashboards / Docker-Compose alternative (API, Prometheus, Grafana)"],
  ], [2300, 7338]),
  gap(),
  p("Windows without Docker: create the venv, `pip install -r requirements.txt`, run the three `python -m heart.*` modules, then `uvicorn heart.serving:app`."),
);

// 3 Data & EDA
add(
  h1("3. Data acquisition and exploratory analysis"),
  h2("3.1 Acquisition and cleaning"),
  p("`heart.data.download()` fetches `processed.{cleveland,hungarian,switzerland,va}.data` from the UCI repository and caches them in `data/raw/`. `heart.data.clean()` then applies the following rules (unit-tested in `tests/test_data.py`):"),
  table(["Issue found in raw data", "Treatment"], [
    ["`?` used as missing marker", "parsed as NaN"],
    ["Diagnosis `num` is a 0-4 severity", "binary target = `num > 0` (presence of disease)"],
    ["`chol = 0` in 172 rows (all 123 Swiss rows + 49 VA), `trestbps = 0` once", "physiologically impossible, so set to NaN"],
    ["Codes outside the data dictionary", "set to NaN (e.g. thal must be 3, 6 or 7)"],
    ["2 exact duplicate rows", "dropped, leaving 918 rows (55.3 % positive)"],
  ], [5000, 4638]),
  gap(),
  p("Imputation is intentionally **not** done during cleaning. It is part of the model pipeline (Section 4), so identical logic is applied to API requests."),
  h2("3.2 Findings"),
  ...figure(path.join(FIG, "eda", "01_class_balance.png"), "Class balance overall and per collection site."),
  ...figure(path.join(FIG, "eda", "02_missingness_by_site.png"), "Missing values (%) per feature and site - missingness is site-driven.", 340),
  ...figure(path.join(FIG, "eda", "03_numeric_histograms.png"), "Distributions of numeric features by diagnosis."),
  ...figure(path.join(FIG, "eda", "04_categorical_disease_rate.png"), "Disease rate per category level (n shown above each bar)."),
  ...figure(path.join(FIG, "eda", "05_correlation_heatmap.png"), "Spearman correlation heat-map (pairwise complete observations).", 380),
  table(["Observation", "Consequence for modelling"], [
    ["Balanced overall (55/45) but prevalence ranges from 36 % (Hungary) to 93 % (Switzerland)", "stratified splits; `site` excluded from features because it would leak centre prevalence"],
    ["`ca` 66 %, `thal` 53 %, `slope` 33 % missing, mostly outside Cleveland", "median/constant imputation plus missing-indicator features"],
    ["Asymptomatic chest pain, exercise angina, ST depression, flat ST slope, `ca` > 0 and reversible thal defect are the strongest risk factors; max heart rate is protective", "relationships are largely monotone, so linear models are competitive"],
    ["`oldpeak` is skewed; `chol` has outliers", "standard scaling; tree models are robust to the rest"],
  ], [4900, 4738]),
);

// 4 Features & models
add(
  h1("4. Feature engineering and model development"),
  h2("4.1 Feature pipeline"),
  p("`heart.features.build_preprocessor()` returns one `ColumnTransformer` that turns the 13 inputs into 31 model features:"),
  table(["Group", "Columns", "Steps"], [
    ["Numeric", "age, trestbps, chol, thalach, oldpeak, ca", "median impute + missing indicator, then StandardScaler"],
    ["Binary", "sex, fbs, exang", "most-frequent impute"],
    ["Categorical", "cp, restecg, slope, thal", "missing becomes level -1, then one-hot with a fixed category list (unknown codes map to all-zeros)"],
  ], [1500, 3400, 4738]),
  gap(),
  p("An ablation in notebook `02_model_training` (logistic regression, 5-fold CV) confirms the design: ROC-AUC **0.880** without indicators, **0.896** with indicators, and **0.899** with indicators plus a 'missing' category level."),
  h2("4.2 Models and tuning"),
  p("Three model families are tuned with `GridSearchCV` (5-fold stratified CV, `refit='roc_auc'`; accuracy, precision, recall and F1 are also scored). ROC-AUC is the selection metric because it does not depend on the decision threshold and is robust to the prevalence shift between sites. A majority-class baseline scores 0.500 ROC-AUC and 0.553 accuracy."),
  table(["Model", "Grid searched", "Candidates"], [
    ["Logistic regression (liblinear)", "C {0.01, 0.1, 0.5, 1, 5} x penalty {l1, l2} x class_weight {none, balanced}", "20"],
    ["Random forest", "n_estimators {200, 400} x max_depth {none, 5, 8} x min_samples_leaf {1, 4} x max_features {sqrt, 0.5}", "24"],
    ["Hist gradient boosting", "learning_rate {0.03, 0.1} x max_depth {3, none} x max_iter {100, 250} x l2 {0, 1}", "16"],
  ], [2500, 6138, 1000]),
  h2("4.3 Results"),
  table(["Model", "CV ROC-AUC", "CV acc.", "CV prec.", "CV recall", "Test ROC-AUC", "Test acc.", "Test recall"],
    models.map((m) => [pretty(m.name) + (m.name === meta.model_name ? " *" : ""), `${f3(m.cv_roc_auc)} ± ${f3(m.cv_roc_auc_std)}`,
      f3(m.cv_accuracy), f3(m.cv_precision), f3(m.cv_recall), f3(m.test_roc_auc), f3(m.test_accuracy), f3(m.test_recall)]),
    [2140, 1400, 950, 950, 950, 1150, 1000, 1048]),
  p("* champion", { spacing: { after: 140 } }),
  ...figure(path.join(FIG, "model_comparison_roc.png"), "Hold-out ROC curves of the three tuned models.", 330),
  ...figure(path.join(FIG, "models", "logistic_regression_diagnostics.png"), "Champion diagnostics on the hold-out set: confusion matrix, ROC and precision-recall."),
  h2("4.4 Model selection"),
  bullet(`The three families are within about 0.01 CV ROC-AUC of each other, which is less than one CV standard deviation (about 0.05).`),
  bullet(`**${pretty(meta.model_name)}** (penalty=${meta.best_params.penalty}, C=${meta.best_params.C}) has the best mean CV ROC-AUC (${f3(mm.cv_roc_auc)}). It is also the simplest and fastest model and gives interpretable coefficients, which matters in a clinical setting, so it was chosen as champion. The L1 penalty zeroes 8 of the 31 features, which acts as built-in feature selection.`),
  bullet(`On the untouched hold-out set it reaches ROC-AUC ${f3(mm.test_roc_auc)}, accuracy ${f3(mm.test_accuracy)}, precision ${f3(mm.test_precision)} and recall ${f3(mm.test_recall)}. Recall is the more important clinical metric, and the API returns the raw probability so the decision threshold can be adjusted.`),
  bullet("The champion is then refit on all 918 labelled rows and exported for serving."),
  ...figure(path.join(FIG, "models", "logistic_regression_features.png"), "Largest logistic-regression coefficients (red increases risk).", 340),
);

// 5 MLflow
add(
  h1("5. Experiment tracking (MLflow)"),
  p("`heart.train` logs every run to MLflow (SQLite backend `mlflow.db`, artifacts in `mlruns/`) under the experiment `heart-disease-classification`, **64 runs** in total:"),
  bullet("**3 parent runs** (one per model family): best parameters, CV mean and standard deviation for 5 metrics, hold-out metrics, fit time, diagnostics and feature-importance plots, the full `cv_results_` CSV, and the sklearn model with an input signature and example."),
  bullet("**60 nested child runs**, one per grid candidate, with its parameters, CV metrics and rank. These give the full tuning history (notebook 02 plots them)."),
  bullet("**1 champion-final run**: comparison plot and table plus the refit model, registered as `heart-disease-classifier` version 1 with alias **`@champion`**."),
  ...figure(path.join(SHOTS, "10_mlflow_experiment_runs.png"), "MLflow experiment: parent runs (expandable to nested candidates) and the registered champion."),
  ...figure(path.join(SHOTS, "11_mlflow_run_lr_overview.png"), "Logistic-regression parent run: parameters and metrics.", 480),
  ...figure(path.join(SHOTS, "12_mlflow_run_lr_artifacts.png"), "Plots logged as run artifacts.", 480),
  ...figure(path.join(SHOTS, "14_mlflow_model_registry.png"), "Model registry entry with the @champion alias."),
);

// 6 Packaging
add(
  h1("6. Model packaging and reproducibility"),
  table(["Requirement", "Implementation"], [
    ["Reusable model format", "`models/heart_model.joblib` (complete sklearn Pipeline) plus MLflow `sklearn` flavour in the registry; `models/model_meta.json` stores the run id, parameters, metrics and timestamp"],
    ["Environment", "pinned `requirements.txt` (training, tests, notebooks) and slim `requirements-serve.txt` (container); installable package defined in `pyproject.toml`"],
    ["Preprocessing", "the `ColumnTransformer` is inside the persisted Pipeline, so a raw JSON request is all that is needed for inference"],
    ["Determinism", "`random_state=42` for splits, CV folds and models; data download is versioned by URL"],
    ["Inference paths", "`python -m heart.predict samples/patient_high_risk.json`, `mlflow.sklearn.load_model('models:/heart-disease-classifier@champion')`, REST API (notebook 03 shows all three agree)"],
  ], [2400, 7238]),
);

// 7 CI/CD & tests
add(
  h1("7. CI/CD pipeline and automated testing"),
  h2("7.1 Unit tests"),
  p("`tests/` contains **34 pytest tests (91 % line coverage)**. They use a synthetic UCI-shaped dataset with injected faults, so they run offline in about a minute:"),
  table(["File", "What is verified"], [
    ["test_data.py", "target binarisation, impossible zeros and invalid codes become NaN, duplicates removed, schema, no input mutation, '?' parsing"],
    ["test_features.py", "no NaNs after transform, standardisation, missing indicators, fixed one-hot width, all-missing row"],
    ["test_model.py", "learns signal, valid probabilities, metric helper, every model family fits, joblib round-trip, quality bar on the saved artefact"],
    ["test_api.py", "health/ready, /predict contract (prediction + confidence + request id), optional fields, 422 validation, batch, /metrics, 503 without a model"],
    ["test_training_pipeline.py", "full `heart.train` run in a sandbox (MLflow runs + registry alias), EDA writes all figures"],
  ], [2600, 7038]),
  ...figure(path.join(SHOTS, "30_pytest_coverage.png"), "pytest run with coverage report.", 470),
  h2("7.2 GitHub Actions workflow"),
  p("`.github/workflows/ci.yml` runs on every push and pull request. The jobs are chained with `needs`, so a lint error, failing test, failed quality gate, broken image or failed deployment stops the pipeline with a clear log:"),
  table(["Job", "Steps", "Artifacts / logging"], [
    ["lint", "`ruff check` (GitHub annotations)", "inline annotations"],
    ["test", "`pytest` with coverage and JUnit output", "`test-reports` (junit.xml, coverage.xml), job summary table"],
    ["train", "download, EDA, train with MLflow, **quality gate CV ROC-AUC >= 0.80**", "`model`, `mlflow-tracking`, `reports`; model-comparison table in the summary"],
    ["docker", "build the image with the freshly trained model, run it, smoke test", "`docker-image` tarball, container logs"],
    ["deploy-kind", "kind cluster, load image, `deploy_k8s.sh`, smoke test through the Service", "`kubectl get all` and pod logs"],
  ], [1500, 4600, 3538]),
  ...figure(path.join(SHOTS, "50_github_actions_run.png"), "GitHub Actions run - all five jobs green."),
  ...figure(path.join(SHOTS, "51_github_actions_artifacts.png"), "Workflow run summary and uploaded artifacts."),
);

// 8 Container
add(
  h1("8. Model containerisation"),
  p("The `Dockerfile` builds from `python:3.12-slim` with serving-only dependencies (no MLflow or plotting libraries), installs the `heart` package, bakes in the model artefact, runs as a **non-root user** (uid 10001) and declares a `HEALTHCHECK` on `/health`. Uvicorn serves `heart.serving:app` on port 8000."),
  table(["Endpoint", "Purpose"], [
    ["POST /predict", "one patient as JSON; returns `prediction`, `label`, `confidence`, `probability_disease`, `risk_band`, model name/version"],
    ["POST /predict/batch", "up to 500 patients"],
    ["GET /health, /ready", "liveness (process up) and readiness (model loaded), used by Docker and k8s probes"],
    ["GET /model-info", "metadata and training metrics of the served model"],
    ["GET /metrics", "Prometheus exposition"],
  ], [2400, 7238]),
  gap(),
  p("Inputs are validated by Pydantic with clinical ranges and the allowed category codes. Bad input returns **HTTP 422** with field-level messages, and optional measurements may be omitted."),
  ...code([
    "docker build -t heart-disease-api:1.0.0 .",
    "docker run -d -p 8000:8000 --name heart-api heart-disease-api:1.0.0",
    "curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @samples/patient_high_risk.json",
    '{"prediction":1,"label":"heart disease","confidence":0.9624,"probability_disease":0.9624,"risk_band":"high",...}',
  ]),
  ...figure(path.join(SHOTS, "40_docker_build_run.png"), "Docker build, container run and smoke test in the lab (bash scripts/lab_run.sh docker)."),
  ...figure(path.join(SHOTS, "21_api_swagger_predict_executed.png"), "Interactive OpenAPI docs (/docs): /predict executed with the sample patient.", 520),
  ...figure(path.join(SHOTS, "31_api_smoke_test.png"), "scripts/smoke_test.sh: 9 checks including validation (422) and metrics.", 520),
);

// 9 Deployment
add(
  h1("9. Production deployment (Kubernetes)"),
  p("The API is deployed to a local **Minikube** cluster (the same manifests work on Docker Desktop, kind, GKE, EKS or AKS). `scripts/deploy_k8s.sh` applies the kustomization and the monitoring stack, then waits for the rollouts."),
  table(["Manifest", "Highlights"], [
    ["k8s/00-namespace.yaml", "isolated namespace `heart-mlops`"],
    ["k8s/api-deployment.yaml", "2 replicas, RollingUpdate with maxUnavailable 0, startup/readiness/liveness probes, CPU/memory requests and limits, non-root, read-only root filesystem, Prometheus scrape annotations"],
    ["k8s/api-service.yaml", "**type LoadBalancer** port 80 to 8000 (EXTERNAL-IP via `minikube tunnel`)"],
    ["k8s/api-ingress.yaml", "**nginx Ingress** host `heart.local`"],
    ["k8s/api-hpa.yaml", "HorizontalPodAutoscaler 2-5 replicas at 70 % CPU"],
    ["k8s/monitoring/*.yaml", "Prometheus (pod discovery, RBAC), Grafana (provisioned datasource and dashboard), NodePorts 30090/30300"],
  ], [2600, 7038]),
  ...code([
    "minikube start --driver=docker && minikube addons enable ingress && minikube addons enable metrics-server",
    "minikube image load heart-disease-api:1.0.0",
    "bash scripts/deploy_k8s.sh",
    "kubectl -n heart-mlops get all,ingress,hpa",
    "minikube service heart-api -n heart-mlops --url      # or: minikube tunnel / heart.local ingress",
    "bash scripts/smoke_test.sh http://localhost:8080      # via kubectl port-forward svc/heart-api 8080:80",
  ]),
  ...figure(path.join(SHOTS, "41_k8s_resources.png"), "kubectl get all,ingress,hpa in namespace heart-mlops."),
  ...figure(path.join(SHOTS, "42_k8s_endpoint_verification.png"), "Endpoint verification through the Kubernetes Service / Ingress."),
  p("**Access instructions (local):** API `http://localhost:8080/docs` (port-forward), `minikube service heart-api -n heart-mlops --url`, or `http://heart.local` after adding `$(minikube ip) heart.local` to `/etc/hosts`."),
);

// 10 Monitoring
add(
  h1("10. Monitoring and logging"),
  h2("10.1 Logging"),
  p("An HTTP middleware gives every request an `x-request-id` and writes one **JSON log line** with method, path, status, latency in ms and client. Predictions are logged with their probability. The logs go to stdout, so they are available through `docker logs` and `kubectl logs` and can be shipped to Loki or ELK without changes."),
  ...figure(path.join(SHOTS, "32_api_structured_request_logs.png"), "Structured request logs (including a rejected 422 request).", 600),
  h2("10.2 Metrics, dashboard and alerts"),
  table(["Metric", "Type", "Use"], [
    ["heart_api_requests_total{method,path,status}", "counter", "traffic and error rate"],
    ["heart_api_request_latency_seconds{path}", "histogram", "p50/p95/p99 latency"],
    ["heart_api_predictions_total{outcome}", "counter", "prediction mix (drift signal)"],
    ["heart_api_disease_probability", "histogram", "score distribution"],
    ["heart_api_model_loaded", "gauge", "readiness"],
  ], [4600, 1300, 3738]),
  gap(),
  p("Prometheus discovers and scrapes every API pod every 10 s. Grafana is provisioned with the *Heart Disease API - Service & Model Monitoring* dashboard. `monitoring/alert_rules.yml` defines four alerts: API down, 5xx rate > 5 %, p95 latency > 500 ms, and a positive-prediction rate above 85 %, which would suggest input drift since the training prevalence is 55 %. `scripts/generate_traffic.py` replays real patients plus 5 % invalid payloads to populate the panels."),
  ...figure(path.join(SHOTS, "43_prometheus_targets.png"), "Prometheus targets - each API pod discovered and UP."),
  ...figure(path.join(SHOTS, "44_grafana_dashboard.png"), "Grafana dashboard under generated traffic."),
);

// 11 Conclusion
add(
  h1("11. Conclusion and deliverables"),
  p(`The project delivers a heart-disease classifier (CV ROC-AUC ${f3(mm.cv_roc_auc)}, hold-out ROC-AUC ${f3(mm.test_roc_auc)}) behind a validated, observable API. It is reproducible from a clean checkout with one command and automatically linted, tested, trained, quality-gated, containerised and deployed by CI.`),
  table(["Deliverable", "Location"], [
    ["Code, Dockerfile, requirements", "`src/heart/`, `Dockerfile`, `requirements*.txt`, `pyproject.toml`"],
    ["Cleaned dataset + download script", "`data/processed/heart_clean.csv`, `python -m heart.data`"],
    ["Notebooks / scripts", "`notebooks/01_eda`, `02_model_training`, `03_inference`; `scripts/`"],
    ["Unit tests", "`tests/` (34 tests)"],
    ["CI/CD workflow", "`.github/workflows/ci.yml`"],
    ["Deployment manifests", "`k8s/`, `k8s/monitoring/`, `docker-compose.yml`"],
    ["Screenshots", "`screenshots/`"],
    ["Repository", REPO_URL],
    ["Video", VIDEO_URL],
  ], [3400, 6238]),
  gap(),
  h2("Possible extensions"),
  bullet("Probability calibration and a cost-sensitive threshold chosen with clinicians."),
  bullet("Feature-level drift detection (e.g. Evidently) on logged requests, with automated retraining through the existing CI train job."),
  bullet("Push the image to a registry (GHCR) and add a GitOps continuous-delivery step (Argo CD) to a managed cluster."),
);

// ------------------------------------------------------------------ document
const doc = new Document({
  creator: STUDENT,
  title: "Heart Disease Risk Prediction - MLOps Assignment 1",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: ACCENT, font: FONT },
        paragraph: { spacing: { before: 300, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "2E75B6", font: FONT },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "AIMLCZG523 MLOps - Assignment 01 - Heart Disease Risk Prediction", size: 16, color: "808080" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: "808080" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT);
});

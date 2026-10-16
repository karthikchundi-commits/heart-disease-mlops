"""Render docs/architecture.png - the end-to-end system diagram used in the report."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"
C = {
    "data": "#d6eaf8", "ml": "#d5f5e3", "ci": "#fdebd0",
    "run": "#e8daef", "mon": "#fadbd8", "lane": "#f4f6f7",
}

fig, ax = plt.subplots(figsize=(16, 9.2))
ax.set_xlim(0, 160)
ax.set_ylim(0, 92)
ax.axis("off")


def lane(y, h, title):
    ax.add_patch(FancyBboxPatch((1, y), 158, h, boxstyle="round,pad=0.2,rounding_size=1.5",
                                fc=C["lane"], ec="#aab7b8", lw=1))
    ax.text(3, y + h - 2.2, title, fontsize=11, fontweight="bold", color="#34495e", va="top")


def box(x, y, w, h, text, color, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=color, ec="#566573", lw=1.2))
    ax.text(x + w / 2, y + h / 2 + (1.4 if sub else 0), text, ha="center", va="center",
            fontsize=9.5, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.0, sub, ha="center", va="center", fontsize=7.8, color="#424949")


def arrow(x1, y1, x2, y2, label=None, style="-|>", color="#2c3e50", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, lw=1.4,
                                 color=color, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.3, label, fontsize=7.5, ha="center", color=color,
                style="italic")


# lanes
lane(62, 28, "1  Experimentation & training  (python -m heart.data | heart.eda | heart.train)")
lane(33, 27, "2  CI/CD  (GitHub Actions: .github/workflows/ci.yml)")
lane(2, 29, "3  Serving & monitoring  (Docker -> Kubernetes / Minikube)")

# lane 1
box(4, 70, 20, 11, "UCI repository", C["data"], "4 sites, 920 rows")
box(30, 70, 22, 11, "heart.data", C["data"], "download + clean\nheart_clean.csv")
box(58, 75, 20, 9, "heart.eda", C["data"], "6 EDA figures")
box(58, 64, 20, 9, "heart.features", C["ml"], "ColumnTransformer")
box(84, 70, 24, 11, "heart.train", C["ml"], "GridSearchCV x 3 models\n5-fold stratified CV")
box(114, 75, 20, 9, "MLflow tracking", C["ml"], "64 runs, plots, models")
box(114, 64, 20, 9, "MLflow registry", C["ml"], "@champion alias")
box(139, 69, 18, 11, "models/", C["ml"], "heart_model.joblib\nmodel_meta.json")
arrow(24, 75.5, 30, 75.5)
arrow(52, 77, 58, 79)
arrow(52, 74, 58, 68.5)
arrow(78, 68.5, 84, 74)
arrow(108, 77, 114, 79)
arrow(108, 74, 114, 68.5)
arrow(134, 69, 139, 73)

# lane 2
steps = [("push / PR", "trigger"), ("lint", "ruff"), ("test", "pytest + coverage"),
         ("train", "data > EDA > MLflow\nquality gate AUC>=0.80"), ("docker", "build + smoke test"),
         ("deploy-kind", "kubectl apply\nsmoke via Service")]
x = 5
for i, (t, s) in enumerate(steps):
    w = 20 if i in (3, 5) else 17
    box(x, 39, w, 12, t, C["ci"], s)
    if i:
        arrow(x - 4, 45, x, 45)
    x += w + 4
ax.text(144, 45, "artifacts:\njunit/coverage\nmodel, mlruns\nreports, image", fontsize=8, va="center",
        ha="center", bbox={"boxstyle": "round", "fc": "white", "ec": "#aab7b8"})
arrow(148, 69, 120, 51, "model artefact", rad=-0.2, color="#7f8c8d")

# lane 3
box(5, 10, 20, 12, "Client", C["run"], "curl / Swagger /\ngenerate_traffic.py")
box(32, 10, 22, 12, "Ingress / LB", C["run"], "heart.local\nService type=LoadBalancer")
ax.add_patch(FancyBboxPatch((60, 5), 46, 20.5, boxstyle="round,pad=0.3,rounding_size=1.2",
                            fc="white", ec="#7d3c98", lw=1.5, ls="--"))
ax.text(83, 23.2, "Deployment heart-api (2-5 replicas, HPA)", ha="center", fontsize=8.5, color="#7d3c98",
        fontweight="bold")
box(63, 7, 19, 13, "FastAPI pod", C["run"], "/predict /health\n/ready /metrics")
box(85, 7, 19, 13, "FastAPI pod", C["run"], "JSON request logs\nnon-root, probes")
box(114, 15, 18, 10, "Prometheus", C["mon"], "pod discovery")
box(139, 15, 18, 10, "Grafana", C["mon"], "API dashboard")
box(114, 4, 43, 8, "kubectl logs / docker logs  (structured JSON)", C["mon"])
arrow(25, 16, 32, 16, "JSON")
arrow(54, 16, 63, 16)
arrow(114, 20, 104, 17, "scrape", style="<|-")
arrow(132, 20, 139, 20)
arrow(104, 10, 114, 8)
arrow(122, 39, 95, 26, "image", rad=0.15, color="#7f8c8d")

ax.set_title("Heart Disease Risk Prediction - MLOps architecture", fontsize=15, fontweight="bold", pad=8)
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("wrote", OUT)

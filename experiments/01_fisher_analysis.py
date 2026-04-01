"""
Exp 01 — Three-Certificate Comparison on All Existing Hidden States

Runs three separability certificates at every layer on all existing models:
  (A) Euclidean Fisher ratio J(L) — our primary certificate
  (B) Causal Fisher ratio J_causal(L) — Park et al. ICML 2024 correction
  (C) Local Intrinsic Dimension |ΔLID(L)| — Yin et al. ICML 2024 alternative

For each: compares the certificate to the actual OOF probe AUROC.
Produces the certificate-vs-AUROC scatter plot (central diagnostic figure).

Existing data used (zero new compute):
  - Qwen 2.5 3B: harp/results/hidden_states/hidden_states.npz
  - GPT-2 Medium 345M: harp/results/hidden_states/25_gpt2_medium_hidden_states.npz
  - GPT-2 117M: MECH-INT known results (separate extraction pass needed for full curve)

Cost: $0 — CPU only, ~3–4 hours total.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.fisher import fisher_curve
from src.lid import lid_curve

RESULTS_DIR = ROOT / "results" / "logs"
PLOTS_DIR = ROOT / "results" / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

HARP_DIR = Path("/Users/chakrivijayarao/Desktop/harp")
MECH_INT_DIR = Path("/Users/chakrivijayarao/Desktop/MECH-INT")

# LLM-as-Judge labels (from Exp 07) override ROUGE labels if available
JUDGE_LABELS_PATH = RESULTS_DIR / "07_judge_labels.npy"

HS_DIR = ROOT / "results" / "hidden_states"

MODELS = {
    "Qwen 2.5 3B": {
        "hidden_states_path": HS_DIR / "00_halueval_qwen3b.npz",
        "labels_path": None,  # embedded in npz under key 'labels'
        "n_params": 3e9,
        "known_best_auroc": 0.775,
        "source": "HaluEval Job A",
        "use_judge_labels": False,
        "unembedding_model_id": "Qwen/Qwen2.5-3B-Instruct",
    },
    "GPT-2 Medium 345M": {
        "hidden_states_path": HS_DIR / "00_halueval_gpt2med.npz",
        "labels_path": None,  # embedded in npz under key 'labels'
        "n_params": 345e6,
        "known_best_auroc": 0.579,
        "source": "HaluEval Job A",
        "use_judge_labels": False,
        "unembedding_model_id": "openai-community/gpt2-medium",
    },
    "GPT-2 117M": {
        "hidden_states_path": None,     # activations.pkl — handled via known results
        "labels_path": None,
        "n_params": 117e6,
        "known_best_auroc": 0.604,
        "source": "MECH-INT Step 4C",
        "use_judge_labels": False,
        "unembedding_model_id": None,
    },
}


def load_unembedding_matrix(model_id: str) -> np.ndarray | None:
    """
    Load the unembedding (lm_head) weight matrix W_U for causal Fisher computation.
    Returns None if model is unavailable or loading fails.
    """
    try:
        from transformers import AutoModelForCausalLM
        import torch
        print(f"    Loading unembedding matrix from {model_id} ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, device_map="cpu"
        )
        W_U = model.lm_head.weight.detach().float().numpy()  # (vocab_size, d)
        del model
        print(f"    W_U shape: {W_U.shape}")
        return W_U
    except Exception as e:
        print(f"    Could not load unembedding matrix: {e}")
        return None


def load_hidden_states(cfg: dict) -> tuple[np.ndarray, np.ndarray] | None:
    path = cfg["hidden_states_path"]
    if path is None or not path.exists():
        return None
    data = np.load(path)
    key = "hidden_states" if "hidden_states" in data else list(data.keys())[0]
    H = data[key].astype(np.float32)
    if cfg["labels_path"] is not None:
        y = np.load(cfg["labels_path"]).astype(int)
    elif "labels" in data:
        y = data["labels"].astype(int)
    else:
        raise ValueError(f"No labels found in {path}")
    n = min(H.shape[0], y.shape[0])
    return H[:n], y[:n]


def probe_auroc_per_layer(H: np.ndarray, y: np.ndarray, n_splits: int = 5) -> list[float]:
    n_samples, n_layers, d = H.shape
    aurocs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for L in range(n_layers):
        H_L = H[:, L, :]
        oof = np.zeros(n_samples)
        for tr, te in skf.split(H_L, y):
            sc = StandardScaler()
            X_tr = sc.fit_transform(H_L[tr])
            X_te = sc.transform(H_L[te])
            lr = LogisticRegression(max_iter=1000, C=1.0)
            lr.fit(X_tr, y[tr])
            oof[te] = lr.predict_proba(X_te)[:, 1]
        aurocs.append(float(roc_auc_score(y, oof)))
    return aurocs


def run() -> dict:
    all_results = {}

    # Check if LLM-as-Judge labels are available (from Exp 07)
    judge_labels = None
    if JUDGE_LABELS_PATH.exists():
        judge_labels = np.load(JUDGE_LABELS_PATH).astype(int)
        print(f"Using LLM-as-Judge labels from Exp 07 "
              f"(hall_rate={1-judge_labels.mean():.3f})")

    for model_name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        loaded = load_hidden_states(cfg)
        if loaded is None:
            print("  Skipping — no hidden states on disk (MECH-INT needs separate pass).")
            all_results[model_name] = {
                "fisher_computed": False,
                "n_params": cfg["n_params"],
                "known_best_auroc": cfg["known_best_auroc"],
                "source": cfg["source"],
            }
            continue

        H, y_rouge = loaded
        # Use judge labels if available and model is TruthfulQA-sourced
        if cfg["use_judge_labels"] and judge_labels is not None:
            n = min(H.shape[0], judge_labels.shape[0])
            y = judge_labels[:n]
            H = H[:n]
            print(f"  Using judge labels: n={n}, hall_rate={1-y.mean():.3f}")
        else:
            y = y_rouge
            print(f"  Using ROUGE labels: n={len(y)}, hall_rate={1-y.mean():.3f}")

        print(f"  Hidden states: {H.shape}")

        # Load unembedding matrix for causal Fisher
        W_U = None
        if cfg["unembedding_model_id"]:
            W_U = load_unembedding_matrix(cfg["unembedding_model_id"])

        # ── Certificate A + B: Euclidean + Causal Fisher ──────────────────────
        print("\n  Certificate A+B: Euclidean + Causal Fisher ...")
        curve = fisher_curve(H, y, method="pca", n_components=100,
                             verbose=True, W_U=W_U)

        # ── Certificate C: Local Intrinsic Dimension ───────────────────────────
        print("\n  Certificate C: Local Intrinsic Dimension ...")
        k_nn = min(20, H.shape[0] // 10)
        lid_result = lid_curve(H, y, k=k_nn, verbose=True)

        # ── OOF probe at each layer ────────────────────────────────────────────
        print("\n  OOF probe at each layer ...")
        probe_aurocs = probe_auroc_per_layer(H, y, n_splits=5)

        n_layers = H.shape[1]
        result = {
            "model_name": model_name,
            "n_params": cfg["n_params"],
            "n_layers": n_layers,
            "label_source": "judge" if (cfg["use_judge_labels"] and judge_labels is not None) else "rouge",
            "fisher_computed": True,
            "source": cfg["source"],
            # Euclidean Fisher
            "J_per_layer": curve["J"],
            "auroc_bound_per_layer": curve["auroc_bound"],
            "best_J_layer": curve["best_layer"],
            "best_J": curve["best_J"],
            "best_auroc_bound": curve["best_auroc_bound"],
            "best_depth_fraction_fisher": curve["depth_fraction"],
            # Causal Fisher
            "J_causal_per_layer": curve["J_causal"],
            "auroc_bound_causal_per_layer": curve["auroc_bound_causal"],
            # LID
            "lid_auroc_per_layer": lid_result["lid_auroc_per_layer"],
            "lid_diff_per_layer": lid_result["lid_diff_per_layer"],
            "best_lid_layer": lid_result["best_layer"],
            "best_lid_auroc": lid_result["best_lid_auroc"],
            "best_depth_fraction_lid": lid_result["best_depth_fraction"],
            # Probe
            "probe_auroc_per_layer": probe_aurocs,
            "best_probe_auroc": float(np.max(probe_aurocs)),
            "best_probe_layer": int(np.argmax(probe_aurocs)),
            "best_depth_fraction_probe": float(np.argmax(probe_aurocs) / (n_layers - 1)),
            "known_best_auroc": cfg["known_best_auroc"],
        }

        # Bound errors (primary certificate vs probe)
        result["bound_error_euclidean"] = abs(result["best_auroc_bound"] - result["best_probe_auroc"])
        if not np.isnan(curve["auroc_bound_causal"][curve["best_layer"]]):
            result["bound_error_causal"] = abs(
                curve["auroc_bound_causal"][curve["best_layer"]] - result["best_probe_auroc"]
            )

        print(f"\n  Summary:")
        print(f"    Euclidean Fisher bound: {result['best_auroc_bound']:.4f} "
              f"(error: {result['bound_error_euclidean']:.4f})")
        if "bound_error_causal" in result:
            print(f"    Causal Fisher bound:    {result['auroc_bound_causal_per_layer'][curve['best_layer']]:.4f} "
                  f"(error: {result['bound_error_causal']:.4f})")
        print(f"    LID best AUROC:         {result['best_lid_auroc']:.4f}")
        print(f"    Probe AUROC:            {result['best_probe_auroc']:.4f}")
        print(f"    Depth fraction (probe): {result['best_depth_fraction_probe']:.3f}")

        all_results[model_name] = result

    # Save
    def _to_native(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_to_native(v) for v in obj]
        return obj

    out_path = RESULTS_DIR / "01_fisher_analysis.json"
    with open(out_path, "w") as f:
        json.dump(_to_native(all_results), f, indent=2)
    print(f"\nResults saved to {out_path}")

    _plot(all_results)
    _print_certificate_comparison(all_results)
    return all_results


def _print_certificate_comparison(results: dict) -> None:
    computed = {k: v for k, v in results.items() if v.get("fisher_computed", False)}
    if not computed:
        return
    print("\n" + "="*60)
    print("CERTIFICATE COMPARISON SUMMARY")
    print(f"{'Model':<25} {'Fisher err':>12} {'Causal err':>12} {'LID AUROC':>10} {'Probe AUROC':>12}")
    print("─"*75)
    for model_name, r in computed.items():
        fe = f"{r['bound_error_euclidean']:.4f}"
        ce = f"{r.get('bound_error_causal', float('nan')):.4f}"
        la = f"{r['best_lid_auroc']:.4f}"
        pa = f"{r['best_probe_auroc']:.4f}"
        print(f"  {model_name:<23} {fe:>12} {ce:>12} {la:>10} {pa:>12}")


def _plot(results: dict) -> None:
    computed = {k: v for k, v in results.items() if v.get("fisher_computed", False)}
    if not computed:
        return

    n_models = len(computed)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    colors = {"Fisher": "steelblue", "Causal Fisher": "darkorange",
              "LID": "green", "Probe": "black"}

    for ax, (model_name, r) in zip(axes, computed.items()):
        n_layers = r["n_layers"]
        norm_depth = [L / (n_layers - 1) for L in range(n_layers)]

        ax.plot(norm_depth, r["auroc_bound_per_layer"],
                label="Fisher bound", color=colors["Fisher"], linewidth=2)
        if any(not np.isnan(v) for v in r["auroc_bound_causal_per_layer"]):
            ax.plot(norm_depth, r["auroc_bound_causal_per_layer"],
                    label="Causal Fisher bound", color=colors["Causal Fisher"],
                    linewidth=2, linestyle="-.")
        ax.plot(norm_depth, r["lid_auroc_per_layer"],
                label="LID AUROC", color=colors["LID"], linewidth=1.5, linestyle="--")
        ax.plot(norm_depth, r["probe_auroc_per_layer"],
                label="Probe AUROC (OOF)", color=colors["Probe"],
                linewidth=2, linestyle=":")

        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.axvline(0.89, color="red", linestyle=":", linewidth=1, alpha=0.4, label="89% depth")
        ax.set_xlabel("Normalized Depth")
        ax.set_ylabel("AUROC")
        ax.set_title(f"{model_name}")
        ax.legend(fontsize=7)
        ax.set_ylim(0.4, 1.0)
        ax.set_xlim(0, 1)

    plt.suptitle("Three Certificates vs Probe AUROC — All Models", fontsize=13)
    plt.tight_layout()
    out = PLOTS_DIR / "01_three_certificates.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    run()

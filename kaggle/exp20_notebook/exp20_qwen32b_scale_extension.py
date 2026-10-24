"""
GEOM-PROOF -- Kaggle Exp 20: Qwen 2.5 14B Scale Extension (4-bit quantized)
============================================================================

Extends this paper's scale curve past its current ceiling (Qwen 2.5 7B,
7.6B params) to Qwen2.5-14B-Instruct (14.8B params), 4-bit quantized via
bitsandbytes.

Round-6/7/8 note: an earlier version of this script targeted
Qwen2.5-32B-Instruct on a requested T4 x2 (2x14.5GB) allocation, but
`device_map="auto"` (with and without explicit `max_memory` splits)
repeatedly placed the entire 4-bit model on GPU 0 alone and OOM'd --
Accelerate's automatic multi-GPU balancing did not behave as documented
for this model/quantization combination, and burning further iterations
debugging multi-GPU placement was not a good use of a fixed compute
budget. Qwen2.5-14B-Instruct in 4-bit (~7.4GB weights) fits comfortably
on a SINGLE T4 (14.5GB) with no multi-GPU device_map logic needed at
all, eliminating this entire failure class. This is a smaller scale jump
than originally planned (7B -> 14B, 2x, not 7B -> 32B, ~4.3x) but a real
and reliably-obtainable one.

Identical protocol to Exp 13 (Qwen 2.5 7B): balanced HaluEval forced-choice
(Pipeline A) extraction, Fisher certificate curve, linear + MLP OOF probe
AUROC per layer, split conformal calibration -- same hyperparameters
throughout, so this is a directly comparable scale-curve data point, not
a new methodology.

N_SAMPLES reduced from 1000 to 500 (1000 vectors) relative to Exp 13.

Requires: autoawq (pip install -q autoawq) for the pre-quantized checkpoint.
Qwen models are NOT gated on HuggingFace Hub -- no HF_TOKEN required.

Outputs saved to /kaggle/working/:
  20_halueval_qwen14b.npz        <- hidden states (download -> results/hidden_states/)
  20_qwen14b_analysis.json       <- full results  (download -> results/logs/)
  20_qwen14b_certificate.png     <- plot          (download -> results/plots/)
"""

import os, sys, json, gc, time, subprocess

# Rounds 6-11 all hit the same ~14.3-14.5GB wall on a 14.56GB T4 regardless
# of device_map/max_memory/cpu-offload tweaks to on-the-fly bitsandbytes
# 4-bit quantization -- a real bug in this Kaggle environment's
# transformers/bitsandbytes combination, not a config problem. Round 12:
# switch to the OFFICIAL pre-quantized AWQ checkpoint (a fundamentally
# different loading path -- no on-the-fly quantization at all, weights
# are already 4-bit on disk) rather than tweaking the same broken path
# further.
try:
    import gptqmodel  # noqa: F401
except ImportError:
    # This transformers version routes AWQ loading through gptqmodel, not
    # autoawq directly -- the actual (shallow, easy) error round 12 hit.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gptqmodel"], check=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from scipy.stats import norm as scipy_norm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}  : {torch.cuda.get_device_name(i)}  "
              f"VRAM: {torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB")

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)

HS_PATH   = OUT / "20_halueval_qwen14b.npz"
JSON_PATH = OUT / "20_qwen14b_analysis.json"
PLOT_PATH = OUT / "20_qwen14b_certificate.png"

MODEL_ID     = "Qwen/Qwen2.5-14B-Instruct-AWQ"  # official pre-quantized 4-bit checkpoint
N_SAMPLES    = 500     # QA pairs -> 1000 balanced vectors
N_COMPONENTS = 100
N_SPLITS     = 5
RANDOM_STATE = 42

MLP_KWARGS = dict(
    hidden_layer_sizes=(64,), activation="relu", solver="adam", max_iter=500,
    random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=15, tol=1e-4,
)
LR_KWARGS = dict(max_iter=1000, C=1.0, random_state=RANDOM_STATE)

print(f"\nConfiguration:")
print(f"  Model     : {MODEL_ID}  (4-bit quantized)")
print(f"  N pairs   : {N_SAMPLES} -> {2*N_SAMPLES} vectors (balanced 50/50)")
print(f"  PCA k     : {N_COMPONENTS}")
print(f"  CV folds  : {N_SPLITS}")


def load_halueval(n):
    ds = load_dataset("pminervini/HaluEval", "qa")["data"]
    return list(ds.select(range(min(n, len(ds)))))


def extract_hidden_states():
    if HS_PATH.exists():
        print(f"\nStep 1: Hidden states already at {HS_PATH} -- skipping.")
        return

    print(f"\n{'='*60}\nStep 1: Extracting hidden states (4-bit Qwen 2.5 14B)\n{'='*60}")
    t0 = time.time()

    print("\nLoading HaluEval ...")
    samples = load_halueval(N_SAMPLES)
    print(f"  {len(samples)} pairs loaded.")

    print(f"\nLoading {MODEL_ID} (official pre-quantized AWQ, no on-the-fly quant) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Round 12: AWQ weights are already 4-bit on disk (~8.5GB for 14B) --
    # no BitsAndBytesConfig, no on-the-fly quantization, no max_memory
    # juggling. This sidesteps rounds 6-11's on-the-fly-bnb-quantization
    # OOM entirely rather than continuing to patch the same loading path.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"  Loaded: {n_layers} transformer layers, d={d}")
    if DEVICE == "cuda":
        print(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    all_hs, all_labels, skipped = [], [], 0
    for i, sample in enumerate(samples):
        if i % 25 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (len(samples) - i) / 60
            print(f"  [{i:4d}/{len(samples)}]  elapsed={elapsed/60:.1f}min  ETA={eta:.0f}min")

        for text, label in [
            (f"Q: {sample['question']}\nA: {sample['right_answer']}", 1),
            (f"Q: {sample['question']}\nA: {sample['hallucinated_answer']}", 0),
        ]:
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=256, padding=False)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            try:
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at sample {i} -- clearing cache and retrying ...")
                torch.cuda.empty_cache(); gc.collect()
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)

            if out.hidden_states is None:
                skipped += 1
                continue
            hs = np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])
            all_hs.append(hs)
            all_labels.append(label)

    H = np.stack(all_hs).astype(np.float32)
    y = np.array(all_labels, dtype=np.int32)
    total_min = (time.time() - t0) / 60
    print(f"\n  Extraction complete in {total_min:.1f} min")
    print(f"  Shape: {H.shape}  hall_rate={1-y.mean():.3f}  skipped={skipped}")

    np.savez_compressed(HS_PATH, hidden_states=H, labels=y)
    print(f"  Saved: {HS_PATH}  ({HS_PATH.stat().st_size/1e6:.0f} MB)")

    del model
    gc.collect()
    torch.cuda.empty_cache()


def fisher_ratio_pca(H_L, y, k=N_COMPONENTS):
    k = min(k, H_L.shape[1], H_L.shape[0] - 2)
    H_L = H_L.astype(np.float64)
    pca = PCA(n_components=k, random_state=RANDOM_STATE)
    H_proj = pca.fit_transform(H_L)
    mu_c = H_proj[y == 1].mean(0)
    mu_h = H_proj[y == 0].mean(0)
    Sigma_w = LedoitWolf().fit(np.vstack([H_proj[y==1]-mu_c, H_proj[y==0]-mu_h])).covariance_
    diff = mu_c - mu_h
    try:
        w = np.linalg.solve(Sigma_w, diff)
        J = float(diff @ w)
    except np.linalg.LinAlgError:
        J = float(diff @ np.linalg.lstsq(Sigma_w, diff, rcond=None)[0])
    return max(J, 0.0)


def compute_fisher_curve(H, y):
    print(f"\n{'='*60}\nStep 2: Fisher certificate at every layer\n{'='*60}")
    n_layers = H.shape[1]
    J_list, bound_list = [], []
    for L in range(n_layers):
        J = fisher_ratio_pca(H[:, L, :], y)
        b = float(scipy_norm.cdf(np.sqrt(J / 2)))
        J_list.append(J); bound_list.append(b)
        print(f"  L{L:3d}: J={J:.4f}  bound={b:.4f}")
    best = int(np.argmax(J_list))
    print(f"\n  Best J layer: L{best}  J={J_list[best]:.4f}  bound={bound_list[best]:.4f}")
    return J_list, bound_list


def oof_auroc_per_layer(H, y, probe_cls, probe_kwargs, label="probe"):
    n_samples, n_layers, d = H.shape
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    aurocs = []
    for L in range(n_layers):
        H_L = H[:, L, :]
        oof = np.zeros(n_samples)
        for tr, te in skf.split(H_L, y):
            sc = StandardScaler()
            pca = PCA(n_components=min(N_COMPONENTS, H_L[tr].shape[0]-1, H_L[tr].shape[1]),
                      random_state=RANDOM_STATE)
            X_tr = pca.fit_transform(sc.fit_transform(H_L[tr]))
            X_te = pca.transform(sc.transform(H_L[te]))
            clf = probe_cls(**probe_kwargs)
            clf.fit(X_tr, y[tr])
            oof[te] = clf.predict_proba(X_te)[:, 1]
        auroc = float(roc_auc_score(y, oof))
        aurocs.append(auroc)
        print(f"  L{L:3d}: AUROC={auroc:.4f}")
    print(f"\n  [{label}] best AUROC: {np.max(aurocs):.4f} at L{int(np.argmax(aurocs))}")
    return aurocs


def conformal_calibration(H, y, best_layer):
    print(f"\n{'='*60}\nStep 5: Conformal calibration (layer L{best_layer})\n{'='*60}")
    n = len(y)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(n)
    n_tr, n_cal = int(0.6*n), int(0.2*n)
    tr_idx, cal_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr+n_cal], idx[n_tr+n_cal:]

    H_L = H[:, best_layer, :]
    sc = StandardScaler()
    pca = PCA(n_components=min(N_COMPONENTS, H_L[tr_idx].shape[0]-1, H_L.shape[1]),
              random_state=RANDOM_STATE)
    X_tr = pca.fit_transform(sc.fit_transform(H_L[tr_idx]))
    X_cal = pca.transform(sc.transform(H_L[cal_idx]))
    X_te = pca.transform(sc.transform(H_L[te_idx]))

    clf = LogisticRegression(**LR_KWARGS)
    clf.fit(X_tr, y[tr_idx])
    scores_cal = clf.predict_proba(X_cal)[:, 1]
    scores_te = clf.predict_proba(X_te)[:, 1]

    best_alpha = accept_rate = emp_hall = None
    for alpha in np.arange(0.01, 0.50, 0.01):
        thresh = np.quantile(scores_cal[y[cal_idx] == 0], 1 - alpha)
        accepted = scores_te > thresh
        if accepted.sum() == 0:
            continue
        hall_rate = float((y[te_idx][accepted] == 0).mean())
        if hall_rate <= alpha:
            best_alpha = float(round(alpha, 2))
            accept_rate = float(accepted.mean())
            emp_hall = hall_rate
            break

    result = {"alpha_star": best_alpha, "accept_rate": accept_rate, "emp_hall_rate": emp_hall}
    print(f"  alpha*={best_alpha}  accept_rate={accept_rate}  emp_hall={emp_hall}")
    return result


def main():
    t_start = time.time()
    print("\n" + "="*60)
    print("Exp 20 -- Qwen 2.5 14B (4-bit): Scale Extension")
    print("="*60)

    extract_hidden_states()
    data = np.load(HS_PATH)
    H = data["hidden_states"].astype(np.float32)
    y = data["labels"].astype(int)
    n, n_layers, d = H.shape
    print(f"\nLoaded: H={H.shape}  hall_rate={1-y.mean():.3f}")

    J_per_layer, fisher_bounds = compute_fisher_curve(H, y)
    best_J_layer = int(np.argmax(J_per_layer))
    best_J = float(J_per_layer[best_J_layer])
    best_bound = float(fisher_bounds[best_J_layer])

    print(f"\n{'='*60}\nStep 3: Linear probes\n{'='*60}")
    linear_aurocs = oof_auroc_per_layer(H, y, LogisticRegression, LR_KWARGS, label="linear")

    print(f"\n{'='*60}\nStep 4: MLP probes\n{'='*60}")
    mlp_aurocs = oof_auroc_per_layer(H, y, MLPClassifier, MLP_KWARGS, label="mlp")

    best_linear_auroc = float(np.max(linear_aurocs))
    best_linear_layer = int(np.argmax(linear_aurocs))
    best_mlp_auroc = float(np.max(mlp_aurocs))
    best_mlp_layer = int(np.argmax(mlp_aurocs))

    errors_linear = [abs(b - a) for b, a in zip(fisher_bounds, linear_aurocs) if not np.isnan(b)]
    max_error = float(np.max(errors_linear))
    mean_error = float(np.mean(errors_linear))
    linear_violations = [L for L in range(n_layers) if linear_aurocs[L] > fisher_bounds[L] + 1e-6]
    mlp_violations = [L for L in range(n_layers) if mlp_aurocs[L] > fisher_bounds[L] + 1e-6]
    mlp_gain = float(best_mlp_auroc - best_linear_auroc)

    conformal = conformal_calibration(H, y, best_linear_layer)

    result = {
        "experiment": "20_qwen14b_scale_extension", "model": MODEL_ID,
        "n_params_approx": "14.8B", "quantization": "4-bit AWQ (official pre-quantized checkpoint)",
        "n_samples": n, "n_layers": n_layers, "d": d,
        "best_J": best_J, "best_J_layer": best_J_layer, "best_bound": best_bound,
        "J_per_layer": [float(x) for x in J_per_layer],
        "fisher_bound_per_layer": [float(x) for x in fisher_bounds],
        "best_linear_auroc": best_linear_auroc, "best_linear_layer": best_linear_layer,
        "linear_auroc_per_layer": linear_aurocs,
        "linear_bound_error_max": max_error, "linear_bound_error_mean": mean_error,
        "linear_violations": len(linear_violations), "linear_violation_layers": linear_violations,
        "best_mlp_auroc": best_mlp_auroc, "best_mlp_layer": best_mlp_layer,
        "mlp_auroc_per_layer": mlp_aurocs,
        "mlp_violations": len(mlp_violations), "mlp_violation_layers": mlp_violations,
        "mlp_gain": mlp_gain,
        "conformal": conformal,
        "certificate_holds_linear": len(linear_violations) == 0,
        "certificate_holds_mlp": len(mlp_violations) == 0,
        "total_runtime_min": round((time.time() - t_start) / 60, 1),
    }
    with open(JSON_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved -> {JSON_PATH}")

    print(f"\n{'='*60}\nFINAL SUMMARY -- Qwen 2.5 14B (4-bit)\n{'='*60}")
    print(f"  Fisher bound (best):        {best_bound:.4f}")
    print(f"  Linear probe AUROC (best):  {best_linear_auroc:.4f} at L{best_linear_layer}")
    print(f"  MLP probe AUROC (best):     {best_mlp_auroc:.4f} at L{best_mlp_layer}")
    print(f"  Bound max/mean error:       {max_error:.4f} / {mean_error:.4f}")
    print(f"  Certificate holds (lin/mlp):{result['certificate_holds_linear']} / {result['certificate_holds_mlp']}")
    print(f"  Total runtime:              {result['total_runtime_min']} min")

    for path in [HS_PATH, JSON_PATH]:
        if path.exists():
            print(f"  {path.name:35s} {path.stat().st_size/1e6:.1f} MB")

    return result


if __name__ == "__main__":
    main()

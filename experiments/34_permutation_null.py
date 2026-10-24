"""Exp.34 -- permutation null for Fisher separability, and a solver audit.

Two questions the paper never asked:
 (1) What J does this estimator return when there is NO class signal at all?
     If the null-implied bound is already close to the reported bound, the
     headline "the certificate over-predicts by 0.32-0.41" is substantially a
     measurement of estimator bias, not of representation geometry.
 (2) Is the estimator deterministic? sklearn's PCA dispatches to a *randomized*
     (unseeded) solver at k=100 and a *full* (deterministic) solver at k~=498
     for these (n,d). Any k-dependent "instability" is confounded with that.

CPU-only, uses cached hidden states. No new inference.
"""
import json, sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.fisher import fisher_ratio  # noqa

N_PERM = 200
K = 100

MODELS = {
 "gpt2_117m":  ROOT/"results/hidden_states/15_hidden_states_gpt2_117m.npz",
 "gpt2_medium":ROOT/"results/hidden_states/15_hidden_states_gpt2_medium.npz",
 "qwen_0.5b":  ROOT/"results/hidden_states/15_hidden_states_qwen_0.5b.npz",
 "qwen_1.5b":  ROOT/"results/hidden_states/15_hidden_states_qwen_1.5b.npz",
 "qwen_3b":    ROOT/"results/hidden_states/15_hidden_states_qwen_3b.npz",
 "qwen_7b":    ROOT/"results/hidden_states/15_hidden_states_qwen_7b.npz",
}

def J_det(H, y, k=K):
    """Identical estimator to src/fisher.py 'pca' branch, deterministic solver."""
    H = np.asarray(H, float); y = np.asarray(y, int)
    Hp = PCA(n_components=k, svd_solver="full").fit_transform(H)
    c, h = Hp[y == 1], Hp[y == 0]
    mc, mh = c.mean(0), h.mean(0)
    S = LedoitWolf().fit(np.vstack([c - mc, h - mh])).covariance_
    dlt = mc - mh
    return float(dlt @ np.linalg.solve(S, dlt))

def bound(J): return float(norm.cdf(np.sqrt(max(0.0, J) / 2)))

out = {}
for name, path in MODELS.items():
    if not path.exists():
        print(f"[skip] {name}"); continue
    d = np.load(path); H_all, y = d["hidden_states"], d["labels"]
    # best-J layer under the deterministic estimator
    bestJ, bestL = -1.0, -1
    for L in range(H_all.shape[1]):
        j = J_det(H_all[:, L, :], y)
        if j > bestJ: bestJ, bestL = j, L
    H = H_all[:, bestL, :]

    # determinism check on the SHIPPED estimator (randomized solver at k=100)
    reps = [fisher_ratio(H, y, method="pca", n_components=K) for _ in range(6)]
    nondet = float(max(reps) - min(reps))

    # permutation null
    rng = np.random.default_rng(0)
    nulls = []
    for i in range(N_PERM):
        yp = rng.permutation(y)
        nulls.append(J_det(H, yp))
        if (i+1) % 50 == 0: print(f"  {name}: {i+1}/{N_PERM}", flush=True)
    nulls = np.array(nulls)
    p_emp = float((nulls >= bestJ).mean())

    out[name] = {
        "best_layer": bestL, "J_observed": bestJ, "bound_observed": bound(bestJ),
        "J_null_mean": float(nulls.mean()), "J_null_p95": float(np.percentile(nulls, 95)),
        "J_null_max": float(nulls.max()),
        "bound_from_null_mean": bound(nulls.mean()), "bound_from_null_p95": bound(np.percentile(nulls, 95)),
        "empirical_p": p_emp, "n_perm": N_PERM,
        "shipped_estimator_nondeterminism_spread": nondet,
    }
    print(f"{name}: J={bestJ:.3f} (bound {bound(bestJ):.4f}) | null mean={nulls.mean():.3f} "
          f"(bound {bound(nulls.mean()):.4f}) p95={np.percentile(nulls,95):.3f} | p={p_emp:.3f} "
          f"| shipped-estimator spread={nondet:.4f}", flush=True)

(ROOT/"results/logs").mkdir(parents=True, exist_ok=True)
json.dump(out, open(ROOT/"results/logs/34_permutation_null.json","w"), indent=2)
print("\nSaved results/logs/34_permutation_null.json")

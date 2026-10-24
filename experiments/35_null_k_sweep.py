"""Exp.35 -- does the permutation-null bias grow with PCA rank k, and how?

Exp.34 measures the null at k=100 only. The scientifically load-bearing
question is whether J_null(k) reproduces the paper's own bound-error k-curve
(0.186 -> 0.269 -> 0.351 -> 0.427 at k=25/50/100/200) from data with NO class
signal. If it does, the reported k-sensitivity is an estimator-bias curve, not
a property of the representations.

Deterministic solver throughout (svd_solver='full') so nothing here is
confounded with sklearn's randomized-SVD dispatch. CPU-only, cached data.
"""
import json, sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
KS = [25, 50, 100, 200]
N_PERM = 100

MODELS = {
 "gpt2_117m":  ROOT/"results/hidden_states/15_hidden_states_gpt2_117m.npz",
 "gpt2_medium":ROOT/"results/hidden_states/15_hidden_states_gpt2_medium.npz",
 "qwen_0.5b":  ROOT/"results/hidden_states/15_hidden_states_qwen_0.5b.npz",
 "qwen_1.5b":  ROOT/"results/hidden_states/15_hidden_states_qwen_1.5b.npz",
 "qwen_3b":    ROOT/"results/hidden_states/15_hidden_states_qwen_3b.npz",
 "qwen_7b":    ROOT/"results/hidden_states/15_hidden_states_qwen_7b.npz",
}

def J_det(H, y, k):
    H=np.asarray(H,float); y=np.asarray(y,int)
    Hp=PCA(n_components=k, svd_solver="full").fit_transform(H)
    c,h=Hp[y==1],Hp[y==0]; mc,mh=c.mean(0),h.mean(0)
    S=LedoitWolf().fit(np.vstack([c-mc,h-mh])).covariance_
    dl=mc-mh
    return float(dl@np.linalg.solve(S,dl))

def bnd(J): return float(norm.cdf(np.sqrt(max(0.0,J)/2)))

out={}
for name,path in MODELS.items():
    if not path.exists(): continue
    d=np.load(path); H_all,y=d["hidden_states"],d["labels"]
    # fix the layer at k=100 argmax so k-comparison is like-for-like
    bJ,bL=-1.,-1
    for L in range(H_all.shape[1]):
        j=J_det(H_all[:,L,:],y,100)
        if j>bJ: bJ,bL=j,L
    H=H_all[:,bL,:]
    rec={"layer":bL,"per_k":{}}
    for k in KS:
        obs=J_det(H,y,k)
        rng=np.random.default_rng(0)
        nl=np.array([J_det(H,rng.permutation(y),k) for _ in range(N_PERM)])
        rec["per_k"][str(k)]={
            "J_observed":obs,"bound_observed":bnd(obs),
            "J_null_mean":float(nl.mean()),"bound_from_null_mean":bnd(nl.mean()),
            "empirical_p":float((nl>=obs).mean()),
        }
        print(f"{name} k={k:3d}: J_obs={obs:7.3f} bound={bnd(obs):.4f} | "
              f"J_null={nl.mean():7.3f} null-bound={bnd(nl.mean()):.4f} | p={float((nl>=obs).mean()):.3f}",flush=True)
    out[name]=rec
json.dump(out,open(ROOT/"results/logs/35_null_k_sweep.json","w"),indent=2)
print("\nSaved results/logs/35_null_k_sweep.json")

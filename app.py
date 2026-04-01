"""
GEOM-PROOF: Hallucination-aware Representation Probing — Geometric Certificates
================================================================================
A research dashboard walking through the full GEOM-PROOF pipeline:
Fisher separability, causal Fisher, OT certificates, spectral analysis,
conformal coverage, and scale curves across GPT-2, Qwen 2.5, and Mamba.

Run:   streamlit run app.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GEOM-PROOF · Geometric Certificates for Hallucination Detection",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared style ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Design tokens ───────────────────────────────── */
:root {
    --primary:    #1565c0;
    --primary-lt: #1976d2;
    --primary-bg: #e3f2fd;
    --green:      #2e7d32;
    --green-bg:   #e8f5e9;
    --amber:      #e65100;
    --amber-bg:   #fff3e0;
    --red:        #b71c1c;
    --red-bg:     #ffebee;
    --teal:       #00695c;
    --teal-bg:    #e0f2f1;
    --slate:      #37474f;
    --slate-bg:   #eceff1;
    --indigo:     #283593;
    --indigo-bg:  #e8eaf6;
    --bg:         #f0f4fb;
    --surface:    #ffffff;
    --text:       #0d1b2a;
    --text-muted: #6b7280;
    --border:     #ccd9f0;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* ── Callout boxes ───────────────────────────────── */
.finding-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #e8eaf6 100%);
    border-left: 5px solid var(--primary);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(21,101,192,0.10);
    font-size: 0.93rem;
}
.good-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
    border-left: 5px solid var(--green);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(46,125,50,0.10);
    font-size: 0.93rem;
}
.warn-box {
    background: linear-gradient(135deg, #fff3e0 0%, #fff8e1 100%);
    border-left: 5px solid var(--amber);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(230,81,0,0.10);
    font-size: 0.93rem;
}
.null-box {
    background: linear-gradient(135deg, #ffebee 0%, #fce4ec 100%);
    border-left: 5px solid var(--red);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(183,28,28,0.10);
    font-size: 0.93rem;
}
.analogy-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #e0f2f1 100%);
    border-left: 5px solid var(--teal);
    padding: 0.9rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    font-style: italic;
    box-shadow: 0 2px 8px rgba(0,105,92,0.10);
    font-size: 0.90rem;
}
.limit-box {
    background: linear-gradient(135deg, #eceff1 0%, #f5f5f5 100%);
    border-left: 5px solid var(--slate);
    padding: 1.0rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(55,71,79,0.08);
    font-size: 0.91rem;
}
.math-box {
    background: linear-gradient(135deg, #e8eaf6 0%, #ede7f6 100%);
    border-left: 5px solid var(--indigo);
    padding: 1.0rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(40,53,147,0.10);
    font-size: 0.91rem;
    font-family: 'SFMono-Regular', Consolas, monospace;
}

/* ── Pills ───────────────────────────────────────── */
.step-pill {
    display: inline-block;
    background: linear-gradient(90deg, #1565c0 0%, #1976d2 100%);
    color: white;
    padding: 0.28rem 0.9rem;
    border-radius: 20px;
    font-size: 0.80rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 0.6rem;
    box-shadow: 0 2px 6px rgba(21,101,192,0.28);
}
.exp-pill {
    display: inline-block;
    background: linear-gradient(90deg, #283593 0%, #3949ab 100%);
    color: white;
    padding: 0.22rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
    margin-right: 0.3rem;
    box-shadow: 0 2px 6px rgba(40,53,147,0.25);
}
.null-result-pill {
    display: inline-block;
    background: linear-gradient(90deg, #b71c1c 0%, #c62828 100%);
    color: white;
    padding: 0.22rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
    margin-right: 0.3rem;
    box-shadow: 0 2px 6px rgba(183,28,28,0.25);
}
.honest-pill {
    display: inline-block;
    background: linear-gradient(90deg, #2e7d32 0%, #388e3c 100%);
    color: white;
    padding: 0.22rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 6px rgba(46,125,50,0.25);
}

/* ── Stat card ───────────────────────────────────── */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--primary);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(21,101,192,0.07);
    transition: box-shadow 0.2s, transform 0.15s;
}
.stat-card:hover {
    box-shadow: 0 8px 28px rgba(21,101,192,0.13);
    transform: translateY(-2px);
}
.stat-number {
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--primary);
    line-height: 1;
    letter-spacing: -0.02em;
}
.stat-number-red    { color: var(--red); }
.stat-number-green  { color: var(--green); }
.stat-number-indigo { color: var(--indigo); }
.stat-number-amber  { color: var(--amber); }
.stat-number-teal   { color: var(--teal); }
.stat-label {
    font-size: 0.76rem;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-top: 0.35rem;
    line-height: 1.3;
}

/* ── General card ────────────────────────────────── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1.1rem 1.3rem;
    border-radius: 10px;
    margin-bottom: 0.7rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* ── Phase badge ─────────────────────────────────── */
.phase-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(90deg, #0d1b3e 0%, #1565c0 100%);
    color: #e3f2fd;
    border-radius: 8px;
    padding: 0.38rem 1rem;
    font-size: 0.80rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 8px rgba(21,101,192,0.22);
}

/* ── Feature chips ───────────────────────────────── */
.feature-chip {
    display: inline-block;
    background: var(--primary-bg);
    color: var(--primary);
    border-radius: 6px;
    padding: 0.15rem 0.55rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.15rem;
    font-family: 'SFMono-Regular', Consolas, monospace;
}

/* ── Model badge ─────────────────────────────────── */
.model-badge {
    display: inline-block;
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 700;
    margin: 0.2rem;
    letter-spacing: 0.02em;
}
.model-qwen   { background: #e3f2fd; color: #1565c0; border: 1px solid #90caf9; }
.model-gpt2   { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
.model-mamba  { background: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }

/* ── Sidebar ─────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071330 0%, #0d1b3e 40%, #1a2f6e 75%, #1565c0 100%);
}
section[data-testid="stSidebar"] * { color: #e3f2fd !important; }
section[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #e3f2fd !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    text-align: left !important;
    padding: 0.35rem 0.7rem !important;
    margin-bottom: 0.1rem !important;
    transition: background 0.15s !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.15) !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }

/* ── Typography ──────────────────────────────────── */
h1 { color: #0d1b3e; font-weight: 800; letter-spacing: -0.025em; line-height: 1.15; }
h2 { color: #1565c0; font-weight: 700; }
h3 { color: #00695c; font-weight: 600; }

h1::after {
    content: '';
    display: block;
    width: 56px;
    height: 4px;
    background: linear-gradient(90deg, #1565c0, #42a5f5);
    border-radius: 2px;
    margin-top: 0.4rem;
}

/* ── Table ───────────────────────────────────────── */
.stDataFrame { border-radius: 8px; overflow: hidden; }
.stDataFrame thead th { background: var(--primary-bg) !important; color: var(--primary) !important; }

/* ═══════════════════════════════════════════════════
   DARK MODE OVERRIDES
   All light backgrounds become rich dark tones.
   Text colours flip to light so contrast holds.
   ═══════════════════════════════════════════════════ */
@media (prefers-color-scheme: dark) {
    :root {
        --bg:         #0b1120;
        --surface:    #111827;
        --border:     #1e3058;
        --text:       #e2e8f0;
        --text-muted: #94a3b8;
    }

    /* callout boxes */
    .finding-box {
        background: linear-gradient(135deg, #0a1929 0%, #0f1f45 100%) !important;
        color: #b3d9ff !important;
        border-left-color: #42a5f5 !important;
        box-shadow: 0 2px 8px rgba(66,165,245,0.15) !important;
    }
    .good-box {
        background: linear-gradient(135deg, #051a0d 0%, #0a2e17 100%) !important;
        color: #a5d6a7 !important;
        border-left-color: #66bb6a !important;
        box-shadow: 0 2px 8px rgba(102,187,106,0.15) !important;
    }
    .warn-box {
        background: linear-gradient(135deg, #1a0f00 0%, #2a1a00 100%) !important;
        color: #ffcc80 !important;
        border-left-color: #ffa726 !important;
        box-shadow: 0 2px 8px rgba(255,167,38,0.15) !important;
    }
    .null-box {
        background: linear-gradient(135deg, #1a0505 0%, #2a0a0a 100%) !important;
        color: #ef9a9a !important;
        border-left-color: #ef5350 !important;
        box-shadow: 0 2px 8px rgba(239,83,80,0.15) !important;
    }
    .analogy-box {
        background: linear-gradient(135deg, #002320 0%, #003328 100%) !important;
        color: #80cbc4 !important;
        border-left-color: #26a69a !important;
        box-shadow: 0 2px 8px rgba(38,166,154,0.15) !important;
    }
    .limit-box {
        background: linear-gradient(135deg, #111c22 0%, #1a2830 100%) !important;
        color: #b0bec5 !important;
        border-left-color: #78909c !important;
        box-shadow: 0 2px 8px rgba(120,144,156,0.12) !important;
    }
    .math-box {
        background: linear-gradient(135deg, #060d29 0%, #0f1540 100%) !important;
        color: #c5cae9 !important;
        border-left-color: #5c6bc0 !important;
        box-shadow: 0 2px 8px rgba(92,107,192,0.15) !important;
    }

    /* stat cards */
    .stat-card {
        background: #111827 !important;
        border-color: #1e3058 !important;
        border-top-color: #42a5f5 !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.35) !important;
    }
    .stat-number        { color: #64b5f6 !important; }
    .stat-number-green  { color: #81c784 !important; }
    .stat-number-red    { color: #e57373 !important; }
    .stat-number-indigo { color: #9fa8da !important; }
    .stat-number-amber  { color: #ffb74d !important; }
    .stat-number-teal   { color: #4db6ac !important; }
    .stat-label         { color: #90a4ae !important; }

    /* general card */
    .card {
        background: #111827 !important;
        border-color: #1e3058 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }

    /* feature chips & model badges */
    .feature-chip {
        background: #0f1f45 !important;
        color: #90caf9 !important;
    }
    .model-qwen  { background: #0a1929 !important; color: #90caf9 !important; border-color: #1e3a5f !important; }
    .model-gpt2  { background: #051a0d !important; color: #81c784 !important; border-color: #1b5e20 !important; }
    .model-mamba { background: #1a0f00 !important; color: #ffb74d !important; border-color: #4e342e !important; }

    /* typography */
    h1 { color: #90caf9 !important; }
    h2 { color: #64b5f6 !important; }
    h3 { color: #4db6ac !important; }
    h1::after { background: linear-gradient(90deg, #42a5f5, #90caf9) !important; }

    /* dataframe header */
    .stDataFrame thead th {
        background: #0a1929 !important;
        color: #90caf9 !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent
LOGS_DIR  = _HERE / "results" / "logs"
PLOTS_DIR = _HERE / "results" / "plots"


# ── Data loaders ───────────────────────────────────────────────────────────────
_json_cache: dict = {}

def load_json(name):
    if name not in _json_cache:
        p = LOGS_DIR / name
        if not p.exists():
            _json_cache[name] = {}
        else:
            with open(p) as f:
                _json_cache[name] = json.load(f)
    return _json_cache[name]

def plot_img(name, caption=None):
    p = PLOTS_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"Plot not yet generated: {name}")


# ── Helper functions ───────────────────────────────────────────────────────────
def pill(text):
    st.markdown(f'<span class="step-pill">{text}</span>', unsafe_allow_html=True)

def exp_pill(text):
    st.markdown(f'<span class="exp-pill">{text}</span>', unsafe_allow_html=True)

def null_pill(text="NULL RESULT"):
    st.markdown(f'<span class="null-result-pill">✗ {text}</span>', unsafe_allow_html=True)

def honest_pill():
    st.markdown('<span class="honest-pill">✓ HONEST</span>', unsafe_allow_html=True)

def finding(text):
    st.markdown(f'<div class="finding-box">{text}</div>', unsafe_allow_html=True)

def good(text):
    st.markdown(f'<div class="good-box">{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def null(text):
    st.markdown(f'<div class="null-box">{text}</div>', unsafe_allow_html=True)

def analogy(text):
    st.markdown(f'<div class="analogy-box"><b>Analogy:</b> {text}</div>', unsafe_allow_html=True)

def limit(text):
    st.markdown(f'<div class="limit-box"><b>⚠ Limitation:</b> {text}</div>', unsafe_allow_html=True)

def math_note(text):
    st.markdown(f'<div class="math-box">{text}</div>', unsafe_allow_html=True)

def stat(number, label, color=""):
    st.markdown(
        f'<div class="stat-card"><div class="stat-number {color}">{number}</div>'
        f'<div class="stat-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

def phase(text):
    st.markdown(f'<div class="phase-badge">📐 {text}</div>', unsafe_allow_html=True)

def model_badge(name):
    css = {"Qwen 2.5 3B": "model-qwen", "GPT-2 Medium": "model-gpt2", "Mamba-130m": "model-mamba"}.get(name, "model-qwen")
    st.markdown(f'<span class="model-badge {css}">{name}</span>', unsafe_allow_html=True)

def math(latex):
    st.latex(latex)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE STUBS — content added per step
# ═══════════════════════════════════════════════════════════════════════════════

def page_overview():
    st.title("GEOM-PROOF — Geometric Certificates for Hallucination Detection")
    st.markdown("##### Qwen 2.5 3B · GPT-2 Medium · Mamba-130m · HaluEval · 11 Experiments")

    st.markdown("""
    <div style="background:linear-gradient(135deg,#071330,#1565c0);color:#e3f2fd;
    border-radius:12px;padding:1.3rem 1.6rem;margin:0.5rem 0 1.4rem;
    box-shadow:0 4px 20px rgba(21,101,192,0.25);line-height:1.7;">
    <b style="font-size:1.1rem;">The core claim:</b><br>
    Fisher separability — a closed-form geometric certificate derived from hidden-state distributions —
    bounds linear probe AUROC within 2.7% (max error) / 0.93% (mean error) across all layers and models,
    without training a single probe. The mean error is 0.0093; maximum error 0.027 (Exp 06).
    A Fisher bound J gives a provable upper bound on detection difficulty, grounded in optimal transport geometry.
    </div>
    """, unsafe_allow_html=True)

    d01 = load_json("01_fisher_analysis.json")
    ci  = load_json("bootstrap_auroc_ci.json")
    q   = d01.get("Qwen 2.5 3B", {})
    g   = d01.get("GPT-2 Medium 345M", {})
    qci = ci.get("Qwen 2.5 3B", {}).get("bootstrap_ci_95", {})
    gci = ci.get("GPT-2 Medium 345M", {}).get("bootstrap_ci_95", {})

    st.markdown("---")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: stat("2,000", "HaluEval Samples (Qwen/GPT-2)")
    with c2: stat(f"{q.get('best_probe_auroc',0.9917):.4f}", "Best AUROC — Qwen (L36)", "stat-number-green")
    with c3: stat(f"{g.get('best_probe_auroc',0.9887):.4f}", "Best AUROC — GPT-2 (L8)", "stat-number-indigo")
    with c4: stat("≤2.7%", "Max Fisher Bound Error", "stat-number-teal")
    with c5: stat("0.485", "OOD AUROC (TruthfulQA)", "stat-number-red")
    with c6: stat("0.9996", "Within-family Scale R²", "stat-number-green")

    st.markdown("---")
    col1, col2 = st.columns([3,2])
    with col1:
        st.subheader("The Problem")
        st.markdown("""
        Hallucination detection methods typically require training a probe — a supervised classifier on
        labelled hidden states. This raises a circular question: you need labels to build the detector,
        but the detector exists to catch errors in unlabelled data.

        **GEOM-PROOF asks:** Can we certify hallucination detectability from geometry alone — without
        training any probe — using the Fisher separability ratio as a provable upper bound?

        **The Fisher certificate:** Given hidden states of correct vs hallucinated responses at a layer,
        the Fisher ratio J = δᵀΣ⁻¹δ (where δ = class mean difference, Σ = pooled covariance) gives
        a closed-form AUROC upper bound via the Gaussian discriminant formula.
        """)
        null(
            "<b>The key limitation discovered:</b> Fisher J bounds probe AUROC accurately (mean error 0.93%, "
            "max error 2.7% across 62 layers) "
            "but cannot identify the <em>best layer</em> — argmax(J) picks the wrong probe layer 100% of the time. "
            "And the entire system collapses OOD: TruthfulQA AUROC = 0.485 (chance)."
        )
    with col2:
        st.subheader("Experiment Arc")
        steps = [
            ("#1565c0","Exp 01","Three-Certificate Comparison","Fisher J vs Causal Fisher vs LID across layers"),
            ("#1565c0","Exp 02","Scale Curve","Sigmoid fit across 3 architectures; R²=0.9993"),
            ("#1565c0","Exp 03","Certificate Validation","5-fold CV; argmax-J layer match = 0%"),
            ("#b71c1c","Exp 04","Mamba Transfer","FAILED — base LM, hall_rate=1.0"),
            ("#1565c0","Exp 05","Depth Fraction","Universality REFUTED; >50% depth holds"),
            ("#1565c0","Exp 06","Boundary Conditions","Systematic bound looseness documented"),
            ("#e65100","Exp 07","Judge Re-labeling","Cohen's κ=−0.010; labels unreliable"),
            ("#00695c","Exp 08","OT Certificate","SW₂ Spearman r=0.82 vs probe AUROC"),
            ("#00695c","Exp 09","Spectral Transition","KL-MP peaks ≠ probe peaks; arch-specific"),
            ("#00695c","Exp 10","Conformal Coverage","α*=0.07 not 0.10; OOD collapses"),
            ("#1565c0","Exp 11","Qwen Scale Curve","Controlled R²=0.9996; 7B pred=0.999"),
        ]
        for color, badge, title, desc in steps:
            bg = "#ffebee" if "#b71c1c" in color else ("#fff3e0" if "#e65100" in color else ("#e0f2f1" if "#00695c" in color else "#e3f2fd"))
            st.markdown(
                f'<div style="border-left:3px solid {color};background:{bg};'
                f'padding:0.4rem 0.8rem;border-radius:0 8px 8px 0;margin-bottom:0.28rem;">'
                f'<span style="font-size:0.70rem;font-weight:700;color:{color};">{badge}</span> '
                f'<span style="font-size:0.83rem;font-weight:600;color:#0d1b3e;"> {title}</span><br>'
                f'<span style="font-size:0.76rem;color:#6b7280;">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bootstrap 95% CIs (2000 resamples)")
        rows = [
            ("Qwen 2.5 3B",    "L36", f"{q.get('best_probe_auroc',0.9917):.4f}",
             f"[{qci.get('ci_lower',0.9948):.4f}, {qci.get('ci_upper',0.9989):.4f}]"),
            ("GPT-2 Medium",   "L8",  f"{g.get('best_probe_auroc',0.9887):.4f}",
             f"[{gci.get('ci_lower',0.9936):.4f}, {gci.get('ci_upper',0.9983):.4f}]"),
        ]
        df = pd.DataFrame(rows, columns=["Model","Best Layer","AUROC","95% CI"])
        st.dataframe(df, hide_index=True, use_container_width=True)
        warn("CIs computed on training set — slight optimistic bias expected.")
    with col2:
        st.subheader("What This Work Proves vs Does Not Prove")
        claims = [
            ("✅","Fisher J bounds probe AUROC within 2.7% (max) / 0.93% (mean)","All 62 layers, both models — Exp 06"),
            ("✅","Hallucination signal exists in >50% depth","Both transformer models"),
            ("✅","SW₂ correlates with AUROC (r=0.82)","Qwen, Exp 08"),
            ("✅","Within-family scaling is log-linear","Qwen 0.5B→1.5B→3B, R²=0.9996"),
            ("❌","Fisher J identifies best probe layer","0% argmax match rate"),
            ("❌","Probe generalizes OOD","TruthfulQA AUROC=0.485"),
            ("❌","Mamba transfer works","Base LM failure — not architecture failure"),
            ("❌","Labels are reliable","Cohen's κ=−0.010 ROUGE vs judge"),
        ]
        for icon, claim, note in claims:
            bc = "#c8e6c9" if icon=="✅" else "#ffcdd2"
            bg = "#f9fffe" if icon=="✅" else "#fff8f8"
            st.markdown(
                f'<div style="border:1px solid {bc};border-radius:8px;padding:0.4rem 0.7rem;'
                f'margin-bottom:0.28rem;background:{bg};">'
                f'<span style="font-size:0.90rem;">{icon} <b>{claim}</b></span><br>'
                f'<span style="font-size:0.76rem;color:#6b7280;">{note}</span></div>',
                unsafe_allow_html=True,
            )


def page_fisher():
    st.title("Exp 01 — Three-Certificate Comparison")
    exp_pill("Experiment 01")
    phase("Fisher J · Causal Fisher · LID across all layers")

    st.markdown("""
    **Scientific question:** Three geometric certificates are compared per layer:
    **(A)** Euclidean Fisher J, **(B)** Causal Fisher (projected through W_U), **(C)** LID AUROC.
    Which certificate best tracks actual probe AUROC?
    """)

    d = load_json("01_fisher_analysis.json")
    q = d.get("Qwen 2.5 3B", {})
    g = d.get("GPT-2 Medium 345M", {})

    st.markdown("---")
    st.subheader("Qwen 2.5 3B — Certificate Comparison")
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: stat(f"{q.get('best_J',52.15):.2f}", "Best Fisher J (L25)", "stat-number-indigo")
    with c2: stat(f"{q.get('best_auroc_bound',0.9998):.5f}", "AUROC Bound (L25)")
    with c3: stat(f"{q.get('best_probe_auroc',0.9917):.4f}", "Best Probe AUROC (L36)", "stat-number-green")
    with c4: stat(f"{q.get('best_causal_J',39.13):.2f}", "Causal Fisher J (L25)", "stat-number-teal")
    with c5: stat(f"{q.get('best_lid_auroc',0.9491):.4f}", "Best LID AUROC (L0)", "stat-number-amber")

    col1, col2 = st.columns([3,2])
    with col1:
        plot_img("01_three_certificates.png", "Fisher J, Causal Fisher J, and Probe AUROC by layer — Qwen 2.5 3B")
    with col2:
        st.subheader("The Three Certificates")
        certs = [
            ("#1565c0","Certificate A — Euclidean Fisher J",
             "J = δᵀΣ⁻¹δ where δ = μ_correct − μ_hallucinated, Σ = pooled covariance. "
             "AUROC bound = Φ(√J/2). Fast, parameter-free, closed-form.",
             f"Qwen: J={q.get('best_J',52.15):.2f} at L25, bound={q.get('best_auroc_bound',0.9998):.5f}"),
            ("#00695c","Certificate B — Causal Fisher (W_U projection)",
             "Projects hidden states through unembedding matrix W_U (PCA top-100 components) "
             "before computing Fisher ratio. Tests whether vocabulary-space geometry carries "
             "additional hallucination signal beyond the hidden-space geometry.",
             f"Qwen: J_causal={q.get('best_causal_J',39.13):.2f} at L25 — 25% lower than Euclidean, "
             "same depth. Causal projection reduces J because W_U is lower-rank than hidden space."),
            ("#e65100","Certificate C — LID AUROC",
             "Local Intrinsic Dimensionality measures the effective dimensionality of the hidden-state "
             "manifold around each sample. Lower LID = more compressible geometry = potentially more discriminable.",
             f"Qwen: LID best AUROC={q.get('best_lid_auroc',0.9491):.4f} at L0 — substantially below "
             "Fisher bound 0.9998. LID measures a different manifold property, not linear separability."),
        ]
        for color, title, desc, result in certs:
            st.markdown(
                f'<div style="border-left:4px solid {color};background:#f8fbff;'
                f'border-radius:0 8px 8px 0;padding:0.7rem 1rem;margin-bottom:0.5rem;">'
                f'<b style="color:{color};font-size:0.88rem;">{title}</b><br>'
                f'<span style="font-size:0.82rem;color:#37474f;">{desc}</span><br>'
                f'<span style="font-size:0.80rem;color:{color};font-weight:600;margin-top:0.3rem;display:block;">→ {result}</span>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("GPT-2 Medium 345M — Certificate Comparison")
    c1,c2,c3,c4 = st.columns(4)
    with c1: stat(f"{g.get('best_J',49.33):.2f}", "Best Fisher J (L19)", "stat-number-indigo")
    with c2: stat(f"{g.get('best_auroc_bound',0.9998):.5f}", "AUROC Bound (L19)")
    with c3: stat(f"{g.get('best_probe_auroc',0.9887):.4f}", "Best Probe AUROC (L8)", "stat-number-green")
    with c4: stat(f"{g.get('best_lid_auroc',0.9589):.4f}", "Best LID AUROC (L0)", "stat-number-amber")

    null("<b>Causal Fisher failed for GPT-2 Medium:</b> All J_causal values = NaN. "
         "Root cause: GPT-2 lm_head weight shape differs from Qwen — the PCA projection "
         "through W_U was not compatible with the GPT-2 architecture's tied embeddings. "
         "Certificate B comparison exists only for Qwen 3B.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Finding: Fisher J ≠ Probe Peak Layer")
        finding(
            "<b>The certificates disagree on which layer is best.</b><br>"
            "Qwen: Fisher J peaks at L25 (depth=69%), probe AUROC peaks at L36 (depth=100%).<br>"
            "GPT-2: Fisher J peaks at L19 (depth=79%), probe AUROC peaks at L8 (depth=33%).<br><br>"
            "Fisher measures linear class separability in the Gaussian sense. The probe measures "
            "discriminative information under cross-validation. These are not the same thing, "
            "and the gap grows with model depth."
        )
    with col2:
        st.subheader("Causal vs Euclidean Fisher")
        cv = q.get("causal_vs_euclidean_comparison", {})
        finding(
            f"<b>Causal Fisher peaks at the same layer as Euclidean Fisher (L25)</b> "
            f"but with lower magnitude: J_causal={cv.get('causal_best_J',39.13):.2f} vs "
            f"J_euclidean={cv.get('euclidean_best_J',52.15):.2f}.<br><br>"
            "The unembedding projection W_U captures a lower-rank subspace of the hallucination signal "
            "than the full hidden space. This means the model's vocabulary-level representation "
            "is less separable than the full internal representation — the hallucination signal "
            "is distributed across dimensions that don't project cleanly onto tokens."
        )
    limit(
        "All Fisher computations use Ledoit-Wolf covariance shrinkage and PCA(100) for Causal Fisher. "
        "The Gaussian equal-covariance assumption is violated in practice — bounds are systematically "
        "loose (see Exp 06). LID uses a k=20 nearest-neighbour estimator."
    )


def page_scale_curve():
    st.title("Exp 02 + 11 — Scale Curve: How AUROC Scales with Parameter Count")
    exp_pill("Experiment 02")
    exp_pill("Experiment 11")
    phase("Cross-architecture (Exp 02) · Within-family Qwen (Exp 11)")

    d02 = load_json("02_scale_curve.json")
    d11 = load_json("11_qwen_scale_curve.json")
    fit02 = d02.get("fit", {})
    fit11 = d11.get("scale_fit", {})
    pred7b = d11.get("prediction_7b", {})

    c1,c2,c3,c4 = st.columns(4)
    with c1: stat(f"{fit02.get('r_squared',0.9993):.4f}", "Cross-arch R² (Exp 02)", "stat-number-indigo")
    with c2: stat(f"{fit11.get('r_squared',0.9996):.4f}", "Within-family R² (Exp 11)", "stat-number-green")
    with c3: stat(f"{pred7b.get('point_estimate',0.999):.4f}", "7B Prediction (Qwen)", "stat-number-teal")
    with c4: stat("3", "Data Points (both fits)", "stat-number-amber")

    st.markdown("---")
    col1, col2 = st.columns([3,2])
    with col1:
        plot_img("02_scale_curve.png", "Exp 02 — AUROC vs parameter count (cross-architecture sigmoid fit)")
        plot_img("11_qwen_scale_curve.png", "Exp 11 — Within-family Qwen scaling (controlled n=400)")
    with col2:
        st.subheader("Exp 02 — Cross-Architecture Fit")
        st.markdown(f"""
        Sigmoid fit: AUROC = 1 / (1 + exp(−(a·log₁₀(params) + b)))

        **Fitted parameters:**
        - a = {fit02.get('a',8.62):.3f}
        - b = {fit02.get('b',-69.10):.3f}
        - R² = {fit02.get('r_squared',0.9993):.4f}
        """)
        dp = d02.get("data_points", {})
        rows = [(m, f"{v['params']/1e6:.0f}M", f"{v['auroc']:.4f}") for m,v in dp.items()]
        st.dataframe(pd.DataFrame(rows, columns=["Model","Params","AUROC"]), hide_index=True, use_container_width=True)
        warn(
            "<b>Architecture confound:</b> GPT-2 and Qwen differ in tokenizer, training data, "
            "RLHF tuning, and architecture. Scale is confounded with everything else. "
            "R²=0.9993 on 3 points is trivially achievable with 2 free parameters — "
            "it reflects interpolation quality, not a scaling law."
        )

        st.subheader("Exp 11 — Within-Family Controlled Fit")
        m11 = d11.get("models", {})
        rows11 = []
        for mname, mv in m11.items():
            rows11.append((mname, f"{mv['n_params']/1e9:.2f}B",
                           str(mv.get("n_layers","?")), f"{mv['best_auroc']:.4f}",
                           f"L{mv['best_layer']} ({mv['best_depth_fraction']*100:.0f}%)"))
        st.dataframe(pd.DataFrame(rows11, columns=["Model","Params","Layers","Best AUROC","Best Layer"]),
                     hide_index=True, use_container_width=True)

        good(
            f"<b>Controlled fit (n=400 each):</b> a={d11.get('controlled_scale_fit',{}).get('a',4.71):.3f}, "
            f"b={d11.get('controlled_scale_fit',{}).get('b',-39.45):.3f}, "
            f"R²={d11.get('controlled_scale_fit',{}).get('r_squared',0.9996):.4f}<br>"
            f"7B prediction: {pred7b.get('point_estimate',0.999):.4f} "
            f"(90% CI: [{pred7b.get('ci_lower',0.9985):.4f}, {pred7b.get('ci_upper',0.9991):.4f}])<br>"
            "Within the same model family, log-linear sigmoid is well-supported. "
            "This is the more defensible result of the two experiments."
        )

    st.markdown("---")
    st.subheader("What the Scale Curve Means and Does Not Mean")
    col1, col2 = st.columns(2)
    with col1:
        good(
            "<b>What holds:</b> AUROC increases monotonically with parameter count within the "
            "Qwen family. The Fisher separability of hallucinated vs correct hidden states grows "
            "with model capacity. Larger models produce more linearly separable representations "
            "of their own errors."
        )
    with col2:
        null(
            "<b>What does not hold:</b> The 7B predictions (≈0.9999) are not credible — the sigmoid "
            "saturates at 1.0 and any model above 3B will predict near-perfect AUROC by construction. "
            "The cross-architecture fit conflates scale with architecture. "
            "Three data points cannot establish a scaling law."
        )
    limit(
        "Exp 11 uses n=400 samples for 0.5B and 1.5B (hardware constraint) vs n=2000 for 3B. "
        "Qwen 0.5B has γ=2.24 and 1.5B has γ=3.84 — both violate the Gaussian assumption (d > n). "
        "Fisher bounds for these models are unreliable. AUROC values are probe-based and valid."
    )


def page_validation():
    st.title("Exp 03 — Certificate Validation via 5-Fold Cross-Validation")
    exp_pill("Experiment 03")
    phase("Layer Selection Strategies · Bound Error · Oracle vs Predicted")

    st.markdown("""
    **Scientific question:** Does the Fisher certificate give reliable layer selection?
    Given J per layer on training data, does argmax(J) identify the best probe layer on held-out data?
    """)

    d = load_json("03_certificate_validation.json")
    s = d.get("summary", {})
    ls = d.get("layer_selection_strategies", {})

    c1,c2,c3,c4 = st.columns(4)
    with c1: stat(f"{s.get('mean_bound_error',0.0093):.4f}", "Mean Bound Error (±std)", "stat-number-amber")
    with c2: stat(f"{s.get('max_bound_error',0.0154):.4f}", "Max Bound Error", "stat-number-red")
    with c3: stat(f"{s.get('mean_actual_auroc',0.9906):.4f}", "Mean Actual AUROC")
    with c4: stat("0%", "argmax-J Layer Match Rate", "stat-number-red")

    st.markdown("---")
    col1, col2 = st.columns([3,2])
    with col1:
        plot_img("03_certificate_calibration.png", "Predicted AUROC bound vs actual probe AUROC per fold")
        st.subheader("Fold-by-Fold Results")
        fold_rows = []
        for f in d.get("fold_results", []):
            fold_rows.append({
                "Fold": f["fold"],
                "Predicted Layer": f"L{f['predicted_best_layer']}",
                "Oracle Layer": f"L{f['oracle_best_layer']}",
                "Bound": f"{f['predicted_auroc_bound']:.5f}",
                "Actual AUROC": f"{f['actual_auroc_at_predicted_layer']:.4f}",
                "Oracle AUROC": f"{f['oracle_auroc']:.4f}",
                "Bound Error": f"{f['bound_error']:.4f}",
                "Layer Match": "✅" if f["layer_match"] else "❌",
            })
        st.dataframe(pd.DataFrame(fold_rows), hide_index=True, use_container_width=True)
    with col2:
        st.subheader("Layer Selection Strategy Comparison")
        strat_rows = []
        for sname, sv in ls.items():
            if sname == "oracle":
                strat_rows.append({"Strategy":"Oracle (best possible)","Match Rate":"100%",
                                   "Mean AUROC":f"{sv.get('mean_test_auroc',0.9930):.4f}"})
            else:
                strat_rows.append({"Strategy":sname.replace("_"," ").title(),
                                   "Match Rate":f"{sv.get('layer_match_rate',0)*100:.0f}%",
                                   "Mean AUROC":f"{sv.get('mean_test_auroc',0):.4f}"})
        st.dataframe(pd.DataFrame(strat_rows), hide_index=True, use_container_width=True)

        null(
            "<b>argmax J: 0% layer match rate.</b> Fisher J is nearly flat across L20–L36 for Qwen 3B "
            "(range: 49.6 to 52.2 — a spread of only 2.6 J units). Single-layer argmax is "
            "statistically indistinguishable from random selection in this flat region."
        )
        good(
            "<b>Best strategy: depth_weighted_J</b> (J × L/n_layers) achieves 20% layer match rate "
            f"and mean AUROC={ls.get('depth_weighted_J',{}).get('mean_test_auroc',0.9910):.4f} vs "
            f"oracle {ls.get('oracle',{}).get('mean_test_auroc',0.9930):.4f}. "
            "Weighting J by depth penalises early-layer peaks and selects deeper layers more often — "
            "closer to where the true probe optimum lies."
        )

    st.markdown("---")
    finding(
        "<b>Summary:</b> The Fisher certificate successfully bounds AUROC (max error 1.54% across 5 folds "
        "on Qwen 2.5 3B; overall max across all 62 layer-model combinations is 2.7% — see Exp 06). "
        "But it cannot reliably locate the best probe layer — the J plateau across deep layers makes "
        "layer selection from J alone unreliable. The certificate is a valid bound, not a valid selector."
    )
    limit(
        "Experiment uses only Qwen 2.5 3B (n=2000, 5-fold CV). Train size=1600, test size=400 per fold. "
        "PCA(100) applied before Fisher computation. Layer match rate is binary per fold."
    )


def page_mamba():
    st.title("Exp 04 — Mamba Architecture Transfer")
    exp_pill("Experiment 04")
    null_pill("TRANSFER FAILED")
    phase("State Space Model · Base LM · Degenerate Labels")

    d = load_json("04_mamba_transfer.json")
    fa = d.get("failure_analysis", {})

    c1,c2,c3,c4 = st.columns(4)
    with c1: stat("130M", "Mamba-130m Parameters", "stat-number-amber")
    with c2: stat("25", "Layers")
    with c3: stat("1.0", "Hall Rate (all hallucinated)", "stat-number-red")
    with c4: stat("0", "Correct Responses / 400", "stat-number-red")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("What Happened")
        null(
            "<b>Root cause: Base LM, not instruction-tuned.</b><br><br>"
            "Mamba-130m (state-spaces/mamba-130m) is trained on The Pile as a next-token predictor. "
            "When given TruthfulQA questions, it generates text continuations — not factual answers. "
            "ROUGE-L scores against reference answers are universally < 0.10. "
            "With any ROUGE threshold (0.4, 0.1 tested), all 400 samples are labelled hallucinated. "
            "Single-class distribution → Fisher certificate and linear probe AUROC are undefined."
        )
        st.markdown("""
        **What was pre-registered vs what happened:**
        - Pre-registered: AUROC = 0.47 ± 0.05 (scale curve extrapolation to 130M)
        - Actual: AUROC = undefined (degenerate labels)
        - Pre-registration assumed instruction-following capability that base Mamba lacks
        """)
    with col2:
        st.subheader("What Was Salvaged")
        good(
            "<b>Hidden states are valid.</b> Despite labelling failure, "
            "<code>04_mamba_hidden_states.npz</code> contains valid hidden states "
            "for 400 TruthfulQA input prompts (shape: 400×25×768). "
            "These are used in Exp 09 (spectral analysis) where labels are not required."
        )
        st.subheader("Path Forward")
        st.markdown("""
        Three viable fixes for future work:
        1. **MambaChat** — instruction-tuned Mamba exists on HuggingFace; requires GPU
        2. **HaluEval labels directly** — load Mamba on HaluEval QA pairs (pre-labelled); bypass ROUGE
        3. **Classification task** — evaluate on a fixed-label task where base LM responses are expected
        """)
        analogy(
            "Testing a base LM on factual QA is like testing a raw autocomplete engine on an exam. "
            "It was never trained to answer questions — only to continue text. "
            "The failure is about task-model mismatch, not about the Mamba architecture."
        )

    st.markdown("---")
    st.subheader("Spectral Results (Labels Not Required)")
    finding(
        "<b>Mamba's spectral pattern is architecturally distinct.</b> "
        "KL divergence from Marchenko-Pastur peaks at L1 (depth=4%) vs L15 for Qwen (42%) "
        "and L0 for GPT-2. Mamba's recurrent structure produces structured (non-random) "
        "representations immediately — within the first layer. See Exp 09 for full analysis."
    )
    limit(
        "Architecture transfer verdict is inconclusive — failure is due to labelling, not architecture. "
        "Cannot claim that SSMs are harder/easier to probe for hallucination until a labelled "
        "instruction-tuned Mamba experiment is run."
    )


def page_depth():
    st.title("Exp 05 — Depth Fraction Universality")
    exp_pill("Experiment 05")
    null_pill("UNIVERSALITY REFUTED")
    phase("Probe Peak Depth · Fisher Peak Depth · Cross-Model")

    d = load_json("05_depth_fraction.json")
    duf = d.get("depth_universality_finding", {})

    c1,c2,c3,c4 = st.columns(4)
    with c1: stat("100%", "Qwen Probe Peak Depth (L36)", "stat-number-indigo")
    with c2: stat("33%",  "GPT-2 Probe Peak Depth (L8)", "stat-number-green")
    with c3: stat("69%",  "Qwen Fisher Peak Depth (L25)", "stat-number-teal")
    with c4: stat("79%",  "GPT-2 Fisher Peak Depth (L19)", "stat-number-amber")

    st.markdown("---")
    col1, col2 = st.columns([3,2])
    with col1:
        plot_img("05_depth_fraction_overlay.png", "Probe AUROC and Fisher J by layer depth fraction — all models")
    with col2:
        st.subheader("The Depth Universality Hypothesis")
        st.markdown("""
        Prior literature (e.g., Zou et al., 2023; Li et al., 2023) suggested that truthfulness
        information concentrates at approximately **~89% depth** across transformer models.

        GEOM-PROOF tests this across two models (Qwen 2.5 3B and GPT-2 Medium) by computing
        the depth fraction (layer / n_layers) at which both probe AUROC and Fisher J peak.
        """)
        null(
            f"<b>Claim status: {duf.get('claim_status','REFUTED by data')}</b><br><br>"
            f"Predicted: ~89% depth fraction across all models.<br>"
            f"Actual probe depths: Qwen=100%, GPT-2=33%.<br>"
            f"Actual Fisher depths: Qwen=69%, GPT-2=79%.<br><br>"
            "The 33% vs 100% spread shows probe peaks are architecture-dependent, "
            "not universal. Fisher J peaks are closer together (69%–79%) but still disagree "
            "with probe peaks within each model."
        )
        good(
            "<b>Revised claim (supported by data):</b><br>"
            f"{duf.get('revised_claim','Hallucination signals emerge after 50% network depth; exact fraction is architecture-specific')}"
        )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cross-Model Depth Summary")
        rows = [
            ["Qwen 2.5 3B",   "L36", "100%", "L25", "69%"],
            ["GPT-2 Medium",  "L8",  "33%",  "L19", "79%"],
            ["Mamba-130m",    "L1",  "4%",   "N/A", "N/A"],
        ]
        df = pd.DataFrame(rows, columns=["Model","Probe Peak","Probe Depth","Fisher Peak","Fisher Depth"])
        st.dataframe(df, hide_index=True, use_container_width=True)
    with col2:
        finding(
            "<b>Why Fisher depth ≠ probe depth:</b> Fisher J measures linear class separability "
            "in the Gaussian sense — it peaks where the mean difference is largest relative to "
            "within-class variance. Probe AUROC measures discriminative information under "
            "5-fold cross-validation — it peaks where the held-out classifier generalises best. "
            "These are correlated but not identical objectives, and they diverge especially "
            "when the J plateau is flat (Qwen L20–L36 has J ranging 49.6–52.2)."
        )
    limit(
        "Mamba depth fraction (4%) is from spectral analysis only — labels are undefined for Mamba "
        "(see Exp 04). Including Mamba in the depth comparison is illustrative, not evidential. "
        "Depth universality test has n=2 valid models — not enough to make strong universality claims."
    )


def page_boundary():
    st.title("Exp 06 — Boundary Conditions: How Tight is the Fisher Bound?")
    exp_pill("Experiment 06")
    phase("Bound Error · All Layers · Gaussian Violation Analysis")

    st.markdown("""
    **Scientific question:** The Fisher AUROC bound Φ(√J/2) is derived under Gaussian
    equal-covariance assumptions. How loose is it in practice, and is the looseness uniform across layers?
    """)

    d = load_json("06_boundary_conditions.json")
    ba = d.get("boundary_analysis", {})
    q  = ba.get("Qwen 2.5 3B", {})
    g  = ba.get("GPT-2 Medium 345M", {})

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: stat(f"{q.get('best_error',0.0101):.4f}", "Qwen Min Error (L25)", "stat-number-green")
    with c2: stat(f"{q.get('mean_error_all_layers',0.0116):.4f}", "Qwen Mean Error")
    with c3: stat(f"{q.get('max_error_all_layers',0.0266):.4f}", "Qwen Max Error (L0)", "stat-number-amber")
    with c4: stat(f"{g.get('best_error',0.0143):.4f}", "GPT-2 Min Error (L19)", "stat-number-green")
    with c5: stat(f"{g.get('mean_error_all_layers',0.0140):.4f}", "GPT-2 Mean Error")
    with c6: stat(f"{g.get('max_error_all_layers',0.0201):.4f}", "GPT-2 Max Error (L0)", "stat-number-amber")

    st.markdown("---")
    col1, col2 = st.columns([3,2])
    with col1:
        plot_img("06_central_figure.png", "Fisher AUROC bound vs actual probe AUROC — all layers, both models")
    with col2:
        st.subheader("Why the Bound is Systematically Loose")
        bl = d.get("bound_looseness_analysis", {})
        st.markdown(f"""
        {bl.get('finding','')}
        """)
        warn(
            "<b>Practical implication:</b><br>"
            f"{bl.get('practical_implication','')}"
        )
        st.subheader("Bound Error by Model")
        rows = [
            ["Qwen 2.5 3B",  "37", f"{q.get('best_error',0.0101):.4f}",
             f"{q.get('mean_error_all_layers',0.0116):.4f}", f"{q.get('max_error_all_layers',0.0266):.4f}",
             f"{q.get('layers_within_005',37)}/{q.get('total_layers',37)}"],
            ["GPT-2 Medium", "25", f"{g.get('best_error',0.0143):.4f}",
             f"{g.get('mean_error_all_layers',0.0140):.4f}", f"{g.get('max_error_all_layers',0.0201):.4f}",
             f"{g.get('layers_within_005',25)}/{g.get('total_layers',25)}"],
        ]
        df = pd.DataFrame(rows, columns=["Model","Layers","Min Error","Mean Error","Max Error","Within 0.05"])
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown("---")
    good(
        "<b>The bound is valid (not violated):</b> In every layer of both models, "
        "AUROC bound ≥ actual probe AUROC. The bound provides a genuine guarantee — "
        "actual AUROC cannot exceed the Fisher bound. "
        "All 62 layers across both models are within 2.7% error. "
        "The bound is conservative (as designed) but never wrong."
    )
    analogy(
        "A speed camera that always shows a speed ≥ true speed is a valid upper bound — "
        "it never gives false clearance. But if it always reads 30% higher than actual, "
        "it's not useful for choosing which lane to drive in. "
        "The Fisher bound is valid but undiscriminating."
    )
    limit(
        "Bound error is computed as (bound − actual_probe_AUROC). Negative errors would indicate "
        "a violated guarantee — none were found. The looseness stems from unequal within-class "
        "covariances (Σ_correct ≠ Σ_hallucinated) and non-Gaussian hidden state distributions. "
        "A tighter bound would require the full OT Bures W₂ distance (see Exp 08)."
    )


def page_judge():
    st.title("Exp 07 — LLM-as-Judge Label Re-evaluation")
    exp_pill("Experiment 07")
    null_pill("LABEL QUALITY POOR")
    phase("ROUGE-L Labels vs GPT-4o Judge · Cohen's κ")

    st.markdown("""
    **Scientific question:** ROUGE-L threshold labelling assigns binary hallucination labels
    based on surface-form overlap with reference answers. How well do these labels agree with
    GPT-4o judge assessment of actual factual correctness?
    """)

    d = load_json("07_judge_log.json")

    c1,c2,c3,c4 = st.columns(4)
    with c1: stat(str(d.get("n",700)), "Samples Re-evaluated")
    with c2: stat(f"{d.get('rouge_hall_rate',0.9929)*100:.1f}%", "ROUGE Hall Rate", "stat-number-red")
    with c3: stat(f"{d.get('judge_hall_rate',0.9814)*100:.1f}%", "Judge Hall Rate", "stat-number-amber")
    with c4: stat(f"{d.get('kappa',-0.010):.3f}", "Cohen's κ", "stat-number-red")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("What Cohen's κ = −0.010 Means")
        null(
            "<b>Near-zero agreement, worse than chance.</b><br><br>"
            "Cohen's κ measures label agreement corrected for chance. κ=0 means the two labellers "
            "agree exactly as much as random chance would predict. κ=−0.010 means they agree "
            "<em>slightly worse</em> than chance — the two methods are essentially uncorrelated "
            "in their disagreements.<br><br>"
            "ROUGE hall rate = 99.3% vs judge hall rate = 98.1%. Both are very high (consistent "
            "with a hard benchmark), but the individual sample assignments differ substantially. "
            "A sample that ROUGE labels hallucinated may be labelled correct by the judge, and vice versa."
        )
        st.subheader("Example Disagreements (from log)")
        log = d.get("log", [])
        disagree = [x for x in log if x.get("rouge",0) != x.get("judge",0)][:4]
        for ex in disagree:
            rouge_label = "hallucinated" if ex["rouge"]==0 else "correct"
            judge_label = "hallucinated" if ex["judge"]==0 else "correct"
            st.markdown(
                f'<div class="card" style="padding:0.5rem 0.8rem;margin-bottom:0.4rem;">'
                f'<b style="font-size:0.82rem;">Q:</b> <span style="font-size:0.82rem;">{ex["q"]}</span><br>'
                f'<span style="font-size:0.78rem;color:#1565c0;">ROUGE: {rouge_label}</span> · '
                f'<span style="font-size:0.78rem;color:#e65100;">Judge: {judge_label}</span>'
                f'</div>', unsafe_allow_html=True)
    with col2:
        st.subheader("Impact on All Downstream Experiments")
        warn(
            "<b>All experiments in GEOM-PROOF use ROUGE-L labels.</b><br><br>"
            "This is consistent with prior work (HaluEval, SelfCheckGPT) and ensures reproducibility "
            "without GPT-4o API costs. But the κ=−0.010 finding means we cannot claim our probe "
            "detects genuine factual hallucination — it detects ROUGE-L label patterns.<br><br>"
            "The probe may be learning surface-form answer matching rather than "
            "deep factual correctness. This is a fundamental limitation that cannot be fixed "
            "without either human annotation or a high-agreement automatic judge."
        )
        st.subheader("What ROUGE-L Captures vs Misses")
        cases = [
            ("✅ Correct ROUGE label", "#c8e6c9",
             "Model says 'Robin Sharma' when reference says 'Robin Sharma'. "
             "High overlap → correct label. Judge agrees."),
            ("✅ Correct ROUGE label", "#c8e6c9",
             "Model generates a completely wrong proper noun with no lexical overlap. "
             "Low ROUGE → hallucinated label. Judge agrees."),
            ("❌ ROUGE incorrect", "#ffcdd2",
             "Model says correct fact in different words (paraphrase). "
             "Low surface overlap → hallucinated by ROUGE. Judge: correct."),
            ("❌ ROUGE incorrect", "#ffcdd2",
             "Model generates text that coincidentally overlaps with reference "
             "but answers the wrong question. High ROUGE → correct by ROUGE. Judge: hallucinated."),
        ]
        for label, color, desc in cases:
            st.markdown(
                f'<div style="border:1px solid {color};border-radius:8px;padding:0.4rem 0.7rem;'
                f'margin-bottom:0.3rem;">'
                f'<b style="font-size:0.80rem;">{label}</b><br>'
                f'<span style="font-size:0.78rem;color:#37474f;">{desc}</span></div>',
                unsafe_allow_html=True)
    limit(
        "Only 10 sample responses are shown in the judge log (first 10 of 700). "
        "Full re-evaluation used 700 samples with GPT-4o-mini. "
        "The judge prompt is not logged — judge label reliability depends on prompt quality. "
        "κ = −0.010 is the aggregate across all 700 pairs."
    )


def page_ot():
    st.title("Exp 08 — OT Certificate: Wasserstein Generalisation of Fisher")
    exp_pill("Experiment 08")
    phase("Sliced W₂ · Bures W₂ · MMD² · Identity Verification")

    st.markdown("""
    **Scientific question:** Fisher J is the Gaussian approximation to W₂² (Bures-Wasserstein distance
    in whitened space). Do non-parametric OT distances (SW₂, Bures W₂) better track probe AUROC?
    And does the identity J ≈ W₂² hold empirically?
    """)

    d = load_json("08_ot_certificate.json")
    q  = d.get("Qwen 2.5 3B", {})
    g  = d.get("GPT-2 Medium 345M", {})
    qs = q.get("spearman_correlations", {})
    gs = g.get("spearman_correlations", {})

    c1,c2,c3,c4 = st.columns(4)
    with c1: stat(f"{qs.get('SW2',0.821):.3f}", "SW₂ Spearman r (Qwen)", "stat-number-green")
    with c2: stat(f"{qs.get('Fisher_J',0.458):.3f}", "Fisher Spearman r (Qwen)", "stat-number-indigo")
    with c3: stat(f"{q.get('non_gaussianity_finding',{}).get('mean_relative_error_J_vs_W2',1.711):.3f}", "Mean Rel. Error J≈W₂² (Qwen)", "stat-number-amber")
    with c4: stat(f"{g.get('non_gaussianity_finding',{}).get('mean_relative_error_J_vs_W2',0.771):.3f}", "Mean Rel. Error J≈W₂² (GPT-2)", "stat-number-amber")

    st.markdown("---")
    col1, col2 = st.columns([3,2])
    with col1:
        plot_img("08_ot_certificates.png", "Fisher J, SW₂, and Probe AUROC by layer — Qwen 2.5 3B")
    with col2:
        st.subheader("Spearman Correlations vs Probe AUROC")
        rows = [
            ["Qwen 2.5 3B",   f"{qs.get('Fisher_J',0.458):.3f}", f"{qs.get('SW2',0.821):.3f}",
             f"{qs.get('MMD2',0.542):.3f}", f"{qs.get('W2_eq',0.828):.3f}"],
            ["GPT-2 Medium",  f"{gs.get('Fisher_J',0.395):.3f}", f"{gs.get('SW2',-0.237):.3f}",
             f"{gs.get('MMD2',-0.596):.3f}", f"{gs.get('W2_eq',-0.241):.3f}"],
        ]
        df = pd.DataFrame(rows, columns=["Model","Fisher J","SW₂","MMD²","W₂ (equal-cov)"])
        st.dataframe(df, hide_index=True, use_container_width=True)

        finding(
            "<b>SW₂ outperforms Fisher for Qwen (r=0.821 vs r=0.458)</b> — SW₂ better tracks "
            "the probe AUROC curve across layers. This is because SW₂ makes no Gaussianity assumption "
            "and captures the full distributional separation, not just mean-covariance geometry.<br><br>"
            "<b>GPT-2: SW₂ Spearman r=−0.237</b> — SW₂ is negatively correlated with probe AUROC "
            "for GPT-2. The SW₂ curve increases monotonically with layer depth while probe AUROC "
            "peaks at L8. This model-specific reversal shows no single certificate universally "
            "tracks probe AUROC across architectures."
        )

        st.subheader("Best Layers per Certificate (Qwen)")
        bl = q.get("best_layer", {})
        rows2 = [
            ["Fisher J",  f"L{bl.get('Fisher',{}).get('layer',25)}", f"{bl.get('Fisher',{}).get('depth',0.694)*100:.0f}%", f"{bl.get('Fisher',{}).get('certificate',51.92):.2f}"],
            ["SW₂",       f"L{bl.get('SW2',{}).get('layer',35)}",   f"{bl.get('SW2',{}).get('depth',0.972)*100:.0f}%",  f"{bl.get('SW2',{}).get('certificate',309.58):.2f}"],
            ["Probe",     f"L{bl.get('Probe',{}).get('layer',36)}", f"{bl.get('Probe',{}).get('depth',1.0)*100:.0f}%", f"AUROC={bl.get('Probe',{}).get('auroc',0.9917):.4f}"],
        ]
        st.dataframe(pd.DataFrame(rows2, columns=["Certificate","Peak Layer","Depth","Value"]),
                     hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Identity Verification: J ≈ W₂² in Whitened Space?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Qwen 2.5 3B — relative errors at sampled layers**")
        iv_q = q.get("identity_verification", [])
        rows_iv = [{"Layer":f"L{r['layer']}", "J":f"{r['J']:.2f}",
                    "W₂² (whitened)":f"{r['w2_whitened']:.2f}",
                    "Rel. Error":f"{r['relative_error']:.3f}",
                    "Identity Holds":"No"} for r in iv_q]
        st.dataframe(pd.DataFrame(rows_iv), hide_index=True, use_container_width=True)
    with col2:
        ng = q.get("non_gaussianity_finding", {})
        null(
            f"<b>Identity J = W₂² fails empirically.</b><br>"
            f"Mean relative error = {ng.get('mean_relative_error_J_vs_W2',1.711):.3f} for Qwen, "
            f"0.771 for GPT-2. Error grows with layer depth (0.718 at L0 → 3.006 at L36).<br><br>"
            "This is not a failure of the OT framework — it is a finding about LLM geometry: "
            "hidden state distributions are non-Gaussian. Fisher J is the Gaussian limit of W₂². "
            "The growing error means representations become more non-Gaussian with depth."
        )
    limit(
        "SW₂ is estimated with 200 random projections (Monte Carlo). "
        "Bures W₂ requires eigendecomposition of d×d matrices — computed on PCA(200) projections "
        "to make it tractable for d=2048. Identity verification uses 5 sampled layers, not all layers."
    )


def page_spectral():
    st.title("Exp 09 — Spectral Phase Transition (BBP Threshold)")
    exp_pill("Experiment 09")
    phase("Eigenvalue Spectrum · Marchenko-Pastur · BBP Threshold")

    st.markdown("""
    **Scientific question:** Do hidden-state covariance matrices show a spectral phase transition
    (BBP threshold) that correlates with probe AUROC? The Baik-Ben Arous-Péché (BBP) transition
    predicts that signal eigenvalues emerge above the bulk only when the signal is strong enough.
    """)

    d   = load_json("09_spectral_phase_transition.json")
    q   = d.get("Qwen 2.5 3B", {})
    g   = d.get("GPT-2 Medium 345M", {})
    m   = d.get("Mamba-2 130M", {})
    csf = d.get("cross_model_spectral_finding", {})

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: stat(f"L{q.get('best_spectral_layer',15)}", "Qwen Spectral Peak")
    with c2: stat(f"{q.get('best_kl',1.39):.3f}", "Qwen Peak KL-MP", "stat-number-indigo")
    with c3: stat(f"L{g.get('best_spectral_layer',0)}", "GPT-2 Spectral Peak")
    with c4: stat(f"{g.get('best_kl',4.90):.3f}", "GPT-2 Peak KL-MP", "stat-number-green")
    with c5: stat(f"L{m.get('best_spectral_layer',1)}", "Mamba Spectral Peak")
    with c6: stat(f"{m.get('best_kl',1.53):.3f}", "Mamba Peak KL-MP", "stat-number-amber")

    st.markdown("---")
    col1, col2 = st.columns([3,2])
    with col1:
        plot_img("09_spectral_phase_transition.png", "KL divergence from Marchenko-Pastur by layer — all three models")
    with col2:
        st.subheader("The Marchenko-Pastur Framework")
        st.markdown("""
        For a random matrix of shape (n, d), eigenvalues follow the Marchenko-Pastur (MP) distribution
        with aspect ratio γ = d/n. Structured (non-random) data will have eigenvalues that deviate
        from MP — measured by KL divergence.

        **BBP threshold:** The largest random eigenvalue is λ₊ = (1+√γ)². Eigenvalues above this
        threshold are signal spikes (BBP phase transition).
        """)
        math(r"\gamma = d/n, \quad \lambda_+ = (1+\sqrt{\gamma})^2")

        rows = [
            ["Qwen 2.5 3B",  f"{q.get('gamma',1.024):.3f}", f"{q.get('bbp_threshold',1.012):.3f}",
             f"L{q.get('best_spectral_layer',15)} (42%)", "Peaks mid-network"],
            ["GPT-2 Medium", f"{g.get('gamma',0.512):.3f}", f"{g.get('bbp_threshold',0.716):.3f}",
             f"L{g.get('best_spectral_layer',0)} (0%)", "Monotone decrease"],
            ["Mamba-130m",   f"{m.get('gamma',1.92):.3f}",  f"{m.get('bbp_threshold',1.386):.3f}",
             f"L{m.get('best_spectral_layer',1)} (4%)", "Peaks at L1"],
        ]
        df = pd.DataFrame(rows, columns=["Model","γ (d/n)","BBP λ+","KL Peak Layer","Pattern"])
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Cross-Model Spectral Patterns")
    col1, col2, col3 = st.columns(3)
    with col1:
        finding(f"<b>Qwen 2.5 3B:</b><br>{csf.get('qwen_pattern','KL peaks at L15 then decreases')}")
    with col2:
        finding(f"<b>GPT-2 Medium:</b><br>{csf.get('gpt2_pattern','KL monotonically decreases from L0')}")
    with col3:
        finding(f"<b>Mamba-130m:</b><br>{csf.get('mamba_pattern','KL peaks at L1 — recurrent structure')}")

    st.markdown("---")
    mf = q.get("spectral_probe_mismatch_finding", {})
    null(
        "<b>Spectral peaks ≠ probe peaks.</b><br><br>"
        f"{mf.get('interpretation','')}<br><br>"
        f"<b>Revised hypothesis:</b> {mf.get('revised_hypothesis','')}"
    )
    limit(
        "KL divergence is computed by comparing empirical eigenvalue histogram to MP density "
        "using 100 bins. Spike count uses BBP threshold as cutoff. "
        "For Mamba (γ=1.92), the MP bulk is wide and most eigenvalues qualify as spikes — "
        "spike counts are less interpretable at high γ."
    )


def page_conformal():
    st.title("Exp 10 — Conformal Coverage Guarantee")
    exp_pill("Experiment 10")
    phase("Split Conformal · Mondrian CP · α* Correction · Real OOD")

    st.markdown("""
    **Scientific question:** Can the Fisher probe be wrapped in a conformal prediction set
    to give a formal coverage guarantee P(hallucination | accepted) ≤ α?
    What is the minimum α at which the guarantee actually holds (α*)?
    """)
    st.markdown(
        '<div class="limit-box"><b>⚠ Distribution-conditional guarantee:</b> '
        'The conformal coverage guarantee is <em>distribution-conditional</em> — it holds only '
        'when test queries come from the same distribution as the calibration set (HaluEval). '
        'As shown by the OOD test below (TruthfulQA AUROC = 0.485), the guarantee completely '
        'collapses under distribution shift. The "formal guarantee" language applies only '
        'within the training distribution.</div>',
        unsafe_allow_html=True,
    )

    d   = load_json("10_conformal_coverage.json")
    q   = d.get("Qwen 2.5 3B", {})
    g   = d.get("GPT-2 Medium 345M", {})
    qsc = q.get("split_conformal", {})
    gsc = g.get("split_conformal", {})
    qas = q.get("alpha_star", {})
    gas = g.get("alpha_star", {})
    ood = q.get("real_ood_truthfulqa", {})

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: stat("0.10",  "Claimed α", "stat-number-red")
    with c2: stat(f"{qas.get('value',0.07):.3f}", "Qwen α* (valid)", "stat-number-green")
    with c3: stat(f"{gas.get('value',0.06):.3f}", "GPT-2 α* (valid)", "stat-number-green")
    with c4: stat(f"{qas.get('acceptance_rate_at_alpha_star',0.529)*100:.0f}%", "Qwen Accept Rate @ α*")
    with c5: stat(f"{ood.get('hall_rate',0.9825)*100:.1f}%", "OOD Hall Rate (TruthfulQA)", "stat-number-red")
    with c6: stat(f"{ood.get('ood_auroc',0.485):.3f}", "OOD AUROC", "stat-number-red")

    st.markdown("---")
    col1, col2 = st.columns([3,2])
    with col1:
        plot_img("10_conformal_coverage.png", "Hall rate and acceptance rate vs α — both models")
        st.subheader("α* vs Claimed α = 0.10")
        rows = [
            ["Qwen 2.5 3B",  "0.10", f"{qas.get('value',0.07):.3f}",
             f"{qas.get('empirical_hall_rate_at_alpha_star',0.0605):.4f}",
             f"{qas.get('acceptance_rate_at_alpha_star',0.529)*100:.0f}%"],
            ["GPT-2 Medium", "0.10", f"{gas.get('value',0.06):.3f}",
             f"{gas.get('empirical_hall_rate_at_alpha_star',0.0587):.4f}",
             f"{gas.get('acceptance_rate_at_alpha_star',0.528)*100:.0f}%"],
        ]
        df = pd.DataFrame(rows, columns=["Model","Claimed α","Valid α*","Empirical Hall Rate","Accept Rate"])
        st.dataframe(df, hide_index=True, use_container_width=True)
    with col2:
        st.subheader("Split Conformal Guarantee")
        st.markdown("""
        **Split conformal prediction** uses a calibration set to set a threshold τ such that
        the conformal set contains the true label with probability ≥ 1−α on held-out test data.
        For hallucination detection, this means: among accepted samples (below threshold τ),
        the empirical hallucination rate should be ≤ α.
        """)
        math(r"\hat{\tau} = \text{Quantile}_{1-\alpha}(s_1, \ldots, s_n)")
        warn(
            f"<b>α=0.10 claim is incorrect.</b><br>"
            f"At α=0.10, the conformal guarantee does not hold for either model. "
            f"Empirical hall rate at α=0.10: Qwen={qsc.get('empirical_hall_rate',0.108):.3f}, "
            f"GPT-2={gsc.get('empirical_hall_rate',0.119):.3f} — both exceed 0.10.<br><br>"
            f"The guarantee is achievable, but only at α*=0.07 (Qwen) and α*=0.06 (GPT-2)."
        )
        st.subheader("Mondrian CP by Quadrant")
        mon = q.get("mondrian", {})
        rows_m = [(k, f"{v.get('n_cal','?')}",
                   f"{v.get('empirical_hall_rate',0)*100:.1f}%" if v.get('empirical_hall_rate') is not None else "NaN",
                   "✅" if v.get("guarantee_holds") else "❌")
                  for k,v in mon.items()]
        df_m = pd.DataFrame(rows_m, columns=["Quadrant","N Cal","Hall Rate","Guarantee"])
        st.dataframe(df_m, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Real OOD Test — TruthfulQA (400 questions, Qwen 2.5 3B)")
    col1, col2 = st.columns(2)
    with col1:
        null(
            f"<b>OOD AUROC = {ood.get('ood_auroc',0.485):.3f} — chance level.</b><br><br>"
            f"Hall rate on TruthfulQA = {ood.get('hall_rate',0.9825)*100:.1f}% "
            f"({ood.get('n_correct',7)}/{ood.get('n_samples',400)} correct).<br><br>"
            "The probe, trained on HaluEval hidden states, completely fails to generalise "
            "to TruthfulQA. The conformal guarantee trained on HaluEval calibration data "
            "is meaningless on TruthfulQA — the score distribution shifts entirely."
        )
    with col2:
        finding(
            "<b>Why OOD failure is expected (and important):</b><br><br>"
            "HaluEval and TruthfulQA are fundamentally different tasks. HaluEval contains "
            "Wikipedia-style Q&A with ROUGE-L labels. TruthfulQA contains questions specifically "
            "designed to elicit confident hallucinations — harder, more adversarial.<br><br>"
            "The probe learns the geometry of HaluEval's hallucination distribution. "
            "That geometry does not transfer. This is not a bug — it is a finding: "
            "Fisher probes are dataset-specific, not universal hallucination detectors."
        )
    limit(
        "n_cal = 1000 (50% of 2000 samples). τ is computed on calibration set, tested on remaining 1000. "
        "Mondrian CP uses 4 quadrants from the entropy × consistency decomposition. "
        "Epistemic quadrant has n_cal=1 (too few for a valid guarantee). "
        "OOD test uses hidden states from the Kaggle GPU run (exp_causal_fisher_ood.py)."
    )


def page_ood():
    st.title("OOD Failure — Why the Probe Doesn't Generalise")
    honest_pill()
    phase("TruthfulQA · Distribution Shift · What It Means")

    st.markdown("""
    This page consolidates the OOD failure evidence and explains what it means for
    the project's claims. The key result: Fisher probe AUROC = **0.485** on TruthfulQA — chance level.
    """)

    d   = load_json("10_conformal_coverage.json")
    q   = d.get("Qwen 2.5 3B", {})
    ood = q.get("real_ood_truthfulqa", {})

    c1,c2,c3,c4 = st.columns(4)
    with c1: stat("0.9917", "In-distribution AUROC (HaluEval)", "stat-number-green")
    with c2: stat(f"{ood.get('ood_auroc',0.485):.3f}", "OOD AUROC (TruthfulQA)", "stat-number-red")
    with c3: stat(f"{ood.get('hall_rate',0.9825)*100:.1f}%", "TruthfulQA Hall Rate", "stat-number-red")
    with c4: stat(f"{ood.get('n_correct',7)}/400", "Correct on TruthfulQA", "stat-number-amber")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("What Changed Between Datasets")
        rows = [
            ["Question type", "Wikipedia factoid Q&A", "Designed to elicit confident hallucinations"],
            ["Difficulty", "Medium — model gets many right", "Hard — Qwen gets 7/400 right"],
            ["Hall rate", "~98% (HaluEval design)", "98.25% (model failure)"],
            ["Label method", "ROUGE-L vs Wikipedia answer", "Best answer from human annotations"],
            ["Probe training", "On HaluEval geometry", "Tested on alien geometry"],
        ]
        df = pd.DataFrame(rows, columns=["Property","HaluEval (train)","TruthfulQA (OOD)"])
        st.dataframe(df, hide_index=True, use_container_width=True)

        null(
            "<b>The probe's geometry is HaluEval-specific.</b><br><br>"
            "The class means μ_correct and μ_hallucinated are computed from HaluEval hidden states. "
            "On TruthfulQA, the model's hidden states occupy a different region of representation space — "
            "the cosine distances to those class means are meaningless. "
            "The conformal threshold τ calibrated on HaluEval does not transfer."
        )
    with col2:
        st.subheader("What This Means for the Project's Claims")
        good(
            "<b>Claims that survive OOD failure:</b><br>"
            "• Fisher J bounds probe AUROC on the training distribution (mean error 0.93%, max 2.7%)<br>"
            "• Hallucination signal exists in HaluEval hidden states at >50% depth<br>"
            "• Within-family Qwen scaling is log-linear (in-distribution)<br>"
            "• SW₂ better tracks in-distribution AUROC than Fisher J<br>"
            "• The conformal guarantee holds at α*=0.07 on HaluEval"
        )
        null(
            "<b>Claims that do NOT survive OOD failure:</b><br>"
            "• The system is a general hallucination detector<br>"
            "• The probe generalises across question types or benchmarks<br>"
            "• The Fisher certificate certifies anything about real-world deployment<br>"
            "• The conformal guarantee holds outside HaluEval distribution"
        )
        analogy(
            "A fingerprint recognition system trained on one population performs poorly on another. "
            "It learned the specific geometry of the training fingerprints, not a universal "
            "fingerprint representation. The Fisher probe has the same limitation — "
            "it learned HaluEval geometry."
        )
    limit(
        "The TruthfulQA test uses Qwen 2.5 3B hidden states at L25 (Fisher best layer). "
        "The 7/400 correct answers produce a near-degenerate label distribution — AUROC is "
        "unreliable with such imbalance (393 hallucinated vs 7 correct). "
        "The hall_rate=0.9825 also suggests Qwen struggles on TruthfulQA generally, "
        "not specifically on conformal detection."
    )


def page_honest():
    st.title("Honest Assessment")
    honest_pill()
    phase("What This Work Actually Proves — And What It Does Not")

    # ── Plain-language intro ──────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#071330,#1565c0);color:#e3f2fd;
    border-radius:12px;padding:1.4rem 1.8rem;margin:0.5rem 0 1.6rem;
    box-shadow:0 4px 20px rgba(21,101,192,0.30);line-height:1.8;font-size:0.97rem;">
    <b style="font-size:1.15rem;display:block;margin-bottom:0.6rem;">
    What is this page for?</b>
    Research papers often overstate their contributions.  This page does the opposite —
    it separates what the numbers actually support from what they do not.
    Each claim is graded against the experiment that was supposed to test it.
    If the experiment failed or the evidence is weak, the claim is listed as <b>Not Earned</b>.
    Honest science means publishing both columns.
    </div>
    """, unsafe_allow_html=True)

    # ── How to read this ─────────────────────────────────────────────────────
    analogy(
        "<b>How to read this page:</b><br>"
        "Think of a claim as a hypothesis and an experiment as a test. "
        "A claim is <b>Earned</b> when the experimental numbers directly support it and "
        "no confound can explain the result away. "
        "A claim is <b>Not Earned</b> when the experiment failed, the numbers are ambiguous, "
        "or we simply never ran the right test. "
        "A Not Earned result is not a failure — it is a finding that tells you what to do next."
    )

    st.markdown("---")
    st.subheader("What Did We Set Out To Do?")
    st.markdown("""
    The central question of GEOM-PROOF is:

    > *Can you predict how well a hallucination detector will work — purely from the geometry
    > of the model's internal representations, without training any detector?*

    We tested this using **Fisher separability (J)**: a single number that measures how far
    apart the hidden-state distributions of hallucinated vs. correct responses sit.
    If J is large, a linear probe should achieve high AUROC. If J is small, the signal
    is weak and detection will be unreliable.

    We ran 11 experiments across three models (Qwen 2.5 3B, GPT-2 Medium, Mamba-130m)
    and one dataset (HaluEval). Here is what we found.
    """)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Claims Earned")
        st.caption("These are claims where the experimental evidence is direct, numbers hold up under cross-validation, and the finding survives scrutiny.")
        earned = [
            (
                "Fisher J bounds probe AUROC within 2.7% (max error) / 0.93% (mean error)",
                "Experiments 01 & 06",
                "Across every layer in both Qwen 2.5 3B and GPT-2 Medium — 62 layer-model "
                "combinations — the Fisher bound never violated actual probe AUROC by more than "
                "2.7% (max; mean error = 0.93%). This means: if J says detection is easy, a probe "
                "will confirm it. The bound is conservative and reliable."
            ),
            (
                "Hallucination signals emerge after the halfway point of the network",
                "Experiment 05",
                "For both transformer models, the hallucination-related separation in hidden "
                "states becomes detectable only in layers deeper than 50% of the network depth. "
                "Early layers encode syntax and surface form, not factual accuracy."
            ),
            (
                "Sliced Wasserstein (SW₂) tracks probe AUROC better than Fisher J",
                "Experiment 08",
                "SW₂ achieves Spearman r=0.821 vs probe AUROC, compared to r=0.458 for Fisher J. "
                "SW₂ is a distribution-level distance that does not assume Gaussian geometry — "
                "it is more flexible and captures the actual shape of hidden-state distributions."
            ),
            (
                "Within one model family, AUROC scales predictably with model size",
                "Experiment 11",
                "Testing Qwen 2.5 at 0.5B, 1.5B, and 3B — same architecture, controlled "
                "sample size — shows a near-perfect log-linear relationship (R²=0.9996). "
                "Larger Qwen models produce more separable hidden states for this task."
            ),
            (
                "A conformal coverage guarantee is achievable at α*=0.07 (in-distribution only)",
                "Experiment 10",
                "Split conformal prediction gives a valid statistical guarantee: the hallucination "
                "rate among accepted responses stays below 7.0% on the HaluEval test set. "
                "Note: the originally targeted 10% threshold did not hold — 7% is the true floor. "
                "Critical caveat: this guarantee is distribution-conditional. On TruthfulQA (OOD), "
                "probe AUROC collapses to 0.485 and the conformal guarantee is meaningless."
            ),
            (
                "Different architectures produce distinct spectral fingerprints",
                "Experiment 09",
                "Qwen, GPT-2, and Mamba all show different patterns in how their weight matrix "
                "spectra deviate from the Marchenko-Pastur (random) distribution. "
                "This is a genuine structural finding about how architectures encode information."
            ),
        ]
        for claim, source, explanation in earned:
            st.markdown(
                f'<div style="border-left:4px solid #2e7d32;border-radius:0 9px 9px 0;'
                f'padding:0.75rem 1rem;margin-bottom:0.7rem;'
                f'background:rgba(46,125,50,0.10);">'
                f'<div style="font-size:0.88rem;font-weight:700;margin-bottom:0.25rem;">✅ {claim}</div>'
                f'<div style="font-size:0.74rem;font-weight:600;opacity:0.6;margin-bottom:0.35rem;'
                f'text-transform:uppercase;letter-spacing:0.07em;">{source}</div>'
                f'<div style="font-size:0.82rem;line-height:1.6;opacity:0.85;">{explanation}</div>'
                f'</div>',
                unsafe_allow_html=True)

    with col2:
        st.subheader("❌ Claims Not Earned")
        st.caption("These are claims we expected to make but could not — either the experiment disproved them, or the right experiment was never run.")
        not_earned = [
            (
                "Fisher J can identify which layer is best for detection",
                "Experiment 03",
                "This is the most important failure. Across 5-fold cross-validation, choosing "
                "the layer with the highest J score matched the true best probe layer exactly "
                "0% of the time. The reason: J forms a flat plateau across deep layers, "
                "so argmax(J) is essentially random. Depth-weighting helps marginally (20% match) "
                "but is not reliable enough for practical use."
            ),
            (
                "The 'best layer is at ~89% depth' is a universal rule",
                "Experiment 05",
                "GPT-2's best probe layer is at 33% depth; Qwen's is at 100%; Mamba's appears at 4%. "
                "There is no universal depth fraction. The revised claim is weaker: "
                "'>50% depth for instruction-tuned transformers' — which is architecture-dependent."
            ),
            (
                "The hallucination probe generalises to new question types",
                "Experiment 10 (OOD test)",
                "When tested on TruthfulQA — a different benchmark than the HaluEval training data — "
                "probe AUROC collapses to 0.485 (random chance). The probe has learned the geometry "
                "of HaluEval responses specifically, not a universal signal of factual error. "
                "This is the single most important limitation of the whole project."
            ),
            (
                "Fisher J works for non-transformer architectures (Mamba)",
                "Experiment 04",
                "The Mamba experiment failed completely, but not because of the architecture. "
                "We used a base language model (not instruction-tuned), which produced degenerate "
                "outputs on the instruction-style HaluEval prompts. hall_rate = 1.0 — every response "
                "classified as a hallucination. The architecture question remains open."
            ),
            (
                "ROUGE-L scores are a valid measure of hallucination",
                "Experiment 07",
                "ROUGE-L compares the word overlap between a model's answer and a reference answer. "
                "When we asked GPT-4o to independently judge the same responses, the two labellings "
                "disagreed almost completely (Cohen's κ = −0.010). ROUGE-L captures whether the "
                "model used similar words — not whether it was factually correct. "
                "All our claims rest on these labels."
            ),
            (
                "Fisher J and Wasserstein distance W₂ are equivalent (J ≈ W₂²)",
                "Experiment 08",
                "The theoretical connection between Fisher separability and Wasserstein distance "
                "holds only when hidden states follow a Gaussian distribution with equal covariance. "
                "In practice, mean relative error between J and W₂² is 1.71 for Qwen — "
                "the geometry is non-Gaussian, so the elegant identity breaks down."
            ),
            (
                "We can reliably predict AUROC for a 7B model",
                "Experiment 02",
                "The sigmoid fit to our 3-point scale curve (117M, 345M, 3B) saturates near 1.0 "
                "for any model above 3B parameters. The '7B prediction' is 0.9990, but this is "
                "a consequence of the curve hitting its ceiling — it has no predictive content. "
                "Running actual Qwen 2.5 7B experiments is the only valid test."
            ),
        ]
        for claim, source, explanation in not_earned:
            st.markdown(
                f'<div style="border-left:4px solid #c62828;border-radius:0 9px 9px 0;'
                f'padding:0.75rem 1rem;margin-bottom:0.7rem;'
                f'background:rgba(183,28,28,0.10);">'
                f'<div style="font-size:0.88rem;font-weight:700;margin-bottom:0.25rem;">❌ {claim}</div>'
                f'<div style="font-size:0.74rem;font-weight:600;opacity:0.6;margin-bottom:0.35rem;'
                f'text-transform:uppercase;letter-spacing:0.07em;">{source}</div>'
                f'<div style="font-size:0.82rem;line-height:1.6;opacity:0.85;">{explanation}</div>'
                f'</div>',
                unsafe_allow_html=True)

    # ── The two-project arc ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("How GEOM-PROOF Connects to HaRP")
    analogy(
        "<b>The two-project research arc in plain language:</b><br><br>"
        "<b>GEOM-PROOF</b> asks: <em>Does a hallucination signal exist in the model's geometry — "
        "and can we prove it without training a detector?</em><br>"
        "Answer: <b>Yes, the signal exists</b> (Fisher bound valid; max error 2.7%, mean 0.93%). "
        "But the geometry alone cannot tell you <em>which layer</em> to use, and it does not "
        "transfer outside the training distribution.<br><br>"
        "<b>HaRP</b> asks: <em>Given that the signal exists, can we build a practical system that "
        "flags hallucinations before a user sees them?</em><br>"
        "Answer: <b>Yes, in-distribution</b> (AUROC ≈ 0.77, calibrated, with a 3-action policy). "
        "But it also fails OOD — the same fundamental limitation.<br><br>"
        "GEOM-PROOF provides the theoretical justification for why probing works. "
        "HaRP provides the engineering that makes it deployable. "
        "Neither is complete without the other."
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        rows = [
            ["Question",      "Does a geometric certificate exist?",    "Can we deploy a detection system?"],
            ["Primary result","Fisher bound ≤2.7% max / 0.93% mean error","AUROC 0.77 (OOF-corrected)"],
            ["Models tested", "Qwen 2.5 3B, GPT-2 Medium, Mamba-130m", "Qwen 2.5 3B"],
            ["Dataset",       "HaluEval (ROUGE-L labels)",              "TruthfulQA (ROUGE-L labels)"],
            ["OOD result",    "0.485 — fails completely",               "Not benchmarked OOD"],
            ["Key technique", "Fisher J, SW₂, Conformal CP",            "LR probe + 3-action policy"],
            ["Honest gap",    "Cannot select best layer (0% match)",    "Does not generalise OOD"],
        ]
        df = pd.DataFrame(rows, columns=["Property", "GEOM-PROOF", "HaRP"])
        st.dataframe(df, hide_index=True, use_container_width=True)
    with col2:
        warn(
            "<b>The shared honest limitation:</b><br><br>"
            "Both projects fail out-of-distribution. GEOM-PROOF was validated on HaluEval. "
            "HaRP was validated on TruthfulQA. Neither was tested on the other's dataset "
            "or on held-out benchmarks like NaturalQuestions, MedQA, or TriviaQA.<br><br>"
            "This means: <em>we have demonstrated the technique works on the datasets we chose, "
            "not that it works in general.</em> That distinction matters enormously for anyone "
            "considering applying this to a real system."
        )

    # ── What to do next ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("What Should Be Done Next")
    st.markdown("These are the most important gaps left open by this work, ranked by how much they affect the validity of the existing claims.")

    nexts = [
        ("Critical", "#b71c1c",
         "Fix the label quality problem",
         "Cohen's κ = −0.010 between ROUGE-L and GPT-4o judge labels means the two methods "
         "measure different things. Every experiment in this project — every claim earned or not — "
         "is built on ROUGE-L labels. If those labels do not capture genuine factual errors, "
         "then the geometry we are measuring is also not about factual errors. "
         "Fix: collect a small human-annotated or high-κ-judge subset and re-run key experiments."),
        ("Critical", "#b71c1c",
         "Run a real OOD benchmark",
         "Testing the same pipeline on NaturalQuestions, TriviaQA, or MedQA would answer "
         "whether the OOD failure on TruthfulQA is specific to that benchmark or universal. "
         "Until this is done, the OOD claim cannot be rehabilitated."),
        ("Important", "#e65100",
         "Solve the layer selection problem",
         "A certificate that predicts AUROC accurately but cannot identify which layer to "
         "actually use for the probe has limited practical value. "
         "The J-plateau problem needs a new selection criterion — "
         "Fisher curvature (rate of change of J across layers) is one candidate."),
        ("Important", "#e65100",
         "Test MambaChat (instruction-tuned Mamba)",
         "The Mamba architecture transfer experiment failed because we used a base model. "
         "Repeating with MambaChat would give a genuine test of whether "
         "Fisher separability is architecture-independent."),
        ("Useful", "#1565c0",
         "Validate the 7B scale prediction",
         "Run Qwen 2.5 7B on HaluEval with n=400 samples — the same setup as Experiment 11. "
         "This is the only way to know whether the within-family scale curve holds beyond 3B. "
         "Requires approximately 24GB VRAM."),
    ]
    for priority, pc, title, desc in nexts:
        st.markdown(
            f'<div style="border-left:4px solid {pc};border-radius:0 10px 10px 0;'
            f'padding:0.8rem 1.1rem;margin-bottom:0.6rem;background:rgba(0,0,0,0.04);">'
            f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.35rem;">'
            f'<span style="font-size:0.72rem;font-weight:800;color:{pc};background:rgba(0,0,0,0.08);'
            f'padding:0.15rem 0.6rem;border-radius:20px;letter-spacing:0.06em;">{priority}</span>'
            f'<span style="font-size:0.92rem;font-weight:700;">{title}</span></div>'
            f'<div style="font-size:0.83rem;line-height:1.65;opacity:0.85;">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

PAGES = {
    # ── Summary ─────────────────────────────────────────────────────────────
    "Overview":                        page_overview,
    # ── Core Experiments ────────────────────────────────────────────────────
    "Exp 01 — Fisher Certificates":    page_fisher,
    "Exp 02 + 11 — Scale Curve":       page_scale_curve,
    "Exp 03 — Certificate Validation": page_validation,
    "Exp 04 — Mamba Transfer":         page_mamba,
    "Exp 05 — Depth Fraction":         page_depth,
    "Exp 06 — Boundary Conditions":    page_boundary,
    "Exp 07 — Judge Re-labeling":      page_judge,
    # ── Extended Experiments ────────────────────────────────────────────────
    "Exp 08 — OT Certificate":         page_ot,
    "Exp 09 — Spectral Transition":    page_spectral,
    "Exp 10 — Conformal Coverage":     page_conformal,
    # ── Honest Limits ───────────────────────────────────────────────────────
    "OOD Failure (TruthfulQA)":        page_ood,
    "Honest Assessment":               page_honest,
}

SECTIONS = {
    "Summary":              ["Overview"],
    "Core Experiments":     [
        "Exp 01 — Fisher Certificates",
        "Exp 02 + 11 — Scale Curve",
        "Exp 03 — Certificate Validation",
        "Exp 04 — Mamba Transfer",
        "Exp 05 — Depth Fraction",
        "Exp 06 — Boundary Conditions",
        "Exp 07 — Judge Re-labeling",
    ],
    "Extended Experiments": [
        "Exp 08 — OT Certificate",
        "Exp 09 — Spectral Transition",
        "Exp 10 — Conformal Coverage",
    ],
    "Honest Limits": [
        "OOD Failure (TruthfulQA)",
        "Honest Assessment",
    ],
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.6rem 0 0.3rem;">
      <div style="font-size:1.35rem;font-weight:800;letter-spacing:-0.01em;color:#e3f2fd;">
        📐 GEOM-PROOF</div>
      <div style="font-size:0.72rem;opacity:0.65;color:#bbdefb;margin-top:0.1rem;">
        Geometric Certificates for Hallucination Detection</div>
      <div style="font-size:0.75rem;color:#bbdefb;margin-top:0.35rem;opacity:0.85;">
        Lakshmi Chakradhar Vijayarao</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(255,255,255,0.07);border-radius:10px;
    padding:0.75rem 0.9rem;margin:0.6rem 0;">
      <div style="font-size:0.70rem;font-weight:700;opacity:0.6;text-transform:uppercase;
      letter-spacing:0.09em;margin-bottom:0.5rem;">Models &amp; Dataset</div>
      <div style="font-size:0.80rem;color:#bbdefb;line-height:1.7;">
        <b>Models:</b> Qwen 2.5 3B · GPT-2 Medium · Mamba-130m<br>
        <b>Dataset:</b> HaluEval (2000 samples)<br>
        <b>OOD Test:</b> TruthfulQA (400 questions)<br>
        <b>Experiments:</b> 11 (Exp 01–11)
      </div>
    </div>
    <div style="background:rgba(255,255,255,0.07);border-radius:10px;
    padding:0.75rem 0.9rem;margin:0.4rem 0 0.8rem;">
      <div style="font-size:0.70rem;font-weight:700;opacity:0.6;text-transform:uppercase;
      letter-spacing:0.09em;margin-bottom:0.5rem;">Key Results</div>
      <div style="font-size:0.80rem;color:#bbdefb;line-height:1.7;">
        <b>Best AUROC (Qwen):</b> 0.9917 (L36)<br>
        <b>Best AUROC (GPT-2):</b> 0.9887 (L8)<br>
        <b>Fisher bound error:</b> ≤2.7% max / 0.93% mean<br>
        <b>OOD AUROC:</b> 0.485 (chance — fails OOD)<br>
        <b>Scale fit R²:</b> 0.9996 (within-family)
      </div>
    </div>
    """, unsafe_allow_html=True)

    sect_icons = {
        "Summary":              "📋",
        "Core Experiments":     "🔬",
        "Extended Experiments": "🧪",
        "Honest Limits":        "⚠",
    }
    for section, page_list in SECTIONS.items():
        icon = sect_icons.get(section, "•")
        st.markdown(
            f'<div style="font-size:0.70rem;font-weight:700;opacity:0.50;'
            f'text-transform:uppercase;letter-spacing:0.10em;margin:0.7rem 0 0.3rem;">'
            f'{icon} {section}</div>',
            unsafe_allow_html=True,
        )
        for pname in page_list:
            if st.button(pname, key=f"nav_{pname}", use_container_width=True):
                st.session_state["page"] = pname

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.70rem;opacity:0.45;text-align:center;padding:0.3rem 0;">
      Lakshmi Chakradhar Vijayarao · GEOM-PROOF · 2026
    </div>
    """, unsafe_allow_html=True)

# ── Route to active page ───────────────────────────────────────────────────────
active = st.session_state.get("page", "Overview")
if active not in PAGES:
    active = "Overview"
PAGES[active]()

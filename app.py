"""
app.py  –  Neural Network Explorer  (Streamlit frontend)
=========================================================
Run with:   streamlit run app.py
"""

import io
import sys
import os
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── path setup so backend imports work from any cwd ──────────────
sys.path.insert(0, os.path.dirname(__file__))
from backend.data    import load_ann_data, load_cnn_data, load_rnn_data
from backend.models  import build_ann, build_cnn, build_rnn, get_model_summary, count_params
from backend.trainer import train_model, evaluate_classifier, evaluate_regressor

# ══════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Neural Network Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Minimal CSS tweaks ────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
      padding: 8px 20px; border-radius: 6px 6px 0 0;
      font-weight: 600;
  }
  div[data-testid="metric-container"] {
      background: #f8f9fa; border-radius: 8px; padding: 12px;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════
_KEYS = [
    "ann_history", "ann_eval", "ann_model", "ann_meta",
    "cnn_history", "cnn_eval", "cnn_model", "cnn_meta",
    "rnn_history", "rnn_eval", "rnn_model", "rnn_meta",
    "rnn_t_full",  "rnn_series",
]
for _k in _KEYS:
    if _k not in st.session_state:
        st.session_state[_k] = None


# ══════════════════════════════════════════════════════════════════
# Shared plotting helpers
# ══════════════════════════════════════════════════════════════════

COLORS = dict(train="#4C72B0", val="#DD8452", ref="#55A868")

def _smooth(arr, alpha=0.15):
    """Exponential moving average for smoother curves."""
    s, out = arr[0], [arr[0]]
    for v in arr[1:]:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out


def plot_history(history: dict, is_regression: bool = False) -> go.Figure:
    """Loss + secondary-metric subplot from a Keras history dict."""
    if is_regression:
        p_key, s_key = "loss", "mae"
        p_title, s_title = "Loss (MSE)", "MAE"
    else:
        p_key, s_key = "loss", "accuracy"
        p_title, s_title = "Cross-Entropy Loss", "Accuracy"

    epochs = list(range(1, len(history[p_key]) + 1))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(p_title, s_title),
                        horizontal_spacing=0.12)

    for col, key, title in [(1, p_key, p_title), (2, s_key, s_title)]:
        if key not in history:
            continue
        raw_tr  = history[key]
        raw_val = history.get(f"val_{key}", [])
        fig.add_trace(go.Scatter(
            x=epochs, y=raw_tr, mode="lines",
            name=f"Train {title}",
            line=dict(color=COLORS["train"], width=1.5, dash="dot"),
            opacity=0.45, showlegend=(col == 1),
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=epochs, y=_smooth(raw_tr), mode="lines",
            name=f"Train {title} (smoothed)",
            line=dict(color=COLORS["train"], width=2.5),
            showlegend=(col == 1),
        ), row=1, col=col)
        if raw_val:
            fig.add_trace(go.Scatter(
                x=epochs, y=raw_val, mode="lines",
                name=f"Val {title}",
                line=dict(color=COLORS["val"], width=1.5, dash="dot"),
                opacity=0.45, showlegend=(col == 1),
            ), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=epochs, y=_smooth(raw_val), mode="lines",
                name=f"Val {title} (smoothed)",
                line=dict(color=COLORS["val"], width=2.5),
                showlegend=(col == 1),
            ), row=1, col=col)

    fig.update_layout(
        height=340, margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", y=-0.18),
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Epoch")
    return fig


def plot_confusion_matrix(preds, truth, class_names=None) -> go.Figure:
    n = max(preds.max(), truth.max()) + 1
    cm = np.zeros((n, n), dtype=int)
    for p, t in zip(preds, truth):
        cm[t, p] += 1
    labels = class_names if class_names else [str(i) for i in range(n)]
    # Normalise for colour, keep raw counts as text
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig = go.Figure(go.Heatmap(
        z=cm_norm, x=labels, y=labels,
        text=cm, texttemplate="%{text}",
        colorscale="Blues", showscale=False,
    ))
    fig.update_layout(
        title="Confusion Matrix (counts)",
        xaxis_title="Predicted", yaxis_title="Actual",
        height=360, margin=dict(t=40, b=20, l=20, r=20),
        template="plotly_white",
    )
    return fig


def plot_rnn_predictions(t_full, series, X_te, y_te, preds, seq_length) -> go.Figure:
    n_train = len(series) - len(y_te) - seq_length
    t_te = t_full[n_train + seq_length : n_train + seq_length + len(y_te)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_full[:n_train + seq_length], y=series[:n_train + seq_length],
        mode="lines", name="Train signal",
        line=dict(color=COLORS["train"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=t_te, y=y_te, mode="lines", name="True (test)",
        line=dict(color=COLORS["val"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=t_te, y=preds, mode="lines", name="Predicted",
        line=dict(color=COLORS["ref"], width=2, dash="dash"),
    ))
    fig.update_layout(
        title="Signal vs Predictions",
        xaxis_title="Time", yaxis_title="Value",
        height=320, margin=dict(t=40, b=20, l=20, r=20),
        template="plotly_white",
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


def plot_residuals(residuals) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Residual Distribution", "Residuals over Test Samples"))
    fig.add_trace(go.Histogram(x=residuals, nbinsx=30, name="Residuals",
                               marker_color=COLORS["train"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(residuals))), y=residuals,
                             mode="lines", name="Residual",
                             line=dict(color=COLORS["val"], width=1.2)),
                  row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="grey", row=1, col=2)
    fig.update_layout(height=280, margin=dict(t=35, b=15, l=15, r=15),
                      showlegend=False, template="plotly_white")
    return fig


def lr_schedule_info(name: str) -> str:
    info = {
        "none":              "Constant learning rate throughout training.",
        "step_decay":        "Halves LR every ⅓ of total epochs.",
        "cosine":            "Smooth cosine annealing from base LR → ~0.",
        "reduce_on_plateau": "Cuts LR by ½ when val_loss stagnates.",
        "warmup_cosine":     "Linear warm-up for first 10 %, then cosine decay.",
    }
    return info.get(name, "")


# ══════════════════════════════════════════════════════════════════
# Reusable param widgets (grouped into expanders)
# ══════════════════════════════════════════════════════════════════

def _opt_widgets(prefix: str):
    """Shared optimizer + LR-schedule widgets. Returns a dict of values."""
    cfg = {}
    with st.expander("⚡ Optimisation", expanded=False):
        cfg["optimizer"] = st.selectbox(
            "Optimizer", ["Adam", "AdamW", "SGD", "RMSprop", "Adagrad", "Adadelta"],
            key=f"{prefix}_opt",
            help="Adam: fast, general-purpose.\n"
                 "AdamW: Adam + decoupled weight decay.\n"
                 "SGD: simple, often best with schedules.\n"
                 "RMSprop: good for RNNs & noisy gradients.\n"
                 "Adagrad: sparse/high-dimensional data.\n"
                 "Adadelta: no manual LR needed.",
        )
        cfg["lr"] = st.select_slider(
            "Learning Rate",
            [1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2, 0.1],
            value=1e-3, key=f"{prefix}_lr",
        )

        opt = cfg["optimizer"]
        if opt == "SGD":
            cfg["momentum"] = st.slider("Momentum", 0.0, 0.99, 0.9, 0.01,
                                        key=f"{prefix}_mom")
            cfg["nesterov"] = st.checkbox("Nesterov momentum", True,
                                          key=f"{prefix}_nest")
        else:
            cfg["momentum"] = 0.9; cfg["nesterov"] = False

        if opt in ("Adam", "AdamW"):
            c1, c2 = st.columns(2)
            cfg["beta_1"] = c1.slider("β₁", 0.80, 0.999, 0.9,   0.001, key=f"{prefix}_b1")
            cfg["beta_2"] = c2.slider("β₂", 0.90, 0.9999, 0.999, 0.0001,
                                      format="%.4f", key=f"{prefix}_b2")
        else:
            cfg["beta_1"] = 0.9; cfg["beta_2"] = 0.999

        if opt == "AdamW":
            cfg["weight_decay"] = st.select_slider(
                "Weight Decay", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], value=1e-4,
                key=f"{prefix}_wd",
            )
        else:
            cfg["weight_decay"] = 0.0

        if opt in ("RMSprop", "Adadelta"):
            cfg["rho"] = st.slider("ρ (decay)", 0.7, 0.99, 0.9, 0.01, key=f"{prefix}_rho")
        else:
            cfg["rho"] = 0.9

        cfg["lr_schedule"] = st.selectbox(
            "LR Schedule", ["none", "step_decay", "cosine",
                            "reduce_on_plateau", "warmup_cosine"],
            key=f"{prefix}_lrsched",
        )
        st.caption(lr_schedule_info(cfg["lr_schedule"]))

        cfg["grad_clip"] = st.checkbox("Gradient Clipping", False, key=f"{prefix}_gc")
        if cfg["grad_clip"]:
            cfg["clip_value"] = st.slider("Clip Value", 0.1, 5.0, 1.0, 0.1,
                                          key=f"{prefix}_cv")
        else:
            cfg["clip_value"] = 1.0

        cfg["early_stop"] = st.checkbox("Early Stopping", False, key=f"{prefix}_es")
        if cfg["early_stop"]:
            cfg["patience"] = st.slider("Patience (epochs)", 3, 30, 10,
                                        key=f"{prefix}_pat")
        else:
            cfg["patience"] = 10
    return cfg


def _train_widgets(prefix: str, default_epochs: int = 50):
    cfg = {}
    with st.expander("🎯 Training Hyperparameters", expanded=True):
        cfg["epochs"]     = st.slider("Epochs", 10, 300, default_epochs, key=f"{prefix}_ep")
        cfg["batch_size"] = st.select_slider("Batch Size", [8, 16, 32, 64, 128, 256],
                                              value=32, key=f"{prefix}_bsz")
    return cfg


def _reg_widgets(prefix: str, l1: bool = True):
    cfg = {}
    with st.expander("🛡️ Regularisation", expanded=False):
        cfg["dropout"] = st.slider("Dropout Rate", 0.0, 0.6, 0.0, 0.05,
                                   key=f"{prefix}_drop",
                                   help="Randomly zeros neurons during training "
                                        "to prevent co-adaptation.")
        if l1:
            cfg["l1"] = st.select_slider("L1 λ", [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
                                          value=0.0, key=f"{prefix}_l1",
                                          help="Encourages sparse weight matrices.")
        else:
            cfg["l1"] = 0.0
        cfg["l2"] = st.select_slider("L2 λ", [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
                                      value=0.0, key=f"{prefix}_l2",
                                      help="Keeps weights small; reduces overfitting.")
        cfg["batch_norm"] = st.checkbox("Batch Normalisation", False, key=f"{prefix}_bn",
                                        help="Normalises layer inputs; often allows "
                                             "higher learning rates.")
    return cfg


def _show_model_info(model, prefix: str):
    p = count_params(model)
    c1, c2, c3 = st.columns(3)
    c1.metric("Trainable params", f"{p['trainable']:,}")
    c2.metric("Non-trainable", f"{p['non_trainable']:,}")
    c3.metric("Total", f"{p['total']:,}")
    with st.expander("📐 Model Architecture", expanded=False):
        st.code(get_model_summary(model), language="text")


# ══════════════════════════════════════════════════════════════════
# Page title
# ══════════════════════════════════════════════════════════════════
st.title("🧠 Neural Network Explorer")
st.markdown(
    "Compare **ANN · CNN · RNN** architectures. "
    "Tune every hyperparameter, regularisation knob, and optimiser — "
    "then watch how they affect training dynamics."
)

ann_tab, cnn_tab, rnn_tab = st.tabs([
    "🔴  ANN — Feedforward",
    "🟢  CNN — Convolutional",
    "🔵  RNN — Recurrent",
])


# ══════════════════════════════════════════════════════════════════
# ① ANN TAB
# ══════════════════════════════════════════════════════════════════
with ann_tab:
    st.markdown("### Artificial Neural Network")
    st.caption(
        "Dense feedforward network for tabular classification. "
        "Great for understanding depth, width, regularisation, and optimisers."
    )

    left, right = st.columns([1, 2], gap="large")

    with left:
        with st.expander("📊 Dataset", expanded=True):
            ann_ds = st.selectbox(
                "Dataset",
                ["Iris", "Breast Cancer", "Wine", "Digits"],
                key="ann_ds",
                help="All from scikit-learn. Digits = 64 features (8×8 pixel values).",
            )
            ann_test_pct = st.slider("Test split %", 10, 40, 20, key="ann_split")

        with st.expander("🏗️ Architecture", expanded=True):
            ann_layers  = st.slider("Hidden Layers", 1, 6, 2, key="ann_hl",
                                    help="Network depth: more layers → more abstract features.")
            ann_units   = st.select_slider("Units / Layer", [8,16,32,64,128,256,512],
                                            value=64, key="ann_u",
                                            help="Network width: more units → more capacity.")
            ann_act     = st.selectbox("Activation", ["relu","elu","tanh","sigmoid","gelu"],
                                        key="ann_act")
            ann_skip    = st.checkbox("Residual Skip Connections", False, key="ann_skip",
                                      help="Adds the input to the output of each block "
                                           "(requires all layers to have the same width).")

        tr_cfg  = _train_widgets("ann")
        reg_cfg = _reg_widgets("ann", l1=True)
        opt_cfg = _opt_widgets("ann")

        train_btn = st.button("🚀 Train ANN", type="primary",
                              use_container_width=True, key="ann_train_btn")

    with right:
        if train_btn:
            with st.spinner("Loading data & training …"):
                X_tr, X_te, y_tr, y_te, n_cls, meta = load_ann_data(
                    ann_ds, ann_test_pct / 100
                )
                model = build_ann(
                    input_dim=X_tr.shape[1],
                    n_classes=n_cls,
                    hidden_layers=ann_layers,
                    units=ann_units,
                    activation=ann_act,
                    dropout_rate=reg_cfg["dropout"],
                    l1_lambda=reg_cfg["l1"],
                    l2_lambda=reg_cfg["l2"],
                    use_batch_norm=reg_cfg["batch_norm"],
                    use_skip_conn=ann_skip,
                )
                hist = train_model(
                    model, X_tr, y_tr, X_te, y_te,
                    optimizer_name=opt_cfg["optimizer"],
                    learning_rate=opt_cfg["lr"],
                    epochs=tr_cfg["epochs"],
                    batch_size=tr_cfg["batch_size"],
                    is_regression=False,
                    momentum=opt_cfg["momentum"],
                    nesterov=opt_cfg["nesterov"],
                    beta_1=opt_cfg["beta_1"],
                    beta_2=opt_cfg["beta_2"],
                    weight_decay=opt_cfg["weight_decay"],
                    rho=opt_cfg["rho"],
                    lr_schedule=opt_cfg["lr_schedule"],
                    early_stopping=opt_cfg["early_stop"],
                    patience=opt_cfg["patience"],
                    use_gradient_clip=opt_cfg["grad_clip"],
                    clip_value=opt_cfg["clip_value"],
                )
                ev = evaluate_classifier(model, X_te, y_te)
                st.session_state.ann_history = hist.history
                st.session_state.ann_eval    = ev
                st.session_state.ann_model   = model
                st.session_state.ann_meta    = meta

        if st.session_state.ann_history:
            h  = st.session_state.ann_history
            ev = st.session_state.ann_eval
            meta = st.session_state.ann_meta

            # ── KPI row ──────────────────────────────────────────
            st.markdown("#### Results")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Test Accuracy",  f"{ev['test_accuracy']*100:.1f} %")
            k2.metric("Test Loss",      f"{ev['test_loss']:.4f}")
            best_val = max(h.get("val_accuracy", [0]))
            k3.metric("Best Val Acc",   f"{best_val*100:.1f} %")
            k4.metric("Epochs trained", len(h["loss"]))

            st.plotly_chart(plot_history(h, is_regression=False),
                            use_container_width=True)

            col_a, col_b = st.columns([1, 1])
            with col_a:
                class_names = meta.get("class_names")
                st.plotly_chart(
                    plot_confusion_matrix(ev["predictions"],
                                         ev["ground_truth"],
                                         class_names),
                    use_container_width=True,
                )
            with col_b:
                _show_model_info(st.session_state.ann_model, "ann")
                st.markdown("**Dataset info**")
                st.json({k: v for k, v in meta.items()
                         if k not in ("feature_names", "class_names")},
                        expanded=False)
        else:
            st.info("Configure the parameters on the left, then press **Train ANN**.")


# ══════════════════════════════════════════════════════════════════
# ② CNN TAB
# ══════════════════════════════════════════════════════════════════
with cnn_tab:
    st.markdown("### Convolutional Neural Network")
    st.caption(
        "Spatial feature extraction on 8×8 digit images (sklearn Digits). "
        "Experiment with filter count, pooling, kernel size, and regularisation."
    )

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("**Dataset**: sklearn `load_digits`  —  1797 samples, 10 classes, 8×8 px")

        with st.expander("🏗️ Architecture", expanded=True):
            cnn_filters  = st.select_slider("Base Filters",  [8, 16, 32, 64], value=32,
                                             key="cnn_filt",
                                             help="Filters double each block (base, 2×, 4×, …).")
            cnn_ksize    = st.radio("Kernel Size", [2, 3], index=1,
                                    horizontal=True, key="cnn_ks")
            cnn_blocks   = st.slider("Conv Blocks", 1, 2, 2, key="cnn_blk",
                                     help="Each block = Conv→BN→ReLU→MaxPool.")
            cnn_dense    = st.select_slider("FC Head Units", [32, 64, 128, 256],
                                             value=128, key="cnn_fc")
            cnn_act      = st.selectbox("Activation", ["relu", "elu", "gelu"],
                                         key="cnn_act")
            cnn_gap      = st.checkbox("Global Avg Pooling (replace Flatten)", False,
                                        key="cnn_gap",
                                        help="Reduces spatial dims to 1×1 before FC; "
                                             "fewer params, often better generalisation.")

        tr_cfg  = _train_widgets("cnn")
        reg_cfg = _reg_widgets("cnn", l1=False)
        opt_cfg = _opt_widgets("cnn")

        train_btn_cnn = st.button("🚀 Train CNN", type="primary",
                                   use_container_width=True, key="cnn_train_btn")

    with right:
        if train_btn_cnn:
            with st.spinner("Loading data & training …"):
                X_tr, X_te, y_tr, y_te, meta = load_cnn_data(0.2)
                model = build_cnn(
                    n_classes=10,
                    n_filters=cnn_filters,
                    kernel_size=cnn_ksize,
                    n_conv_blocks=cnn_blocks,
                    dense_units=cnn_dense,
                    activation=cnn_act,
                    dropout_rate=reg_cfg["dropout"],
                    l2_lambda=reg_cfg["l2"],
                    use_batch_norm=reg_cfg["batch_norm"],
                    use_global_avg=cnn_gap,
                )
                hist = train_model(
                    model, X_tr, y_tr, X_te, y_te,
                    optimizer_name=opt_cfg["optimizer"],
                    learning_rate=opt_cfg["lr"],
                    epochs=tr_cfg["epochs"],
                    batch_size=tr_cfg["batch_size"],
                    is_regression=False,
                    momentum=opt_cfg["momentum"],
                    nesterov=opt_cfg["nesterov"],
                    beta_1=opt_cfg["beta_1"],
                    beta_2=opt_cfg["beta_2"],
                    weight_decay=opt_cfg["weight_decay"],
                    rho=opt_cfg["rho"],
                    lr_schedule=opt_cfg["lr_schedule"],
                    early_stopping=opt_cfg["early_stop"],
                    patience=opt_cfg["patience"],
                    use_gradient_clip=opt_cfg["grad_clip"],
                    clip_value=opt_cfg["clip_value"],
                )
                ev = evaluate_classifier(model, X_te, y_te)
                st.session_state.cnn_history = hist.history
                st.session_state.cnn_eval    = ev
                st.session_state.cnn_model   = model
                st.session_state.cnn_meta    = meta

        if st.session_state.cnn_history:
            h  = st.session_state.cnn_history
            ev = st.session_state.cnn_eval

            st.markdown("#### Results")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Test Accuracy",  f"{ev['test_accuracy']*100:.1f} %")
            k2.metric("Test Loss",      f"{ev['test_loss']:.4f}")
            k3.metric("Best Val Acc",   f"{max(h.get('val_accuracy',[0]))*100:.1f} %")
            k4.metric("Epochs trained", len(h["loss"]))

            st.plotly_chart(plot_history(h, is_regression=False),
                            use_container_width=True)

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.plotly_chart(
                    plot_confusion_matrix(ev["predictions"], ev["ground_truth"],
                                         [str(i) for i in range(10)]),
                    use_container_width=True,
                )
            with col_b:
                _show_model_info(st.session_state.cnn_model, "cnn")
                # Show a sample digit grid
                st.markdown("**Sample 8×8 Digit Images**")
                import matplotlib.pyplot as plt
                from sklearn import datasets as skd
                dg = skd.load_digits()
                fig_img, axes = plt.subplots(2, 5, figsize=(6, 2.4))
                for ax, img, label in zip(axes.flat, dg.images[:10], dg.target[:10]):
                    ax.imshow(img, cmap="gray_r"); ax.set_title(label, fontsize=9)
                    ax.axis("off")
                plt.tight_layout(pad=0.3)
                st.pyplot(fig_img, use_container_width=False)
        else:
            st.info("Configure the parameters on the left, then press **Train CNN**.")


# ══════════════════════════════════════════════════════════════════
# ③ RNN TAB
# ══════════════════════════════════════════════════════════════════
with rnn_tab:
    st.markdown("### Recurrent Neural Network")
    st.caption(
        "Next-step regression on a synthetic multi-frequency sine wave. "
        "Compare LSTM vs GRU vs SimpleRNN; explore depth, bidirectionality, "
        "recurrent dropout, and more."
    )

    left, right = st.columns([1, 2], gap="large")

    with left:
        with st.expander("📊 Task & Data", expanded=True):
            rnn_seq  = st.slider("Sequence Length", 10, 100, 40, key="rnn_seq",
                                  help="How many past time steps the model sees.")
            rnn_ns   = st.select_slider("# Samples", [500, 800, 1200, 2000],
                                         value=1200, key="rnn_ns")
            rnn_nz   = st.select_slider("Noise Level", [0.02, 0.05, 0.08, 0.15, 0.25],
                                         value=0.08, key="rnn_nz",
                                         help="Standard deviation of Gaussian noise.")

        with st.expander("🏗️ Architecture", expanded=True):
            rnn_type  = st.radio("Cell Type", ["LSTM", "GRU", "SimpleRNN"],
                                  horizontal=True, key="rnn_type",
                                  help="LSTM: gates for long-range dependencies.\n"
                                       "GRU: lighter LSTM variant.\n"
                                       "SimpleRNN: vanilla; struggles with long sequences.")
            rnn_units = st.select_slider("RNN Units", [8, 16, 32, 64, 128],
                                          value=64, key="rnn_units")
            rnn_depth = st.slider("Stacked Layers", 1, 3, 1, key="rnn_depth")
            rnn_bidir = st.checkbox("Bidirectional", False, key="rnn_bidir",
                                    help="Processes sequence both forwards and backwards; "
                                         "doubles effective unit count.")
            rnn_dense = st.select_slider("Readout FC Units", [8, 16, 32, 64],
                                          value=32, key="rnn_fc")
            rnn_act   = st.selectbox("FC Activation", ["relu", "tanh", "elu"],
                                      key="rnn_act")

        tr_cfg  = _train_widgets("rnn", default_epochs=30)
        with st.expander("🛡️ Regularisation", expanded=False):
            rnn_dropout   = st.slider("Input Dropout",      0.0, 0.5, 0.0, 0.05, key="rnn_drop")
            rnn_rec_drop  = st.slider("Recurrent Dropout",  0.0, 0.4, 0.0, 0.05, key="rnn_rdrop",
                                       help="Dropout on the hidden-to-hidden connections "
                                            "(LSTM/GRU only).")
            rnn_l2        = st.select_slider("L2 λ", [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
                                              value=0.0, key="rnn_l2")
            rnn_bn        = st.checkbox("Batch Norm on RNN outputs", False, key="rnn_bn",
                                         help="Applied between stacked RNN layers "
                                              "(sequence dimension).")

        opt_cfg = _opt_widgets("rnn")

        train_btn_rnn = st.button("🚀 Train RNN", type="primary",
                                   use_container_width=True, key="rnn_train_btn")

    with right:
        if train_btn_rnn:
            with st.spinner("Generating data & training …"):
                X_tr, X_te, y_tr, y_te, t_full, series, meta = load_rnn_data(
                    seq_length=rnn_seq,
                    n_samples=rnn_ns,
                    noise_level=rnn_nz,
                )
                model = build_rnn(
                    seq_length=rnn_seq,
                    rnn_type=rnn_type,
                    rnn_units=rnn_units,
                    n_rnn_layers=rnn_depth,
                    dense_units=rnn_dense,
                    activation=rnn_act,
                    dropout_rate=rnn_dropout,
                    recurrent_drop=rnn_rec_drop,
                    l2_lambda=rnn_l2,
                    use_batch_norm=rnn_bn,
                    bidirectional=rnn_bidir,
                )
                hist = train_model(
                    model, X_tr, y_tr, X_te, y_te,
                    optimizer_name=opt_cfg["optimizer"],
                    learning_rate=opt_cfg["lr"],
                    epochs=tr_cfg["epochs"],
                    batch_size=tr_cfg["batch_size"],
                    is_regression=True,
                    momentum=opt_cfg["momentum"],
                    nesterov=opt_cfg["nesterov"],
                    beta_1=opt_cfg["beta_1"],
                    beta_2=opt_cfg["beta_2"],
                    weight_decay=opt_cfg["weight_decay"],
                    rho=opt_cfg["rho"],
                    lr_schedule=opt_cfg["lr_schedule"],
                    early_stopping=opt_cfg["early_stop"],
                    patience=opt_cfg["patience"],
                    use_gradient_clip=opt_cfg["grad_clip"],
                    clip_value=opt_cfg["clip_value"],
                )
                ev = evaluate_regressor(model, X_te, y_te)
                st.session_state.rnn_history = hist.history
                st.session_state.rnn_eval    = ev
                st.session_state.rnn_model   = model
                st.session_state.rnn_meta    = meta
                st.session_state.rnn_t_full  = t_full
                st.session_state.rnn_series  = series

        if st.session_state.rnn_history:
            h  = st.session_state.rnn_history
            ev = st.session_state.rnn_eval
            meta = st.session_state.rnn_meta

            st.markdown("#### Results")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Test RMSE", f"{ev['test_rmse']:.4f}")
            k2.metric("Test MAE",  f"{ev['test_mae']:.4f}")
            k3.metric("Test MSE",  f"{ev['test_mse']:.5f}")
            k4.metric("Epochs trained", len(h["loss"]))

            st.plotly_chart(plot_history(h, is_regression=True),
                            use_container_width=True)

            st.plotly_chart(
                plot_rnn_predictions(
                    st.session_state.rnn_t_full,
                    st.session_state.rnn_series,
                    None,
                    ev["ground_truth"],
                    ev["predictions"],
                    meta["seq_length"],
                ),
                use_container_width=True,
            )

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.plotly_chart(plot_residuals(ev["residuals"]),
                                use_container_width=True)
            with col_b:
                _show_model_info(st.session_state.rnn_model, "rnn")
                st.markdown("**Dataset info**")
                st.json(meta, expanded=False)
        else:
            st.info("Configure the parameters on the left, then press **Train RNN**.")


# ══════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "Neural Network Explorer · Built with Streamlit + TensorFlow/Keras + scikit-learn  "
    "· All three models train on CPU; reduce epochs or batch size if slow."
)

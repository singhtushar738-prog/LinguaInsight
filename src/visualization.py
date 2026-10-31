"""
Chart builders.

* Plotly figures (interactive) are used by the Streamlit pages.
* Matplotlib figures (static PNG) are written into ``reports/`` by the
  exploration and evaluation scripts.

All charts take plain DataFrames / arrays so they contain no model logic.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend (no display needed)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from config import settings  # noqa: E402

SENTIMENT_COLORS = {"negative": "#D64545", "neutral": "#9A9CA5", "positive": "#2E9E5B"}
MODEL_COLORS = {
    settings.MODEL_LR: "#4C78A8", settings.MODEL_SVM: "#72B7B2", settings.MODEL_PER_LANG: "#9ECAE9",
    settings.MODEL_XLMR_FT: "#F58518", settings.MODEL_XLMR_PRE: "#E45756",
}
SHORT_MODEL_NAMES = {
    settings.MODEL_LR: "TF-IDF + LR", settings.MODEL_SVM: "TF-IDF + SVM",
    settings.MODEL_PER_LANG: "TF-IDF + LR (per-language)", settings.MODEL_XLMR_FT: "XLM-R fine-tuned",
    settings.MODEL_XLMR_PRE: "XLM-R pretrained sentiment",
}


def _layout(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color="#17243e")) if title else None,
                      height=height, margin=dict(l=18, r=18, t=58 if title else 24, b=24),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                  font=dict(size=11), bgcolor="rgba(255,255,255,0)"),
                      font=dict(family="DM Sans, sans-serif", size=12, color="#53647f"),
                      paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
                      hoverlabel=dict(bgcolor="#14213b", font_color="#ffffff"))
    fig.update_xaxes(showgrid=True, gridcolor="#e9eef6", zerolinecolor="#cbd5e3", automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="#e9eef6", zerolinecolor="#cbd5e3", automargin=True)
    return fig


# --------------------------------------------------------------------------
# Dataset / results distributions (Plotly)
# --------------------------------------------------------------------------
def sentiment_distribution(df: pd.DataFrame, col: str = "sentiment", title: str = "Sentiment distribution"):
    counts = df[col].value_counts().reindex(settings.LABELS, fill_value=0).reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.pie(counts, names="sentiment", values="count", hole=0.45, color="sentiment",
                 color_discrete_map=SENTIMENT_COLORS)
    fig.update_traces(textinfo="percent+value", textposition="inside", insidetextorientation="horizontal",
                      marker=dict(line=dict(color="#ffffff", width=3)), sort=False,
                      hovertemplate="%{label}: %{value:,} reviews (%{percent})<extra></extra>")
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center",
                                                   yanchor="top", font=dict(size=11)))
    return _layout(fig, title, height=360)


def language_distribution(df: pd.DataFrame, lang_col: str = "language_name", title: str = "Reviews per language"):
    counts = df[lang_col].value_counts().reset_index()
    counts.columns = ["language", "count"]
    fig = px.bar(counts, x="count", y="language", orientation="h", text="count",
                 color_discrete_sequence=["#4C78A8"])
    fig.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=11)), xaxis_title="Number of reviews",
                      yaxis_title=None, bargap=0.28)
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0,
                      hovertemplate="%{y}: %{x:,} reviews<extra></extra>")
    return _layout(fig, title, height=max(320, 32 * len(counts) + 100))


def sentiment_by_language(df: pd.DataFrame, lang_col: str = "language_name", col: str = "sentiment",
                          normalize: bool = True, title: str = "Sentiment distribution by language"):
    table = pd.crosstab(df[lang_col], df[col], normalize="index" if normalize else False)
    table = table.reindex(columns=settings.LABELS, fill_value=0)
    long = table.reset_index().melt(id_vars=lang_col, var_name="sentiment", value_name="value")
    fig = px.bar(long, y=lang_col, x="value", color="sentiment", orientation="h",
                 color_discrete_map=SENTIMENT_COLORS, category_orders={"sentiment": settings.LABELS},
                 text=long["value"].map(lambda v: f"{v:.0%}" if normalize else f"{int(v)}"))
    fig.update_layout(barmode="stack", xaxis_title="Share of reviews" if normalize else "Number of reviews",
                      yaxis_title=None, xaxis_tickformat=".0%" if normalize else None,
                      yaxis=dict(autorange="reversed", tickfont=dict(size=11)), bargap=0.24)
    fig.update_traces(hovertemplate="%{y} · %{fullData.name}: %{x:.1%}<extra></extra>" if normalize
                      else "%{y} · %{fullData.name}: %{x:,} reviews<extra></extra>")
    return _layout(fig, title, height=max(340, 36 * len(table) + 110))


def positive_vs_negative(df: pd.DataFrame, lang_col: str = "language_name", col: str = "sentiment",
                         title: str = "Positive vs negative reviews across languages"):
    table = pd.crosstab(df[lang_col], df[col]).reindex(columns=settings.LABELS, fill_value=0)
    table = table.assign(net=table["positive"] - table["negative"]).sort_values("net")
    fig = go.Figure()
    fig.add_bar(name="Negative", y=table.index, x=-table["negative"], orientation="h",
                marker_color=SENTIMENT_COLORS["negative"], customdata=table["negative"],
                hovertemplate="%{y}: %{customdata:,} negative reviews<extra></extra>")
    fig.add_bar(name="Positive", y=table.index, x=table["positive"], orientation="h",
                marker_color=SENTIMENT_COLORS["positive"],
                hovertemplate="%{y}: %{x:,} positive reviews<extra></extra>")
    fig.update_layout(barmode="relative", xaxis_title="Reviews  ·  negative ← 0 → positive",
                      yaxis_title=None, yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                      xaxis=dict(tickformat=",d", zeroline=True, zerolinewidth=2, zerolinecolor="#8796ad"),
                      bargap=0.25)
    return _layout(fig, title, height=max(360, 34 * len(table) + 110))


def length_distribution(df: pd.DataFrame, length_col: str = "length_chars", color_col: str | None = "sentiment",
                        title: str = "Review length distribution (characters)", clip_quantile: float = 0.98):
    data = df.copy()
    cap = data[length_col].quantile(clip_quantile)
    data = data[data[length_col] <= cap]
    kwargs = dict(color=color_col, color_discrete_map=SENTIMENT_COLORS,
                  category_orders={color_col: settings.LABELS}) if color_col else {}
    fig = px.histogram(data, x=length_col, nbins=60, opacity=0.8, barmode="overlay", **kwargs)
    fig.update_layout(xaxis_title=f"Review length in characters · upper {100*(1-clip_quantile):.0f}% hidden",
                      yaxis_title="Number of reviews", bargap=0.08)
    fig.update_traces(hovertemplate="Length: %{x:.0f} characters<br>Reviews: %{y:,}<extra></extra>")
    return _layout(fig, title, height=360)


def length_by_language_box(df: pd.DataFrame, lang_col: str = "language_name", length_col: str = "length_chars",
                           title: str = "Review length by language"):
    fig = px.box(df, x=lang_col, y=length_col, points=False, color_discrete_sequence=["#4C78A8"])
    fig.update_yaxes(type="log", title="Characters (log scale)",
                     tickmode="array", tickvals=[1, 10, 100, 1000, 10000],
                     ticktext=["1", "10", "100", "1k", "10k"],
                     showgrid=True, gridcolor="#e9eef6")
    fig.update_xaxes(title=None, tickangle=-25, tickfont=dict(size=10))
    fig.update_traces(line_color="#4169f5", fillcolor="rgba(65,105,245,0.18)",
                      hovertemplate="%{x}<br>Median/range: %{y}<extra></extra>")
    return _layout(fig, title, height=360)


# --------------------------------------------------------------------------
# Predictions (Plotly)
# --------------------------------------------------------------------------
def probability_bars(probabilities: dict[str, float], title: str | None = None):
    labels = settings.LABELS
    vals = [probabilities.get(lab, 0.0) for lab in labels]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=[SENTIMENT_COLORS[l] for l in labels],
                           text=[f"{v:.1%}" for v in vals], textposition="auto"))
    fig.update_layout(xaxis=dict(range=[0, 1], tickformat=".0%"), yaxis=dict(autorange="reversed"))
    return _layout(fig, title, height=200)


def model_probability_comparison(rows: list[dict]):
    """rows: [{"model": name, "negative": p, "neutral": p, "positive": p}, ...]"""
    df = pd.DataFrame(rows).melt(id_vars="model", var_name="sentiment", value_name="probability")
    fig = px.bar(df, x="probability", y="model", color="sentiment", orientation="h", barmode="group",
                 color_discrete_map=SENTIMENT_COLORS, category_orders={"sentiment": settings.LABELS})
    fig.update_layout(xaxis=dict(range=[0, 1], tickformat=".0%"), yaxis_title=None)
    return _layout(fig, "Class probabilities by model", height=140 + 70 * len(rows))


def feature_contributions(features: list[dict], title: str):
    """features: [{"feature": str, "contribution": float}]"""
    if not features:
        return None
    df = pd.DataFrame(features)
    df["label"] = df["feature"].map(lambda f: repr(f) if f != f.strip() else f)
    df["color"] = np.where(df["contribution"] >= 0, "#2E9E5B", "#D64545")
    df = df.sort_values("contribution")
    fig = go.Figure(go.Bar(x=df["contribution"], y=df["label"], orientation="h", marker_color=df["color"]))
    fig.update_layout(xaxis_title="Contribution to class score", yaxis_title=None)
    return _layout(fig, title, height=max(260, 24 * len(df) + 80))


def token_importance_html(tokens: list[dict], text: str) -> str:
    """Colour each word by its occlusion importance (green supports, red opposes)."""
    import html

    if not tokens:
        return html.escape(text)
    scale = max(abs(t["importance"]) for t in tokens) or 1.0
    parts = []
    for t in tokens:
        v = t["importance"] / scale
        color = f"rgba(46,158,91,{abs(v):.2f})" if v >= 0 else f"rgba(214,69,69,{abs(v):.2f})"
        parts.append(f'<span title="{t["importance"]:+.3f}" style="background:{color};padding:2px 4px;'
                     f'margin:2px;border-radius:4px;display:inline-block">{html.escape(t["token"])}</span>')
    return " ".join(parts)


# --------------------------------------------------------------------------
# Evaluation (Plotly)
# --------------------------------------------------------------------------
def confusion_matrix_plotly(cm: np.ndarray, labels: list[str], title: str, normalize: bool = True):
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    text = [[f"{int(cm[i, j])}<br>({shown[i, j]:.0%})" if normalize else f"{int(cm[i, j])}"
             for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    fig = go.Figure(go.Heatmap(z=shown, x=labels, y=labels, text=text, texttemplate="%{text}",
                               colorscale="Blues", showscale=False, zmin=0, zmax=1 if normalize else None))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="True", yaxis=dict(autorange="reversed"))
    return _layout(fig, title, height=360)


def model_comparison_chart(comparison: pd.DataFrame, metrics=("accuracy", "f1_macro", "f1_weighted",
                                                               "precision_macro", "recall_macro")):
    df = comparison.copy()
    df["model_short"] = df["model"].map(SHORT_MODEL_NAMES).fillna(df["model"])
    long = df.melt(id_vars=["model", "model_short"], value_vars=list(metrics), var_name="metric", value_name="score")
    fig = px.bar(long, x="metric", y="score", color="model_short", barmode="group", text=long["score"].round(3),
                 color_discrete_sequence=[MODEL_COLORS.get(m, "#888") for m in df["model"]])
    fig.update_layout(yaxis=dict(range=[0, 1]), xaxis_title=None, yaxis_title="Score", legend_title=None)
    return _layout(fig, "Model comparison on the test set", height=420)


def per_language_chart(per_lang: pd.DataFrame, metric: str = "f1_macro"):
    df = per_lang.copy()
    df["model_short"] = df["model"].map(SHORT_MODEL_NAMES).fillna(df["model"])
    df["language_name"] = df["language"].map(settings.language_name)
    order = list(dict.fromkeys(df["model"]))
    fig = px.bar(df, x="language_name", y=metric, color="model_short", barmode="group",
                 color_discrete_sequence=[MODEL_COLORS.get(m, "#888") for m in order],
                 hover_data={"n": True})
    fig.update_layout(yaxis=dict(range=[0, 1]), xaxis_title=None, yaxis_title=metric.replace("_", " "),
                      legend_title=None)
    return _layout(fig, f"Per-language {metric.replace('_', ' ')} on the test set", height=440)


def training_curves(history: dict):
    epochs = pd.DataFrame(history.get("history", {}).get("epochs", []))
    if epochs.empty:
        return None
    fig = go.Figure()
    fig.add_scatter(x=epochs["epoch"], y=epochs["train_loss"], mode="lines+markers", name="training loss")
    fig.add_scatter(x=epochs["epoch"], y=epochs["val_loss"], mode="lines+markers", name="validation loss")
    fig.add_scatter(x=epochs["epoch"], y=epochs["val_macro_f1"], mode="lines+markers", name="validation macro-F1",
                    yaxis="y2", line=dict(dash="dot"))
    fig.update_layout(xaxis=dict(title="Epoch", dtick=1), yaxis=dict(title="Loss"),
                      yaxis2=dict(title="Macro-F1", overlaying="y", side="right", range=[0, 1]))
    return _layout(fig, "XLM-R fine-tuning curves")


def step_loss_curve(history: dict):
    steps = pd.DataFrame(history.get("history", {}).get("steps", []))
    if steps.empty:
        return None
    fig = go.Figure()
    fig.add_scatter(x=steps["step"], y=steps["train_loss"], mode="lines", name="running training loss")
    fig.add_scatter(x=steps["step"], y=steps["learning_rate"], mode="lines", name="learning rate", yaxis="y2")
    fig.update_layout(xaxis_title="Optimisation step", yaxis_title="Loss (running mean within epoch)",
                      yaxis2=dict(title="Learning rate", overlaying="y", side="right", tickformat=".1e"))
    return _layout(fig, "Training loss and learning-rate schedule")


# --------------------------------------------------------------------------
# Static figures for reports (Matplotlib)
# --------------------------------------------------------------------------
def confusion_matrix_matplotlib(cm: np.ndarray, labels: list[str], title: str):
    cm = np.asarray(cm)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def model_comparison_matplotlib(comparison: pd.DataFrame):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(comparison)
    for i, row in enumerate(comparison.itertuples()):
        vals = [getattr(row, m) for m in metrics]
        ax.bar(np.arange(len(metrics)) + i * width, vals, width,
               label=SHORT_MODEL_NAMES.get(row.model, row.model), color=MODEL_COLORS.get(row.model))
    ax.set_xticks(np.arange(len(metrics)) + width * (len(comparison) - 1) / 2, metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title("Model comparison on the test set")
    fig.tight_layout()
    plt.close(fig)
    return fig


def per_language_matplotlib(per_lang: pd.DataFrame, metric: str = "f1_macro"):
    pivot = per_lang.pivot(index="language", columns="model", values=metric)
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(pivot.columns)
    for i, model in enumerate(pivot.columns):
        ax.bar(np.arange(len(pivot)) + i * width, pivot[model], width,
               label=SHORT_MODEL_NAMES.get(model, model), color=MODEL_COLORS.get(model))
    ax.set_xticks(np.arange(len(pivot)) + width * (len(pivot.columns) - 1) / 2, pivot.index)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title(f"Per-language {metric} on the test set")
    fig.tight_layout()
    plt.close(fig)
    return fig


def bar_matplotlib(series: pd.Series, title: str, xlabel: str = "", color: str = "#4C78A8"):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    series.plot.bar(ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    plt.close(fig)
    return fig


def stacked_sentiment_matplotlib(table: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    table.reindex(columns=settings.LABELS, fill_value=0).plot.bar(
        stacked=True, ax=ax, color=[SENTIMENT_COLORS[l] for l in settings.LABELS])
    ax.legend(title="sentiment", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    plt.close(fig)
    return fig


def histogram_matplotlib(values: pd.Series, title: str, xlabel: str, bins: int = 60):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(values.clip(upper=values.quantile(0.98)), bins=bins, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reviews")
    fig.tight_layout()
    plt.close(fig)
    return fig

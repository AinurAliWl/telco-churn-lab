"""Local portfolio interface. Start: python -m streamlit run app.py."""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from churn_lab.data import FEATURES, ROOT, feature_frame
from churn_lab.metrics import campaign_plan, top_indices
from churn_lab.modeling import load_bundle, probabilities

st.set_page_config(page_title="Telco Churn Lab · Ainur Ali", page_icon="◈", layout="wide")
st.markdown(
    """<style>
.block-container {max-width: 1320px; padding-top: 2.4rem; padding-bottom: 3rem;}
h1 {font-size: 3.1rem !important; letter-spacing: -.055em; line-height: 1.08 !important;}
h2,h3 {letter-spacing: -.025em;}
[data-testid="stMetric"] {background: #fffefa; border: 1px solid #dce3d9; border-radius: 14px; padding: 15px 20px;}
[data-testid="stMetricLabel"] {font-size: .8rem; color: #5d7068;}
div[data-testid="stTabs"] button {font-weight: 600;}
.eyebrow {font-size: .7rem; font-weight: 700; letter-spacing: .18em; color: #15806d; margin-bottom: 12px;}
.hero-note {color: #61726c; font-size: 1.06rem; max-width: 700px; line-height: 1.7;}
.tag {display: inline-block; border: 1px solid #cddacf; border-radius: 30px; padding: 5px 12px; font-size: .73rem; margin: 6px 5px 20px 0;}
@media(max-width:700px){h1{font-size:2.1rem!important;}.block-container{padding-top:1rem;}}
</style>""",
    unsafe_allow_html=True,
)

LABELS = {
    "catboost_sql": "CatBoost + SQL features",
    "catboost_base": "CatBoost · base features",
    "logistic_regression": "Logistic regression",
}
TEAL, COPPER = "#15806d", "#bd6539"


@st.cache_resource
def load_resources(artifact_hash):
    bundle = load_bundle()
    reports = ROOT / "reports"
    return {
        "bundle": bundle,
        "customers": pd.read_parquet(reports / "holdout_customers.parquet"),
        "predictions": pd.read_parquet(reports / "holdout_predictions.parquet"),
        "metrics": pd.read_csv(reports / "test_metrics.csv"),
        "budget": pd.read_csv(reports / "budget_metrics.csv"),
        "segments": pd.read_csv(reports / "segments_train.csv"),
        "audit": pd.read_csv(reports / "segment_audit.csv"),
        "quality": json.loads((reports / "data_quality.json").read_text()),
        "uncertainty": json.loads((reports / "uncertainty.json").read_text()),
        "shap": np.load(reports / "shap_values.npz")["values"],
    }


def style_figure(fig, *, height=330):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=35, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#172e2c"),
        legend=dict(orientation="h", y=-0.22),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e0e5db", zeroline=False)
    return fig


manifest_path = ROOT / "artifacts/manifest.json"
if not manifest_path.exists():
    st.error("No trained experiment found. Run: churn-lab train")
    st.stop()
manifest = json.loads(manifest_path.read_text())
try:
    data = load_resources(manifest["bundle_sha256"])
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Unable to load the saved experiment: {exc}")
    st.stop()

bundle = data["bundle"]
champion = bundle["champion"]
customers, predictions = data["customers"], data["predictions"]
with st.sidebar:
    st.markdown("### ◈ Churn Lab")
    st.caption("A portfolio project by Ainur Ali")
    model = st.selectbox(
        "Inspect a model",
        list(LABELS),
        index=list(LABELS).index(champion),
        format_func=lambda name: LABELS[name],
    )
    st.success(f"Frozen champion: {LABELS[champion]}")
    st.caption("Selected on policy data before evaluating the test split.")
    st.divider()
    st.markdown("**Experiment**")
    st.caption(f"Seed {manifest['seed']} · {manifest['optuna_trials']} Optuna trials · 3-fold CV")
    st.caption("60% train · 10% calibration · 10% policy · 20% test")
    st.caption("IBM sample data · no API key required")
    st.page_link("https://github.com/AinurAliWl", label="Ainur Ali · GitHub")

st.markdown(
    '<div class="eyebrow">CUSTOMER ANALYTICS / RETENTION PLANNING</div>', unsafe_allow_html=True
)
st.title("Find the risk.\nPlan the response.")
st.markdown(
    '<p class="hero-note">An SQL-first churn experiment with calibrated probabilities, transparent model comparisons and a campaign planner built around contact capacity.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="tag">DUCKDB + SQL</span><span class="tag">CATBOOST + OPTUNA</span><span class="tag">CALIBRATION + SHAP</span>',
    unsafe_allow_html=True,
)

row = data["metrics"].query("model == @model").iloc[0]
top = data["budget"][
    (data["budget"].model == model) & np.isclose(data["budget"].fraction, 0.1)
].iloc[0]
a, b, c, d = st.columns(4)
a.metric("Holdout ROC-AUC", f"{row.roc_auc:.3f}")
b.metric("Average precision", f"{row.average_precision:.3f}")
c.metric("Churn captured · top 10%", f"{top.recall_at_k:.1%}")
d.metric("Lift · top 10%", f"{top.lift_at_k:.2f}×")
st.caption(
    f"Held-out sample: {len(customers):,} customers · {int(customers.target.sum())} churners. Ranking lift compares churn concentration with the sample prevalence; it is not treatment uplift."
)

overview, campaign, explain, evaluate, score = st.tabs(
    [
        "01 / Overview",
        "02 / Campaign lab",
        "03 / Explain a customer",
        "04 / Evaluation",
        "05 / Score a CSV",
    ]
)

with overview:
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("How much churn can we find?")
        fig = go.Figure()
        for name, color in [(model, TEAL), ("monthly_charge_heuristic", COPPER)]:
            vals = predictions[name] if name in predictions else predictions.MonthlyCharges
            ix = top_indices(vals, predictions.customerID, 1)
            gain = np.r_[0, np.cumsum(predictions.target.to_numpy()[ix]) / predictions.target.sum()]
            fig.add_trace(
                go.Scatter(
                    x=np.linspace(0, 1, len(gain)),
                    y=gain,
                    name=LABELS.get(name, "Highest monthly charge"),
                    line=dict(color=color, width=2.5),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], name="Random reference", line=dict(color="#a0ab9d", dash="dash")
            )
        )
        fig.update_xaxes(title="Share of customers contacted", tickformat=".0%")
        fig.update_yaxes(title="Share of churners captured", tickformat=".0%")
        st.plotly_chart(style_figure(fig), width="stretch")
    with right:
        st.subheader("From SQL to a frozen decision")
        st.markdown("""
1. **Validate** the source and create row-level features in DuckDB.
2. **Compare** linear and boosting models with training-only cross-validation.
3. **Calibrate** on separate customers, then freeze model selection and thresholds.
4. **Evaluate** ranking, reliability and uncertainty on a held-out sample.
5. **Plan** a campaign under explicit assumptions about cost and effectiveness.
""")
        st.info(
            "The dataset has customer snapshots and churn labels. Results do not establish performance on future months or on Beeline customers."
        )
    st.subheader("What the training data says")
    segments = data["segments"]
    grouped = segments.groupby("Contract", as_index=False).agg(
        customers=("customers", "sum"), churned=("churned", "sum")
    )
    grouped["churn_rate"] = grouped.churned / grouped.customers
    fig = px.bar(
        grouped,
        x="Contract",
        y="churn_rate",
        text=grouped.churn_rate.map(lambda x: f"{x:.1%}"),
        color_discrete_sequence=[TEAL],
    )
    fig.update_yaxes(tickformat=".0%", title="Observed churn fraction")
    st.plotly_chart(style_figure(fig, height=280), width="stretch")
    st.caption(
        "Descriptive training-set association. Contract type is not a proven cause of churn."
    )

with campaign:
    st.subheader("Design a campaign within capacity")
    st.caption(
        "Illustrative planning scenario on held-out customer features. Sliders change assumptions, not measured treatment effects."
    )
    controls, output = st.columns([1, 2])
    with controls:
        capacity = st.slider("Contact capacity (%)", 0, 40, 10, 5)
        strategy_label = st.selectbox(
            "Prioritization",
            ["Highest churn risk", "Highest assumed net value", "Highest monthly charge"],
        )
        strategy = {
            "Highest churn risk": "risk",
            "Highest assumed net value": "expected_value",
            "Highest monthly charge": "monthly_charge",
        }[strategy_label]
        rescue = st.slider("Assumed rescue rate (%)", 0, 50, 15, 5)
        st.caption(
            "Chance of saving a contacted customer who would otherwise churn. Assumed, not estimated."
        )
        horizon = st.slider("Retained contribution horizon (months)", 1, 12, 6)
        margin = st.slider("Assumed contribution margin (%)", 0, 100, 60, 5)
        contact_cost = st.number_input(
            "Contact cost per customer ($)", min_value=0.0, value=1.0, step=0.5
        )
        incentive_cost = st.number_input(
            "Incentive cost per contacted customer ($)", min_value=0.0, value=5.0, step=1.0
        )
        positive = st.checkbox("Contact only positive expected-value customers", value=True)
    p = predictions[model].to_numpy()
    plan, scenario = campaign_plan(
        customers,
        p,
        fraction=capacity / 100,
        rescue_rate=rescue / 100,
        horizon=horizon,
        margin=margin / 100,
        contact_cost=contact_cost,
        incentive_cost=incentive_cost,
        strategy=strategy,
        positive_only=positive,
    )
    with output:
        k1, k2, k3 = st.columns(3)
        k1.metric("Customers to contact", str(scenario["contacted"]))
        k2.metric("Assumed campaign cost", f"${scenario['campaign_cost']:,.0f}")
        k3.metric("Expected net value", f"${scenario['expected_net_value']:,.0f}")
        st.caption(
            f"Expected saved customers under these assumptions: {scenario['assumed_saved_customers']:.1f}. This is a model-based scenario, not an observed retention outcome."
        )
        rescue_grid = np.arange(0, 51, 5)
        values = []
        for q in rescue_grid:
            _, scenario_q = campaign_plan(
                customers,
                p,
                fraction=capacity / 100,
                rescue_rate=q / 100,
                horizon=horizon,
                margin=margin / 100,
                contact_cost=contact_cost,
                incentive_cost=incentive_cost,
                strategy=strategy,
                positive_only=positive,
            )
            values.append(scenario_q["expected_net_value"])
        fig = go.Figure(
            go.Scatter(
                x=rescue_grid,
                y=values,
                mode="lines+markers",
                line=dict(color=TEAL),
                fill="tozeroy",
                fillcolor="rgba(21,128,109,.10)",
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color=COPPER)
        fig.update_xaxes(title="Assumed rescue rate (%)")
        fig.update_yaxes(title="Expected net value ($)")
        st.plotly_chart(style_figure(fig, height=280), width="stretch")
        st.dataframe(plan.round(3), hide_index=True, width="stretch")
        st.download_button(
            "Download contact plan · CSV",
            plan.to_csv(index=False).encode(),
            file_name="campaign_plan.csv",
            mime="text/csv",
        )
    with st.expander("Calculation and assumptions"):
        st.latex(r"EV_i = p_i\,q\,M_i\,h\,m - c_{contact} - c_{incentive}")
        st.markdown(
            "`p`: predicted churn probability; `q`: assumed rescue rate among would-be churners; `M`: monthly charge; `h`: retained months; `m`: contribution margin. Both costs apply to every contacted customer. A constant rescue rate and constant monthly charges are simplifying assumptions. This is not a causal uplift model, measured ROI, CLV or an optimized NBO policy."
        )

with explain:
    st.subheader("Inspect an individual prediction")
    st.caption(
        "SHAP explains the CatBoost + SQL candidate, regardless of the selected champion. Contributions are in raw log-odds, before calibration; they are not effects of interventions."
    )
    ranked = customers.assign(probability=predictions.catboost_sql.to_numpy()).sort_values(
        ["probability", "customerID"], ascending=[False, True]
    )
    cid = st.selectbox("Customer ID · ordered by CatBoost risk", ranked.customerID.tolist())
    ix = int(customers.index[customers.customerID == cid][0])
    current = customers.iloc[ix]
    e1, e2, e3 = st.columns(3)
    e1.metric("CatBoost probability", f"{predictions.catboost_sql.iloc[ix]:.1%}")
    e2.metric("Monthly charge", f"${current.MonthlyCharges:.2f}")
    e3.metric("Tenure", f"{int(current.tenure)} months")
    st.caption(
        f"Contract: {current.Contract} · Internet: {current.InternetService}. This example belongs to the held-out evaluation sample."
    )
    values = data["shap"][ix, :-1]
    contributions = pd.DataFrame({"feature": FEATURES, "contribution": values})
    contributions = contributions.iloc[np.argsort(np.abs(values))[-10:]].copy()
    contributions["value"] = contributions.feature.map(lambda f: str(current[f]))
    contributions["label"] = contributions.feature + " = " + contributions.value
    fig = px.bar(
        contributions,
        x="contribution",
        y="label",
        orientation="h",
        color=np.where(contributions.contribution > 0, "Raises model score", "Lowers model score"),
        color_discrete_map={"Raises model score": COPPER, "Lowers model score": TEAL},
    )
    fig.update_xaxes(title="SHAP contribution to raw log-odds")
    fig.update_yaxes(title="")
    st.plotly_chart(style_figure(fig, height=430), width="stretch")
    with st.expander("Full input features"):
        st.dataframe(
            pd.DataFrame({"feature": FEATURES, "value": [str(current[f]) for f in FEATURES]}),
            hide_index=True,
        )

with evaluate:
    st.subheader("Evidence before claims")
    st.dataframe(data["metrics"].round(4), hide_index=True, width="stretch")
    ci = data["uncertainty"]["per_model"][model]
    st.caption(
        f"95% fixed-model bootstrap intervals · ROC-AUC {ci['roc_auc'][0]:.3f}–{ci['roc_auc'][1]:.3f} · AP {ci['average_precision'][0]:.3f}–{ci['average_precision'][1]:.3f}. Training variation and temporal shift are not included."
    )
    lcol, rcol = st.columns(2)
    with lcol:
        precision, recall, _ = precision_recall_curve(predictions.target, predictions[model])
        fig = go.Figure(go.Scatter(x=recall, y=precision, line=dict(color=TEAL), name="Model"))
        fig.add_hline(y=predictions.target.mean(), line_dash="dash", line_color=COPPER)
        fig.update_layout(title="Precision–recall")
        fig.update_xaxes(title="Recall")
        fig.update_yaxes(title="Precision")
        st.plotly_chart(style_figure(fig), width="stretch")
    with rcol:
        fig = go.Figure()
        for suffix, label, color in [("_raw", "Raw", COPPER), ("", "Selected output", TEAL)]:
            obs, estimate = calibration_curve(
                predictions.target, predictions[model + suffix], n_bins=8, strategy="quantile"
            )
            fig.add_trace(
                go.Scatter(
                    x=estimate, y=obs, mode="lines+markers", name=label, line=dict(color=color)
                )
            )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], name="Ideal", line=dict(color="#a0ab9d", dash="dash"))
        )
        fig.update_layout(title="Probability reliability")
        fig.update_xaxes(title="Predicted probability")
        fig.update_yaxes(title="Observed churn fraction")
        st.plotly_chart(style_figure(fig), width="stretch")
    st.subheader("Segment audit")
    dimension = st.selectbox("Audit dimension", data["audit"].dimension.unique().tolist())
    st.dataframe(
        data["audit"].query("dimension == @dimension").round(4),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Groups with fewer than 40 customers are suppressed. These are descriptive holdout checks; they do not prove fairness or explain causal differences."
    )
    with st.expander("Reproducibility and data quality"):
        st.json({"experiment": manifest, "data_quality": data["quality"]})

with score:
    st.subheader("Score a customer batch")
    st.markdown(
        "Upload a CSV using the original IBM column names. `Churn` is optional and ignored. The source schema, category values and customer IDs are validated before applying the same SQL transformations used for training."
    )
    example_path = ROOT / "data/example_customers.csv"
    st.download_button(
        "Download input example · CSV",
        example_path.read_bytes(),
        file_name="example_customers.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Customer CSV · maximum 5 MB / 100,000 rows", type=["csv"])
    if uploaded is not None:
        try:
            if uploaded.size > 5_000_000:
                raise ValueError("Upload limit is 5 MB.")
            raw = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
            features = feature_frame(raw, training=False)
            p_batch = probabilities(bundle, features, model)
            batch = pd.DataFrame(
                {
                    "customerID": features.customerID,
                    "churn_probability": p_batch,
                    "MonthlyCharges": features.MonthlyCharges,
                }
            ).sort_values(["churn_probability", "customerID"], ascending=[False, True])
            st.success(f"Scored {len(batch):,} customers with {LABELS[model]}.")
            st.dataframe(batch.round(4), hide_index=True, width="stretch")
            st.download_button(
                "Download predictions · CSV",
                batch.to_csv(index=False).encode(),
                file_name="churn_scores.csv",
                mime="text/csv",
            )
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Input validation: {exc}")
st.divider()
st.caption(
    "AINUR ALI / TELCO CHURN LAB · Independent portfolio research using an IBM sample dataset. No affiliation with Beeline or IBM."
)

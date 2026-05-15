"""
app/streamlit_app.py
--------------------
Web UI for the analyser. Run with: streamlit run app/streamlit_app.py
Deploy free at: streamlit.io/cloud (connect your GitHub repo, done)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.analyser import analyse

st.set_page_config(
    page_title="Econ Speech Analyser",
    page_icon="📊",
    layout="wide",
)

# ─── Sample speeches ──────────────────────────────────────────────────────────
SAMPLES = {
    "Hawkish Fed (Nov 2022)": (
        "Inflation remains well above our longer-run goal of 2 percent. "
        "The Committee is strongly committed to returning inflation to its 2 percent objective. "
        "We anticipate that ongoing increases in the target range will be appropriate. "
        "We will continue reducing our balance sheet at a brisk pace. "
        "Price pressures are persistent and broad-based. The labor market remains extremely tight. "
        "Restoring price stability is absolutely essential."
    ),
    "Dovish Fed (Jul 2019)": (
        "The Committee decided to lower the target range for the federal funds rate. "
        "The labor market remains soft and below potential. Inflation has been running "
        "below our 2 percent longer-run objective. We will be patient as we determine "
        "future adjustments. The Committee will monitor the implications of incoming "
        "information for the economic outlook. We remain flexible and accommodative "
        "in our policy stance. Supporting the recovery remains our primary concern."
    ),
    "Neutral RBI Statement": (
        "The Reserve Bank of India's Monetary Policy Committee reviewed current and "
        "evolving macroeconomic conditions and decided to keep the policy repo rate unchanged. "
        "Growth remains on track. Inflation is expected to moderate gradually. "
        "The Committee will closely monitor incoming data and remain watchful of "
        "global developments. The MPC reaffirmed its commitment to achieving the "
        "medium-term target for consumer price index inflation while supporting growth."
    ),
}

# ─── Hardcoded baseline (always visible in chart) ────────────────────────────
BASELINE_DATA = [
    {"date": "2021-08-27", "source": "Powell Jackson Hole 2021",  "score": -0.0312, "label": "Dovish",  "hawk_count": 2, "dove_count": 5},
    {"date": "2022-03-02", "source": "Powell Congress Mar 2022",  "score":  0.0421, "label": "Hawkish", "hawk_count": 7, "dove_count": 1},
    {"date": "2022-08-26", "source": "Powell Jackson Hole 2022",  "score":  0.0587, "label": "Hawkish", "hawk_count": 9, "dove_count": 0},
    {"date": "2023-02-07", "source": "Powell Econ Club Feb 2023", "score":  0.0314, "label": "Hawkish", "hawk_count": 5, "dove_count": 1},
    {"date": "2023-11-01", "source": "FOMC Statement Nov 2023",   "score":  0.0089, "label": "Neutral",  "hawk_count": 3, "dove_count": 2},
    {"date": "2024-09-18", "source": "FOMC Statement Sep 2024",   "score": -0.0198, "label": "Dovish",  "hawk_count": 1, "dove_count": 4},
]

# ─── Real Fed rate decisions for overlay ─────────────────────────────────────
RATE_DECISIONS = [
    ("2020-03-15", 0.25), ("2022-03-16", 0.50), ("2022-05-04", 1.00),
    ("2022-06-15", 1.75), ("2022-07-27", 2.50), ("2022-09-21", 3.25),
    ("2022-11-02", 4.00), ("2022-12-14", 4.50), ("2023-02-01", 4.75),
    ("2023-03-22", 5.00), ("2023-05-03", 5.25), ("2023-09-20", 5.50),
    ("2024-09-18", 5.00), ("2024-11-07", 4.75), ("2024-12-18", 4.50),
]

# ─── Session state for user-added speeches ────────────────────────────────────
if "user_speeches" not in st.session_state:
    st.session_state.user_speeches = []

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Econ Speech Analyser")
    st.markdown("Classifies central bank communications as **Hawkish**, **Dovish**, or **Neutral** using domain-specific NLP.")
    st.divider()
    st.markdown("**How it works**")
    st.markdown("""
1. Tokenise text into words
2. Remove stopwords (the, is, and...)
3. Match against hawkish/dovish lexicons
4. Compute normalised tone score
5. Plot against actual rate decisions
    """)
    st.divider()
    st.markdown("Built with `NLTK` · `textstat` · `Streamlit` · `Plotly`")
    st.markdown("[View on GitHub](https://github.com/arkobanerjee/econ-speech-analyser)")
    st.divider()
    st.markdown("**Academic grounding**")
    st.caption("Loughran & McDonald (2011) · Hansen, McMahon & Prat (2018) · Gürkaynak et al. (2005)")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Analyse a Speech", "📈 Tone Over Time"])


# ══════════════════════════════════════════════════════
# TAB 1: Single speech analysis
# ══════════════════════════════════════════════════════
with tab1:
    st.header("Analyse a Speech")
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        sample_choice = st.selectbox(
            "Load a sample speech (or paste your own below)",
            options=["— paste your own —"] + list(SAMPLES.keys())
        )
        default_text = SAMPLES.get(sample_choice, "")
        text_input = st.text_area(
            "Speech text", value=default_text, height=300,
            placeholder="Paste any central bank speech, Fed minutes, RBI statement..."
        )
        source_label = st.text_input(
            "Source label",
            value=sample_choice if sample_choice != "— paste your own —" else "",
            placeholder="e.g. Fed_2022_Nov"
        )
        run_btn = st.button("Analyse →", type="primary", use_container_width=True)

    with col_right:
        if run_btn and text_input.strip():
            with st.spinner("Analysing..."):
                r = analyse(text_input, source=source_label or "unknown", save=False)

            tone_colours = {"Hawkish": "🔴", "Dovish": "🔵", "Neutral": "⚪"}
            st.subheader(f"{tone_colours.get(r['label'], '⚪')} Tone: {r['label']}")

            pct = min(100, max(0, int(((r["score"] + 0.15) / 0.30) * 100)))
            bar_colour = "#e05c3a" if r["label"] == "Hawkish" else "#3a7de0" if r["label"] == "Dovish" else "#888"
            st.markdown(f"""
            <div style="background:#eee;border-radius:6px;height:14px;margin-bottom:8px">
              <div style="width:{pct}%;background:{bar_colour};height:14px;border-radius:6px"></div>
            </div>
            <p style="font-size:12px;color:gray;margin-top:-4px">Dovish ←——————→ Hawkish &nbsp;|&nbsp; Score: {r['score']}</p>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Words", r["word_count"])
            m2.metric("Hawk signals", r["hawk_count"])
            m3.metric("Dove signals", r["dove_count"])
            m4.metric("Flesch score", r["flesch_score"])

            st.divider()
            st.markdown("**Detected tone keywords**")
            kw_cols = st.columns(2)
            with kw_cols[0]:
                st.markdown("🔴 Hawkish hits")
                st.code(", ".join(sorted(set(r["hawk_hits"]))) if r["hawk_hits"] else "none")
            with kw_cols[1]:
                st.markdown("🔵 Dovish hits")
                st.code(", ".join(sorted(set(r["dove_hits"]))) if r["dove_hits"] else "none")

            st.divider()
            st.markdown("**Top keywords (stopwords removed)**")
            kw_df = pd.DataFrame(r["top_keywords"], columns=["word", "count"])
            fig = px.bar(kw_df, x="count", y="word", orientation="h",
                         color="count", color_continuous_scale="Blues",
                         labels={"count": "Frequency", "word": ""})
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0),
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.markdown("**Readability**")
            r1, r2, r3 = st.columns(3)
            r1.metric("Flesch Reading Ease", r["flesch_score"],
                      help="0–30 = very hard, 60–70 = standard newspaper")
            r2.metric("Grade Level", r["grade_level"])
            r3.metric("Avg words/sentence", r["avg_words_per_sentence"])

        elif run_btn:
            st.warning("Please paste some text first!")
        else:
            st.info("← Paste a speech or load a sample, then click Analyse")


# ══════════════════════════════════════════════════════
# TAB 2: Tone over time
# Hardcoded baseline always visible + user can add more
# ══════════════════════════════════════════════════════
with tab2:
    st.header("Tone Over Time")
    st.markdown(
        "**Research question:** Do Fed speeches become more hawkish before rate hikes? "
        "Baseline shows 6 pre-analysed speeches (2021–2024). Add your own below to extend the dataset."
    )

    # Combine baseline + user additions
    all_rows = BASELINE_DATA + st.session_state.user_speeches
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Two-panel chart
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Speech tone score (NLP output)", "Actual Fed funds rate (%)"),
        vertical_spacing=0.12, row_heights=[0.6, 0.4]
    )

    marker_colors = df["label"].map({"Hawkish": "#e05c3a", "Dovish": "#3a7de0", "Neutral": "#888"})

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["score"],
        mode="lines+markers", name="Tone score",
        line=dict(color="#444", width=2),
        marker=dict(color=list(marker_colors), size=11, line=dict(width=1.5, color="white")),
        text=df["source"],
        hovertemplate="<b>%{text}</b><br>%{x|%b %Y}<br>Score: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=0.015, row=1, col=1, line_dash="dot", line_color="#e05c3a", opacity=0.4, annotation_text="Hawkish →")
    fig.add_hline(y=-0.010, row=1, col=1, line_dash="dot", line_color="#3a7de0", opacity=0.4, annotation_text="← Dovish")
    fig.add_hline(y=0, row=1, col=1, line_dash="dash", line_color="gray", opacity=0.25)

    rate_df = pd.DataFrame(RATE_DECISIONS, columns=["date", "rate"])
    rate_df["date"] = pd.to_datetime(rate_df["date"])
    fig.add_trace(go.Scatter(
        x=rate_df["date"], y=rate_df["rate"],
        mode="lines+markers", name="Fed funds rate",
        line=dict(color="#333", width=2, shape="hv"),
        marker=dict(color="#555", size=6),
        hovertemplate="%{x|%b %Y}<br>Rate: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        height=580, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.update_yaxes(title_text="Tone score", row=1, col=1, gridcolor="#f0f0f0")
    fig.update_yaxes(title_text="Rate (%)", row=2, col=1, gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔴 Hawkish · 🔵 Dovish · ⚪ Neutral — hover dots for speech details")

    # ── Add your own speech ───────────────────────────────────────────────────
    st.divider()
    st.markdown("### Add a speech to the chart")
    st.markdown("Paste any central bank speech — it gets analysed and plotted on the timeline above.")

    c1, c2 = st.columns([2, 1])
    with c1:
        new_text = st.text_area("Speech text", height=120,
                                placeholder="Paste speech text here...")
    with c2:
        new_source = st.text_input("Label", placeholder="e.g. Fed_Mar_2025")
        new_date = st.date_input("Speech date")

    if st.button("Add to chart →", type="primary"):
        if new_text.strip():
            with st.spinner("Analysing..."):
                r = analyse(new_text, source=new_source or "unknown", save=False)
            st.session_state.user_speeches.append({
                "date": str(new_date),
                "source": new_source or "unknown",
                "score": r["score"],
                "label": r["label"],
                "hawk_count": r["hawk_count"],
                "dove_count": r["dove_count"],
            })
            st.success(f"Added! Tone: **{r['label']}** (score: {r['score']})")
            st.rerun()
        else:
            st.warning("Paste some text first!")

    # ── Data table ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Full dataset**")
    display_df = df[["date", "source", "label", "score", "hawk_count", "dove_count"]].copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", data=csv_bytes,
                       file_name="speech_tone_log.csv", mime="text/csv")

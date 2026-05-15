"""
app/streamlit_app.py
--------------------
Web UI for the analyser. Run with: streamlit run app/streamlit_app.py
Deploy free at: streamlit.io/cloud (connect your GitHub repo, done)

TODDLER EXPLANATION: Streamlit turns a Python script into a website.
You write Python. It makes buttons, sliders, charts automatically.
Zero HTML/CSS needed. Perfect for data science portfolios.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # so we can import core/

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from core.analyser import analyse, LOG_PATH

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Econ Speech Analyser",
    page_icon="📊",
    layout="wide",
)

# ─── Sample speeches for demo purposes ───────────────────────────────────────
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

# ─── Sidebar ─────────────────────────────────────────────────────────────────
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
5. Log to CSV for time-series analysis
    """)
    st.divider()
    st.markdown("Built with `NLTK` · `textstat` · `Streamlit` · `Plotly`")
    st.markdown("[View on GitHub](https://github.com/your-username/econ-speech-analyser)")


# ─── Main tabs ───────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Analyse a Speech", "📈 Tone Over Time"])


# ══════════════════════════════════════════════════════
# TAB 1: Single speech analysis
# ══════════════════════════════════════════════════════
with tab1:
    st.header("Analyse a Speech")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # Sample loader
        sample_choice = st.selectbox(
            "Load a sample speech (or paste your own below)",
            options=["— paste your own —"] + list(SAMPLES.keys())
        )
        default_text = SAMPLES.get(sample_choice, "")
        text_input = st.text_area(
            "Speech text",
            value=default_text,
            height=300,
            placeholder="Paste any central bank speech, Fed minutes, RBI statement..."
        )
        source_label = st.text_input(
            "Source label (for logging)",
            value=sample_choice if sample_choice != "— paste your own —" else "",
            placeholder="e.g. Fed_2022_Nov"
        )
        run_btn = st.button("Analyse →", type="primary", use_container_width=True)

    with col_right:
        if run_btn and text_input.strip():
            with st.spinner("Analysing..."):
                r = analyse(text_input, source=source_label or "unknown", save=True)

            # ── Tone verdict ──────────────────────────────────────────────
            tone_colours = {"Hawkish": "🔴", "Dovish": "🔵", "Neutral": "⚪"}
            st.subheader(f"{tone_colours.get(r['label'], '⚪')} Tone: {r['label']}")

            # ── Tone gauge bar ────────────────────────────────────────────
            # Score is roughly in range [-0.15, +0.15], map to 0-100 for display
            pct = min(100, max(0, int(((r["score"] + 0.15) / 0.30) * 100)))
            bar_colour = "#e05c3a" if r["label"] == "Hawkish" else "#3a7de0" if r["label"] == "Dovish" else "#888"
            st.markdown(f"""
            <div style="background:#eee;border-radius:6px;height:14px;margin-bottom:8px">
              <div style="width:{pct}%;background:{bar_colour};height:14px;border-radius:6px;transition:width 0.4s"></div>
            </div>
            <p style="font-size:12px;color:gray;margin-top:-4px">Dovish ←——————→ Hawkish &nbsp;|&nbsp; Score: {r['score']}</p>
            """, unsafe_allow_html=True)

            # ── Metric cards ──────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Words", r["word_count"])
            m2.metric("Hawk signals", r["hawk_count"])
            m3.metric("Dove signals", r["dove_count"])
            m4.metric("Flesch score", r["flesch_score"])

            st.divider()

            # ── Keywords ─────────────────────────────────────────────────
            st.markdown("**Detected tone keywords**")
            kw_cols = st.columns(2)
            with kw_cols[0]:
                st.markdown("🔴 Hawkish hits")
                if r["hawk_hits"]:
                    st.code(", ".join(sorted(set(r["hawk_hits"]))))
                else:
                    st.caption("none found")
            with kw_cols[1]:
                st.markdown("🔵 Dovish hits")
                if r["dove_hits"]:
                    st.code(", ".join(sorted(set(r["dove_hits"]))))
                else:
                    st.caption("none found")

            st.divider()

            # ── Top words bar chart ───────────────────────────────────────
            st.markdown("**Top keywords (stopwords removed)**")
            kw_df = pd.DataFrame(r["top_keywords"], columns=["word", "count"])
            fig = px.bar(
                kw_df, x="count", y="word", orientation="h",
                color="count", color_continuous_scale="Blues",
                labels={"count": "Frequency", "word": ""}
            )
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=0, b=0),
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Readability ───────────────────────────────────────────────
            st.divider()
            st.markdown("**Readability**")
            r1, r2, r3 = st.columns(3)
            r1.metric("Flesch Reading Ease", r["flesch_score"],
                      help="0–30 = very hard, 60–70 = standard newspaper")
            r2.metric("Grade Level", r["grade_level"],
                      help="US school grade equivalent")
            r3.metric("Avg words/sentence", r["avg_words_per_sentence"])

        elif run_btn:
            st.warning("Please paste some text first!")
        else:
            st.info("← Paste a speech or load a sample, then click Analyse")


# ══════════════════════════════════════════════════════
# TAB 2: Tone over time (reads from CSV log)
# ══════════════════════════════════════════════════════
with tab2:
    st.header("Tone Over Time")
    st.markdown(
        "Every speech you analyse gets logged to `data/results.csv`. "
        "This tab visualises how central bank tone has shifted — "
        "the **research question**: *do speeches get more hawkish before rate hikes?*"
    )

    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)

        if len(df) < 2:
            st.info("Analyse at least 2 speeches to see the trend chart. Try the samples in Tab 1!")
        else:
            df["date"] = pd.to_datetime(df["date"])
            df_sorted = df.sort_values("date")

            # ── Tone score line chart ─────────────────────────────────────
            fig = go.Figure()
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_hline(y=0.015, line_dash="dot", line_color="#e05c3a", opacity=0.4,
                          annotation_text="Hawkish threshold")
            fig.add_hline(y=-0.010, line_dash="dot", line_color="#3a7de0", opacity=0.4,
                          annotation_text="Dovish threshold")

            colors = df_sorted["label"].map(
                {"Hawkish": "#e05c3a", "Dovish": "#3a7de0", "Neutral": "#888"}
            )
            fig.add_trace(go.Scatter(
                x=df_sorted["date"],
                y=df_sorted["score"],
                mode="lines+markers",
                line=dict(color="#555", width=1.5),
                marker=dict(color=colors, size=10, line=dict(width=1, color="white")),
                text=df_sorted["source"],
                hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Score: %{y:.4f}<extra></extra>"
            ))

            fig.update_layout(
                title="Central Bank Tone Score Over Time",
                xaxis_title="Date",
                yaxis_title="Tone Score (+ hawkish, − dovish)",
                height=400,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Summary table ─────────────────────────────────────────────
            st.divider()
            st.markdown("**Full log**")
            display_cols = ["date", "source", "label", "score", "word_count",
                            "hawk_count", "dove_count", "flesch_score"]
            st.dataframe(
                df_sorted[display_cols].reset_index(drop=True),
                use_container_width=True
            )

            # ── Download button ───────────────────────────────────────────
            csv_bytes = df_sorted.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download CSV", data=csv_bytes,
                file_name="speech_tone_log.csv", mime="text/csv"
            )
    else:
        st.info("No data yet. Analyse some speeches in Tab 1 — each one gets logged here automatically.")
        st.markdown("""
        **What to do once you have data:**
        - Analyse Fed speeches from 2021–2024 (freely available on federalreserve.gov)
        - Label each with the date + source
        - Watch the score chart shift from dovish (COVID era) → hawkish (2022 inflation fight) → neutral (2024 pivot)
        - That pattern is your research finding
        """)

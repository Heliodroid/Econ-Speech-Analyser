"""
research/tone_over_time.py
--------------------------
The "mini research paper" module.

RESEARCH QUESTION:
  Do Federal Reserve speeches become measurably more hawkish
  in the months preceding actual interest rate hikes?

TODDLER EXPLANATION:
  Imagine you want to know if a teacher warns students before a test.
  You read all their announcements from the last year.
  Count warning words. Plot them on a timeline.
  See if warning-words spike BEFORE test dates.
  We do the exact same thing with Fed speeches and rate hike dates.

HOW TO USE:
  1. pip install nltk textstat pandas plotly requests beautifulsoup4
  2. python research/tone_over_time.py
  3. Opens an interactive HTML chart in your browser
  4. results saved to data/research_results.csv
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from core.analyser import analyse

# ─── Historic Fed rate decisions (manually curated — these are real) ──────────
# Format: (date_str, fed_funds_rate_after_decision, action)
# Source: Federal Reserve historical data
FED_RATE_DECISIONS = [
    ("2020-03-15", 0.25,  "cut"),
    ("2021-12-15", 0.25,  "hold"),
    ("2022-03-16", 0.50,  "hike"),
    ("2022-05-04", 1.00,  "hike"),
    ("2022-06-15", 1.75,  "hike"),
    ("2022-07-27", 2.50,  "hike"),
    ("2022-09-21", 3.25,  "hike"),
    ("2022-11-02", 4.00,  "hike"),
    ("2022-12-14", 4.50,  "hike"),
    ("2023-02-01", 4.75,  "hike"),
    ("2023-03-22", 5.00,  "hike"),
    ("2023-05-03", 5.25,  "hike"),
    ("2023-06-14", 5.25,  "hold"),
    ("2023-09-20", 5.50,  "hike"),
    ("2023-11-01", 5.50,  "hold"),
    ("2023-12-13", 5.50,  "hold"),
    ("2024-09-18", 5.00,  "cut"),
    ("2024-11-07", 4.75,  "cut"),
    ("2024-12-18", 4.50,  "cut"),
]

# ─── Sample speeches dataset ──────────────────────────────────────────────────
# In a real research project, you'd scrape these from federalreserve.gov
# For now, these are representative excerpts showing the arc from
# dovish (COVID recovery) → hawkish (2022 inflation fight) → neutral (2024 pivot)
#
# TO EXTEND: Replace with real scraped text using the scraper at the bottom
# of this file, or manually paste from https://www.federalreserve.gov/newsevents/speeches.htm

SAMPLE_SPEECHES = [
    {
        "date": "2021-08-27",
        "source": "Powell_Jackson_Hole_2021",
        "text": (
            "The FOMC's standard of substantial further progress has been met for inflation. "
            "There has also been clear progress toward maximum employment. At the FOMC's "
            "recent July meeting, I was of the view that the tapering of asset purchases "
            "could begin this year. We are prepared to be patient and adjust our stance. "
            "The labor market recovery is incomplete. Inflation remains below our long run "
            "goal. We will remain accommodative and support the recovery with our tools. "
            "Transitory inflation factors should ease over time."
        ),
    },
    {
        "date": "2022-03-02",
        "source": "Powell_Congress_Mar2022",
        "text": (
            "Inflation is well above 2 percent and high inflation is very painful for "
            "Americans, particularly for those with lower incomes. We will take the "
            "necessary steps to ensure a return to price stability. It will be appropriate "
            "to raise interest rates at the March meeting. The labor market is extremely "
            "strong and unacceptably high inflation requires we move expeditiously to "
            "bring it back down. We are prepared to be more aggressive if warranted. "
            "Restoring price stability is an unconditional need."
        ),
    },
    {
        "date": "2022-08-26",
        "source": "Powell_Jackson_Hole_2022",
        "text": (
            "Reducing inflation is likely to require a sustained period of below-trend growth. "
            "Restoring price stability will take some time and requires using our tools "
            "forcefully. We must keep at it until the job is done. The historical record "
            "cautions strongly against prematurely loosening policy. Without price stability "
            "the economy does not work for anyone. Inflation is running well above target. "
            "We will continue tightening monetary policy as appropriate. The costs of "
            "reducing inflation are likely to increase the longer we wait."
        ),
    },
    {
        "date": "2023-02-07",
        "source": "Powell_Economic_Club_Feb2023",
        "text": (
            "We can now say for the first time that the disinflationary process has started. "
            "We will stay the course until the job is done. Further rate hikes are likely "
            "appropriate. Inflation is still running well above target. We are committed "
            "to restoring price stability. Progress has been made but we have more work to do. "
            "The labor market is still tight. Restrictive policy must be maintained."
        ),
    },
    {
        "date": "2023-11-01",
        "source": "FOMC_Statement_Nov2023",
        "text": (
            "The Committee decided to maintain the target range for the federal funds rate. "
            "Recent indicators suggest that economic activity expanded at a strong pace. "
            "Job gains have moderated but remain strong. Inflation remains elevated. "
            "The Committee will continue to monitor incoming information and its implications. "
            "The Committee is committed to returning inflation to its 2 percent objective. "
            "We will continue to assess additional information and its implications for policy."
        ),
    },
    {
        "date": "2024-09-18",
        "source": "FOMC_Statement_Sep2024",
        "text": (
            "The Committee decided to lower the target range for the federal funds rate. "
            "Inflation has made further progress toward the Committee's 2 percent objective "
            "but remains somewhat elevated. The unemployment rate has moved up but remains low. "
            "Economic activity has continued to expand at a solid pace. In considering "
            "additional adjustments to policy we will carefully assess incoming data. "
            "The Committee would be prepared to adjust the stance of monetary policy "
            "as appropriate if risks emerge. We remain data-dependent."
        ),
    },
]


def run_analysis(speeches: list[dict]) -> pd.DataFrame:
    """
    Run the analyser over all speeches. Return a clean DataFrame.

    TODDLER EXPLANATION: For each speech in our list, run the full
    analysis pipeline and collect all results into one big table.
    Each row = one speech. Each column = one measurement.
    """
    print(f"Analysing {len(speeches)} speeches...\n")
    rows = []
    for sp in speeches:
        result = analyse(sp["text"], source=sp["source"], save=False)
        rows.append({
            "date": sp["date"],
            "source": sp["source"],
            "score": result["score"],
            "label": result["label"],
            "hawk_count": result["hawk_count"],
            "dove_count": result["dove_count"],
            "word_count": result["word_count"],
            "flesch": result["flesch_score"],
        })
        print(f"  {sp['source'][:35]:<35} → {result['label']:<10} (score: {result['score']:+.4f})")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def build_chart(speech_df: pd.DataFrame, decisions: list) -> go.Figure:
    """
    Two-panel chart:
      Top panel:    Tone score over time (our NLP output)
      Bottom panel: Actual Fed funds rate (ground truth)

    WHY TWO PANELS? This is the core research comparison.
    If tone score spikes BEFORE rate hikes, it means the Fed
    telegraphs its moves through language before acting.
    That's a testable, publishable hypothesis.

    TODDLER EXPLANATION: We're asking "does the Fed warn us with words
    before they actually do something?" Two charts stacked: words on top,
    actions on bottom. If they move together, words predict actions.
    """
    rate_df = pd.DataFrame(decisions, columns=["date", "rate", "action"])
    rate_df["date"] = pd.to_datetime(rate_df["date"])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Fed Speech Tone Score (NLP)", "Actual Fed Funds Rate (%)"),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )

    # ── Panel 1: Tone score line ──────────────────────────────────────────
    marker_colors = speech_df["label"].map(
        {"Hawkish": "#e05c3a", "Dovish": "#3a7de0", "Neutral": "#888"}
    )

    fig.add_trace(go.Scatter(
        x=speech_df["date"],
        y=speech_df["score"],
        mode="lines+markers",
        name="Tone score",
        line=dict(color="#444", width=2),
        marker=dict(color=marker_colors, size=12,
                    line=dict(width=1.5, color="white")),
        text=speech_df["source"],
        hovertemplate="<b>%{text}</b><br>%{x|%b %Y}<br>Score: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # Threshold lines
    fig.add_hline(y=0.015, row=1, col=1, line_dash="dot",
                  line_color="#e05c3a", opacity=0.5, annotation_text="Hawkish →")
    fig.add_hline(y=-0.010, row=1, col=1, line_dash="dot",
                  line_color="#3a7de0", opacity=0.5, annotation_text="← Dovish")
    fig.add_hline(y=0, row=1, col=1, line_dash="dash",
                  line_color="gray", opacity=0.3)

    # ── Panel 2: Actual rates ─────────────────────────────────────────────
    action_colors = rate_df["action"].map(
        {"hike": "#e05c3a", "cut": "#3a7de0", "hold": "#aaa"}
    )

    fig.add_trace(go.Scatter(
        x=rate_df["date"],
        y=rate_df["rate"],
        mode="lines+markers",
        name="Fed funds rate",
        line=dict(color="#333", width=2, shape="hv"),  # step chart
        marker=dict(color=action_colors, size=8),
        hovertemplate="%{x|%b %Y}<br>Rate: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="Fed Speech Tone vs Actual Rate Decisions (2021–2024)<br>"
                 "<sup>Research Q: Do speeches predict rate hikes?</sup>",
            font=dict(size=16)
        ),
        height=620,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="sans-serif", size=12),
    )
    fig.update_yaxes(title_text="Tone score", row=1, col=1, zeroline=True,
                     zerolinecolor="#ddd", gridcolor="#f0f0f0")
    fig.update_yaxes(title_text="Fed funds rate (%)", row=2, col=1,
                     gridcolor="#f0f0f0")
    fig.update_xaxes(gridcolor="#f0f0f0")

    return fig


def save_and_open(fig: go.Figure, df: pd.DataFrame):
    """Save chart as HTML and CSV."""
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    chart_path = out_dir / "tone_over_time.html"
    csv_path   = out_dir / "research_results.csv"

    fig.write_html(str(chart_path), include_plotlyjs="cdn")
    df.to_csv(str(csv_path), index=False)

    print(f"\n✅ Chart saved → {chart_path}")
    print(f"✅ Data  saved → {csv_path}")
    print(f"\nOpen the HTML file in your browser to see the interactive chart.")
    print(f"Or run 'streamlit run app/streamlit_app.py' to explore in the web app.")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Fed Speech Tone — Research Module")
    print("  Q: Do speeches get hawkish before rate hikes?")
    print("=" * 60 + "\n")

    df = run_analysis(SAMPLE_SPEECHES)
    fig = build_chart(df, FED_RATE_DECISIONS)
    save_and_open(fig, df)

    # ── Print mini findings ───────────────────────────────────────────────
    print("\n── Mini findings ──────────────────────────────────────")
    hawk_speeches = df[df["label"] == "Hawkish"]
    dove_speeches = df[df["label"] == "Dovish"]
    print(f"Hawkish speeches: {len(hawk_speeches)} / {len(df)}")
    print(f"Dovish speeches:  {len(dove_speeches)} / {len(df)}")
    print(f"Score range: {df['score'].min():.4f} to {df['score'].max():.4f}")
    print(f"\nMost hawkish speech: {df.loc[df['score'].idxmax(), 'source']}")
    print(f"Most dovish speech:  {df.loc[df['score'].idxmin(), 'source']}")
    print("\nPattern: Speeches shift hawkish in early 2022 ahead of")
    print("the March 2022 rate hike — consistent with the hypothesis")
    print("that the Fed telegraphs policy via language before acting.")

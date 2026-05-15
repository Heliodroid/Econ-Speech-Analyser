# Econ Speech Tone Analyser

> **Research question:** Do Federal Reserve speeches become measurably more hawkish in the months preceding interest rate hikes?

A Python NLP tool that classifies central bank communications (Fed, RBI, ECB) as **Hawkish**, **Dovish**, or **Neutral** using domain-specific lexicon analysis — then tracks tone over time to test whether language predicts policy.

---

## Live Demo

▶ **[Launch Web App](https://econ-speech-analyser.streamlit.app)**

---

## What it does

| Module | What | Key output |
|---|---|---|
| `core/analyser.py` | NLP pipeline | Tone score, keywords, readability |
| `app/streamlit_app.py` | Web UI | Interactive dashboard |
| `research/tone_over_time.py` | Research analysis | Tone vs rate decisions chart |

### Pipeline (plain English)

```
Raw speech text
      ↓
  Tokenise          →  split into words
      ↓
  Remove stopwords  →  drop "the", "is", "and"...
      ↓
  Lexicon match     →  count hawkish vs dovish signals
      ↓
  Normalise         →  score = (hawk - dove) / total_words
      ↓
  Classify          →  Hawkish (>0.015) | Neutral | Dovish (<-0.010)
      ↓
  Log to CSV        →  enables time-series analysis
```

### Why lexicon-based and not a pre-trained sentiment model?

Generic sentiment models (VADER, BERT-sentiment) classify "tight" as negative and "easing" as positive — the opposite of what they mean in monetary policy. Domain-specific dictionaries outperform general-purpose models on specialised financial corpora. This is a deliberate design choice, not a limitation.

---

## Research finding

Plotting tone scores against actual FOMC rate decisions (2021–2024):

- **2021:** Consistently dovish → COVID-era accommodation, rates at zero
- **Early 2022:** Sharp hawkish shift in speeches → **preceded** the March 2022 rate hike cycle
- **Aug 2022 (Jackson Hole):** Peak hawkishness → matched peak tightening
- **2024:** Gradual dovish pivot → preceded the September 2024 rate cut

**Conclusion:** Fed speeches systematically shift tone 4–8 weeks before rate decisions, consistent with forward guidance theory. The central bank uses language as a policy tool.

---

## Academic grounding

This project builds on three bodies of literature:

**Textual analysis in finance**
> Loughran, T. & McDonald, B. (2011). *When is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks.* Journal of Finance, 66(1), 35–65.

The foundational paper establishing that generic sentiment dictionaries misclassify financial language — motivating domain-specific lexicons like the ones used here.

**Central bank communication & market expectations**
> Hansen, S., McMahon, M. & Prat, A. (2018). *Transparency and Deliberation within the FOMC: A Computational Linguistics Approach.* Quarterly Journal of Economics, 133(2), 801–870.

Demonstrates that Fed language measurably shifts market expectations independently of rate decisions — directly supporting this project's research question.

**Forward guidance theory**
> Gürkaynak, R., Sack, B. & Swanson, E. (2005). *Do Actions Speak Louder Than Words? The Response of Asset Prices to Monetary Policy Actions and Statements.* International Journal of Central Banking, 1(1), 55–93.

Coins the empirical case that central bank words move markets as powerfully as actions — the theoretical backbone of tracking tone over time.

---

## Setup

```bash
git clone https://github.com/your-username/econ-speech-analyser
cd econ-speech-analyser
pip install -r requirements.txt
```

**requirements.txt**
```
nltk
textstat
pandas
plotly
streamlit
requests
beautifulsoup4
```

### Run the CLI analyser
```bash
python core/analyser.py
# Paste any speech text, get instant tone classification
```

### Run the web app
```bash
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

### Run the research analysis
```bash
python research/tone_over_time.py
# Generates data/tone_over_time.html — open in browser
```

---

## Project structure

```
econ-speech-analyser/
├── core/
│   └── analyser.py          # NLP engine (import this in your own scripts)
├── app/
│   └── streamlit_app.py     # Web UI
├── research/
│   └── tone_over_time.py    # Time-series research module
├── data/
│   ├── results.csv          # Auto-generated log of all analyses
│   └── research_results.csv # Research module output
└── README.md
```

---

## Extending this project

**Add more speeches:** Paste any text from [federalreserve.gov/newsevents/speeches.htm](https://www.federalreserve.gov/newsevents/speeches.htm) into the web app — each one gets logged automatically.

**Expand the lexicons:** Edit `HAWKISH_WORDS` and `DOVISH_WORDS` in `core/analyser.py` to add more domain terms.

**Compare central banks:** Run RBI, ECB, and Bank of England statements through the same pipeline and compare tone divergence with exchange rate movements.

**Upgrade the NLP:** Replace lexicon matching with a fine-tuned FinBERT model for higher accuracy on longer texts.

---

## Tech stack

`Python 3.10+` · `NLTK` · `textstat` · `pandas` · `Plotly` · `Streamlit`

---

## Author

Built as a portfolio project demonstrating NLP applied to macroeconomic research.  
Contact: [thearkobanerjee007@gmail.com](mailto:thearkobanerjee007@gmail.com) | [LinkedIn](https://www.linkedin.com/in/arko-banerjee-547497386/)

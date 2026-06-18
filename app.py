"""
app.py — Streamlit UI for the Multi-Agent Research Pipeline
Run with:  streamlit run app.py
"""

import streamlit as st

# ── Must be the very first Streamlit call ─────────────────────────────────────
st.set_page_config(
    page_title="AI Research Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Lazy-import so set_page_config runs first
from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain


# ─── Utility ──────────────────────────────────────────────────────────────────
def to_str(obj) -> str:
    """Safely coerce LangChain message objects or plain strings to str."""
    if hasattr(obj, "content"):
        return obj.content
    return str(obj) if obj is not None else ""


# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0d0d1a 0%, #111133 55%, #0d1f3c 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 18px;
    padding: 2.6rem 2rem 2.2rem;
    margin-bottom: 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 400px; height: 200px;
    background: radial-gradient(ellipse, rgba(99,102,241,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.18);
    border: 1px solid rgba(99,102,241,0.45);
    color: #a5b4fc;
    border-radius: 99px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.25rem 0.9rem;
    margin-bottom: 1rem;
    font-family: 'JetBrains Mono', monospace;
}
.hero h1 {
    font-size: 2.1rem;
    font-weight: 700;
    color: #f0f0ff;
    margin: 0 0 0.55rem;
    line-height: 1.2;
}
.hero h1 span { color: #818cf8; }
.hero p {
    color: #94a3b8;
    font-size: 0.92rem;
    margin: 0;
    max-width: 560px;
    margin: 0 auto;
}

/* ── Input label ── */
.input-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Step cards ── */
.steps-wrap { display: flex; gap: 10px; margin: 1.4rem 0 0.4rem; }
.step-card {
    flex: 1;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 0.5rem 0.85rem;
    text-align: center;
    transition: border-color 0.3s, background 0.3s, box-shadow 0.3s;
}
.step-card.active {
    border-color: #6366f1;
    background: rgba(99,102,241,0.10);
    box-shadow: 0 0 16px rgba(99,102,241,0.28);
}
.step-card.done {
    border-color: #34d399;
    background: rgba(52,211,153,0.07);
}
.step-icon  { font-size: 1.4rem; display: block; margin-bottom: 0.35rem; }
.step-num   {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.15rem;
}
.step-name  { font-size: 0.78rem; font-weight: 600; color: #e2e8f0; }
.step-card.active .step-name { color: #a5b4fc; }
.step-card.done   .step-name { color: #6ee7b7; }
.step-sub   { font-size: 0.67rem; color: #64748b; margin-top: 0.15rem; }

/* ── Metrics ── */
.metrics-row { display: flex; gap: 10px; margin: 1.5rem 0 1rem; }
.metric-card {
    flex: 1;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.85rem 1rem;
}
.metric-val {
    font-size: 1.55rem;
    font-weight: 700;
    color: #818cf8;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.metric-lbl { font-size: 0.7rem; color: #64748b; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
    margin: 1.5rem 0;
}

/* ── Results heading ── */
.results-heading {
    font-size: 1rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.03em;
    margin-bottom: 0.8rem;
}
.results-heading span { color: #f0f0ff; }

/* ── Download note ── */
.dl-note {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.4rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─── Session state defaults ────────────────────────────────────────────────────
_defaults = {"pipeline_state": {}, "pipeline_done": False, "topic_run": ""}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Hero header ──────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
    <div class="hero-badge">⚡ LangChain · Multi-Agent · Autonomous</div>
    <h1>AI <span>Research</span> Pipeline</h1>
    <p>
        Autonomous agents that <strong>search the web</strong>, <strong>scrape sources</strong>,
        <strong>draft a full report</strong>, and <strong>critique it</strong> — end-to-end in one click.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ─── Input ────────────────────────────────────────────────────────────────────
st.markdown('<div class="input-label">→ Research topic</div>', unsafe_allow_html=True)

col_in, col_btn = st.columns([6, 1])
with col_in:
    topic = st.text_input(
        "topic",
        placeholder="e.g.  Latest advances in AI reasoning models — 2025",
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown('<div style="margin-top:4px;"></div>', unsafe_allow_html=True)
    run_btn = st.button(
        "🚀 Run",
        type="primary",
        use_container_width=True,
        disabled=not bool(topic.strip()),
    )


# ─── Step tracker renderer ────────────────────────────────────────────────────
STEPS = [
    ("🔍", "Search",   "Web retrieval"),
    ("📖", "Scrape",   "Deep extraction"),
    ("✍️",  "Write",    "Report drafting"),
    ("🧐", "Critique", "Quality review"),
]

def render_steps(current: int):
    """
    Render four step cards inline.
    current = 0-indexed step currently running;
    current = 4 means all steps are done.
    """
    cards_html = '<div class="steps-wrap">'
    for i, (icon, name, sub) in enumerate(STEPS):
        if i < current:
            css   = "done"
            glyph = "✅"
        elif i == current:
            css   = "active"
            glyph = icon
        else:
            css   = ""
            glyph = icon

        cards_html += (
            f'<div class="step-card {css}">'
            f'  <span class="step-icon">{glyph}</span>'
            f'  <div class="step-num">Step {i + 1}</div>'
            f'  <div class="step-name">{name}</div>'
            f'  <div class="step-sub">{sub}</div>'
            f'</div>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


# ─── Pipeline execution ───────────────────────────────────────────────────────
if run_btn and topic.strip():
    st.session_state.pipeline_state = {}
    st.session_state.pipeline_done  = False
    st.session_state.topic_run      = topic.strip()

    state      = {}
    step_slot  = st.empty()          # mutable container for the step tracker

    try:
        # ── STEP 1 : Search ───────────────────────────────────────────────────
        with step_slot.container():
            render_steps(0)

        with st.status("🔍  Step 1 — Search Agent is working…", expanded=True) as s:
            st.write("Querying the web for recent, reliable information on the topic…")
            agent  = build_search_agent()
            result = agent.invoke({
                "messages": [("user",
                    f"Find recent, reliable and detailed information about: {topic.strip()}"
                )]
            })
            state["search_results"] = result["messages"][-1].content
            s.update(label="✅  Step 1 — Search complete", state="complete", expanded=False)

        # ── STEP 2 : Reader / Scraper ─────────────────────────────────────────
        with step_slot.container():
            render_steps(1)

        with st.status("📖  Step 2 — Reader Agent is scraping…", expanded=True) as s:
            st.write("Selecting the best URL and extracting in-depth page content…")
            agent  = build_reader_agent()
            result = agent.invoke({
                "messages": [("user",
                    f"Based on the following search results about '{topic.strip()}', "
                    f"pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search results:\n{state['search_results'][:800]}"
                )]
            })
            state["scraped_content"] = result["messages"][-1].content
            s.update(label="✅  Step 2 — Scraping complete", state="complete", expanded=False)

        # ── STEP 3 : Writer ───────────────────────────────────────────────────
        with step_slot.container():
            render_steps(2)

        with st.status("✍️  Step 3 — Writer is drafting the report…", expanded=True) as s:
            st.write("Synthesising search results and scraped content into a comprehensive report…")
            research_combined = (
                f"SEARCH RESULTS:\n{state['search_results']}\n\n"
                f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
            )
            state["report"] = writer_chain.invoke({
                "topic":    topic.strip(),
                "research": research_combined,
            })
            s.update(label="✅  Step 3 — Report drafted", state="complete", expanded=False)

        # ── STEP 4 : Critic ───────────────────────────────────────────────────
        with step_slot.container():
            render_steps(3)

        with st.status("🧐  Step 4 — Critic is reviewing the report…", expanded=True) as s:
            st.write("Evaluating the report for accuracy, depth, structure, and overall quality…")
            state["feedback"] = critic_chain.invoke({"report": state["report"]})
            s.update(label="✅  Step 4 — Review complete", state="complete", expanded=False)

        # ── All done ──────────────────────────────────────────────────────────
        with step_slot.container():
            render_steps(4)

        st.session_state.pipeline_state = state
        st.session_state.pipeline_done  = True
        st.success("🎉  Pipeline complete!  Scroll down to explore your results.", icon="✅")

    except Exception as exc:
        st.error(f"❌  Pipeline error: {exc}")
        st.exception(exc)


# ─── Results display ──────────────────────────────────────────────────────────
if st.session_state.pipeline_done and st.session_state.pipeline_state:
    state     = st.session_state.pipeline_state
    topic_run = st.session_state.topic_run

    # Compute word counts
    search_words  = len(to_str(state.get("search_results",  "")).split())
    scraped_words = len(to_str(state.get("scraped_content", "")).split())
    report_words  = len(to_str(state.get("report",          "")).split())

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Metric strip
    st.markdown(
        f"""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-val">{search_words:,}</div>
                <div class="metric-lbl">Search words</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{scraped_words:,}</div>
                <div class="metric-lbl">Scraped words</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{report_words:,}</div>
                <div class="metric-lbl">Report words</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">4 / 4</div>
                <div class="metric-lbl">Agents done</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="results-heading">📊 Results — <span>{topic_run}</span></div>',
        unsafe_allow_html=True,
    )

    # Tabs
    t1, t2, t3, t4 = st.tabs([
        "🔍 Search Results",
        "📖 Scraped Content",
        "📄 Final Report",
        "🧐 Critic Feedback",
    ])

    with t1:
        content = to_str(state.get("search_results", ""))
        if content:
            st.markdown(content)
        else:
            st.info("No search results captured.")

    with t2:
        content = to_str(state.get("scraped_content", ""))
        if content:
            st.markdown(content)
        else:
            st.info("No scraped content captured.")

    with t3:
        report_str = to_str(state.get("report", ""))
        if report_str:
            st.markdown(report_str)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            safe_name = topic_run[:40].replace(" ", "_").replace("/", "-")
            st.download_button(
                label="⬇️  Download Report (.md)",
                data=report_str,
                file_name=f"report_{safe_name}.md",
                mime="text/markdown",
            )
            st.markdown(
                '<div class="dl-note">Markdown format · ready for Notion, Obsidian, or GitHub</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No report generated.")

    with t4:
        feedback_str = to_str(state.get("feedback", ""))
        if feedback_str:
            st.markdown(feedback_str)
        else:
            st.info("No critic feedback captured.")
"""
AI Drilling Advisor Chat — integrated into the Streamlit dashboard.
Call render_advisor_chat(df) from app.py to add the chat section.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constants (same as notebook)
# ---------------------------------------------------------------------------
COL_MAP = {
    'Time': 'Time', 'Weight on Bit': 'WOB', 'ROP Depth/Hour': 'ROP',
    'Top Drive RPM': 'RPM', 'Top Drive Torque (ft-lbs)': 'TRQ',
    'Flow In': 'FLOW_IN', 'Pump Pressure': 'SPP', 'Depth Hole TVD': 'DEPTH',
    'Differential Pressure': 'DIFF_P', 'Downhole Torque': 'DH_TRQ',
    'Hookload': 'HOOKLOAD', 'Gas Total - units': 'GAS',
    'Return Flow': 'RETURN_FLOW', 'Pit G/L Active': 'PIT_GL',
    'Block Position': 'BLOCK_POS', 'MWD Inclination': 'MWD_INC',
    'On Bottom': 'ON_BOTTOM', 'MUD TEMP': 'MUD_TEMP',
    'Trip G/L': 'TRIP_GL', 'Trip Volume Active': 'TRIP_VOL',
}
UNITS = {
    'WOB': 'klbs', 'ROP': 'm/hr', 'RPM': 'rpm', 'TRQ': 'ft-lbs',
    'FLOW_IN': 'gpm', 'SPP': 'psi', 'HOOKLOAD': 'klbs', 'DEPTH': 'm',
}
KEY_PARAMS = ['WOB', 'ROP', 'RPM', 'TRQ', 'SPP', 'FLOW_IN', 'HOOKLOAD', 'DH_TRQ', 'DIFF_P']

SYSTEM_PROMPT = """You are an expert drilling engineering advisor for well FORGE 16A(78)-32.
Your PRIMARY job is to advise whether a wiper trip should be performed.

AVAILABLE TOOLS:
- wiper_trip_assessment: Full risk scoring on latest sensor window. Call FIRST.
- report_search: Semantic search over 163 historical daily drilling reports.
- sensor_analysis: Single-parameter trend analysis.
- well_context: Well identification, depth, trajectory, data coverage.
- query_by_time_range: Get sensor data between two timestamps.
- compute_statistics: Compute mean/median/std/min/max/percentiles on parameters.
- detect_anomalies: Find sigma-based anomalies in sensor channels.
- correlate_parameters: Pearson correlation matrix between parameters.
- query_by_depth: Slice sensor data by depth interval.

WORKFLOW:
1. Call wiper_trip_assessment FIRST for current risk score
2. Call report_search for historical precedents
3. Use other tools as needed for deeper analysis
4. Synthesize a recommendation

OUTPUT FORMAT:
### CURRENT SENSOR STATUS
Summarize the risk score and key flags

### HISTORICAL PRECEDENT
Cite specific dates, depths, and outcomes from reports

### RECOMMENDATION
Clear GO / NO-GO with confidence level (HIGH/MEDIUM/LOW)

### MONITORING PLAN
What to watch if continuing without a wiper trip

Be specific. Cite actual values and dates. Safety first."""


# ---------------------------------------------------------------------------
# Agent setup (cached so it only runs once)
# ---------------------------------------------------------------------------
@st.cache_resource
def setup_agent(_df):
    """Build LangGraph agent with all 9 tools. Returns agent or None."""
    try:
        from langchain_core.tools import tool
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:
        return None, "Missing packages. Run: pip install langchain langchain-ollama langgraph"

    sensor_df = _df.copy()

    # --- ChromaDB (optional) ---
    collection = None
    try:
        import chromadb
        from chromadb.config import Settings
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        chroma_dir = os.path.join(os.path.dirname(__file__), 'chroma_db')
        if os.path.isdir(chroma_dir):
            client = chromadb.PersistentClient(path=chroma_dir, settings=Settings(anonymized_telemetry=False))
            emb_fn = SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
            collection = client.get_or_create_collection('daily_reports', embedding_function=emb_fn)
    except Exception:
        pass

    # ---- Define tools ----
    @tool
    def wiper_trip_assessment() -> str:
        """Perform a comprehensive wiper trip risk assessment using current sensor data."""
        flags, risk_score = [], 0
        n = min(100, len(sensor_df))
        window = sensor_df.iloc[-n:]
        latest = sensor_df.iloc[-1]

        for param, thresh_warn, thresh_caut, weight, label in [
            ('TRQ', 10, 5, 25, 'TORQUE'), ('HOOKLOAD', 10, 5, 20, 'HOOKLOAD'),
            ('SPP', 10, 5, 20, 'SPP'),
        ]:
            s, e = window[param].iloc[0], window[param].iloc[-1]
            pct = ((e - s) / abs(s) * 100) if s else 0
            if pct > thresh_warn:
                flags.append(f'WARNING {label} rising {pct:+.1f}%')
                risk_score += weight
            elif pct > thresh_caut:
                flags.append(f'CAUTION {label} up {pct:+.1f}%')
                risk_score += weight // 3
            else:
                flags.append(f'OK {label} stable ({pct:+.1f}%)')

        rop_s, rop_e = window['ROP'].iloc[0], window['ROP'].iloc[-1]
        rop_pct = ((rop_e - rop_s) / abs(rop_s) * 100) if rop_s else 0
        if rop_pct < -15:
            flags.append(f'WARNING ROP dropping {rop_pct:+.1f}%'); risk_score += 15
        elif rop_pct < -5:
            flags.append(f'CAUTION ROP declining {rop_pct:+.1f}%'); risk_score += 5
        else:
            flags.append(f'OK ROP normal ({rop_pct:+.1f}%)')

        risk_score = min(risk_score, 100)
        if risk_score >= 60: decision = 'RECOMMEND WIPER TRIP'
        elif risk_score >= 35: decision = 'CONSIDER WIPER TRIP'
        else: decision = 'NO WIPER TRIP NEEDED'

        depth = latest.get('DEPTH', 0)
        inc = latest.get('MWD_INC', 0) if 'MWD_INC' in sensor_df.columns else 0
        lines = [f'RISK ASSESSMENT | Score: {risk_score}/100',
                 f'Depth: {depth:,.0f} m | Inc: {inc:.1f}°', '']
        lines.extend(flags)
        lines.append(f'\nDecision: {decision}')
        lines.append('\nCurrent Readings:')
        for p in KEY_PARAMS:
            if p in sensor_df.columns:
                lines.append(f'  {p}: {latest[p]:.2f} {UNITS.get(p, "")}')
        return '\n'.join(lines)

    @tool
    def report_search(query: str) -> str:
        """Search historical daily drilling reports."""
        if collection is None:
            return 'Knowledge base not available. Run 01_knowledge_base.ipynb first.'
        try:
            results = collection.query(query_texts=[query], n_results=5)
            if not results or not results['documents'][0]:
                return 'No relevant reports found.'
            lines = []
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                date = meta.get('date_display', meta.get('date', 'Unknown'))
                lines.append(f'Report {i+1} ({date}): {doc[:400]}\n')
            return '\n'.join(lines)
        except Exception as e:
            return f'Error: {e}'

    @tool
    def sensor_analysis(query: str) -> str:
        """Analyze a specific sensor parameter. Available: WOB, ROP, RPM, TRQ, SPP, FLOW_IN, HOOKLOAD, DH_TRQ, DIFF_P."""
        q = query.lower()
        param = 'TRQ'
        aliases = {'torque': 'TRQ', 'pressure': 'SPP', 'weight': 'WOB', 'rop': 'ROP',
                   'hookload': 'HOOKLOAD', 'flow': 'FLOW_IN', 'gas': 'GAS', 'rpm': 'RPM'}
        for a, p in aliases.items():
            if a in q: param = p; break
        for p in KEY_PARAMS:
            if p.lower() in q: param = p; break
        n = min(100, len(sensor_df))
        window = sensor_df.iloc[-n:]
        cur, start_val = window[param].iloc[-1], window[param].iloc[0]
        pct = ((cur - start_val) / abs(start_val) * 100) if start_val else 0
        trend = 'INCREASING' if pct > 2 else 'DECREASING' if pct < -2 else 'STABLE'
        mean, std = sensor_df[param].mean(), sensor_df[param].std()
        return (f'{param}: Current={cur:.2f} {UNITS.get(param, "")}\n'
                f'Trend: {trend} ({pct:+.1f}%)\n'
                f'Stats: mean={mean:.2f}, std={std:.2f}, '
                f'range=[{sensor_df[param].min():.2f}, {sensor_df[param].max():.2f}]')

    @tool
    def well_context() -> str:
        """Get well identification, depth, trajectory, and data coverage."""
        row = sensor_df.iloc[-1]
        inc = row.get('MWD_INC', 0) if 'MWD_INC' in sensor_df.columns else 0
        return (f'Well: FORGE 16A(78)-32 (Utah FORGE geothermal)\n'
                f'Depth: {row.get("DEPTH", 0):,.0f} m TVD | Inc: {inc:.1f}°\n'
                f'Data: {len(sensor_df):,} points, {sensor_df["Time"].min()} to {sensor_df["Time"].max()}')

    # Import the 5 data-access tools
    try:
        from agent_tools import create_data_tools
        data_tools = create_data_tools(sensor_df, KEY_PARAMS, UNITS)
    except Exception:
        data_tools = []

    all_tools = [wiper_trip_assessment, report_search, sensor_analysis, well_context] + data_tools

    # Build agent
    try:
        llm = ChatOllama(model='qwen2.5:7b', temperature=0)
        memory = MemorySaver()
        agent = create_react_agent(
            model=llm, tools=all_tools,
            prompt=SystemMessage(content=SYSTEM_PROMPT),
            checkpointer=memory,
        )
        return agent, None
    except Exception as e:
        return None, f"Could not connect to Ollama: {e}"


# ---------------------------------------------------------------------------
# Chat UI renderer — Premium Glassmorphic Design
# ---------------------------------------------------------------------------
CHAT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ---- Advisor Outer Container ---- */
.advisor-outer {
    margin-top: 40px;
    padding-top: 24px;
    border-top: 1px solid #1e293b;
}

/* ---- Advisor Chat Box (Glassmorphic) ---- */
.advisor-box {
    background: linear-gradient(145deg, rgba(15,22,41,0.95) 0%, rgba(19,27,46,0.92) 100%);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 16px;
    padding: 28px 28px 20px;
    box-shadow:
        0 0 30px rgba(56,189,248,0.04),
        0 8px 32px rgba(0,0,0,0.4),
        inset 0 1px 0 rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
    margin-bottom: 16px;
}
.advisor-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8, #818cf8, #38bdf8, transparent);
    opacity: 0.6;
}

/* ---- Header ---- */
.advisor-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}
.advisor-icon {
    width: 40px; height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, #1e3a5f, #0f2847);
    border: 1px solid rgba(56,189,248,0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 0 16px rgba(56,189,248,0.1);
}
.advisor-title {
    font-family: 'Inter', sans-serif;
    color: #f1f5f9;
    margin: 0;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.3px;
}
.advisor-title span {
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ---- Status Bar ---- */
.advisor-status-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    color: #64748b;
    font-size: 11px;
    font-family: 'Inter', sans-serif;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid rgba(30,41,59,0.8);
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 20px;
    padding: 3px 12px 3px 8px;
    font-size: 11px;
    color: #22c55e;
    font-weight: 500;
}
.status-pill.offline {
    background: rgba(239,68,68,0.08);
    border-color: rgba(239,68,68,0.2);
    color: #ef4444;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #22c55e;
    animation: glow-pulse 2s ease-in-out infinite;
}
.status-pill.offline .status-dot {
    background: #ef4444;
    animation: none;
}
@keyframes glow-pulse {
    0%, 100% { box-shadow: 0 0 4px rgba(34,197,94,0.6); opacity: 1; }
    50% { box-shadow: 0 0 8px rgba(34,197,94,0.3); opacity: 0.5; }
}
.status-tag {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 10px;
    color: #38bdf8;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* ---- Chat Area ---- */
.chat-area {
    background: rgba(10,14,23,0.5);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px;
    max-height: 520px;
    overflow-y: auto;
    margin-bottom: 4px;
    scrollbar-width: thin;
    scrollbar-color: #1e293b transparent;
}
.chat-area::-webkit-scrollbar { width: 5px; }
.chat-area::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
.chat-area::-webkit-scrollbar-track { background: transparent; }

/* ---- Message Row ---- */
.msg-row {
    display: flex;
    gap: 14px;
    margin-bottom: 20px;
    animation: msg-in 0.3s ease-out;
}
.msg-row.user { flex-direction: row-reverse; }
@keyframes msg-in {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---- Avatar ---- */
.msg-avatar {
    width: 38px; height: 38px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
    margin-top: 2px;
}
.msg-avatar.assistant {
    background: linear-gradient(135deg, #1e3a5f, #0f2847);
    border: 1px solid rgba(56,189,248,0.2);
}
.msg-avatar.user {
    background: linear-gradient(135deg, #312e81, #1e1b4b);
    border: 1px solid rgba(129,140,248,0.2);
}

/* ---- Bubble ---- */
.msg-bubble {
    max-width: 70%;
    font-family: 'Inter', sans-serif;
}
.msg-row.user .msg-bubble {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
}
.msg-name {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 5px;
}
.msg-name.assistant { color: #38bdf8; }
.msg-name.user { color: #818cf8; text-align: right; }

.msg-content {
    padding: 14px 20px;
    border-radius: 18px;
    font-size: 15px;
    line-height: 1.7;
    word-wrap: break-word;
    white-space: pre-wrap;
    width: fit-content;
}
.msg-content.assistant {
    background: linear-gradient(145deg, #111827, #0f172a);
    color: #e2e8f0;
    border: 1px solid #1e293b;
    border-top-left-radius: 4px;
}
.msg-content.user {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    color: #f0f4ff;
    border-top-right-radius: 4px;
    box-shadow: 0 2px 12px rgba(59,130,246,0.15);
}
.msg-tools {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 8px;
}
.tool-chip {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 10px;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
}

/* ---- Welcome ---- */
.welcome-msg {
    text-align: center;
    padding: 48px 20px;
}
.welcome-icon {
    font-size: 44px;
    margin-bottom: 16px;
    display: block;
}
.welcome-title {
    font-family: 'Inter', sans-serif;
    font-size: 20px;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 8px;
}
.welcome-sub {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    color: #94a3b8;
    line-height: 1.6;
    max-width: 480px;
    margin: 0 auto;
}
/* ---- Quick Action Cards ---- */
.qa-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 18px;
}
.qa-card {
    background: linear-gradient(145deg, #111827, #0f172a);
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s ease;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
}
.qa-card:hover {
    border-color: rgba(56,189,248,0.4);
    background: linear-gradient(145deg, #151d2e, #111827);
    box-shadow: 0 4px 20px rgba(56,189,248,0.08);
    transform: translateY(-2px);
}
.qa-icon {
    font-size: 32px;
    line-height: 1;
}
.qa-label {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: #e2e8f0;
}
.qa-desc {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    color: #64748b;
    line-height: 1.3;
}
</style>
"""


def _render_message(role, content, tools=None):
    """Return HTML for a single chat message with avatar."""
    import re
    avatar = "🤖" if role == "assistant" else "👤"
    name = "Advisor" if role == "assistant" else "You"

    if role == "assistant":
        # Process markdown for assistant responses
        safe = content
        # Headers: ### → section header, ## → subheader
        safe = re.sub(r'^\s*#{3,}\s*(.+)$', r'<div style="font-size:15px;font-weight:700;color:#38bdf8;margin:14px 0 8px;text-transform:uppercase;letter-spacing:0.5px;border-bottom:1px solid #1e293b;padding-bottom:5px;">\1</div>', safe, flags=re.MULTILINE)
        safe = re.sub(r'^\s*#{2}\s*(.+)$', r'<div style="font-size:16px;font-weight:700;color:#e2e8f0;margin:12px 0 8px;">\1</div>', safe, flags=re.MULTILINE)
        safe = re.sub(r'^\s*#\s*(.+)$', r'<div style="font-size:17px;font-weight:700;color:#f1f5f9;margin:12px 0 8px;">\1</div>', safe, flags=re.MULTILINE)
        
        # Remove LaTeX inline math markers \( and \)
        safe = re.sub(r'\\\((.*?)\\\)', r'\1', safe)
        
        # Bold: **text**
        safe = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#f1f5f9;font-weight:600;">\1</strong>', safe)
        # Italic: *text*
        safe = re.sub(r'\*(.+?)\*', r'<em>\1</em>', safe)
        
        # Bullet lists: - item (allow leading spaces)
        safe = re.sub(r'^\s*[\-\*]\s+(.+)$', r'<div style="padding:3px 0 3px 14px;border-left:2px solid #334155;margin:3px 0;font-size:15px;">• \1</div>', safe, flags=re.MULTILINE)
        # Numbered lists: 1. item (allow leading spaces)
        safe = re.sub(r'^\s*(\d+)\.\s+(.+)$', r'<div style="padding:3px 0 3px 14px;border-left:2px solid #334155;margin:3px 0;font-size:15px;">\1. \2</div>', safe, flags=re.MULTILINE)
        # Line breaks
        safe = safe.replace("\n", "<br>")
        # Clean up double <br> after block elements
        safe = re.sub(r'</div><br>', '</div>', safe)
    else:
        safe = (content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\n", "<br>"))

    tools_html = ""
    if tools:
        chips = "".join(f'<span class="tool-chip">{t}</span>' for t in tools)
        tools_html = f'<div class="msg-tools">{chips}</div>'
    return (
        f'<div class="msg-row {role}">'
        f'  <div class="msg-avatar {role}">{avatar}</div>'
        f'  <div class="msg-bubble">'
        f'    <div class="msg-name {role}">{name}</div>'
        f'    <div class="msg-content {role}">{safe}{tools_html}</div>'
        f'  </div>'
        f'</div>'
    )


def render_advisor_chat(df):
    """Render the AI advisor chat section — premium glassmorphic design."""
    st.markdown(CHAT_CSS, unsafe_allow_html=True)

    agent, error = setup_agent(df)
    online = error is None

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_thread" not in st.session_state:
        st.session_state.chat_thread = "streamlit-default"

    # ---- Outer Container ----
    st.markdown('<div class="advisor-outer">', unsafe_allow_html=True)

    # ---- Box ----
    pill_cls = "" if online else " offline"
    status_label = "Online" if online else "Offline"
    st.markdown(
        f'<div class="advisor-box">'
        f'  <div class="advisor-header">'
        f'    <div class="advisor-icon">🤖</div>'
        f'    <div>'
        f'      <div class="advisor-title"><span>AI Drilling Advisor</span></div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="advisor-status-bar">'
        f'    <div class="status-pill{pill_cls}"><span class="status-dot"></span>{status_label}</div>'
        f'    <span class="status-tag">Qwen 2.5 · 9 Tools</span>'
        f'    <span class="status-tag">Well 16A(78)-32</span>'
        f'  </div>',
        unsafe_allow_html=True,
    )

    if not online:
        st.markdown(
            f'<div style="color:#ef4444;font-size:13px;padding:12px;">{error}</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return

    # ---- Quick Action Cards ----
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        st.markdown(
            '<div class="qa-card">'
            '  <div class="qa-icon">🔍</div>'
            '  <div class="qa-label">Risk Assessment</div>'
            '  <div class="qa-desc">Evaluate wiper trip need</div>'
            '</div>', unsafe_allow_html=True,
        )
        if st.button("Run", key="qa_risk", use_container_width=True):
            st.session_state.chat_pending = "Should we perform a wiper trip right now? Analyze current sensor risk and historical precedents."
    with qa2:
        st.markdown(
            '<div class="qa-card">'
            '  <div class="qa-icon">📊</div>'
            '  <div class="qa-label">Torque Analysis</div>'
            '  <div class="qa-desc">Trends & anomalies</div>'
            '</div>', unsafe_allow_html=True,
        )
        if st.button("Run", key="qa_torque", use_container_width=True):
            st.session_state.chat_pending = "Analyze the torque and hookload trends in detail. Are they concerning?"
    with qa3:
        st.markdown(
            '<div class="qa-card">'
            '  <div class="qa-icon">📋</div>'
            '  <div class="qa-label">Historical Events</div>'
            '  <div class="qa-desc">Past incidents & reports</div>'
            '</div>', unsafe_allow_html=True,
        )
        if st.button("Run", key="qa_hist", use_container_width=True):
            st.session_state.chat_pending = "What historical wiper trip events happened at similar depths and conditions?"
    with qa4:
        st.markdown(
            '<div class="qa-card">'
            '  <div class="qa-icon">🔗</div>'
            '  <div class="qa-label">Correlations</div>'
            '  <div class="qa-desc">Parameter relationships</div>'
            '</div>', unsafe_allow_html=True,
        )
        if st.button("Run", key="qa_corr", use_container_width=True):
            st.session_state.chat_pending = "Correlate torque, hookload, SPP, and ROP. Are there concerning relationships?"

    # ---- Chat Area ----
    if st.session_state.chat_messages:
        html = '<div class="chat-area">'
        for msg in st.session_state.chat_messages:
            html += _render_message(msg["role"], msg["content"], msg.get("tools"))
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="chat-area">'
            '  <div class="welcome-msg">'
            '    <span class="welcome-icon">⛽</span>'
            '    <div class="welcome-title">What can I help you with?</div>'
            '    <div class="welcome-sub">'
            '      Ask about wiper trip decisions, sensor anomalies, torque trends, '
            '      historical events, or parameter correlations for well 16A(78)-32.'
            '    </div>'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Close box + outer
    st.markdown('</div></div>', unsafe_allow_html=True)

    # ---- Input ----
    pending = st.session_state.pop("chat_pending", None)
    
    with st.form(key="advisor_input_form", clear_on_submit=True, border=False):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("Ask the drilling advisor...", label_visibility="collapsed", placeholder="Ask the drilling advisor...")
        with col2:
            submit_btn = st.form_submit_button("Send", use_container_width=True)
            
    question = pending or (user_input if submit_btn and user_input.strip() else None)

    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})

        with st.spinner("🤖 Analyzing with sensor data & historical reports..."):
            try:
                from langchain_core.messages import HumanMessage, AIMessage
                config = {'configurable': {'thread_id': st.session_state.chat_thread}}
                result = agent.invoke(
                    {'messages': [HumanMessage(content=question)]}, config=config
                )
                response, tools_used = "", []
                for msg in result['messages']:
                    if isinstance(msg, AIMessage):
                        if msg.content:
                            response = msg.content
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tools_used.append(tc['name'])
            except Exception as e:
                response = f"Error: {e}"
                tools_used = []

        st.session_state.chat_messages.append({
            "role": "assistant", "content": response,
            "tools": tools_used if tools_used else None,
        })
        st.rerun()

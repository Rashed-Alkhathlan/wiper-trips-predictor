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
# Chat UI renderer
# ---------------------------------------------------------------------------
def render_advisor_chat(df):
    """Render the AI advisor chat section. Call from app.py."""
    st.markdown(
        '<div style="margin-top:32px; padding-top:16px; border-top:2px solid #1e293b;">'
        '<h3 style="color:#38bdf8; margin-bottom:4px;">🤖 AI Drilling Advisor</h3>'
        '<p style="color:#64748b; font-size:13px; margin-bottom:16px;">'
        'Ask questions about wiper trips, sensor trends, or historical events. '
        'Powered by Qwen 2.5 + 9 analysis tools.</p></div>',
        unsafe_allow_html=True,
    )

    # Init agent
    agent, error = setup_agent(df)
    if error:
        st.error(f"⚠️ Agent unavailable: {error}")
        return

    # Session state for chat
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_thread" not in st.session_state:
        st.session_state.chat_thread = "streamlit-default"

    # Quick actions
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("🔍 Risk Assessment", use_container_width=True, key="qa_risk"):
            st.session_state.chat_pending = "Should we perform a wiper trip right now? Analyze current sensor risk and historical precedents."
    with qa2:
        if st.button("📊 Torque Analysis", use_container_width=True, key="qa_torque"):
            st.session_state.chat_pending = "Analyze the torque and hookload trends in detail. Are they concerning?"
    with qa3:
        if st.button("📋 Historical Events", use_container_width=True, key="qa_hist"):
            st.session_state.chat_pending = "What historical wiper trip events happened at similar depths and conditions?"
    with qa4:
        if st.button("🔗 Correlations", use_container_width=True, key="qa_corr"):
            st.session_state.chat_pending = "Correlate torque, hookload, SPP, and ROP. Are there concerning relationships?"

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle input (from quick action or text input)
    pending = st.session_state.pop("chat_pending", None)
    user_input = st.chat_input("Ask the drilling advisor...", key="advisor_input")
    question = pending or user_input

    if question:
        # Show user message
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    from langchain_core.messages import HumanMessage, AIMessage
                    config = {'configurable': {'thread_id': st.session_state.chat_thread}}
                    result = agent.invoke(
                        {'messages': [HumanMessage(content=question)]}, config=config
                    )
                    # Extract final response
                    response = ""
                    tools_used = []
                    for msg in result['messages']:
                        if isinstance(msg, AIMessage):
                            if msg.content:
                                response = msg.content
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    tools_used.append(tc['name'])
                    if tools_used:
                        st.caption(f"🔧 Tools used: {', '.join(tools_used)}")
                    st.markdown(response)
                except Exception as e:
                    response = f"Error: {e}"
                    st.error(response)

        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.rerun()

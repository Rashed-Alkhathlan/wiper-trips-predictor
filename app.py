import os
import warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*__path__.*")

import streamlit as st

# We must define the pages
def fleet_dashboard():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        .stApp {
            background-color: #0a0e17;
            color: #e2e8f0;
        }
        
        .fleet-header {
            font-family: 'Inter', sans-serif;
            margin-top: 0rem;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .fleet-title {
            font-size: 36px;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }
        
        .fleet-subtitle {
            font-size: 16px;
            color: #64748b;
        }

        .rig-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 24px;
            padding: 20px;
        }
        
        .rig-card {
            background: linear-gradient(145deg, rgba(15,22,41,0.95) 0%, rgba(19,27,46,0.92) 100%);
            border: 1px solid #1e293b;
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        
        .rig-card:hover {
            transform: translateY(-5px);
            border-color: rgba(56,189,248,0.4);
            box-shadow: 0 10px 30px rgba(56,189,248,0.1);
        }
        
        .rig-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #334155, #1e293b);
            opacity: 0.8;
        }
        
        .rig-card.active-real::before {
            background: linear-gradient(90deg, #38bdf8, #818cf8);
        }

        .rig-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        
        .rig-name {
            font-family: 'Inter', sans-serif;
            font-size: 20px;
            font-weight: 700;
            color: #f1f5f9;
        }
        
        .rig-status {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-drilling { background: rgba(34,197,94,0.1); color: #22c55e; border: 1px solid rgba(34,197,94,0.2); }
        .status-tripping { background: rgba(245,158,11,0.1); color: #f59e0b; border: 1px solid rgba(245,158,11,0.2); }
        .status-maintenance { background: rgba(239,68,68,0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.2); }

        .rig-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }
        
        .detail-item {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .detail-label {
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .detail-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 15px;
            font-weight: 600;
            color: #e2e8f0;
        }

        .risk-bar-container {
            width: 100%;
            height: 6px;
            background: #1e293b;
            border-radius: 3px;
            margin-top: 8px;
            overflow: hidden;
        }
        
        .risk-bar {
            height: 100%;
            border-radius: 3px;
        }
        
        .risk-low { width: 15%; background: #22c55e; }
        .risk-med { width: 45%; background: #f59e0b; }
        .risk-high { width: 85%; background: #ef4444; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="fleet-header">
            <div class="fleet-title">Global Fleet Operations</div>
            <div class="fleet-subtitle">Live Monitoring & Advisory System</div>
        </div>
    """, unsafe_allow_html=True)

    rigs = [
        {"name": "FORGE 16A(78)-32", "loc": "Utah, USA", "depth": "2,450 m", "status": "Drilling", "cls": "drilling", "risk": "med", "real": True},
        {"name": "Permian Alpha-1", "loc": "Texas, USA", "depth": "4,120 m", "status": "Drilling", "cls": "drilling", "risk": "low", "real": False},
        {"name": "North Sea Horizon", "loc": "Offshore UK", "depth": "3,890 m", "status": "Tripping", "cls": "tripping", "risk": "high", "real": False},
        {"name": "Deepwater Titan-9", "loc": "Gulf of Mexico", "depth": "5,600 m", "status": "Drilling", "cls": "drilling", "risk": "low", "real": False},
        {"name": "Bakken Explorer-2", "loc": "North Dakota", "depth": "3,150 m", "status": "Maintenance", "cls": "maintenance", "risk": "low", "real": False},
        {"name": "Eagle Ford Sigma-4", "loc": "Texas, USA", "depth": "2,980 m", "status": "Drilling", "cls": "drilling", "risk": "med", "real": False},
        {"name": "Gulf Coast Pioneer", "loc": "Louisiana", "depth": "1,200 m", "status": "Tripping", "cls": "tripping", "risk": "low", "real": False},
        {"name": "Arctic Voyager", "loc": "Alaska, USA", "depth": "4,800 m", "status": "Drilling", "cls": "drilling", "risk": "high", "real": False},
        {"name": "Marcellus Driller", "loc": "Pennsylvania", "depth": "2,100 m", "status": "Drilling", "cls": "drilling", "risk": "low", "real": False},
        {"name": "Sahara Prospector", "loc": "Algeria", "depth": "3,400 m", "status": "Maintenance", "cls": "maintenance", "risk": "low", "real": False},
    ]

    cols = st.columns(3)
    
    for i, rig in enumerate(rigs):
        col = cols[i % 3]
        with col:
            # We use an empty container to hold the card HTML
            card_html = f'''
            <div class="rig-card {'active-real' if rig['real'] else ''}">
                <div class="rig-header">
                    <div class="rig-name">{rig["name"]}</div>
                    <div class="rig-status status-{rig["cls"]}">{rig["status"]}</div>
                </div>
                <div class="rig-details">
                    <div class="detail-item">
                        <span class="detail-label">Location</span>
                        <span class="detail-value">{rig["loc"]}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Current Depth</span>
                        <span class="detail-value">{rig["depth"]}</span>
                    </div>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Wiper Trip Risk Level</span>
                    <div class="risk-bar-container">
                        <div class="risk-bar risk-{rig["risk"]}"></div>
                    </div>
                </div>
            </div>
            '''
            
            # Using st.button with width="stretch" to act as a clickable invisible overlay
            # To make the whole card clickable nicely in Streamlit, we render the HTML and put a button over it, 
            # or just use a button for the logic.
            
            # Since Streamlit buttons are hard to style as complex cards, we use standard buttons
            # wait, we can just use a container. Streamlit 1.30+ has st.button(..., type="tertiary")
            
            if st.button(f"Enter {rig['name']}", key=f"btn_{i}", width="stretch"):
                if rig["real"]:
                    st.switch_page(rig_page)
                else:
                    st.warning(f"{rig['name']} is a simulated rig for dashboard demonstration only.")
            
            st.markdown(card_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

# Define pages
fleet_page = st.Page(fleet_dashboard, title="Fleet Dashboard", icon="🌍", default=True)
rig_page = st.Page("app/rig_dashboard.py", title="FORGE 16A(78)-32", icon="⚙️")

pg = st.navigation([fleet_page, rig_page])

st.set_page_config(
    page_title="Global Fleet Operations",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom hide sidebar via CSS if needed
st.markdown("""
    <style>
    [data-testid="collapsedControl"] { display: none; }
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        height: 0px !important;
        min-height: 0px !important;
        padding: 0 !important;
        visibility: hidden !important;
        display: none !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

pg.run()

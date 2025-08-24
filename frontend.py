# ---------- frontend.py ----------
import threading, time, random
import streamlit as st
import pandas as pd, numpy as np
from datetime import datetime
import openai
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
import hashlib

# ---------------- CONFIG ----------------
OPENAI_API_KEY = "YOUR_OPENAI_KEY"  # Replace with your OpenAI key
openai.api_key = OPENAI_API_KEY

# ---------------- GLOBAL STATE ----------------
if "attacks" not in st.session_state: st.session_state.attacks = []
if "tx_results" not in st.session_state: st.session_state.tx_results = []
if "code_suggestions" not in st.session_state: st.session_state.code_suggestions = []
if "auto_simulation" not in st.session_state: st.session_state.auto_simulation = True
if "auto_loop_thread" not in st.session_state: st.session_state.auto_loop_thread = None

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide", page_title="🚀 AI-Powered Secure FinTech Dashboard")
st.markdown("<h1 style='text-align:center;color:#4B0082;'>🚀 AI-Powered Secure FinTech Platform</h1>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Simulation Controls")
st.session_state.auto_simulation = st.sidebar.checkbox("Auto Simulation", value=True)
if st.sidebar.button("Stop Auto Simulation"):
    st.session_state.auto_simulation = False
    st.success("✅ Auto Simulation Stopped")

# ---------------- METRICS CARDS ----------------
col1, col2, col3 = st.columns(3)
with col1: st.metric("🤖 AI Code Suggestions", len(st.session_state.code_suggestions))
with col2: st.metric("🛡️ High Severity Attacks", sum(1 for a in st.session_state.attacks if a["Severity"]=="High"), delta=len(st.session_state.attacks))
with col3: st.metric("💰 Total Transactions", len(st.session_state.tx_results), delta=sum(1 for t in st.session_state.tx_results if t["Status"]=="Fraud Detected"))

# ---------------- AI CODE GENERATOR ----------------
st.markdown("## 🤖 Smart Automation & Risk Analyzer")
col1, col2 = st.columns([2,1])
with col1:
    prompt = st.text_area("Describe function for automation:")
with col2:
    if st.button("Generate Code"):
        if prompt.strip() != "":
            suggestions=[]
            with st.spinner("Generating 3 code suggestions..."):
                for _ in range(3):
                    resp = openai.Completion.create(
                        engine="code-davinci-002",
                        prompt=f"Write Python code for: {prompt}",
                        max_tokens=150
                    )
                    suggestions.append(resp.choices[0].text)
            st.session_state.code_suggestions = suggestions
if st.session_state.code_suggestions:
    for idx, code in enumerate(st.session_state.code_suggestions):
        st.markdown(f"#### Suggestion {idx+1}")
        st.code(code, language="python")
        st.download_button(f"💾 Download Suggestion {idx+1}", code, file_name=f"code_suggestion_{idx+1}.py", key=idx)

# ---------------- HONEYPOT ----------------
st.markdown("## 🛡️ Real-Time Threat Detection")
attack_placeholder = st.empty()

def simulate_attack():
    severity = random.choice(["Low","Medium","High"])
    attack = {"Time":datetime.now().strftime("%H:%M:%S"),
              "IP":f"192.168.1.{random.randint(2,254)}",
              "User":f"user{random.randint(1,50)}",
              "Pass":f"pass{random.randint(100,999)}",
              "Severity":severity}
    st.session_state.attacks.append(attack)
    if severity=="High": st.error(f"⚠️ High severity attack from {attack['IP']}")

col_a1, col_a2 = st.columns(2)
with col_a1:
    if st.button("Simulate Attack"):
        simulate_attack()
with col_a2:
    if st.session_state.attacks: attack_placeholder.dataframe(pd.DataFrame(st.session_state.attacks))

def plot_attacks():
    if st.session_state.attacks:
        df = pd.DataFrame(st.session_state.attacks)
        fig = px.histogram(df, x="Severity", color="Severity", title="Attacks per Severity",
                           color_discrete_map={"Low":"green","Medium":"orange","High":"red"})
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.pie(df, names="Severity", title="Severity Distribution",
                      color_discrete_map={"Low":"green","Medium":"orange","High":"red"})
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- TRANSACTIONS & BLOCKCHAIN ----------------
st.markdown("## 💰 Transaction Processing & Animated Blockchain")
X = np.array([[100,0],[10000,5],[50,0],[50000,8]])
y = np.array([0,1,0,1])
model = RandomForestClassifier()
model.fit(X,y)

def run_transactions():
    transactions = [[random.randint(50,20000), random.randint(0,10)] for _ in range(5)]
    results=[]
    for tx in transactions:
        pred = model.predict([tx])[0]
        status = "Fraud Detected" if pred else "Approved"
        color = "❌" if status=="Fraud Detected" else "✅"
        # Generate a mock blockchain hash
        block_hash = hashlib.sha256(str(tx).encode()).hexdigest()[:10]
        results.append({"Transaction":tx,"Status":status,"Indicator":color,"BlockHash":block_hash})
        if status=="Fraud Detected": st.error(f"⚠️ Fraud detected for transaction {tx}")
    st.session_state.tx_results = results

col_tx1, col_tx2 = st.columns(2)
with col_tx1:
    if st.button("Run Transactions"): run_transactions()
    if st.button("Download Blockchain CSV"):
        if st.session_state.tx_results:
            pd.DataFrame(st.session_state.tx_results).to_csv("blockchain.csv", index=False)
            st.success("💾 CSV downloaded")
with col_tx2:
    if st.session_state.tx_results: st.dataframe(pd.DataFrame(st.session_state.tx_results))

# ---------------- ANIMATED BLOCKCHAIN ----------------
def plot_animated_blockchain():
    if st.session_state.tx_results:
        fig = go.Figure()
        y_level = 0
        for idx, tx in enumerate(st.session_state.tx_results):
            x_pos = idx * 2
            color = "green" if tx["Status"]=="Approved" else "red"
            fig.add_trace(go.Scatter(
                x=[x_pos], y=[y_level],
                mode='markers+text',
                marker=dict(size=50, color=color),
                text=[f"{tx['Indicator']}\n{tx['Transaction']}\n{tx['BlockHash']}"],
                textposition="bottom center"
            ))
            if idx > 0:
                fig.add_trace(go.Scatter(
                    x=[(idx-1)*2, x_pos], y=[y_level, y_level],
                    mode='lines', line=dict(color="#888", width=2)
                ))
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=300, margin=dict(l=20,r=20,t=20,b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

plot_animated_blockchain()

# ---------------- AUTO SIMULATION ----------------
def auto_loop():
    while st.session_state.auto_simulation:
        simulate_attack()
        plot_attacks()
        run_transactions()
        plot_animated_blockchain()
        if st.session_state.attacks:
            attack_placeholder.dataframe(pd.DataFrame(st.session_state.attacks))
        time.sleep(3)

if st.session_state.auto_simulation and st.session_state.auto_loop_thread is None:
    t = threading.Thread(target=auto_loop, daemon=True)
    t.start()
    st.session_state.auto_loop_thread = t

"""
PlaceAI — Prediction Page + Grok AI Chatbot
Single-page Streamlit app
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, time, requests, json

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="PlaceAI · Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── LOAD MODEL ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    BASE = os.path.dirname(os.path.abspath(__file__))
    return (
        pickle.load(open(os.path.join(BASE, "placement_model.pkl"), "rb")),
        pickle.load(open(os.path.join(BASE, "scaler.pkl"),          "rb")),
        pickle.load(open(os.path.join(BASE, "encoders.pkl"),        "rb")),
    )

model, scaler, enc = load_model()
M, S = scaler.mean_, scaler.scale_

FEAT = ["college_id","prev_sem_result","cgpa","academic_performance",
        "internship_experience","extra_curricular_score",
        "communication_skills","projects_completed"]

def scale(v, i):
    return (v - M[i]) / S[i]

def run_predict(cid, prev_sem, cgpa, acad, intern_yn, extra, comm, proj):
    le = enc["college_id"]
    if cid in le.classes_:
        cid_e = int(le.transform([cid])[0])
    else:
        digits = "".join(filter(str.isdigit, cid))
        cid_e = max(0, min(int(digits) - 1, len(le.classes_) - 1)) if digits else len(le.classes_) // 2

    row = pd.DataFrame([[
        cid_e,
        scale(prev_sem, 0), scale(cgpa, 1), scale(acad, 2),
        1 if intern_yn == "Yes" else 0,
        scale(extra, 3), scale(comm, 4), scale(proj, 5)
    ]], columns=FEAT)

    pred  = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    placed = bool(pred == 1)
    conf   = round(float(proba[pred]) * 100, 1)
    placed_prob = round(float(proba[1]) * 100, 1)
    return placed, conf, placed_prob

# ── GROK API ───────────────────────────────────────────────────
def grok_chat(messages: list, api_key: str) -> str:
    """Call xAI Grok API."""
    try:
        r = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-beta",
                "messages": messages,
                "max_tokens": 400,
                "temperature": 0.7,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else "?"
        if code == 401:
            return "❌ Invalid API key. Please check your Grok API key."
        elif code == 429:
            return "⚠️ Rate limit reached. Please wait a moment."
        return f"❌ API error {code}. Please try again."
    except Exception as e:
        return f"❌ Error: {str(e)}"

SYSTEM_PROMPT = """You are PlaceAI Assistant, an expert AI career counselor specializing in campus placements and student career guidance. You help students with:
- Improving their placement chances
- Technical interview preparation (DSA, system design, coding)
- Soft skills and communication tips
- Resume and project advice
- Internship guidance
- Understanding their placement prediction results

Be concise, friendly, encouraging and practical. Use emojis occasionally. Keep answers under 200 words unless asked for detail."""

# ── SESSION STATE ───────────────────────────────────────────────
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
if "chat_visible" not in st.session_state:
    st.session_state.chat_visible = False
if "result" not in st.session_state:
    st.session_state.result = None
if "grok_key" not in st.session_state:
    st.session_state.grok_key = ""

# ══════════════════════════════════════════════════════════════
#  FULL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg: #04040f;
  --s:  rgba(255,255,255,0.048);
  --b:  rgba(255,255,255,0.085);
  --b2: rgba(0,245,255,0.22);
  --cy: #00f5ff;
  --bl: #4f7fff;
  --pu: #a855f7;
  --mg: #f059ff;
  --gr: #00ffaa;
  --rd: #ff4f6a;
  --am: #ffb340;
  --tx: #e8e8f8;
  --mu: rgba(200,200,230,0.5);
  --fh: 'Syne', sans-serif;
  --fb: 'DM Sans', sans-serif;
  --gc: 0 0 24px rgba(0,245,255,.4), 0 0 70px rgba(0,245,255,.15);
  --gp: 0 0 24px rgba(168,85,247,.4), 0 0 70px rgba(168,85,247,.15);
}

/* ── KILL STREAMLIT CHROME ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebar"],
.stDeployButton { display: none !important; }

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: var(--fb) !important;
  color: var(--tx) !important;
}
.stApp, .stApp > div { background: var(--bg) !important; }
.main .block-container {
  padding: 0 !important;
  max-width: 100% !important;
  background: transparent !important;
}
[data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="column"] { padding: 0 8px !important; }

::-webkit-scrollbar { width: 5px; background: #07071a; }
::-webkit-scrollbar-thumb { background: rgba(0,245,255,.18); border-radius: 3px; }

/* ── FIXED BACKGROUNDS ── */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: -2;
  background:
    radial-gradient(ellipse 80% 60% at 10% 30%, rgba(0,245,255,.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 70% at 90% 15%, rgba(168,85,247,.07) 0%, transparent 55%),
    radial-gradient(ellipse 50% 50% at 50% 95%, rgba(79,127,255,.04) 0%, transparent 50%),
    #04040f;
}
body::after {
  content: '';
  position: fixed; inset: 0; z-index: -1; pointer-events: none;
  background-image:
    linear-gradient(rgba(0,245,255,.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,245,255,.02) 1px, transparent 1px);
  background-size: 64px 64px;
  mask-image: radial-gradient(ellipse 85% 85% at 50% 40%, #000 0%, transparent 72%);
  -webkit-mask-image: radial-gradient(ellipse 85% 85% at 50% 40%, #000 0%, transparent 72%);
}

/* ── PAGE WRAPPER ── */
.page-wrap {
  max-width: 1260px;
  margin: 0 auto;
  padding: 28px 40px 60px;
}

/* ── TOP HEADER BAR ── */
.top-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 40px;
  height: 62px;
  background: rgba(4,4,15,.85);
  backdrop-filter: blur(24px) saturate(1.5);
  -webkit-backdrop-filter: blur(24px) saturate(1.5);
  border-bottom: 1px solid var(--b);
  position: sticky; top: 0; z-index: 400;
}
.top-logo {
  font-family: var(--fh); font-weight: 800; font-size: 1.1rem;
  background: linear-gradient(90deg, var(--cy), var(--pu));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; letter-spacing: -.01em;
}
.top-badges { display: flex; gap: 10px; align-items: center; }
.top-badge {
  padding: 5px 14px; border-radius: 20px; font-size: .7rem;
  letter-spacing: .1em; text-transform: uppercase;
  border: 1px solid var(--b); color: var(--mu);
}
.top-badge.live {
  border-color: rgba(0,255,170,.25);
  background: rgba(0,255,170,.07);
  color: var(--gr);
}
.top-badge.live::before {
  content: '●'; margin-right: 5px;
  animation: blink 1.4s ease infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

/* ── PAGE TITLE ── */
.pg-title { margin-bottom: 32px; padding-top: 32px; }
.pg-kicker {
  font-size: .7rem; letter-spacing: .2em; text-transform: uppercase;
  color: var(--cy); font-weight: 600; margin-bottom: 10px;
}
.pg-h1 {
  font-family: var(--fh);
  font-size: clamp(1.9rem, 3.5vw, 2.9rem);
  font-weight: 800; letter-spacing: -.03em; line-height: 1.1;
  margin-bottom: 10px;
}
.pg-h1 .grad {
  background: linear-gradient(130deg, var(--cy) 0%, var(--bl) 40%, var(--pu) 70%, var(--mg) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; background-size: 200% 200%;
  animation: gs 4s ease infinite;
}
@keyframes gs { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
.pg-sub { font-size: .9rem; color: var(--mu); line-height: 1.7; max-width: 540px; }

/* ── GLASS CARD ── */
.gc {
  background: linear-gradient(135deg, rgba(255,255,255,.056) 0%, rgba(255,255,255,.018) 100%);
  border: 1px solid var(--b);
  border-radius: 20px;
  backdrop-filter: blur(28px);
  -webkit-backdrop-filter: blur(28px);
  padding: 28px;
  margin-bottom: 16px;
  transition: border-color .3s;
}
.gc:hover { border-color: var(--b2); }
.gc-title {
  font-family: var(--fh); font-weight: 700; font-size: .95rem;
  margin-bottom: 20px;
  display: flex; align-items: center; gap: 9px;
}
.gc-icon {
  width: 32px; height: 32px; border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  font-size: .95rem; flex-shrink: 0;
}

/* ── SECTION DIVIDER LABEL ── */
.sec-div {
  font-size: .65rem; letter-spacing: .18em; text-transform: uppercase;
  color: var(--mu); margin: 18px 0 10px;
  padding-bottom: 7px; border-bottom: 1px solid var(--b);
}

/* ── STREAMLIT WIDGETS OVERRIDE ── */
[data-testid="stWidgetLabel"] p {
  font-size: .71rem !important;
  letter-spacing: .09em !important;
  text-transform: uppercase !important;
  color: var(--mu) !important;
  font-family: var(--fb) !important;
  font-weight: 500 !important;
}
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div:first-child {
  background: rgba(255,255,255,.04) !important;
  border: 1px solid rgba(255,255,255,.1) !important;
  border-radius: 11px !important;
  transition: all .25s !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:first-child:focus-within {
  border-color: var(--cy) !important;
  box-shadow: 0 0 0 3px rgba(0,245,255,.1), 0 0 20px rgba(0,245,255,.14) !important;
  background: rgba(0,245,255,.038) !important;
}
input, textarea {
  background: transparent !important;
  color: var(--tx) !important;
  font-family: var(--fb) !important;
}
input::placeholder { color: rgba(200,200,230,.22) !important; }
[data-testid="stSlider"] [role="slider"] {
  background: linear-gradient(135deg, var(--cy), var(--bl)) !important;
  border: none !important;
  box-shadow: 0 0 12px rgba(0,245,255,.45) !important;
}
[data-testid="stNumberInput"] button {
  background: rgba(255,255,255,.05) !important;
  border: 1px solid var(--b) !important;
  color: var(--tx) !important;
  border-radius: 8px !important;
}
[data-testid="stNumberInput"] button:hover {
  background: rgba(0,245,255,.09) !important;
  border-color: rgba(0,245,255,.28) !important;
}
div[data-baseweb="popover"] {
  background: #0b0b20 !important;
  border: 1px solid var(--b) !important;
  border-radius: 13px !important;
  backdrop-filter: blur(24px) !important;
}
div[data-baseweb="menu"] li {
  background: transparent !important;
  color: var(--tx) !important;
  font-family: var(--fb) !important;
}
div[data-baseweb="menu"] li:hover {
  background: rgba(0,245,255,.07) !important;
  color: var(--cy) !important;
}
[data-baseweb="select"] svg { color: var(--mu) !important; }

/* ── PREDICT BUTTON ── */
.stButton > button {
  width: 100% !important;
  padding: 16px 32px !important;
  border-radius: 13px !important;
  margin-top: 6px !important;
  background: linear-gradient(135deg, #00f5ff 0%, #4f7fff 50%, #a855f7 100%) !important;
  background-size: 200% 200% !important;
  animation: gs 3s ease infinite !important;
  border: none !important;
  color: #04040a !important;
  font-family: var(--fh) !important;
  font-weight: 800 !important;
  font-size: .96rem !important;
  letter-spacing: .06em !important;
  cursor: pointer !important;
  box-shadow: 0 0 26px rgba(0,245,255,.3), 0 8px 28px rgba(0,0,0,.35) !important;
  transition: transform .25s, box-shadow .25s !important;
}
.stButton > button:hover {
  transform: translateY(-2px) scale(1.01) !important;
  box-shadow: 0 0 44px rgba(0,245,255,.5), 0 16px 42px rgba(0,0,0,.42) !important;
}
.stButton > button:active { transform: scale(.98) !important; }

/* ── RESULT AREA ── */
.result-card {
  border-radius: 18px;
  padding: 26px;
  border: 1.5px solid;
  backdrop-filter: blur(28px);
  text-align: center;
  margin-bottom: 16px;
  animation: fadeUp .5s cubic-bezier(.22,1,.36,1);
}
@keyframes fadeUp { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:none} }
.result-card.placed {
  background: linear-gradient(135deg, rgba(0,255,170,.1), rgba(0,245,255,.06));
  border-color: rgba(0,255,170,.32);
}
.result-card.notplaced {
  background: linear-gradient(135deg, rgba(255,79,106,.1), rgba(255,60,60,.05));
  border-color: rgba(255,79,106,.32);
}
.result-emoji { font-size: 3rem; display: block; margin-bottom: 10px; }
.result-verdict {
  font-family: var(--fh); font-weight: 800; font-size: 1.55rem; margin-bottom: 7px;
}
.result-verdict.placed { color: #00ffaa; }
.result-verdict.notplaced { color: #ff4f6a; }
.result-sub { font-size: .84rem; color: var(--mu); line-height: 1.55; }

/* Confidence ring */
.ring-box {
  padding: 20px; border-radius: 15px;
  background: var(--s); border: 1px solid var(--b);
  text-align: center; margin-bottom: 16px;
}
.ring-box h5 {
  font-family: var(--fh); font-size: .74rem; font-weight: 700;
  color: var(--mu); text-transform: uppercase; letter-spacing: .12em;
  margin-bottom: 14px;
}

/* Probability bar */
.prob-bar-wrap {
  padding: 16px 20px; border-radius: 14px;
  background: var(--s); border: 1px solid var(--b);
  margin-bottom: 16px;
}
.prob-bar-label {
  display: flex; justify-content: space-between;
  font-size: .74rem; color: var(--mu); margin-bottom: 8px;
}
.prob-bar-track {
  height: 8px; border-radius: 4px;
  background: rgba(255,255,255,.07); overflow: hidden;
}
.prob-bar-fill {
  height: 100%; border-radius: 4px;
  background: linear-gradient(90deg, var(--cy), var(--pu));
  transition: width 1.4s cubic-bezier(.22,1,.36,1);
}

/* Profile summary */
.prof-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; margin-bottom: 14px; }
.prof-cell {
  padding: 11px 13px; border-radius: 11px;
  background: rgba(255,255,255,.03); border: 1px solid var(--b);
}
.prof-cell-label {
  font-size: .58rem; text-transform: uppercase; letter-spacing: .12em;
  color: var(--mu); margin-bottom: 4px;
}
.prof-cell-val { font-family: var(--fh); font-weight: 700; font-size: .98rem; }

/* Suggestion chips */
.sug-title {
  font-size: .67rem; text-transform: uppercase; letter-spacing: .14em;
  color: var(--mu); margin-bottom: 9px; font-weight: 600;
}
.chips { display: flex; flex-wrap: wrap; gap: 7px; }
.chip {
  padding: 6px 13px; border-radius: 20px; font-size: .76rem;
  border: 1px solid var(--b); background: rgba(168,85,247,.08); color: var(--tx);
}
.chip.good {
  border-color: rgba(0,255,170,.28); background: rgba(0,255,170,.08); color: #00ffaa;
}
.chip.warn {
  border-color: rgba(255,179,64,.28); background: rgba(255,179,64,.07); color: var(--am);
}

/* Acc badge */
.acc-badge {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 7px 15px; border-radius: 30px;
  background: rgba(0,255,170,.08); border: 1px solid rgba(0,255,170,.22);
  color: #00ffaa; font-family: var(--fh); font-size: .76rem;
  font-weight: 700; margin-bottom: 14px;
}
.warn-box {
  padding: 11px 14px; border-radius: 11px;
  background: rgba(255,79,106,.07); border: 1px solid rgba(255,79,106,.2);
  font-size: .78rem; color: rgba(255,120,140,.95); line-height: 1.55; margin-top: 8px;
}

/* Orb animation */
.orb-wrap {
  display: flex; flex-direction: column; align-items: center;
  padding: 20px 0; margin-bottom: 14px;
}
.orb-label {
  font-family: var(--fh); font-size: .74rem; font-weight: 700;
  color: var(--mu); text-transform: uppercase; letter-spacing: .1em;
  margin-bottom: 14px;
}

/* ── CHATBOT PANEL ── */
.chat-fab {
  position: fixed; bottom: 28px; right: 28px; z-index: 700;
  width: 56px; height: 56px; border-radius: 50%;
  background: linear-gradient(135deg, var(--cy), var(--pu));
  border: none; cursor: pointer; font-size: 1.3rem;
  box-shadow: var(--gp); transition: all .3s;
  display: flex; align-items: center; justify-content: center;
}
.chat-fab:hover { transform: scale(1.12); box-shadow: 0 0 44px rgba(168,85,247,.65); }
.chat-panel {
  position: fixed; bottom: 94px; right: 28px; z-index: 700;
  width: 360px; border-radius: 20px;
  background: linear-gradient(160deg, rgba(8,8,22,.97), rgba(5,5,15,.99));
  border: 1px solid rgba(0,245,255,.15);
  backdrop-filter: blur(32px); -webkit-backdrop-filter: blur(32px);
  box-shadow: 0 24px 70px rgba(0,0,0,.55), var(--gp);
  display: flex; flex-direction: column;
  overflow: hidden; max-height: 540px;
  animation: panelIn .3s cubic-bezier(.22,1,.36,1);
}
@keyframes panelIn { from{opacity:0;transform:scale(.9) translateY(12px)} to{opacity:1;transform:none} }
.chat-head {
  padding: 14px 16px; border-bottom: 1px solid var(--b);
  display: flex; align-items: center; gap: 10px; flex-shrink: 0;
}
.chat-av {
  width: 36px; height: 36px; border-radius: 50%;
  background: linear-gradient(135deg, var(--cy), var(--pu));
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem; flex-shrink: 0;
}
.chat-head-info h4 { font-family: var(--fh); font-size: .85rem; font-weight: 700; }
.chat-head-info p { font-size: .69rem; color: #00ffaa; }
.chat-msgs {
  flex: 1; overflow-y: auto; padding: 14px;
  display: flex; flex-direction: column; gap: 10px;
}
.cmsg {
  max-width: 85%; padding: 10px 14px; border-radius: 14px;
  font-size: .8rem; line-height: 1.55;
}
.cmsg.bot {
  background: rgba(255,255,255,.055); border: 1px solid var(--b);
  align-self: flex-start;
}
.cmsg.user {
  background: linear-gradient(135deg, rgba(0,245,255,.14), rgba(79,127,255,.14));
  border: 1px solid rgba(0,245,255,.22);
  align-self: flex-end; color: #b8f0ff;
}
.cmsg.typing {
  background: rgba(255,255,255,.04); border: 1px solid var(--b);
  align-self: flex-start; color: var(--mu); font-style: italic;
}
.chat-input-row {
  padding: 12px; border-top: 1px solid var(--b);
  display: flex; gap: 8px; flex-shrink: 0;
}
.chat-input {
  flex: 1; padding: 10px 14px; border-radius: 11px;
  background: rgba(255,255,255,.05); border: 1px solid var(--b);
  color: var(--tx); font-family: var(--fb); font-size: .82rem;
  outline: none; transition: border-color .2s;
}
.chat-input:focus {
  border-color: rgba(0,245,255,.35);
  box-shadow: 0 0 0 2px rgba(0,245,255,.1);
}
.chat-input::placeholder { color: rgba(200,200,230,.25); }
.chat-send {
  padding: 10px 16px; border-radius: 11px;
  background: linear-gradient(135deg, var(--cy), var(--bl));
  border: none; color: #04040a; font-weight: 700;
  cursor: pointer; font-size: .82rem; transition: all .2s;
  white-space: nowrap;
}
.chat-send:hover { transform: scale(1.05); box-shadow: 0 0 14px rgba(0,245,255,.4); }
.api-key-row {
  padding: 10px 14px; border-top: 1px solid var(--b);
  background: rgba(0,0,0,.2);
}
.api-key-row input {
  width: 100%; padding: 8px 12px; border-radius: 9px;
  background: rgba(255,255,255,.04) !important; border: 1px solid var(--b) !important;
  color: var(--mu) !important; font-family: var(--fb) !important;
  font-size: .75rem !important; outline: none;
}
.api-key-row input::placeholder { color: rgba(200,200,230,.2) !important; }
.api-key-label {
  font-size: .62rem; color: rgba(200,200,230,.3); letter-spacing: .1em;
  text-transform: uppercase; margin-bottom: 5px;
}

@media (max-width: 860px) {
  .page-wrap { padding: 20px 18px 50px; }
  .top-bar { padding: 0 18px; }
  .chat-panel { width: calc(100vw - 32px); right: 16px; }
}
</style>
""", unsafe_allow_html=True)

# ── CANVAS / CURSOR JS ─────────────────────────────────────────
st.markdown("""
<canvas id="pc" style="position:fixed;inset:0;z-index:0;pointer-events:none"></canvas>
<div id="cur"  style="position:fixed;width:10px;height:10px;border-radius:50%;background:#00f5ff;pointer-events:none;z-index:9999;transform:translate(-50%,-50%);mix-blend-mode:screen;transition:width .12s,height .12s"></div>
<div id="ring" style="position:fixed;width:34px;height:34px;border-radius:50%;border:1.5px solid rgba(0,245,255,.4);pointer-events:none;z-index:9998;transform:translate(-50%,-50%);transition:all .2s"></div>
<script>
(function(){
  /* cursor */
  var c=document.getElementById('cur'),r=document.getElementById('ring');
  if(!c||!r)return;
  var mx=0,my=0,rx=0,ry=0;
  document.addEventListener('mousemove',function(e){
    mx=e.clientX;my=e.clientY;c.style.left=mx+'px';c.style.top=my+'px';
  });
  (function af(){
    rx+=(mx-rx)*.1;ry+=(my-ry)*.1;
    r.style.left=rx+'px';r.style.top=ry+'px';
    requestAnimationFrame(af);
  })();
  document.addEventListener('mouseover',function(e){
    var t=e.target&&e.target.closest&&e.target.closest('a,button,input,select,.chip');
    c.style.width=t?'18px':'10px';c.style.height=t?'18px':'10px';
    r.style.width=t?'48px':'34px';r.style.height=t?'48px':'34px';
    r.style.borderColor=t?'rgba(168,85,247,.55)':'rgba(0,245,255,.4)';
  });
  /* particles */
  var cv=document.getElementById('pc');
  if(!cv)return;
  var ctx=cv.getContext('2d'),W,H,pts=[];
  function rsz(){W=cv.width=window.innerWidth;H=cv.height=window.innerHeight;}
  rsz();window.addEventListener('resize',rsz);
  var C=['#00f5ff','#4f7fff','#a855f7','#f059ff','#00ffaa'];
  function mk(){return{x:Math.random()*W,y:Math.random()*H,r:Math.random()*1.2+.3,
    dx:(Math.random()-.5)*.18,dy:(Math.random()-.5)*.18,
    op:Math.random()*.32+.05,col:C[Math.floor(Math.random()*C.length)]};}
  for(var i=0;i<75;i++)pts.push(mk());
  (function draw(){
    ctx.clearRect(0,0,W,H);
    for(var i=0;i<pts.length;i++){
      var p=pts[i];p.x+=p.dx;p.y+=p.dy;
      if(p.x<0)p.x=W;if(p.x>W)p.x=0;
      if(p.y<0)p.y=H;if(p.y>H)p.y=0;
      ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle=p.col;ctx.globalAlpha=p.op;ctx.fill();
    }
    ctx.globalAlpha=1;
    for(var i=0;i<pts.length;i++)for(var j=i+1;j<pts.length;j++){
      var d=Math.hypot(pts[i].x-pts[j].x,pts[i].y-pts[j].y);
      if(d<110){ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);
        ctx.strokeStyle='rgba(0,245,255,'+(0.032*(1-d/110))+')';ctx.lineWidth=.4;ctx.stroke();}
    }
    requestAnimationFrame(draw);
  })();
})();
</script>
""", unsafe_allow_html=True)

# ── TOP BAR ────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <div class="top-logo">&#x2B21; PlaceAI</div>
  <div class="top-badges">
    <div class="top-badge live">Prediction Engine</div>
    <div class="top-badge">Random Forest · 91.6% Accuracy</div>
    <div class="top-badge">10,000 Records</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── MAIN CONTENT ───────────────────────────────────────────────
st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

# Title
st.markdown("""
<div class="pg-title">
  <div class="pg-kicker">&#x26A1; AI Prediction Engine</div>
  <h1 class="pg-h1">Student <span class="grad">Placement Predictor</span></h1>
  <p class="pg-sub">
    Enter your academic profile below. The Random Forest model — trained on
    10,000 real student records — will predict your placement probability with precision.
  </p>
</div>
""", unsafe_allow_html=True)

# ── TWO-COLUMN LAYOUT ──────────────────────────────────────────
form_col, result_col = st.columns([55, 45], gap="large")

with form_col:

    # ── SECTION 1: IDENTITY ──────────────────────────────────
    st.markdown("""
    <div class="gc">
      <div class="gc-title">
        <div class="gc-icon" style="background:rgba(0,245,255,.1);border:1px solid rgba(0,245,255,.2)">&#x1F464;</div>
        Student Identity
      </div>
    </div>
    """, unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        name = st.text_input("Full Name", placeholder="e.g. Rahul Sharma", key="inp_name")
    with r1c2:
        college_code = st.text_input("College Code", placeholder="e.g. CLG0042", key="inp_code",
                                     help="Use CLG0001–CLG0100 format")
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        age = st.number_input("Age", min_value=17, max_value=40, value=21, key="inp_age")
    with r2c2:
        backlogs = st.number_input("Active Backlogs (0 = none)", min_value=0, max_value=20,
                                   value=0, key="inp_back")

    if backlogs > 0:
        st.markdown(f"""
        <div class="warn-box">
          ⚠️ <strong>{backlogs} active backlog(s) detected.</strong>
          This significantly reduces your placement probability.
          Clearing all backlogs is your #1 priority.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    # ── SECTION 2: ACADEMIC ──────────────────────────────────
    st.markdown("""
    <div class="gc">
      <div class="gc-title">
        <div class="gc-icon" style="background:rgba(168,85,247,.1);border:1px solid rgba(168,85,247,.2)">&#x1F393;</div>
        Academic Performance
      </div>
    </div>
    """, unsafe_allow_html=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        prev_sem = st.slider("Previous Semester CGPA", 5.0, 10.0, 7.5, 0.1, key="inp_prev")
    with r3c2:
        cgpa = st.slider("Cumulative CGPA", 4.5, 10.5, 7.5, 0.1, key="inp_cgpa")

    acad_options = {
        "Excellent (9–10)": 9,
        "Very Good (8)":    8,
        "Good (7)":         7,
        "Average (6)":      6,
        "Below Average (5)":5,
        "Poor (≤ 4)":       4,
    }
    acad_label = st.selectbox("Academic Performance Grade",
                              list(acad_options.keys()), index=2, key="inp_acad")
    acad_val = acad_options[acad_label]
    acad_adj = max(1, acad_val - int(backlogs * 1.2))  # backlog penalty

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    # ── SECTION 3: SKILLS ────────────────────────────────────
    st.markdown("""
    <div class="gc">
      <div class="gc-title">
        <div class="gc-icon" style="background:rgba(79,127,255,.1);border:1px solid rgba(79,127,255,.2)">&#x26A1;</div>
        Skills &amp; Experience
      </div>
    </div>
    """, unsafe_allow_html=True)

    internship = st.selectbox("Internship Experience", ["No", "Yes"], key="inp_intern")

    r4c1, r4c2 = st.columns(2)
    with r4c1:
        comm = st.slider("Communication Skills", 1, 10, 6, key="inp_comm",
                         help="1 = very poor  |  10 = excellent")
    with r4c2:
        extra = st.slider("Seminars / Certifications / Workshops", 0, 10, 5, key="inp_extra")

    projects = st.number_input("Projects Completed", min_value=0, max_value=5, value=2,
                               key="inp_proj",
                               help="Model trained on 0–5 projects")

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    # Accuracy badge
    st.markdown("""
    <div class="acc-badge">
      ✅ Random Forest · 91.6% Test Accuracy · n_estimators=100 · random_state=42
    </div>
    """, unsafe_allow_html=True)

    predict_clicked = st.button("⚡  Predict My Placement Now", key="btn_predict")


# ── RIGHT COLUMN: RESULT ────────────────────────────────────────
with result_col:

    if predict_clicked:
        disp_name = name.strip() or "Student"
        disp_code = college_code.strip() or "CLG0050"

        # Show profile summary
        st.markdown(f"""
        <div class="gc">
          <div class="gc-title">
            <div class="gc-icon" style="background:rgba(0,245,255,.1);border:1px solid rgba(0,245,255,.18)">📋</div>
            Profile Summary · {disp_name}
          </div>
          <div class="prof-grid">
            <div class="prof-cell">
              <div class="prof-cell-label">CGPA</div>
              <div class="prof-cell-val" style="color:#00f5ff">{cgpa:.1f}</div>
            </div>
            <div class="prof-cell">
              <div class="prof-cell-label">Prev. Semester</div>
              <div class="prof-cell-val" style="color:#4f7fff">{prev_sem:.1f}</div>
            </div>
            <div class="prof-cell">
              <div class="prof-cell-label">Projects</div>
              <div class="prof-cell-val" style="color:#a855f7">{int(projects)}</div>
            </div>
            <div class="prof-cell">
              <div class="prof-cell-label">Communication</div>
              <div class="prof-cell-val" style="color:#f059ff">{comm}/10</div>
            </div>
          </div>
          <div style="display:flex;gap:7px;flex-wrap:wrap;margin-top:10px">
            <span style="padding:4px 11px;border-radius:18px;font-size:.7rem;border:1px solid var(--b);background:var(--s);color:var(--mu)">
              Internship <strong style="color:var(--tx);font-family:var(--fh)">{"✓" if internship=="Yes" else "✗"}</strong>
            </span>
            <span style="padding:4px 11px;border-radius:18px;font-size:.7rem;border:1px solid var(--b);background:var(--s);color:var(--mu)">
              Backlogs <strong style="color:var(--tx);font-family:var(--fh)">{"None ✓" if backlogs==0 else str(backlogs)}</strong>
            </span>
            <span style="padding:4px 11px;border-radius:18px;font-size:.7rem;border:1px solid var(--b);background:var(--s);color:var(--mu)">
              Extra Activities <strong style="color:var(--tx);font-family:var(--fh)">{extra}/10</strong>
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Processing animation
        spin_ph = st.empty()
        spin_ph.markdown("""
        <div class="orb-wrap">
          <div class="orb-label">🔄 Running Random Forest...</div>
          <svg width="150" height="150" viewBox="0 0 150 150" style="overflow:visible">
            <defs>
              <radialGradient id="og">
                <stop offset="0%"   stop-color="#00f5ff" stop-opacity=".28"/>
                <stop offset="100%" stop-color="#a855f7" stop-opacity="0"/>
              </radialGradient>
            </defs>
            <circle cx="75" cy="75" r="65" fill="none" stroke="rgba(0,245,255,.05)" stroke-width="1"/>
            <circle cx="75" cy="75" r="40" fill="url(#og)">
              <animate attributeName="r" values="40;48;40" dur="2.4s" repeatCount="indefinite"/>
            </circle>
            <circle cx="75" cy="75" r="13" fill="#00f5ff" opacity=".85">
              <animate attributeName="r" values="13;18;13" dur="2.4s" repeatCount="indefinite"/>
            </circle>
            <ellipse cx="75" cy="75" rx="58" ry="20" fill="none" stroke="rgba(0,245,255,.2)" stroke-width="1.2">
              <animateTransform attributeName="transform" type="rotate" from="0 75 75" to="360 75 75" dur="7s" repeatCount="indefinite"/>
            </ellipse>
            <circle r="4" fill="#a855f7">
              <animateMotion dur="5s" repeatCount="indefinite" path="M75,75 m0,-56 a56,20 0 1,0 .001,0"/>
            </circle>
          </svg>
          <div style="font-size:.72rem;color:var(--mu);margin-top:8px;text-transform:uppercase;letter-spacing:.1em">
            Analyzing 100 decision trees...</div>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(1.3)

        # ── ACTUAL RF PREDICTION ──────────────────────────────
        placed, conf, placed_prob = run_predict(
            disp_code, prev_sem, cgpa, acad_adj,
            internship, float(extra), float(comm), float(projects)
        )
        st.session_state.result = (placed, conf, placed_prob)

        # Update orb to result state
        oc = "#00ffaa" if placed else "#ff4f6a"
        spin_ph.markdown(f"""
        <div class="orb-wrap">
          <div class="orb-label">✓ Analysis Complete</div>
          <svg width="150" height="150" viewBox="0 0 150 150" style="overflow:visible">
            <defs>
              <radialGradient id="rg2">
                <stop offset="0%"   stop-color="{oc}" stop-opacity=".34"/>
                <stop offset="100%" stop-color="#4f7fff" stop-opacity="0"/>
              </radialGradient>
              <filter id="glo">
                <feGaussianBlur stdDeviation="3.5" result="b"/>
                <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
              </filter>
            </defs>
            <circle cx="75" cy="75" r="65" fill="none" stroke="rgba(0,245,255,.04)" stroke-width="1"/>
            <circle cx="75" cy="75" r="44" fill="url(#rg2)" filter="url(#glo)">
              <animate attributeName="r" values="44;52;44" dur="2.8s" repeatCount="indefinite"/>
            </circle>
            <circle cx="75" cy="75" r="15" fill="{oc}" opacity=".9">
              <animate attributeName="r" values="15;20;15" dur="2.8s" repeatCount="indefinite"/>
            </circle>
            <circle cx="75" cy="75" r="7" fill="{oc}"/>
            <ellipse cx="75" cy="75" rx="58" ry="20" fill="none" stroke="{oc}55" stroke-width="1.4">
              <animateTransform attributeName="transform" type="rotate" from="0 75 75" to="360 75 75" dur="6s" repeatCount="indefinite"/>
            </ellipse>
            <circle r="4.5" fill="{oc}" filter="url(#glo)">
              <animateMotion dur="3.8s" repeatCount="indefinite" path="M75,75 m0,-56 a56,20 0 1,0 .001,0"/>
            </circle>
          </svg>
        </div>
        """, unsafe_allow_html=True)

        # ── RESULT CARD ──────────────────────────────────────
        vc   = "placed" if placed else "notplaced"
        em   = "🎉" if placed else "⚠️"
        vt   = "Likely PLACED ✓"    if placed else "Placement at Risk ✗"
        vsub = "Your profile aligns with placement criteria. Start applying now!" \
               if placed else \
               "Key areas need improvement. See the recommendations below."

        st.markdown(f"""
        <div class="result-card {vc}">
          <span class="result-emoji">{em}</span>
          <div class="result-verdict {vc}">{vt}</div>
          <div class="result-sub">{disp_name} · {vsub}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── CONFIDENCE RING ──────────────────────────────────
        R    = 50
        circ = 2 * 3.14159 * R
        off  = circ * (1 - conf / 100)
        sk   = "url(#ringGrad)" if placed else "#ff4f6a"
        lc   = "#00ffaa" if placed else "#ff4f6a"

        st.markdown(f"""
        <div class="ring-box">
          <h5>Confidence Score</h5>
          <svg width="136" height="136" viewBox="0 0 136 136"
               style="display:block;margin:0 auto;overflow:visible">
            <defs>
              <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%"   stop-color="#00f5ff"/>
                <stop offset="100%" stop-color="#a855f7"/>
              </linearGradient>
            </defs>
            <circle cx="68" cy="68" r="{R}"
              fill="none" stroke="rgba(255,255,255,.065)" stroke-width="9"/>
            <circle cx="68" cy="68" r="{R}"
              fill="none" stroke="{sk}" stroke-width="9"
              stroke-linecap="round"
              stroke-dasharray="{circ:.2f}" stroke-dashoffset="{off:.2f}"
              transform="rotate(-90 68 68)"/>
            <text x="68" y="63" text-anchor="middle"
              font-family="Syne,sans-serif" font-weight="800"
              font-size="20" fill="{lc}">{conf:.1f}%</text>
            <text x="68" y="79" text-anchor="middle"
              font-family="DM Sans,sans-serif" font-size="9"
              fill="rgba(200,200,230,.45)" letter-spacing="1.8">CONFIDENCE</text>
          </svg>
        </div>
        """, unsafe_allow_html=True)

        # ── PLACEMENT PROBABILITY BAR ─────────────────────────
        bar_col = "#00ffaa" if placed_prob >= 50 else "#ff4f6a"
        st.markdown(f"""
        <div class="prob-bar-wrap">
          <div class="prob-bar-label">
            <span>Placement Probability</span>
            <span style="color:{bar_col};font-family:var(--fh);font-weight:700">{placed_prob:.1f}%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width:{placed_prob:.1f}%;background:linear-gradient(90deg,{bar_col},{bar_col}99)"></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:5px">
            <span style="font-size:.62rem;color:var(--mu)">Not Placed</span>
            <span style="font-size:.62rem;color:var(--mu)">Placed</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── AI RECOMMENDATIONS ───────────────────────────────
        sugs = []
        if placed:
            sugs.append(("✅ Strong profile — apply to top companies!", "good"))
            if comm >= 7:           sugs.append(("✅ Excellent communication skills", "good"))
            if int(projects) >= 3:  sugs.append(("✅ Solid project portfolio", "good"))
            if internship == "Yes": sugs.append(("✅ Internship experience is a big plus", "good"))
        if cgpa < 7.0:          sugs.append(("📚 Target CGPA above 7.0", "warn"))
        if prev_sem < 6.5:      sugs.append(("📖 Improve last semester score", "warn"))
        if int(projects) < 3:   sugs.append(("🛠️ Build 2–3 more projects", "warn"))
        if comm < 6:            sugs.append(("🎤 Work on communication skills", "warn"))
        if extra < 4:           sugs.append(("🏆 Attend seminars / get certified", "warn"))
        if internship == "No":  sugs.append(("💼 Complete an internship ASAP", "warn"))
        if backlogs > 0:        sugs.append((f"📋 Clear {backlogs} backlog(s) urgently", "warn"))
        if not sugs:            sugs.append(("🚀 Your profile is outstanding!", "good"))

        chips_html = "".join(
            f'<span class="chip {cls}">{text}</span>'
            for text, cls in sugs[:7]
        )
        st.markdown(f"""
        <div class="gc" style="margin-top:0">
          <div class="gc-title">
            <div class="gc-icon" style="background:rgba(0,245,255,.1);border:1px solid rgba(0,245,255,.18)">💡</div>
            AI Recommendations
          </div>
          <div class="chips">{chips_html}</div>
          <div style="margin-top:14px;font-size:.74rem;color:var(--mu);line-height:1.6">
            💬 <strong style="color:var(--tx)">Ask the chatbot</strong> (bottom-right) for
            personalized advice on any recommendation above.
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── DEFAULT STATE ────────────────────────────────────
        st.markdown("""
        <div class="orb-wrap" style="padding:30px 0">
          <div class="orb-label">Neural Analysis Engine</div>
          <svg width="200" height="200" viewBox="0 0 200 200" style="overflow:visible">
            <defs>
              <radialGradient id="dg">
                <stop offset="0%"   stop-color="#00f5ff" stop-opacity=".22"/>
                <stop offset="55%"  stop-color="#4f7fff" stop-opacity=".08"/>
                <stop offset="100%" stop-color="#a855f7" stop-opacity="0"/>
              </radialGradient>
            </defs>
            <circle cx="100" cy="100" r="90" fill="none" stroke="rgba(0,245,255,.04)" stroke-width="1"/>
            <circle cx="100" cy="100" r="70" fill="none" stroke="rgba(168,85,247,.04)" stroke-width="1"/>
            <circle cx="100" cy="100" r="50" fill="none" stroke="rgba(79,127,255,.04)" stroke-width="1"/>
            <circle cx="100" cy="100" r="52" fill="url(#dg)">
              <animate attributeName="r" values="52;62;52" dur="3.2s" repeatCount="indefinite"/>
              <animate attributeName="opacity" values=".7;1;.7" dur="3.2s" repeatCount="indefinite"/>
            </circle>
            <circle cx="100" cy="100" r="18" fill="#00f5ff" opacity=".8">
              <animate attributeName="r" values="18;24;18" dur="3s" repeatCount="indefinite"/>
            </circle>
            <circle cx="100" cy="100" r="8" fill="#00f5ff"/>
            <ellipse cx="100" cy="100" rx="78" ry="27" fill="none" stroke="rgba(0,245,255,.18)" stroke-width="1.3">
              <animateTransform attributeName="transform" type="rotate" from="0 100 100" to="360 100 100" dur="9s" repeatCount="indefinite"/>
            </ellipse>
            <ellipse cx="100" cy="100" rx="78" ry="27" fill="none" stroke="rgba(168,85,247,.11)" stroke-width="1">
              <animateTransform attributeName="transform" type="rotate" from="0 100 100" to="-360 100 100" dur="14s" repeatCount="indefinite"/>
            </ellipse>
            <circle r="5" fill="#00f5ff" opacity=".9">
              <animateMotion dur="4.5s" repeatCount="indefinite" path="M100,100 m0,-76 a76,27 0 1,0 .001,0"/>
            </circle>
            <circle r="4" fill="#a855f7" opacity=".8">
              <animateMotion dur="7s" repeatCount="indefinite" path="M100,100 m0,76 a76,27 0 1,1 .001,0"/>
            </circle>
            <circle r="3.5" fill="#f059ff" opacity=".7">
              <animateMotion dur="5.8s" repeatCount="indefinite" path="M100,100 m0,-50 a50,76 0 1,0 .001,0"/>
            </circle>
          </svg>
          <div style="font-size:.74rem;color:var(--mu);margin-top:10px;text-transform:uppercase;letter-spacing:.1em">
            Awaiting your profile input...
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature importance bars
        FI = [
            ("Communication Skills",  26.7, "#00f5ff"),
            ("CGPA",                  20.6, "#4f7fff"),
            ("Previous Sem. Result",  16.6, "#a855f7"),
            ("Projects Completed",    15.1, "#f059ff"),
            ("College ID",             9.1, "#00ffaa"),
            ("Extra Curricular Score", 5.5, "#ffb340"),
            ("Academic Performance",   4.9, "#ff8c40"),
            ("Internship Experience",  1.5, "#ff4f6a"),
        ]
        rows_html = "".join(f"""
        <div style="display:flex;align-items:center;gap:9px;margin-bottom:10px">
          <div style="font-size:.71rem;color:var(--mu);min-width:130px;white-space:nowrap;
            overflow:hidden;text-overflow:ellipsis">{l}</div>
          <div style="flex:1;height:5px;border-radius:3px;background:rgba(255,255,255,.05);overflow:hidden">
            <div style="width:{min(p*3.6, 100):.0f}%;height:100%;border-radius:3px;
              background:linear-gradient(90deg,{c},{c}88)"></div>
          </div>
          <div style="font-size:.71rem;min-width:34px;text-align:right;
            font-family:var(--fh);font-weight:700;color:{c}">{p}%</div>
        </div>""" for l, p, c in FI)

        st.markdown(f"""
        <div class="gc">
          <div class="gc-title">
            <div class="gc-icon" style="background:rgba(168,85,247,.1);border:1px solid rgba(168,85,247,.2)">📊</div>
            Feature Importance (RF Model)
          </div>
          {rows_html}
        </div>
        <div style="padding:13px 16px;border-radius:13px;
          background:rgba(255,179,64,.07);border:1px solid rgba(255,179,64,.2);
          font-size:.77rem;color:rgba(255,183,64,.9);line-height:1.6;margin-top:0">
          <strong>💡 Tip:</strong> Communication skills &amp; CGPA are the strongest predictors.
          Fill the form on the left and click <em>Predict</em> to see your result.
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close page-wrap

# ══════════════════════════════════════════════════════════════
#  CHATBOT PANEL (Grok-powered)
# ══════════════════════════════════════════════════════════════

# FAB button toggle
if st.button("💬", key="chat_fab", help="Open AI Career Advisor"):
    st.session_state.chat_visible = not st.session_state.chat_visible

# Inject FAB styling override
st.markdown("""
<style>
[data-testid="stBaseButton-secondary"]:has(div:contains("💬")) {
  position: fixed !important; bottom: 28px !important; right: 28px !important;
  z-index: 700 !important; width: 56px !important; height: 56px !important;
  border-radius: 50% !important;
  background: linear-gradient(135deg, #00f5ff, #a855f7) !important;
  border: none !important; font-size: 1.3rem !important;
  box-shadow: 0 0 24px rgba(168,85,247,.4) !important;
  padding: 0 !important; min-height: unset !important;
  color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# Floating chat button via HTML (more reliable positioning)
if not st.session_state.chat_visible:
    st.markdown("""
    <style>
    #chat-open-btn { cursor: pointer; }
    </style>
    """, unsafe_allow_html=True)

# ── CHAT UI ────────────────────────────────────────────────────
if st.session_state.chat_visible:

    # API KEY input at top of chat
    with st.sidebar:
        pass  # sidebar hidden

    st.markdown("""
    <div style="position:fixed;bottom:28px;right:28px;z-index:800">
    """, unsafe_allow_html=True)

    # Chat container
    chat_container = st.container()
    with chat_container:
        # Outer wrapper for floating panel
        st.markdown("""<div class="chat-panel" id="chatPanel">""", unsafe_allow_html=True)

        # Header
        st.markdown("""
        <div class="chat-head">
          <div class="chat-av">🤖</div>
          <div class="chat-head-info">
            <h4>PlaceAI · Grok AI Advisor</h4>
            <p>● Powered by xAI Grok · Real-time</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # API Key input
        st.markdown('<div class="api-key-row"><div class="api-key-label">Grok API Key (xAI)</div>', unsafe_allow_html=True)
        api_key = st.text_input(
            "grok_api_key",
            value=st.session_state.grok_key,
            placeholder="xai-xxxxxxxxxxxxxxxxxxxx",
            type="password",
            label_visibility="collapsed",
            key="chat_api_key_input",
        )
        if api_key:
            st.session_state.grok_key = api_key
        st.markdown("</div>", unsafe_allow_html=True)

        # Chat history display
        st.markdown('<div class="chat-msgs" id="chatMsgs">', unsafe_allow_html=True)

        # Display existing messages (skip system prompt)
        display_msgs = [m for m in st.session_state.chat_messages if m["role"] != "system"]
        if not display_msgs:
            st.markdown("""
            <div class="cmsg bot">
              👋 Hi! I'm your AI career advisor powered by Grok.<br><br>
              Ask me anything about:<br>
              • Improving your placement chances<br>
              • DSA &amp; interview prep<br>
              • Resume &amp; project advice<br>
              • Understanding your prediction result
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in display_msgs:
                role_cls = "bot" if msg["role"] == "assistant" else "user"
                content = msg["content"].replace("\n", "<br>")
                st.markdown(f'<div class="cmsg {role_cls}">{content}</div>',
                            unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close chat-msgs

        # Input row
        st.markdown('<div class="chat-input-row">', unsafe_allow_html=True)
        user_input = st.text_input(
            "chat_input",
            placeholder="Ask anything about placement...",
            label_visibility="collapsed",
            key="chat_user_input",
        )
        send_btn = st.button("Send →", key="chat_send_btn")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close chat-panel

    # Handle send
    if send_btn and user_input.strip():
        if not st.session_state.grok_key:
            st.warning("⚠️ Please enter your Grok API key to use the chatbot.")
        else:
            # Add context about prediction if available
            user_msg = user_input.strip()
            if st.session_state.result and "placement" in user_msg.lower():
                placed_r, conf_r, prob_r = st.session_state.result
                ctx = (f"\n[Context: This student's RF model prediction — "
                       f"{'Placed' if placed_r else 'Not Placed'}, "
                       f"confidence {conf_r}%, placement probability {prob_r}%]")
                user_msg_with_ctx = user_msg + ctx
            else:
                user_msg_with_ctx = user_msg

            # Add user message to history
            st.session_state.chat_messages.append(
                {"role": "user", "content": user_msg_with_ctx}
            )

            # Call Grok
            with st.spinner("Grok is thinking..."):
                reply = grok_chat(
                    st.session_state.chat_messages,
                    st.session_state.grok_key
                )

            # Add assistant reply
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": reply}
            )

            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ── FAB BUTTON (always visible) ───────────────────────────────
fab_label = "✕ Close Chat" if st.session_state.chat_visible else "💬 AI Advisor"
fab_col = st.columns([1])[0]
st.markdown(f"""
<button
  onclick="window.location.href=window.location.href"
  style="
    position:fixed;bottom:28px;right:28px;z-index:750;
    padding:0 18px;height:50px;border-radius:25px;
    background:linear-gradient(135deg,#00f5ff,#a855f7);
    border:none;cursor:pointer;
    color:#04040a;font-family:'Syne',sans-serif;font-weight:700;font-size:.82rem;
    letter-spacing:.04em;white-space:nowrap;
    box-shadow:0 0 22px rgba(168,85,247,.45);
    transition:all .3s;display:flex;align-items:center;gap:8px;
  "
  title="AI Career Advisor"
>
  {fab_label}
</button>
""", unsafe_allow_html=True)

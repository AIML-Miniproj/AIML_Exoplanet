"""Home: what the tool does, a 30-second explainer, and one example per outcome."""

import random

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui import state
from ui import text as T
from ui.components import badge

state.init()

# One example per outcome card (all precomputed, open instantly).
OUTCOME_EXAMPLE = {"PLANET": 420814525, "EB": 300871545, "VARIABLE": 441584047}


def open_example(tic: int) -> None:
    ex = state.example_for(tic)
    demo = state.load_demo(tic)
    st.session_state.update(target={"tic": tic, "name": ex["display_name"] if ex else f"TIC {tic}",
                                    "source": "example"},
                            report=demo, report_is_demo=demo is not None,
                            example_sel=tic, pick_mode=T.PICK_EXAMPLE)
    st.switch_page("pages/2_Analyze_a_Star.py")


st.title(T.APP_TITLE)
st.markdown(f"#### {T.TAGLINE}")
if st.button(T.TRY_EXAMPLE, type="primary"):
    examples = state.load_examples()["examples"]
    if examples:
        open_example(int(random.choice(examples)["tic_id"]))

cols = st.columns(3)
for i, (col, (title, body)) in enumerate(zip(cols, T.STEPS), start=1):
    with col, st.container(border=True):
        st.markdown(f"**{i}. {title}**")
        st.write(body)

st.subheader(T.EXPLAINER_TITLE)
left, right = st.columns([1, 1.3])
left.write(T.EXPLAINER)
demo = state.load_demo(420814525)
fz = ((demo or {}).get("plot_data") or {}).get("fold_zoom") or {}
if fz.get("hours"):
    x, y = np.asarray(fz["hours"], float), np.asarray(fz["flux"], float)
    edges = np.linspace(x.min(), x.max(), 50)
    idx = np.digitize(x, edges)
    xb = [x[idx == k].mean() for k in range(1, len(edges)) if (idx == k).sum() >= 3]
    yb = [y[idx == k].mean() for k in range(1, len(edges)) if (idx == k).sum() >= 3]
    fig = go.Figure(go.Scatter(x=xb, y=yb, mode="lines+markers",
                               line=dict(color=T.CLASS_INFO["PLANET"][1], width=3)))
    fig.add_annotation(x=0, y=min(yb), text=T.EXPLAINER_CAPTION, showarrow=True, arrowhead=2,
                       ay=-60, bgcolor="rgba(255,255,255,0.8)")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis_title="Hours", yaxis_title="Brightness",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    right.plotly_chart(fig, config={"displayModeBar": False})
    right.caption("Real data: the star HD 209458 dims by about 1.4% every 3.5 days.")

st.subheader("What can a dip turn out to be?")
cards = st.columns(4)
for col, cls in zip(cards, T.OUTCOME_ORDER):
    label, _, _, meaning = T.CLASS_INFO[cls]
    with col, st.container(border=True):
        st.markdown(badge(cls), unsafe_allow_html=True)
        st.write(meaning)
        if cls in OUTCOME_EXAMPLE:
            if st.button(T.SEE_EXAMPLE, key=f"see_{cls}"):
                open_example(OUTCOME_EXAMPLE[cls])
        else:
            st.caption("Most stars without a planet end up here. Try any star from the TESS "
                       "candidate list on the Analyze page.")

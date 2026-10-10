"""Streamlit entry point: page config, navigation, sidebar (mode toggle, recent history).

    streamlit run app.py
"""

import streamlit as st

from src.log import setup_logging
from ui import state
from ui import text as T

setup_logging("WARNING")
st.set_page_config(page_title=T.APP_TITLE, page_icon=":material/travel_explore:",
                   layout="wide")
state.init()

home = st.Page("pages/1_Home.py", title="Home", icon=":material/home:", default=True)
analyze = st.Page("pages/2_Analyze_a_Star.py", title="Analyze a Star", icon=":material/search:")
nav = st.navigation([home, analyze])

with st.sidebar:
    st.toggle(T.MODE_TOGGLE, key="expert", help=T.MODE_HELP)
    st.markdown(f"**{T.RECENT}**")
    if not st.session_state["recent"]:
        st.caption(T.RECENT_EMPTY)
    for i, r in enumerate(st.session_state["recent"]):
        label = T.CLASS_INFO.get(r["verdict"] or "NEEDS_REVIEW", T.CLASS_INFO["NEEDS_REVIEW"])[0]
        if st.button(f"{r['name']} · {label}", key=f"recent_{i}", width="stretch"):
            st.session_state.update(report=r["report"], report_is_demo=False,
                                    target={"tic": r["tic"], "name": r["name"],
                                            "source": "recent"})
            st.switch_page(analyze)
    st.caption(T.FOOTER)

nav.run()

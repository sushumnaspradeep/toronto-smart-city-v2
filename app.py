# app.py

import streamlit as st
from core_engine import SpatialContextEngine

st.set_page_config(
    page_title = "Toronto Smart City Planning Tool",
    page_icon  = "T",
    layout     = "wide",
)

st.title("Toronto Smart City Planning Tool")
st.write("Who are you?")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Government**")
    st.caption("I am a city planner")
    if st.button("Enter", key="gov", use_container_width=True):
        st.switch_page("pages/1_Government.py")

with col2:
    st.write("**Constructor**")
    st.caption("I am a developer")
    if st.button("Enter", key="con", use_container_width=True):
        st.switch_page("pages/2_Constructor.py")

with col3:
    st.write("**Resident**")
    st.caption("I am a resident")
    if st.button("Enter", key="usr", use_container_width=True):
        st.switch_page("pages/3_User_Chat.py")
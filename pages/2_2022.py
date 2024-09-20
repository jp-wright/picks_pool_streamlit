import streamlit as st
from utils.utilities import *
from utils.palettes import *


def main():
    st.set_page_config(page_title="NFL Picks Pool", layout="wide", page_icon='🏈', initial_sidebar_state="expanded")
    local_css("style/style.css")

    gradient(blue_bath1[1], blue_bath1[3], blue_bath1[5], '#fcfbfb', f"🏈 NFL Picks Pool", "A Swimming Pool of Interceptions", 27)

    st.markdown(f"""<h6 align=center>We've gone global: Colorado, Texas, California, England, Japan, and Sweden</h6><BR>""", unsafe_allow_html=True)
    st.markdown(f"<h5 align=center>There was no pool in 2022 :(</h5>", unsafe_allow_html=True)


        
     
if __name__ == '__main__':
    main()
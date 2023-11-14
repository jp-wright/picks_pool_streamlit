import altair as alt
# import pandas as pd
import streamlit as st
# import base64
# from utils.constant import *
from utils.utilities import *
from utils.palettes import *
# from utils.image_refs import *
from utils.data_prepper import DataPrepper


DP = DataPrepper()



class PageLayout():
    def __init__(self):
        st.set_page_config(page_title="NFL Picks Pool", layout="wide", page_icon='🏈', initial_sidebar_state="expanded")
        local_css("style/style.css")
        # st.sidebar.markdown(info['Photo'], unsafe_allow_html=True)
        self.intro()
        # self.manager_ranking()
        # self.manager_by_round()
        # st.markdown("***")
        # self.draft_overview_chart()
        # self.best_worst_picks()
        # self.wins_by_round()



    def intro(self):
        gradient(blue_bath1[1], blue_bath1[3], blue_bath1[5], '#fcfbfb', f"🏈 NFL Picks Pool", "A Swimming Pool of Interceptions", 27)

        st.markdown(f"""<h6 align=center>We've gone global: Colorado, Texas, California, England, Japan, and Sweden</h6><BR>""", unsafe_allow_html=True)
        st.markdown(f"<h5 align=center>There was no pool in 2022 :(</h5>", unsafe_allow_html=True)
        
     
if __name__ == '__main__':
    PageLayout()            
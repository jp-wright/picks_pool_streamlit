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
        # st.sidebar.markdown("🏈", unsafe_allow_html=True)

        self.intro()
        self.top_10_playoffs()
        self.top_10_total_wins()
        # self.manager_ranking()
        # self.manager_by_round()
        # st.markdown("***")
        # self.draft_overview_chart()
        # self.best_worst_picks()
        # self.wins_by_round()



    def intro(self):
        gradient(blue_bath1[1], blue_bath1[3], blue_bath1[5], '#fcfbfb', f"🏈 NFL Picks Pool", "A Swimming Pool of Interceptions", 27)

        st.markdown(f"""<h6 align=center>We've gone global: Colorado, Texas, California, England, Japan, and Sweden</h6><BR>""", unsafe_allow_html=True)

        st.markdown(f"<h5 align=center>This page has records for playoff wins and total wins.</h5>", unsafe_allow_html=True)
        
    def top_10_playoffs(self):
        # st.write(body_dct['pohist_txt'])
        st.markdown(f"""<h6 align=center>How about the top 10 Playoff runs in Pool history?</h6>""", unsafe_allow_html=True)

        # st.dataframe(DP.style_frame(DP.hist_frames[1].sort_values(['Win_Rk', 'Win%', 'Year'], ascending=[True, False, True]).head(10), DP.bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win']), width=620)
        st.table(DP.style_frame(DP.hist_frames[1].sort_values(['Win_Rk', 'Win%', 'Year'], ascending=[True, False, True]).head(10), DP.bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win']))

    def top_10_total_wins(self):
        # st.write(body_dct['tothist_txt'])
        st.write(f"""And what about the top 10 regular season and playoffs combined (for a single season) -- i.e. a player's total wins? 
            """)
        
        st.dataframe(DP.style_frame(DP.hist_frames[2].sort_values(['Win%_Rk', 'Win%', 'Year'], ascending=[True, False, True]).head(10), DP.bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win']), width=620)        
                    


if __name__ == '__main__':
    PageLayout()
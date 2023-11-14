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
        self.champions()





    def intro(self):
        gradient(blue_bath1[1], blue_bath1[3], blue_bath1[5], '#fcfbfb', f"🏈 NFL Picks Pool", "A Swimming Pool of Interceptions", 27)

        st.markdown(f"""<h6 align=center>We've gone global: Colorado, Texas, California, England, Japan, and Sweden</h6><BR>""", unsafe_allow_html=True)
        st.markdown(f"<h5 align=center>Standings {DP.po_inc} as of Week {int(DP.the_wk)} - {DP.the_ssn}!</h5>", unsafe_allow_html=True)

    def champions(self):
        st.write("""#### Champions""")
        st.write("""Past champions and their results, as well as projected champion for the current year (highlighted in blue).
            """)
        st.write("""I'm not sure how to parse the added week 18 in the regular season except to use win percent as opposed to wins.  
        """)
        
        st.dataframe(DP.style_frame(DP.champs, DP.bg_clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=DP.curr_year, bold_cols=['Total_Win']))  

if __name__ == '__main__':
    PageLayout()
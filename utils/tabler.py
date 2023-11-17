"""
Format Frames/Tables to display
"""
import streamlit as st
from utils.palettes import *
# from typing import List, Tuple, Dict, Sequence, Optional



def show_player_hist_table(name):
    st.dataframe(style_frame(player_hist[player_hist['Player'] == name].drop(['Reg_Win', 'Playoff_Win'], axis=1), bg_clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=curr_year, bold_cols=['Total_Win']), width=700)


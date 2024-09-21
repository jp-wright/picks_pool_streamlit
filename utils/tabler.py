"""
Format Frames/Tables to display on streamlit
"""
import streamlit as st
from utils.palettes import *
from utils.styler import style_frame
# from typing import List, Tuple, Dict, Sequence, Optional



def show_player_hist_table(player_hist, name, year):
    st.dataframe(style_frame(player_hist[player_hist['Player'] == name].drop(['Reg_Win', 'Playoff_Win'], axis=1), bg_clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=year, bold_cols=['Total_Win']), width=700)



def show_picks_by_round_table(frame, best_worst): 
    idx_max = frame.groupby('Round')['Total_Win'].transform('max') == frame['Total_Win']
    idx_min = frame.groupby('Round')['Total_Win'].transform('min') == frame['Total_Win']
    
    for rd_res in [(rd, best_worst) for rd in range(1,5)]:
        rd, res = rd_res[0], rd_res[1]
        # components.html(f'<div style="text-align: center"> Round {rd} </div>')
        st.write(f""" Round {rd}""")
        idx = idx_max if res == 'Best' else idx_min
        st.dataframe(style_frame(frame[idx].query("""Round==@rd"""), bg_clr_dct, frmt_dct={'Total_Win': '{:.0f}'}), width=495)

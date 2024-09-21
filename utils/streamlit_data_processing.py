# import streamlit as st
# import streamlit.components.v1 as components
# import pandas as pd
# import numpy as np
from pandas import DataFrame, read_csv
from numpy import where as np_where
# import altair as alt
import os
import time
from pathlib import Path
# import statsmodels.api as sm
from typing import Optional, List, Tuple, Dict, Sequence
# from sports_modules.nfl_league_refs import afc, nfc
# from utils.palettes import *
# from utils.palettes import bg_clr_dct, plot_bg_clr_dct, conf_clr_dct
# from utils.reference import champ_hist
# from utils.utilities import export_to_csv
from utils.constants import CURR_SEASON, INT_COLS, FLOAT_COLS

# import logging
# logging.basicConfig(filename='logs/dplog.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
# logging.info(print(frame.columns))


class DataProcessor():
    def __init__(self, year: Optional[int]=None):
        self.ST_ROOT = Path.cwd()
        self.year = year
        self.load_frames()
        self.curr_week = self.wow[self.wow['Year'] == self.year]['Week_Int'].max()
    
    def set_year(self) -> None:
        self.curr_year = CURR_SEASON 
        if self.curr_year not in self.df['Year'].unique(): 
            self.curr_year = self.df['Year'].unique().max()
        
        if self.year is None:
            self.year = self.curr_year

    def load_frames(self):
        OUTPUT = os.path.join(self.ST_ROOT, 'data/output')
        self.df = read_csv(os.path.join(OUTPUT, 'history/nfl_picks_pool_player_standings_history.csv'))
        self.set_year()
        self.df_years = read_csv(os.path.join(OUTPUT, 'history/picks_pool_stats_by_year.csv'))
        self.df_rounds = read_csv(os.path.join(OUTPUT, 'history/picks_pool_stats_by_round.csv'))
        self.df_careers = read_csv(os.path.join(OUTPUT, 'history/picks_pool_stats_by_career.csv'))
        self.df_years_site = read_csv(os.path.join(OUTPUT, 'streamlit_tables/year_data_site.csv'))
        self.df_rounds_site = read_csv(os.path.join(OUTPUT, 'streamlit_tables/draft_rounds_site.csv'))
        self.df_careers_hist = read_csv(os.path.join(OUTPUT, 'streamlit_tables/career_history.csv'))
        self.df_mgr_tms = read_csv(os.path.join(OUTPUT, 'streamlit_tables/manager_teams_data.csv'))
        self.df_mgr_tms_proj = read_csv(os.path.join(OUTPUT, 'streamlit_tables/manager_teams_projwins_data.csv'))
        self.df_playoffs = read_csv(os.path.join(OUTPUT, 'streamlit_tables/playoff_teams_data.csv'))
        self.df_best_worst_rd = self.prep_best_worst_picks_by_rd(self.df, self.year)
        self.df_top10 = read_csv(os.path.join(OUTPUT, 'streamlit_tables/reg_po_tot_top10.csv')) 
        self.player_hist = read_csv(os.path.join(OUTPUT, 'streamlit_tables/year_data_history.csv'))
        self.champs = read_csv(os.path.join(OUTPUT, 'streamlit_tables/champ_history.csv'))
        self.wow = read_csv(os.path.join(OUTPUT, 'history/WoW_wins_history.csv'))
        self.wow_curr = read_csv(os.path.join(OUTPUT, 'streamlit_tables/wow_metrics_curr_week_site.csv'))\
        

    def prep_best_worst_picks_by_rd(self, frame: DataFrame, year: int):
        frame = frame[(frame["Year"]==year) & (frame['Player']!='Leftover')][['Round', 'Pick', 'Player', 'Team', 'Total_Win', 'Full_Ssn_Pace', 'Playoff_Seed']]\
                    .replace('\s\(\d+\)', '', regex=True)
        neg = 'No' if frame['Playoff_Seed'].sum() > 0 else 'TBD'
        frame['Playoffs'] = np_where(frame['Playoff_Seed'] > 0, 'Yes', neg)
        return frame.drop('Playoff_Seed', axis=1)


    # def prep_WoW_metrics_table(self, year: int):
    #     """
    #     """
    #     # wow = read_csv(self.ST_ROOT.joinpath('data', 'output', f'{year}', f'{year}_WoW_wins.csv'))
    #     # wow = self.wow.merge(self.df_years[['Year', 'Player', 'Total_Win', 'Total_Win%']], on=['Year', 'Player'], how='left')
    #     wow = read_csv(os.path.join(self.ST_ROOT, 'data/output/history/WoW_wins_history.csv'))\
    #             .query(f"Year=={year} and Week_Int=={self.curr_week}")\
    #             .assign(Total_Wins=lambda f: f['Total_Wins'].astype(int),
    #                     WoW_Wins=lambda f: f['WoW_Wins'].astype(int))\
    #             .sort_values(['Total_Wins', 'WoW_Wins'], ascending=False)
    #     return wow
        
    #     # wow = self.wow[(self.wow['Year'] == self.year) & (self.wow['Week_Int']==self.curr_week)]\
    #     #         .sort_values(['Total_Wins', 'WoW_Wins'], ascending=False)

    #     # wow[['WoW_Wins', 'Total_Wins']] = wow[['WoW_Wins', 'Total_Wins']].astype(int)
    #     # return wow[(wow['Week_Int']==wow['Week_Int'].max())].sort_values(['Total_Wins', 'WoW_Wins'], ascending=False)
    #     # return wow[(wow['Week_Int']==wow['Week_Int'].max())].sort_values(['Total_Wins', 'Total_Win%', 'WoW_Wins'], ascending=False)

        
     

# if __name__ == '__main__':
    # pass
    # import sys
    # D = DataProcessor(int(sys.argv[1]))
    # D.prep_WoW_metrics_table(2023)
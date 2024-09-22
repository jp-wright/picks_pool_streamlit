from pandas import DataFrame, read_csv
from numpy import where as np_where
import os
# import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Sequence
from utils.constants import CURR_SEASON, INT_COLS, FLOAT_COLS


class DataProcessor():
    def __init__(self, year: Optional[int]=None):
        self.ST_ROOT = Path.cwd()
        self.set_year(year)
        self.assign_class_frames(year)
        
        self.curr_week = self.wow[self.wow['Year'] == self.year]['Week_Int'].max()
    
    def set_year(self, year) -> None:
        ## will need to add logic for all years for the history pages
        self.year = year
        if self.year is None:
            self.year = CURR_SEASON

    def check_curr_year(self):
        assert self.year in self.df['Year'].unique(), f"Year {self.year} not in data"

    def assign_class_frames(self, year: int):
        self.load_frames()
        self.check_curr_year()
        if year is not None:
            self.filter_frames_by_year(year)

    def load_frames(self):
        OUTPUT = os.path.join(self.ST_ROOT, 'data/output')
        self.df = read_csv(os.path.join(OUTPUT, 'history/nfl_picks_pool_player_standings_history.csv'))
        self.df_years = read_csv(os.path.join(OUTPUT, 'history/picks_pool_stats_by_year.csv'))
        self.df_rounds = read_csv(os.path.join(OUTPUT, 'history/picks_pool_stats_by_round.csv'))
        self.df_careers = read_csv(os.path.join(OUTPUT, 'history/picks_pool_stats_by_career.csv'))
        self.df_years_site = read_csv(os.path.join(OUTPUT, 'streamlit_tables/year_data_history_site.csv'))
        self.df_rounds_site = read_csv(os.path.join(OUTPUT, 'streamlit_tables/draft_rounds_site.csv'))
        self.df_careers_hist = read_csv(os.path.join(OUTPUT, 'streamlit_tables/career_history.csv'))
        self.df_mgr_tms = read_csv(os.path.join(OUTPUT, 'streamlit_tables/manager_teams_data.csv'))
        self.df_mgr_tms_proj = read_csv(os.path.join(OUTPUT, 'streamlit_tables/manager_teams_projwins_data.csv'))
        self.df_playoffs = read_csv(os.path.join(OUTPUT, 'streamlit_tables/playoff_teams_data.csv'))
        self.df_top10 = read_csv(os.path.join(OUTPUT, 'streamlit_tables/reg_po_tot_top10.csv')) 
        self.player_hist = read_csv(os.path.join(OUTPUT, 'streamlit_tables/year_data_history.csv'))
        self.champs = read_csv(os.path.join(OUTPUT, 'streamlit_tables/champ_history.csv'))
        self.wow = read_csv(os.path.join(OUTPUT, 'history/WoW_wins_history.csv'))
        self.wow_curr = read_csv(os.path.join(OUTPUT, 'streamlit_tables/wow_metrics_curr_week_site.csv'))
        self.df_best_worst_rd = self.prep_best_worst_picks_by_rd_table(self.df, self.year)

    def filter_frames_by_year(self, year):
        self.df = self.df[self.df['Year'] == year]
        self.df_years = self.df_years[self.df_years['Year'] == year]
        self.df_rounds = self.df_rounds[self.df_rounds['Year'] == year]
        self.df_careers = self.df_careers[self.df_careers['Year'] == year]
        self.df_years_site = self.df_years_site[self.df_years_site['Year'] == year]
        self.df_rounds_site = self.df_rounds_site[self.df_rounds_site['Year'] == year]
        # self.df_careers_hist = self.df_careers_hist[self.df_careers_hist['Year'] == year]
        # self.df_mgr_tms = self.df_mgr_tms[self.df_mgr_tms['Year'] == year]
        # self.df_mgr_tms_proj = self.df_mgr_tms_proj[self.df_mgr_tms_proj['Year'] == year]
        # self.df_playoffs = self.df_playoffs[self.df_playoffs['Year'] == year]
        # self.df_best_worst_rd = self.df_best_worst_rd[self.df_best_worst_rd['Year'] == year]
        self.df_top10 = self.df_top10[self.df_top10['Year'] == year]
        self.player_hist = self.player_hist[self.player_hist['Year'] == year]
        self.champs = self.champs[self.champs['Year'] == year]
        self.wow = self.wow[self.wow['Year'] == year]
        self.wow_curr = self.wow_curr[self.wow_curr['Year'] == year]
    
    def prep_best_worst_picks_by_rd_table(self, frame: DataFrame, year: int):
        frame = frame[(frame["Year"]==year) & (frame['Player']!='Leftover')][['Round', 'Pick', 'Player', 'Team', 'Total_Win', 'Full_Ssn_Proj_Wins', 'Playoff_Seed']]\
                    .replace('\s\(\d+\)', '', regex=True)
        neg = 'No' if frame['Playoff_Seed'].sum() > 0 else 'TBD'
        frame['Playoffs'] = np_where(frame['Playoff_Seed'] > 0, 'Yes', neg)
        return frame.drop('Playoff_Seed', axis=1)




# if __name__ == '__main__':
    # pass
    # import sys
    # D = DataProcessor(int(sys.argv[1]))
    # D.prep_WoW_metrics_table(2023)
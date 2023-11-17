# import streamlit as st
# import streamlit.components.v1 as components
import pandas as pd
import numpy as np
# import altair as alt
# import os
import time
from pathlib import Path
from utils.palettes import *
from typing import List, Tuple, Dict, Sequence, Optional
import logging
from utils.reference import champ_hist
from utils.utilities import get_curr_year


# import logging
# logging.basicConfig(filename='logs/dplog.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
# logging.info(print(frame.columns))


class DataPrepper():
    def __init__(self, year: int):
        self.ROOT_PATH = Path.cwd()
        self.bg_clr_dct = bg_clr_dct
        self.plot_bg_clr_dct = plot_bg_clr_dct
        
        self.conf_dct = {
            'Chiefs': 'AFC',
            'Bills': 'AFC',
            'Patriots': 'AFC',
            'Browns': 'AFC',
            'Ravens': 'AFC',
            'Titans': 'AFC',
            'Chargers': 'AFC',
            'Colts': 'AFC',
            'Dolphins': 'AFC',
            'Broncos': 'AFC',
            'Steelers': 'AFC',
            'Jets': 'AFC',
            'Raiders': 'AFC',
            'Jaguars': 'AFC',
            'Bengals': 'AFC',
            'Texans': 'AFC',
            'Buccaneers': 'NFC',
            'Rams': 'NFC',
            'Packers': 'NFC',
            '49ers': 'NFC',
            'Seahawks': 'NFC',
            'Cowboys': 'NFC',
            'Saints': 'NFC',
            'Falcons': 'NFC',
            'Vikings': 'NFC',
            'Redskins': 'NFC',
            'Cardinals': 'NFC',
            'Bears': 'NFC',
            'Giants': 'NFC',
            'Panthers': 'NFC',
            'Eagles': 'NFC',
            'Lions': 'NFC',
            }

        self.int_cols = ['Win', 'Loss', 'Tie', 'Games', 'Reg_Games_Left', 'Full_Ssn_Pace', 'Playoff_Teams', 'Total_Win', 'Total_Loss', 'Total_Tie', 'Total_Games', 'Playoff_Win', 'Playoff_Loss', 'Reg_Win', 'Reg_Loss', 'Reg_Tie']
        self.float_cols = ['Win%', 'Total_Win%', 'Playoff_Win%', 'Total Win%']
        self.year = year
        self.curr_year = get_curr_year()

        self.df = self.load_and_prep_data()
        self.dfy = self.stats_by_year(self.df)
        self.dfr = self.stats_by_round(self.df)
        self.dfc = self.stats_by_career(self.df)

        self.dfy_ = self.create_mgr_rk_table_for_site(self.dfy, self.year)
        self.dfr_ = self.prep_round_data_for_website(self.dfr, self.year)
        self.dfc_ = self.prep_career_data_for_website(self.dfc)
        
        self.dfpt = self.prep_manager_teams_this_year(self.df, self.year)
        self.dfpo = self.prep_playoff_teams_this_year(self.df, self.year)
        self.dfd = self.prep_best_worst_picks_by_rd(self.df, self.year)
        
        self.hist_frames = self.prep_year_history(self.dfy, self.curr_year)
        self.player_hist = self.prep_player_history(self.dfy, self.curr_year)
        self.champs = self.prep_champ_history(self.dfy, self.curr_year)
        self.wow = self.prep_WoW_metrics(self.curr_year)
        
        self.po_inc = '(playoffs included)' if 'Playoff Win' in str(self.dfy_) else ''
        self.the_date = time.strftime("%A, %d %b %Y", time.localtime())
        self.the_time = time.strftime("%H:%M CST", time.localtime())


    def enforce_int_cols(self, frame: pd.DataFrame, extra_cols: Optional[List[str]]=None):
        """
        asd
        """
        if extra_cols is None:
            extra_cols = []
        int_cols = np.array(self.int_cols + extra_cols)
        int_cols = int_cols[np.isin(int_cols, frame.columns)]
        
        for col in int_cols:
            try:
                frame[int_cols] = frame[int_cols].fillna(0).astype(int)
            except ValueError:
                logging.error(col)
                logging.error(frame[col])
        return frame

    def enforce_float_cols(self, frame: pd.DataFrame, extra_cols: Optional[List[str]]=None):
        """
        asd
        """
        if extra_cols is None:
            extra_cols = []        
        float_cols = np.array(self.float_cols + extra_cols)
        float_cols = float_cols[np.isin(float_cols, frame.columns)]
        
        for col in float_cols:
            try:
                frame[float_cols] = frame[float_cols].fillna(0).astype(float).round(1)
            except ValueError:
                logging.error(col)
                logging.error(frame[col])
        return frame

    def load_and_prep_data(self): 
        """
        a
        """
        ## Read from but never write to this file. Ref only.
        # ROOT_PATH = Path(os.getcwd())
        # st.write(os.getcwd())
        
        # p = self.ROOT_PATH.joinpath('data', 'input', 'nfl_picks_pool_draft_history.xlsx')
        # st.write(p)
        # st.write('is file?', p.is_file())

        # dfref = pd.read_excel(self.ROOT_PATH.joinpath('data', 'input', 'nfl_picks_pool_draft_history.xlsx'), sheet_name='draft_history')
        dfref = pd.read_csv(self.ROOT_PATH.joinpath('data/input/nfl_picks_pool_draft_history.csv')).replace('Redskins', 'Commanders')
        dfref.rename(columns=lambda col: col.title().replace(' ', '_'), inplace=True)
        df = dfref.copy()
        df.loc[df['Player'] == 'LEFTOVER', 'Player'] = 'Leftover'
        df = df[['Year', 'Round', 'Pick', 'Player', 'Team']]

        ## get regular ssn, post ssn, and total win/loss info
        dfreg = pd.read_csv(self.ROOT_PATH.joinpath('data/input/nfl_regular_ssn_standings_pool_years.csv')).drop('Team', axis=1).replace('Redskins', 'Commanders')
        dfpost = pd.read_csv(self.ROOT_PATH.joinpath('data/input/nfl_post_ssn_standings_pool_years.csv')).replace('Redskins', 'Commanders')
        dftot = pd.read_csv(self.ROOT_PATH.joinpath('data/input/nfl_regular_plus_post_ssn_standings_pool_years.csv')).replace('Redskins', 'Commanders')

        dfreg['Playoffs'] = [True if seed > 0 else False for seed in dfreg['Playoff_Seed']]
        
        dfreg.rename(columns={'Team_Name': 'Team', 'Win': 'Reg_Win', 'Loss': 'Reg_Loss', 'Tie': 'Reg_Tie', 'Games_Played': 'Reg_Games'}, inplace=True)
        dfpost.rename(columns={'Team_Name': 'Team', 'Win': 'Playoff_Win', 'Loss': 'Playoff_Loss', 'Games_Played': 'Playoff_Games'}, inplace=True)
        dftot.rename(columns={'Team_Name': 'Team', 'Win': 'Total_Win', 'Loss': 'Total_Loss', 'Tie': 'Total_Tie', 'Games_Played': 'Total_Games'}, inplace=True)

        ## Join all into one DF with Picks Pool data as index
        reg_cols = ['Reg_Win', 'Reg_Loss', 'Reg_Tie', 'Reg_Games', 'Playoff_Seed']
        post_cols = ['Playoff_Win', 'Playoff_Loss', 'Playoff_Games']
        tot_cols = ['Total_Win', 'Total_Loss', 'Total_Tie', 'Total_Games']
        join_cols = ['Year', 'Team']

        df = df.merge(dfreg[join_cols + reg_cols], on=join_cols, how='left')
        df = df.merge(dfpost[join_cols + post_cols], on=join_cols, how='left')
        df = df.merge(dftot[join_cols + tot_cols], on=join_cols, how='left')

        ## Recalc win% now that all data is combined
        for kind in ['Reg', 'Playoff', 'Total']:
            df[f"{kind}_Win%"] = df[f"{kind}_Win"].div(df[f"{kind}_Games"])

        df.to_csv(self.ROOT_PATH.joinpath('data', 'output', 'nfl_picks_pool_player_standings_history.csv'), index=False)
        
        ## No pool in 2022 (hello depressionnnnnn). Need to remove for processing.
        df = df[df['Year'] != 2022]
        # print(df.head(10))
        
        df['Tm_Yr'] = df['Team'] + " " + df['Year'].astype(str)
        df['Tm_Yr_Win'] = df['Tm_Yr'] + " (" + df['Total_Win'].fillna(0).astype(int).astype(str) + ")"
        df['Tm_Win'] = df['Team'] + " (" + df['Total_Win'].fillna(0).astype(int).astype(str) + ")"

        return df

    def stats_by_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Yearly Stats
        """
        dfy = df.groupby(['Year', 'Player'], as_index=False).sum().copy()
        idx = dfy.columns.get_loc('Reg_Games')
        dfy.insert(idx+1, 'Reg_Games_Left', (16*4) - dfy['Reg_Games'])
        mask = (dfy['Year'] >= 2021) ## start of 17-game seasons -- overwrite 16gms
        dfy.loc[mask, 'Reg_Games_Left'] = (17*4) - dfy.loc[mask, 'Reg_Games']

        for kind in ['Reg', 'Playoff', 'Total']:
            dfy[f"{kind}_Win%"] = dfy[f"{kind}_Win"].div(dfy[f"{kind}_Games"])

        dfy['Full_Ssn_Pace'] = dfy['Reg_Win'] ## overwrite only if games remaining...
        dfy.loc[dfy['Reg_Games'] < 16*4, 'Full_Ssn_Pace'] = dfy.loc[dfy['Reg_Games'] < 16*4, 'Reg_Win%'] * (16 * 4)
        dfy.loc[mask & (dfy['Reg_Games'] < 17*4), 'Full_Ssn_Pace'] = dfy.loc[mask & (dfy['Reg_Games'] < 17*4), 'Reg_Win%'] * (17 * 4)

        dfy.set_index(['Year', 'Player'], inplace=True)
        dfy['Playoff_Teams'] = df.groupby(['Year', 'Player'])['Playoff_Seed'].count()
        dfy['Champ'] = dfy.apply(lambda row: row.name[1] == champ_hist.get(row.name[0], None), axis=1)
        # dfy['Champ'] = dfy.apply(lambda row: row['Player'] == champ_hist.get(row['Year'], None), axis=1)

        pct_cols = [c for c in dfy.columns if '%' in c]
        dfy[pct_cols] = (dfy[pct_cols] * 100).round(1)
        dfy.to_csv(self.ROOT_PATH.joinpath('data', 'output', 'picks_pool_stats_by_year.csv'))
        return dfy.reset_index()

    def stats_by_round(self, df):
        """
        asd
        """
        dfr = df.groupby(['Year', 'Round']).sum()
        for kind in ['Reg', 'Playoff', 'Total']:
            dfr[f"{kind}_Win%"] = dfr[f"{kind}_Win"].div(dfr[f"{kind}_Games"])
            dfr[f"{kind}_Win%"] = (dfr[f"{kind}_Win%"] * 100).round(1)

        dfr['Playoff_Teams'] = df.groupby(['Year', 'Round'])['Playoff_Seed'].count()

        dfr.to_csv(self.ROOT_PATH.joinpath('data', 'output', 'picks_pool_stats_by_round.csv'))
        return dfr.reset_index()

    def stats_by_career(self, df):
        """
        asd
        """
        dfc = df.groupby(['Player']).sum().drop(['Round', 'Pick', 'Reg_Win%', 'Playoff_Win%', 'Total_Win%'], axis=1)

        for kind in ['Reg', 'Playoff', 'Total']:
            dfc[f"{kind}_Win%"] = dfc[f"{kind}_Win"].div(dfc[f"{kind}_Games"])
            dfc[f"{kind}_Win%"] = (dfc[f"{kind}_Win%"] * 100).round(1)

        dfc['Playoff_Teams'] = df.groupby(['Player'])['Playoff_Seed'].count()

        dfc.to_csv(self.ROOT_PATH.joinpath('data', 'output', 'picks_pool_stats_by_career.csv'))
        return dfc.reset_index().sort_values(['Total_Win', 'Total_Win%'], ascending=False)

    def create_mgr_rk_table_for_site(self, dfy: pd.DataFrame, year: int) -> pd.DataFrame:
        '''advanced formatting possible via df.style (requires jinja2).
        https://code.i-harness.com/en/q/df3234
        '''
        frame = dfy[dfy['Year'] == year].drop(['Round', 'Pick'], axis=1)

        ## Still in Reg Ssn
        if frame['Reg_Games_Left'].sum() > 0:
            frame = frame.drop([c for c in frame.columns if 'Playoff' in c or 'Total' in c], axis=1)\
                        .round({'Full_Ssn_Pace': 1}).sort_values('Reg_Win%', ascending=False)
            frame.columns = [c.replace('Reg_', '') if c != 'Reg_Games_Left' else c for c in frame.columns]
            ## Use winning % when still reg season to account for BYE weeks which give some managers fewer games played
            cols = np.array(['Rank', 'Year', 'Player', 'Win', 'Loss', 'Tie', 'Win%', 'Games', 'Reg_Games_Left', 'Full_Ssn_Pace'])
            cols = cols[np.isin(cols, frame.columns)]
            sort_col = 'Win%'
        else:
            ## switch to tot win once PO begins b/c bye weeks are done
            cols = ['Year', 'Player', 'Total_Win', 'Total_Loss', 'Total_Tie', 'Total_Games', 'Total_Win%', 'Playoff_Teams', 'Playoff_Win', 'Playoff_Loss', 'Reg_Win', 'Reg_Loss', 'Reg_Tie']
            sort_col = 'Total_Win'

        frame = frame[cols].sort_values(sort_col, ascending=False)
        frame.insert(0, 'Rank', frame[sort_col].rank(ascending=False, method='dense').fillna(0).astype('int'))
        frame = self.enforce_int_cols(frame)
        frame = self.enforce_float_cols(frame)

        ## Drop Tie cols if no ties exist.  Just visual clutter.
        ties = [c for c in frame.columns if 'Tie' in c]
        if frame[ties].sum().sum() == 0:
            frame = frame.drop(ties, axis=1)

        frame.to_csv(self.ROOT_PATH.joinpath('data', 'output', 'year_data_site.csv'))
        return frame.rename(columns={c: c.replace('_', ' ') for c in frame.columns})

    def prep_round_data_for_website(self, dfr: pd.DataFrame, year: Optional[int]=None):
        '''advanced formatting possible via df.style (requires jinja2).
        https://code.i-harness.com/en/q/df3234
        '''
        if year is None: 
            year = self.curr_year
        ## Remove LEFTOVER b/c they're in round "99"
        frame = dfr[(dfr['Year'] == year) & (dfr['Round'].isin([1,2,3,4]))].drop(['Pick'], axis=1)
        frame = frame.sort_values('Total_Win%', ascending=False)
        if frame['Playoff_Games'].sum() == 0:
            frame = frame.drop([c for c in frame.columns if all(['Playoff' in c, 'Teams' not in c]) or 'Total' in c], axis=1)
            frame = frame.sort_values('Reg_Win%', ascending=False)
            frame.columns = [c.replace('Reg_', '') for c in frame.columns]
            frame = frame.loc[:, ['Year', 'Round', 'Win', 'Loss', 'Tie', 'Games', 'Win%', 'Playoff_Teams']]
        else:
            frame.columns = [c.replace('Total_', '') for c in frame.columns]
            frame = frame.loc[:, ['Year', 'Round', 'Playoff_Teams', 'Win', 'Loss', 'Tie', 'Games', 'Win%']]
            frame.drop(['Full_Ssn_Pace'], axis=1, errors='ignore', inplace=True)

        int_cols = [c for c in frame.columns if '16' not in c and 'Win%' not in c and 'Player' not in c]
        frame[int_cols] = frame[int_cols].astype(int)

        frame.insert(0, 'Rank', frame['Win%'].rank(ascending=False).fillna(0).astype('int'))
        frame.columns = [c.replace('_', ' ') for c in frame.columns]
        return frame
        
    def prep_career_data_for_website(self, dfc: pd.DataFrame):
        '''advanced formatting possible via df.style (requires jinja2).
        https://code.i-harness.com/en/q/df3234
        '''

        frame = dfc.loc[:, ['Player', 'Total_Win', 'Total_Games', 'Reg_Win', 'Playoff_Teams', 'Playoff_Win', 'Total_Win%']]
        frame[['Reg_Win', 'Playoff_Teams', 'Playoff_Win','Total_Win', 'Total_Games']] = frame[['Reg_Win', 'Playoff_Teams', 'Playoff_Win','Total_Win', 'Total_Games']].astype(int)

        frame.insert(0, 'Rank', frame['Total_Win'].rank(ascending=False).astype('int'))
        frame.columns = [c.replace('_', ' ') for c in frame.columns]
        return frame

    def prep_year_history(self, dfy: pd.DataFrame, highlight_year: int):
        '''Do Reg, Playoff, and Tot as separate tables?'''

        def create_year_hist_frame(kind: str):
            drops = {'Reg': ['Reg_Games_Left', 'Reg_Games'], 'Playoff': ['Playoff_Seed']}
            drop_me = drops.get(kind, []) + ['Full_Ssn_Pace']
            frame = dfy.set_index(['Year', 'Player'])[[c for c in dfy.columns if kind in c]].drop(drop_me, axis=1, errors='ignore')
            
            ## Must use W
            if kind in ['Reg', 'Total']:
                sort_col = f'{kind}_Win%_Rk'
                frame.insert(0, sort_col, frame[f'{kind}_Win%'].rank(ascending=False, method='dense').fillna(0).astype(int))
            else:
                sort_col = f'{kind}_Win_Rk'
                frame.insert(0, sort_col, frame[f'{kind}_Win'].rank(ascending=False, method='dense').astype(int))
            ints = [c for c in frame.columns if '%' not in c]
            frame[ints] = frame[ints].astype(int)
            return frame.sort_values(sort_col, ascending=True).head(10)

        frames = []
        for kind in ['Reg', 'Playoff', 'Total']:
            hist = create_year_hist_frame(kind).reset_index()
            # hist.columns = [c.replace(f"{kind}_", '') if 'Win_Rk' not in c else c for c in hist.columns]
            hist.columns = [c.replace(f"{kind}_", '') for c in hist.columns]
            col_order = [f"{kind}_Win_Rk"] + [c for c in hist.columns if c != f"{kind}_Win_Rk"]
            frames.append(hist)
        return frames

    def prep_player_history(self, dfy: pd.DataFrame, highlight_year: int):
        '''The dilemma is whether to rank each player's history by total_wins or total_win%.  Prior to 2021, total_wins was best.  But starting in 2021, with 17 games in the season, raw count of wins is misleading, and win% is the more fair metric.
        '''
        dff = dfy.sort_values(['Player', 'Total_Win'], ascending=[True, False])[['Year', 'Player', 'Total_Win%', 'Total_Win', 'Total_Loss', 'Reg_Win', 'Playoff_Win']]
        int_cols = [c for c in dff.columns if any(['Win' in c, 'Loss' in c]) and '%' not in c]
        dff[int_cols] = dff[int_cols].astype(int)
        frames = []
        for plyr in dff['Player'].unique():
            frame = dff[dff['Player']==plyr]
            frame.insert(0, 'Rank', frame['Total_Win%'].rank(ascending=False, method='dense').fillna(0).astype(int))
            frame = frame.sort_values('Rank', ascending=True)
            frames.append(frame)
        res = pd.concat(frames)
        res['Total_Win%'] = res['Total_Win%'].fillna(0.0)
        champs = pd.DataFrame.from_dict(champ_hist, orient='index')\
                            .reset_index()\
                            .rename(columns={'index': 'Year', 0: 'Player'})
        
        ## Insert dummy Victoria years so ST Plotting functions are happy downstream
        # vic = res[(res['Player']=='JP') & (res['Year']<2023)].copy().assign(Player='Victoria')#.assign(Champ=False)
        # vic[['Rank', 'Total_Win%', 'Total_Win', 'Total_Loss', 'Reg_Win', 'Playoff_Win']] = 0
        # res = pd.concat([res, vic.sort_values(by='Year', ascending=False)], axis=0)

        return res.merge(champs.assign(Champ=True)[['Player', 'Year', 'Champ']], on=['Player', 'Year'], how='left').fillna(False)

    def prep_manager_teams_this_year(self, df: pd.DataFrame, year: Optional[int]=None):
        if year is None:
            year = self.curr_year
        df['Team'] = df['Team'] + " (" + df['Total_Win'].fillna(0).astype(int).astype(str) + ")"
        # if year not in df['Year'].unique(): curr_year = df['Year'].unique().max()
        frame = df[df['Year'] == year].sort_values(['Player', 'Pick'], ascending=[True, True])[['Player', 'Round', 'Team']].copy()
        # global ff
        # ff = frame
        if 'Leftover' in frame['Player'].unique():
            frame.loc[frame['Player']=='Leftover', 'Round'] = [1,2,3,4]
        frame = frame.set_index(['Round', 'Player']).unstack()
        frame.columns = frame.columns.droplevel(0)
        frame.columns.name = ''
        leftovers = frame['Leftover'] if 'Leftover' in frame.columns else None
        frame = frame[[c for c in frame.columns if c != 'Leftover']]
        if leftovers is not None:
            frame = pd.concat([frame, leftovers], axis=1)
        frame.loc[0] = frame.columns
        frame.index = frame.index + 1
        frame = frame.sort_index()
        frame.index.name = 'Round'
        frame.reset_index(inplace=True)
        frame['Round'] = frame['Round'].shift(1).fillna(0).astype(int).astype(str)
        frame.loc[0, 'Round'] = 'Round'
        frame.columns = [''.join([' ']*i) for i in range(len(frame.columns))]
        return frame

    def prep_champ_history(self, dfy: pd.DataFrame, highlight_year: int):
        frame = dfy.copy().sort_values(['Total_Win', 'Total_Win%'], ascending=False)
        
        def get_champ(grp):
            ## first row will be max of both b/c of sort above
            return grp[(grp['Total_Win'] == grp.loc[grp.index[0], 'Total_Win']) & (grp['Total_Win%'] == grp.loc[grp.index[0], 'Total_Win%'])]
            
            
        frame = frame.groupby('Year', as_index=False).apply(get_champ)[['Year', 'Player', 'Total_Win%', 'Total_Win', 'Total_Loss', 'Reg_Win', 'Playoff_Win']]\
            .sort_values(['Year', 'Player'], ascending=[False, True])\
            .reset_index(level=0, drop=True)
        # frame = frame.loc[frame.groupby('Year')['Total_Win'].idxmax()][['Year', 'Player', 'Total_Win%', 'Total_Win', 'Total_Loss', 'Reg_Win', 'Playoff_Win']]\
        #     .sort_values('Year', ascending=False)
        int_cols = [c for c in frame.columns if any(['Win' in c, 'Loss' in c]) and '%' not in c]
        frame[int_cols] = frame[int_cols].astype(int)
        return frame

    def prep_playoff_teams_this_year(self, df: pd.DataFrame, year: Optional[int]=None) -> pd.DataFrame:
        df['Team'] = df['Team'].str.replace('\s\(\d+\)', '', regex=True)
        
        if year is None:
            year = self.curr_year
        ## use 'zLeftover' so Mike isn't after Leftover
        frame = df.loc[(df['Year']==year) & (df['Playoff_Seed']>0), ['Round', 'Player', 'Team', 'Playoff_Win', 'Playoff_Loss', 'Playoff_Seed']]\
            .replace('Leftover', 'zLeftover')\
            .sort_values(['Player', 'Playoff_Seed'])\
            .replace('zLeftover', 'Leftover')\
            .fillna(0)

        # frame['Potential_Wins'] = [3 if seed==1 else 4 for seed in frame['Playoff_Seed']]
        frame['Playoff_Seed'] = frame['Playoff_Seed'].astype(int)
        frame['Conference'] = [self.conf_dct[tm] for tm in frame['Team']]
        frame.rename(columns={'Playoff_Loss': 'Eliminated'}, inplace=True)
        frame['Eliminated'] = frame['Eliminated'] > 0
        # if frame.empty:
        #     frame = pd.DataFrame({' ':'No Playoff Teams Yet'}, index=[0])
        return frame

    def get_count_teams_over_n_wins(self, nwins):
        aa = pd.read_excel("""~/Dropbox/Data_Science/datasets/sports_data/NFL_databases/Pythag Spreadsheets/JP_Pythag_Big_List.xlsx""")
        aa['Year'] = pd.to_datetime(aa['Year'], format="%Y")
        aa['Decade'] = aa['Year'].dt.year // 10

        ## Pro-rate the wins for past seasons
        decade = aa[(aa['Win %']>=(nwins/16))].groupby('Decade').apply(lambda g: g['W'].count() / g['Year'].nunique()).round(1)
        decade = decade.reset_index().rename(columns={0: f'{nwins}-win Tms'})
        decade['Decade'] = decade['Decade'] * 10
        decade = decade.set_index('Decade')


        years = aa[(aa['Win %']>=(nwins/16))].groupby('Year').apply(lambda g: g['W'].count() / g['Year'].nunique()).astype(int)

        years = years.reset_index()
        years = years.set_index(years['Year'].dt.year).drop('Year', axis=1).rename(columns={0: f'{nwins}-win Tms'})

        ## temp fix since Pythag big list isn't updated yet
        years.loc[2019] = 7
        years.loc[2020] = 11

        clrs = ['#fff2cc', '#f4cccc', '#cfe2f3', '#d9d2e9', '#fce5cd', '#d9ead3', '#e6b8af',   '#d9d9d9', ]
        bg_clr_dct = {}
        frames = []
        for idx, period in enumerate([(1960, 1970), (1970, 1980), (1980, 1990), (1990, 2000), (2000, 2010), (2010, 2020), (2020, 2030)]):
            frames.append(years.reindex(range(period[0], period[1])).reset_index().fillna(0).astype(int).replace(0, '-'))
            bg_clr_dct.update({y: clrs[idx] for y in range(period[0], period[1])})

        years = pd.concat(frames, 1)
        years = convert_frame_to_html(years)
        decade = convert_frame_to_html(decade.reset_index())

        def clr_decades(table):
            for year in bg_clr_dct.keys():
                table = table.replace(f'<th>{year}', f'<th style="background-color:{bg_clr_dct[year]}">{year}')
                table = table.replace(f'<td>{year}', f'<td style="background-color:{bg_clr_dct[year]}">{year}')
            return table

        years = clr_decades(years)
        decade = clr_decades(decade)
        return decade, years


    def prep_best_worst_picks_by_rd(self, frame: pd.DataFrame, year: Optional[int]=None):
        """
        """
        if year is None:
            year = self.curr_year
        frame = frame.query(f"Year=={year} and Player!='Leftover'")[['Round', 'Pick', 'Player', 'Team', 'Total_Win', 'Playoff_Seed']].replace('\s\(\d+\)', '', regex=True)
        # frame['Playoffs'] = frame['Playoff_Seed'] > 0  ##return bools now rendered as checkbox in Streamlit (blah)
        neg = 'No' if frame['Playoff_Seed'].sum() > 0 else 'TBD'
        frame['Playoffs'] = np.where(frame['Playoff_Seed'] > 0, 'Yes', neg)
        return frame.drop('Playoff_Seed', axis=1)

    def prep_WoW_metrics(self, year: int):
        """
        """
        wow = pd.read_csv(self.ROOT_PATH.joinpath('data', 'output', f'{year}', f'{year}_WoW_wins.csv'))
        wow = wow.merge(self.dfy[['Year', 'Player', 'Total_Win', 'Total_Win%']], on=['Year', 'Player'], how='left')
        wow[['WoW_Wins', 'Total_Win']] = wow[['WoW_Wins', 'Total_Win']].astype(int)
        return wow[(wow['Week_Int']==wow['Week_Int'].max())].sort_values(['Total_Win', 'Total_Win%', 'WoW_Wins'], ascending=False)

        
     

if __name__ == '__main__':
    # pass
    import sys
    D = DataPrepper(int(sys.argv[1]))
    # D.prep_WoW_metrics(2023)
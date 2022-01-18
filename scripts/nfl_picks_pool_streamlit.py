import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import altair as alt
import time
from pathlib import Path
import re
import os
import sys
from typing import List, Tuple, Dict, Sequence, Optional


# @st.cache
def load_and_prep_data():
    ## Read from but never write to this file. Ref only.
    ROOT_PATH = Path(os.getcwd())
    # st.write(os.getcwd())
    dfref = pd.read_excel(ROOT_PATH.joinpath('data', 'input', 'nfl_picks_pool_draft_history.xlsx'), sheet_name='draft_history')
    dfref.rename(columns=lambda col: col.title().replace(' ', '_'), inplace=True)
    df = dfref.copy()
    df.loc[df['Player'] == 'LEFTOVER', 'Player'] = 'Leftover'
    df = df[['Year', 'Round', 'Pick', 'Player', 'Team']]

    ## get regular ssn, post ssn, and total win/loss info
    dfreg = pd.read_csv('data/input/nfl_regular_ssn_standings_pool_years.csv')\
        .drop('Team', 1)
    dfpost = pd.read_csv('data/input/nfl_post_ssn_standings_pool_years.csv')
    dftot = pd.read_csv('data/input/nfl_regular_plus_post_ssn_standings_pool_years.csv')

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

    df.to_csv(ROOT_PATH.joinpath('data', 'output', 'nfl_picks_pool_player_standings_history.csv'), index=False)
    return df

# @st.cache
def stats_by_year(df: pd.DataFrame, champ_hist: Dict[int, str]) -> pd.DataFrame:
    ## Yearly Stats
    dfy = df.groupby(['Year', 'Player'], as_index=False).sum().copy()
    idx = dfy.columns.get_loc('Reg_Games')
    dfy.insert(idx+1, 'Reg_Games_Left', (16*4) - dfy['Reg_Games'])
    mask = (dfy['Year'] >= 2021) ## start of 17-game seasons
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
    dfy.to_csv(ROOT_PATH.joinpath('data', 'output', 'picks_pool_stats_by_year.csv'))
    return dfy.reset_index()

# @st.cache
def stats_by_round(df):
    dfr = df.groupby(['Year', 'Round']).sum()
    for kind in ['Reg', 'Playoff', 'Total']:
        dfr[f"{kind}_Win%"] = dfr[f"{kind}_Win"].div(dfr[f"{kind}_Games"])
        dfr[f"{kind}_Win%"] = (dfr[f"{kind}_Win%"] * 100).round(1)

    dfr['Playoff_Teams'] = df.groupby(['Year', 'Round'])['Playoff_Seed'].count()

    dfr.to_csv(ROOT_PATH.joinpath('data', 'output', 'picks_pool_stats_by_round.csv'))
    return dfr.reset_index()

# @st.cache
def stats_by_career(df):
    dfc = df.groupby(['Player']).sum().drop(['Round', 'Pick', 'Reg_Win%', 'Playoff_Win%', 'Total_Win%'], 1)

    for kind in ['Reg', 'Playoff', 'Total']:
        dfc[f"{kind}_Win%"] = dfc[f"{kind}_Win"].div(dfc[f"{kind}_Games"])
        dfc[f"{kind}_Win%"] = (dfc[f"{kind}_Win%"] * 100).round(1)

    dfc['Playoff_Teams'] = df.groupby(['Player'])['Playoff_Seed'].count()

    dfc.to_csv(ROOT_PATH.joinpath('data', 'output', 'picks_pool_stats_by_career.csv'))
    return dfc.reset_index().sort_values(['Total_Win', 'Total_Win%'], ascending=False)


# @st.cache
def prep_year_data_for_website(dfy: pd.DataFrame) -> pd.DataFrame:
    '''advanced formatting possible via df.style (requires jinja2).
    https://code.i-harness.com/en/q/df3234
    '''
    frame = dfy[dfy['Year'] == dfy['Year'].max()].drop(['Round', 'Pick'], axis=1)

    if frame['Playoff_Games'].sum() == 0:
        sort_col = 'Total_Win%'
        frame = frame.drop([c for c in frame.columns if all(['Playoff' in c, 'Teams' not in c]) or 'Total' in c], 1)
        if frame['Reg_Games_Left'].sum() == 0:
            frame.drop(['Full_Ssn_Pace'], 1, inplace=True)
        else:
            frame = frame.round({'Full_Ssn_Pace': 1}).sort_values('Reg_Win%', ascending=False)

        frame.columns = [c.replace('Reg_', '') if 'Left' not in c else c for c in frame.columns]
        sort_col = 'Total_Win%' if 'Total_Win%' in frame.columns else 'Win%'
    else:
        sort_col = 'Total_Win'
        frame = frame[['Year', 'Player', 'Total_Win', 'Total_Loss', 'Total_Tie', 'Total_Games', 'Total_Win%', 'Playoff_Teams', 'Playoff_Win', 'Playoff_Loss', 'Reg_Win', 'Reg_Loss', 'Reg_Tie']]

    frame = frame.sort_values(sort_col, ascending=False)

    int_cols = [c for c in frame.columns if '16' not in c and 'Win%' not in c and 'Player' not in c]
    frame[int_cols] = frame[int_cols].astype(int)

    win_cols = np.array(['Win%', 'Total_Win'])
    win_col = win_cols[np.isin(win_cols, frame.columns)]
    frame.insert(0, 'Rank', frame[win_col].rank(ascending=False).astype('int'))
    frame.columns = [c.replace('_', ' ') for c in frame.columns]
    cols = ['Win%', 'Full_Ssn_Pace'] if 'Full_Ssn_Pace' in frame.columns else ['Win%']
    frame[cols] = frame[cols].round(1)
    return frame


# @st.cache
def prep_round_data_for_website(dfr: pd.DataFrame):
    '''advanced formatting possible via df.style (requires jinja2).
    https://code.i-harness.com/en/q/df3234
    '''
    frame = dfr[(dfr['Year'] == dfr['Year'].max()) & (dfr['Round'].isin([1,2,3,4]))].drop(['Pick'], 1)
    frame = frame.sort_values('Total_Win%', ascending=False)
    if frame['Playoff_Games'].sum() == 0:
        frame = frame.drop([c for c in frame.columns if all(['Playoff' in c, 'Teams' not in c]) or 'Total' in c], 1)
        frame = frame.sort_values('Reg_Win%', ascending=False)
        frame.columns = [c.replace('Reg_', '') for c in frame.columns]
        frame = frame.loc[:, ['Year', 'Round', 'Win', 'Loss', 'Tie', 'Games', 'Win%', 'Playoff_Teams']]
    else:
        frame.columns = [c.replace('Total_', '') for c in frame.columns]
        frame = frame.loc[:, ['Year', 'Round', 'Playoff_Teams', 'Win', 'Loss', 'Tie', 'Games', 'Win%']]
        frame.drop(['Full_Ssn_Pace'], 1, errors='ignore', inplace=True)

    int_cols = [c for c in frame.columns if '16' not in c and 'Win%' not in c and 'Player' not in c]
    frame[int_cols] = frame[int_cols].astype(int)

    frame.insert(0, 'Rank', frame['Win%'].rank(ascending=False).astype('int'))
    frame.columns = [c.replace('_', ' ') for c in frame.columns]
    return frame
    
    
# @st.cache
def prep_career_data_for_website(dfc: pd.DataFrame):
    '''advanced formatting possible via df.style (requires jinja2).
    https://code.i-harness.com/en/q/df3234
    '''

    frame = dfc.loc[:, ['Player', 'Total_Win', 'Total_Games', 'Reg_Win', 'Playoff_Teams', 'Playoff_Win', 'Total_Win%']]
    frame[['Reg_Win', 'Playoff_Teams', 'Playoff_Win','Total_Win', 'Total_Games']] = frame[['Reg_Win', 'Playoff_Teams', 'Playoff_Win','Total_Win', 'Total_Games']].astype(int)

    frame.insert(0, 'Rank', frame['Total_Win'].rank(ascending=False).astype('int'))
    frame.columns = [c.replace('_', ' ') for c in frame.columns]
    return frame


# @st.cache
def prep_year_history(dfy: pd.DataFrame, highlight_year: int):
    '''Do Reg, Playoff, and Tot as separate tables?'''

    def create_year_hist_frame(kind: str):
        drops = {'Reg': ['Reg_Games_Left', 'Reg_Games'], 'Playoff': ['Playoff_Seed']}
        drop_me = drops.get(kind, []) + ['Full_Ssn_Pace']
        frame = dfy.set_index(['Year', 'Player'])[[c for c in dfy.columns if kind in c]].drop(drop_me, 1, errors='ignore')
        
        ## Must use W
        if kind in ['Reg', 'Total']:
            sort_col = f'{kind}_Win%_Rk'
            frame.insert(0, sort_col, frame[f'{kind}_Win%'].rank(ascending=False, method='dense').astype(int))
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


# @st.cache
def prep_player_history(dfy: pd.DataFrame, highlight_year: int):
    '''The dilemma is whether to rank each player's history by total_wins or total_win%.  Prior to 2021, total_wins was best.  But starting in 2021, with 17 games in the season, raw count of wins is misleading, and win% is the more fair metric.
    '''
    dff = dfy.sort_values(['Player', 'Total_Win'], ascending=[True, False])[['Year', 'Player', 'Total_Win%', 'Total_Win', 'Total_Loss', 'Reg_Win', 'Playoff_Win']]
    int_cols = [c for c in dff.columns if any(['Win' in c, 'Loss' in c]) and '%' not in c]
    dff[int_cols] = dff[int_cols].astype(int)
    frames = []
    for plyr in dff['Player'].unique():
        frame = dff[dff['Player']==plyr]
        frame.insert(0, 'Rank', frame['Total_Win%'].rank(ascending=False, method='dense').astype(int))
        frame = frame.sort_values('Rank', ascending=True)
        frames.append(frame)
    return pd.concat(frames)


# @st.cache
def prep_player_teams_this_year(df: pd.DataFrame, curr_year: int):
    df['Team'] = df['Team'] + " (" + df['Total_Win'].astype(int).astype(str) + ")"
    frame = df[df['Year'] == curr_year].sort_values(['Player', 'Pick'], ascending=[True, True])[['Player', 'Round', 'Team']].copy()
    frame.loc[frame['Player']=='Leftover', 'Round'] = [1,2,3,4]
    frame = frame.set_index(['Round', 'Player']).unstack()
    frame.columns = frame.columns.droplevel(0)
    frame.columns.name = ''
    frame = frame[[c for c in frame.columns if c != 'Leftover'] + ['Leftover']]
    return frame


# @st.cache
def prep_champ_history(dfy: pd.DataFrame, highlight_year: int):
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


# def prep_playoff_teams_this_year(df: pd.DataFrame) -> pd.DataFrame:
# 

def get_count_teams_over_n_wins(nwins):
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
    years = years.set_index(years['Year'].dt.year).drop('Year', 1).rename(columns={0: f'{nwins}-win Tms'})

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



# def get_curr_weekly_standings_text():
#     return f"""On the heels of a perfect 4-win week (thanks Chargers...), Dan has pulled far, far ahead of the rest of the pool heading into the playoffs.  Dan's current lead is 7 wins, over Brandon. Given how the playoff teams broke out (Dan has 3, the most this year), that means the only way Brandon could catch him is if both of the following things happen:
# 
#     1. The Buccaneers, Cowboys, and Raiders cannot win a single playoff game.
#     2. The Eagles and Chiefs have to meet in the Super Bowl. 
# 
#     If these two things happen, Brandon will **tie** Dan.  Otherwise, Dan wins.
# 
#     Perhaps the more interesting result to watch for is whether Dan can overtake Alex's pool record, set last year, of 65.2% win percentage.  Recall, Alex had a disappointing result from his playoff teams last year, with his only two wins coming from KC.  
# 
#     If all three of Dan's playoff teams win their first game, he will be ahead of Alex's record, with a win percentage of 66.2%, else, he will be behind Alex's mark.  Depending on what happens in round one, apart from all three teams of his losing, he could also have a chance each successive round to best Alex's win percentage -- something for us plebians to watch for, at least.
# 
#     Brandon and Mike both finish the regular season dead in the middle of their historical performance, while Jackson, JP, and Jordan all finished with their second-to-worst years.
# 
# 
#     That said, every player has at least one playoff team (including the *LEFTOVERS*!), so we all have the potential to alter our final ranking.
#     """
# 
# def get_historical_nugget_text():
#     decade, years = get_count_teams_over_n_wins(11)
# 
#     return f"""
#         <BR><BR>
#         Here's how many there have been in the SB era (pre-1978 pro-rated to 16-game seasons)
#         {years}
# 
#         <BR><BR>
#         Yes, that's right -- over 1/3 of the league won ELEVEN or more games this year.  As it turns out, this IS a record in the SB era.  So, if ever someone was going to break our Pool record, this was the likeliest year.
# 
#         <BR><BR>
#         By Decade:<BR>
#         {decade}
# 
#         <BR>
#         (* 1970s using num of teams that would have had 11+ wins projected to 16-games)
# 
#         <BR><BR>
#         As we see, it's definitely been climbing each decade since the 80s.  This alone can open up a lot of interesting questions about the state of the league itself during these periods.  Setting that aside for now, we seem to be in a top-heavy/bottom-heavy split for the league.
# 
#         <BR><BR>
#         We would expect next year to return to the running mean, so it would be less likely for any single player to get all 4 teams to hit 11 wins next year than it was for this year.
#         <BR><BR>
#         """
# 
# def get_hist_intro_text():
#     return f"""<BR><BR>
#         But what about historical placements? Is anyone this year on an historical win pace?
# 
#         2021 -- will need to update history to also show for Win% b/c 17 games....
#         <BR><BR>
#         """
# 
# def get_reg_ssn_hist_text():
#     return f"""<BR><BR>
#         Let's take a look at the top 10 Regular Season finishes.<BR>
# 
#         <BR><BR>
#         It's hard to draw too many hard conclusions here given the small amount of data we have for the Pool's history (4 years x 7 players = 28 entries + 4 Leftover entries = 32 total records).
# 
#         <BR><BR>
#         Mike replicated his 2018 performance, Jordan repeated his from last year, and Jackson logged his 3rd entry in the top 10.
#         <BR><BR>
#         {hist_frames[0]}
#         <BR><BR>
#         """
# 
# def get_playoff_hist_text():
#     return f"""
#        How about the top 10 Playoff runs?
# 
#        <BR><BR>
#        {hist_frames[1]}
#        <BR><BR>
#     """
# 
# def get_tot_hist_text():
#     return f"""And what about the top 10 regular season and playoffs combined (for a single season) -- i.e. a player's total wins? <BR>
# 
#         <BR><BR>
#         {hist_frames[2]}
#         <BR><BR>
#         """
# 
# def get_champs_hist_text():
#     return """Past champions and their results, as well as projected champion for the current year (highlighted in blue).
#         """
# 
# def get_career_text():
#     return """Career standings...
#         """
# 
# def get_personal_records_text():
#     return """Last, here are the personal records for each player.  Blue highlight is for this season and shows who might have a chance at setting a new personal record for total wins."""
# 
# 
# def message_body_wrapper(hist_frames):
#     wk_txt = get_curr_weekly_standings_text()
#     # nug_txt = get_historical_nugget_text()
#     nug_txt = ''
#     # hist_intro_txt = get_hist_intro_text()
#     hist_intro_txt = ''
#     # reghist_txt = get_reg_ssn_hist_text()
#     reghist_txt = ''
#     # pohist_txt = get_playoff_hist_text()
#     pohist_txt = ''
#     # tothist_txt = get_tot_hist_text()
#     tothist_txt = ''
#     champs_txt = get_champs_hist_text()
#     career_txt = get_career_text()
#     pr_txt = get_personal_records_text()
#     return {'wk_txt': wk_txt, 'nug_txt': nug_txt, 'hist_intro_txt': hist_intro_txt, 'reghist_txt': reghist_txt, 'pohist_txt': pohist_txt, 'tothist_txt': tothist_txt, 'champs_txt': champs_txt, 'career_txt': career_txt, 'pr_txt': pr_txt}



def streamlit_layout():
    pass
    





if __name__ == '__main__':
    st.set_page_config(
        page_title="V-town FF",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
            }
        )


    ROOT_PATH = Path(os.getcwd())


    bg_clr_dct = {
        'Alex': '#fff2cc',
        'Mike': '#cfe2f3',
        'JP': '#d9ead3',
        'Jordan': '#f4cccc',
        'Brandon': '#e6b8af',
        'Jackson': '#d9d2e9',
        'Dan': '#fce5cd',
        'LEFTOVER': '#d9d9d9',
        'Leftover': '#d9d9d9',
        }

    plot_bg_clr_dct = {
        'Alex': '#ffd966',
        'Brandon': '#da988b',
        'Dan': '#f7b56e',
        'JP': '#a6cd98',
        'Jackson': '#a898cd',
        'Jordan': '#e48181',
        'Leftover': '#b3b3b3',
        'Mike': '#85b6e0',
        }
        # 'LEFTOVER': '#b3b3b3',
    
    txt_clr_dct = {
        'AFC': 'red',
        'NFC': 'blue',
        }
        
    conf_dct = {
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


    champ_hist = {
        2017: 'Jackson',
        2018: 'Brandon',
        2019: 'Jordan',
        2020: 'Alex',
        2021: 'Dan',
        }



    # def enforce_bool(arg):
    #     '''convert 'True' or 'False' command-line arg from crontab into actual Python bool'''
    #     if isinstance(arg, bool):
    #         return arg
    #     if arg.lower() in ('yes', 'true', 'y', '1'):
    #         return True
    #     elif arg.lower() in ('no', 'false', 'n', '0'):
    #         return False
    #     else:
    #         raise TypeError(f'Boolean value expected. {type(arg)} received.')


    curr_year = 2021
    df = load_and_prep_data()
    dfy = stats_by_year(df, champ_hist)
    dfr = stats_by_round(df)
    dfc = stats_by_career(df)
    dfy_ = prep_year_data_for_website(dfy)
    dfr_ = prep_round_data_for_website(dfr)
    dfc_ = prep_career_data_for_website(dfc)
    dfpt = prep_player_teams_this_year(df, curr_year)
    hist_frames = prep_year_history(dfy, curr_year)
    player_hist = prep_player_history(dfy, curr_year)
    champs = prep_champ_history(dfy, curr_year)

    
#     add_slider = st.sidebar.slider(
#     'Select a range of values',
#     0.0, 100.0, (25.0, 75.0)
# )

    # left_column, right_column = st.columns(2)
    # # You can use a column just like st.sidebar:
    # left_column.button('Press me!')
    # 
    # # Or even better, call Streamlit functions inside a "with" block:
    # with right_column:
    #     chosen = st.radio(
    #         'Sorting hat',
    #         ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    #     st.write(f"You are in {chosen} house!")

    # st.write(df.groupby('Round').agg({'Total_Win': 'nunique'}))
    # st.write("AHIOIJAOIJD")
    # df.groupby('Pick').agg({'Total_Win': 'nunique'})

                # .mark_point(strokeWidth=4, color='grey')\
    df['Team'] = df['Team'].str.replace('\s\(\d+\)', '', regex=True)
    df['Tm_Yr'] = df['Team'] + " " + df['Year'].astype(str)
    df['Tm_Yr_Win'] = df['Tm_Yr'] + " (" + df['Total_Win'].astype(int).astype(str) + ")"
    df['Tm_Win'] = df['Team'] + " (" + df['Total_Win'].astype(int).astype(str) + ")"
                






    # champ_pts = alt.Chart(df)\
    #             .mark_point(filled=True, stroke='black', strokeWidth=1, size=100, opacity=1)\
    #             .encode(
    #                 alt.X('Year:O'),
    #                 alt.Y('Total_Win:Q', axis=alt.Axis(title=None)),
    #                 color=alt.Color('Champ', scale=alt.Scale(domain=[True, False], range=['firebrick', plot_bg_clr_dct[frame['Player'].unique().item()]]))
    #                 )
    
    # text = points.mark_text(
    #             align='center',
    #             baseline='top',
    #             dx=0,
    #             dy=10
    #         )\
    #         .encode(
    #             text='Total_Win'
    #         )
    
    # st.altair_chart(points + text, use_container_width=False)
    # st.altair_chart(points, use_container_width=False)







    
    
    the_date = time.strftime("%A, %d %b %Y", time.localtime())
    the_time = time.strftime("%H:%M CST", time.localtime())
    the_ssn = time.localtime().tm_year - 1 if time.localtime().tm_mon < 9 else time.localtime().tm_year
    the_wk = df.loc[df['Year'] == the_ssn, 'Reg_Games'].max()
    if the_wk > 12: the_wk += 1

    po_inc = '(playoffs included)' if 'Playoff Win' in str(dfy_) else ''

    # msg_dct = create_email_connection_info(df, send_to_pool)
    # body_dct = message_body_wrapper(hist_frames)
    

    def colorize_frame(cell, year, bg_clr_dct):
        if cell == year:
            res = colorize_curr_year(cell, year)
        else:
            res = colorize_player_names_new(cell, bg_clr_dct)
            # res = f"background-color: #FAFAFF; color: black"
        return res
        
    
    def colorize_curr_year(cell, year):
        # if cell == year:
        #     res = f"background-color: blue; color: white"
        # else:
        #     res = ''
            # res = f"background-color: #FAFAFF; color: black"
        # return res
        # return f{"background-color: blue; color: white" if cell == year else '#FAFAFF'}"
        return "background-color: blue; color: white"
    
    
    def colorize_player_names_new(cell, bg_clr_dct):
        # return f'text-align:center; color:black; background-color:{bg_clr_dct.get(cell, "")};'
        return f'text-align:center; color:{txt_clr_dct.get(cell, "black")}; background-color:{bg_clr_dct.get(cell, "#FAFAFF")};'
        # return f'text-align:center; color:black; background-color:{bg_clr_dct.get(cell, "#FAFAFF")};'
        # return f'text-align:center; color:black; background-color:{bg_clr_dct.get(cell, "#EAEAEE")};'
        # return f'text-align:center; color:black; background-color:{bg_clr_dct.get(cell, "#EDEDEE")};'
    
    
    def style_frame(frame, bg_clr_dct, frmt_dct={}, bold_cols=[], clr_yr=None):
        return frame.reset_index(drop=True).style\
                .applymap(lambda cell: colorize_frame(cell, clr_yr, bg_clr_dct))\
                .format(frmt_dct)\
                .set_properties(**{'font-weight': 'bold'}, subset=bold_cols)
                # .applymap(lambda cell: colorize_curr_year(cell, clr_yr))\
                # .applymap(lambda cell: colorize_player_names_new(cell, bg_clr_dct))\

    
    dfpt.loc[0] = dfpt.columns
    dfpt.index = dfpt.index + 1
    dfpt = dfpt.sort_index()
    dfpt.index.name = 'Round'
    dfpt.reset_index(inplace=True)
    dfpt['Round'] = dfpt['Round'].shift(1).fillna(0).astype(int).astype(str)
    dfpt.loc[0, 'Round'] = 'Round'
    dfpt.columns = [''.join([' ']*i) for i in range(len(dfpt.columns))]
    
    
    dfpt = style_frame(dfpt, bg_clr_dct)
        
    st.write(f"""
    ## Global NFL Picks Pool
    ##### Texas, Wisconsin, Colorado, Sweden, England, Japan
    Picks Pool {po_inc} as of week {the_wk} - {the_ssn}!  
    #
    """)
    
    
    st.write("""#### Win Totals this Season""")
    st.write("""###### Team by Round""")
    st.dataframe(dfpt, width=1100) ## sortable, honors center alignment and bold
    # st.table(dfpt)  ## better formatting for undesired index

    
    st.write(""" # """)
    st.write("""###### Player Totals""")
    st.dataframe(style_frame(dfy_, bg_clr_dct, frmt_dct={'Win%': '{:.1f}', 'Full_Ssn_Pace': '{:.1f}'}), width=900)

    # st.write(body_dct['wk_txt'])
    st.write("""
On the heels of a perfect 4-win week (thanks Chargers...), Dan has pulled far, far ahead of the rest of the pool heading into the playoffs.  Dan's current lead is 7 wins, over Brandon. Given how the playoff teams broke out (Dan has 3, the most this year), that means the only way Brandon could catch him is if both of the following things happen:
    
1. The Buccaneers, Cowboys, and Raiders cannot win a single playoff game.
2. The Eagles and Chiefs have to meet in the Super Bowl. 

If these two things happen, Brandon will **tie** Dan.  Otherwise, Dan wins.

Perhaps the more interesting result to watch for is whether Dan can overtake Alex's pool record, set last year, of 65.2% win percentage.  Recall, Alex had a disappointing result from his playoff teams last year, with his only two wins coming from KC.  

If all three of Dan's playoff teams win their first game, he will be ahead of Alex's record, with a win percentage of 66.2%, else, he will be behind Alex's mark.  Depending on what happens in round one, apart from all three teams of his losing, he could also have a chance each successive round to best Alex's win percentage -- something for us plebians to watch for, at least.

Brandon and Mike both finish the regular season dead in the middle of their historical performance, while Jackson, JP, and Jordan all finished with their second-to-worst years.


That said, every player has at least one playoff team (including the *LEFTOVERS*!), so we all have the potential to alter our final ranking.
    """)



    ## use 'zLeftover' so Mike isn't after Leftover
    st.write("""#### Playoff Teams Tracker""")
    dfp = df.loc[(df['Year']==curr_year) & (df['Playoff_Seed']>0), ['Round', 'Player', 'Team', 'Playoff_Win', 'Playoff_Seed']]\
        .replace('Leftover', 'zLeftover')\
        .sort_values(['Player', 'Playoff_Seed'])\
        .replace('zLeftover', 'Leftover')\
        .fillna(0)
    # dfp['Potential_Wins'] = [3 if seed==1 else 4 for seed in dfp['Playoff_Seed']]
    dfp['Playoff_Seed'] = dfp['Playoff_Seed'].astype(int)
    dfp['Conference'] = [conf_dct[tm] for tm in dfp['Team']]
    st.dataframe(style_frame(dfp, bg_clr_dct, frmt_dct={'Playoff_Win': '{:.0f}'}), width=635, height=620)




    ## Draft Overview Chart
    st.write("""#### Draft Overview """)
    # print(df.head())
    source = df[df['Year']==curr_year]
    points = alt.Chart()\
                .mark_point(strokeWidth=1, filled=True, stroke='black', size=185)\
                .encode(
                    alt.X('Pick:O', axis=alt.Axis(format='.0f', tickMinStep=1, labelFlush=True, grid=True)),
                    alt.Y('Total_Win:Q', scale=alt.Scale(zero=True)),
                    tooltip="Player:N"
                    )\
                # .properties(
                #     title='Wins by Year',
                #     width=800,
                #     height=200
                #     )

    text_wins = points.mark_text(
                align='center',
                baseline='top',
                dx=0,
                dy=12
            )\
            .encode(
                text='Total_Win'
            )
            
    text_tm = points.mark_text(
                align='center',
                baseline='bottom',
                dx=0,
                dy=-10
            )\
            .encode(
                text='Team'
            )

    rule1 = alt.Chart().mark_rule(color='black')\
            .encode(
                x=alt.X('rd2:O', title='pick'),
                size=alt.value(2),
                # x='rd2:O',
            )
    rule2 = alt.Chart().mark_rule(color='black')\
            .encode(
                # x='rd3:O',
                x=alt.X('rd3:O', title=''),
                size=alt.value(2),
                # title=''
            )
    rule3 = alt.Chart().mark_rule(color='black')\
            .encode(
                # x='rd4:O',
                x=alt.X('rd4:O', title=''),
                size=alt.value(2),
            )
    rule4 = alt.Chart().mark_rule(color='black')\
            .encode(
                # x=,
                x=alt.X('leftover:O', title=''),
                size=alt.value(2),
                # name=''
            )

    ## color changing marks via radio buttons
    # input_checkbox = alt.binding_checkbox()
# checkbox_selection = alt.selection_single(bind=input_checkbox, name="Big Budget Films")

# size_checkbox_condition = alt.condition(checkbox_selection,
#                                         alt.SizeValue(25),
#                                         alt.Size('Hundred_Million_Production:Q')
#                      
                  # )
# selection = alt.selection_multi(fields=['name'])
# color = alt.condition(selection, alt.Color('name:N'), alt.value('lightgray'))
# make_selector = alt.Chart(make).mark_rect().encode(y='name', color=color).add_selection(selection)
# fuel_chart = alt.Chart(fuel).mark_line().encode(x='index', y=alt.Y('fuel', scale=alt.Scale(domain=[0, 10])), color='name').transform_filter(selection)
                                       
                                       
    player_selection = alt.selection_multi(fields=['Player'])

    domain_ = list(plot_bg_clr_dct.keys())
    range_ = list(plot_bg_clr_dct.values())
    opacity_ = alt.condition(player_selection, alt.value(1.0), alt.value(.4))
    
    player_color_condition = alt.condition(player_selection,
                                alt.Color('Player:N', 
                                    scale=alt.Scale(domain=domain_, range=range_)),
                                alt.value('lightgray')
                            )

    highlight_players = points.add_selection(player_selection)\
                            .encode(
                                color=player_color_condition,
                                opacity=opacity_
                                )\
                            .properties(title=f"{curr_year} Picks by Player")
    
    player_selector = alt.Chart(source).mark_rect()\
            .encode(x='Player', color=player_color_condition)\
            .add_selection(player_selection)
    
    
    
    
    # ## color changing marks via radio buttons - WORKS
    # player_radio = alt.binding_radio(options=df['Player'].unique())
    # player_selection = alt.selection_single(fields=['Player'], bind=player_radio, name=".")
    # 
    # domain_ = list(plot_bg_clr_dct.keys())
    # range_ = list(plot_bg_clr_dct.values())
    # opacity_ = alt.condition(player_selection, alt.value(1.0), alt.value(.4))
    # 
    # player_color_condition = alt.condition(player_selection,
    #                             alt.Color('Player:N', 
    #                                 scale=alt.Scale(domain=domain_, range=range_)),
    #                             alt.value('lightgray')
    #                         )
    # 
    # highlight_players = points.add_selection(player_selection)\
    #                         .encode(
    #                             color=player_color_condition,
    #                             opacity=opacity_
    #                             )\
    #                         .properties(title=f"{curr_year} Picks by Player")
    # 
    # 
    
    
    
    # ## PLAYOFFS ? color changing marks via radio buttons
    # po_radio = alt.binding_radio(options=['Playoffs'])
    # po_select = alt.selection_single(fields=['Playoffs'], bind=po_radio, name="po!")
    # 
    # # domain_ = list(plot_bg_clr_dct.keys())
    # # range_ = list(plot_bg_clr_dct.values())
    # opacity_ = alt.condition(po_select, alt.value(1.0), alt.value(.4))
    # 
    # po_color_condition = alt.condition(po_select,
    #                             alt.Color('Playoffs:N', 
    #                                 scale=alt.Scale(domain=domain_, range=range_)),
    #                             alt.value('lightgray')
    #                         )
    # 
    # highlight_po = points.add_selection(po_select)\
    #                         .encode(
    #                             color=po_color_condition,
    #                             opacity=opacity_
    #                             )\
    #                         .properties(title=f"{curr_year} PO")
    
    
    
    
    
    
    
    
    
    res = alt.layer(
        rule1, rule2, rule3, rule4, text_wins, text_tm, highlight_players,
        data=source, width=1200
        ).transform_calculate(
            rd2="7.5",          ## use pick halfway b/w rounds to draw vert line
            rd3="14.5",
            rd4="21.5",
            leftover="28.5"
        )
        
    st.altair_chart(res) 
    # st.altair_chart(player_selector)
    st.write("*Shift-Click dots to add more players; double-click to reset.*")


    st.write("""# """)
    st.write(""" Looking at this, click on **Dan's button** to highlight only his draft picks.  You can see that, by round, he picked a team that finished   
    (1) tied for the most wins    
    (2) tied for the most wins    
    (3) tied for the most wins    
    (4) with the second most wins!   
    Can't beat that....""")









    ## BEST/WORST PICKS BY ROUND
    st.write('  #')
    
    dfd = df.query(f"Year=={curr_year} and Player!='Leftover'")[['Round', 'Pick', 'Player', 'Team', 'Total_Win', 'Playoff_Seed']].replace('\s\(\d+\)', '', regex=True)
    dfd['Playoffs'] = dfd['Playoff_Seed'] > 0
    dfd = dfd.drop('Playoff_Seed', axis=1)
    idx_max = dfd.groupby('Round')['Total_Win'].transform('max') == dfd['Total_Win']
    idx_min = dfd.groupby('Round')['Total_Win'].transform('min') == dfd['Total_Win']

    left_column, right_column = st.columns([1, 1])
    def picks_by_round(frame, best_worst): 
        for rd_res in [(rd, best_worst) for rd in range(1,5)]:
            rd, res = rd_res[0], rd_res[1]
            # components.html(f'<div style="text-align: center"> Round {rd} </div>')
            st.write(f""" Round {rd}""")
            idx = idx_max if res == 'Best' else idx_min
            st.dataframe(style_frame(dfd[idx].query("""Round==@rd"""), bg_clr_dct, frmt_dct={'Total_Win': '{:.0f}'}), width=495)

    with left_column:
        st.write("""**Here are the best picks by round:**""")
        picks_by_round(dfd, 'Best')
    
    with right_column:
        st.write("""**And here are the worst picks by round:**""")
        picks_by_round(dfd, 'Worst')
















    st.write(""" # """)
    st.write("""#### Wins by Round""")
    st.write("""How did we do in our draft, by rounds? 
    Did we use our early draft picks wisely (does Round 1 have a higher win% than Round 2, etc.)?""")
    
    st.dataframe(style_frame(dfr_, bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win%']))

    
    
    
    
    
    
    
    
    
    # st.write(body_dct['reghist_txt'])
    st.write(f"""
Let's take a look at the top 10 Regular Season finishes.
    """)
    st.dataframe(style_frame(hist_frames[0].sort_values(['Win%_Rk', 'Win', 'Year'], ascending=[True, False, True]), bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win%']), width=620, height=550)
    
    
    
    
    # st.write(body_dct['pohist_txt'])
    st.write(f"""
How about the top 10 Playoff runs?
""")


    st.dataframe(style_frame(hist_frames[1].sort_values(['Win_Rk', 'Win%', 'Year'], ascending=[True, False, True]), bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win']), width=620, height=550)
    
    
    
    
    # st.write(body_dct['tothist_txt'])
    st.write(f"""And what about the top 10 regular season and playoffs combined (for a single season) -- i.e. a player's total wins? 
        """)
    
    st.dataframe(style_frame(hist_frames[2].sort_values(['Win%_Rk', 'Win%', 'Year'], ascending=[True, False, True]), bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win']), width=620, height=550)        
        
    
    
    st.write("""#### Champions""")
    st.write("""Past champions and their results, as well as projected champion for the current year (highlighted in blue).
        """)
    st.write("""I'm not sure how to parse the added week 18 in the regular season except to use win percent as opposed to wins.  
    
The playoffs can make or break a Pool champion, and the fewest playoff wins a champion has had was Alex last year, with two.  Dan has three playoff teams, and none have the bye, meaning they all three play in round one.  Due to how the playoff seeding wound up, Dan could have a max of nine (9) playoff wins (the record is 6 by Jordan in his title year) if the Super Bowl is between the Raiders and either Cowboys/Bucs, with them both winning their round one games.  If he did, he'd land at 53 total wins, a record that would be pretty darn hard to beat!""")
    
    
    st.dataframe(style_frame(champs, bg_clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=2021, bold_cols=['Total_Win']))
    
    
    st.write("""#""")
    st.write("""#### Career Performance""")
    st.write("Who in our pool has been the best over their careers (sorted by Wins)?")
    
    
    st.dataframe(style_frame(dfc_, bg_clr_dct, frmt_dct={'Total Win%': '{:.1f}'}))
    
    
    st.write("""Jackson, Alex, and Brandon have all amassed 180 wins or more, with Jackson currently above Alex by a single win.  Mike and Jordan are next, while thanks to his current season, Dan has officially passed Pizzard via winning percentage, leaving Pize in last place in the very pool he created. Yay!""")
    
    
    
    
    
    
    
    
    
    
    
    st.write("""#""")
    # dfs_ = player_hist.sort_values('Year', ascending=True).groupby(['Player', 'Year']).sum().groupby('Player').cumsum().reset_index().sort_values(['Player', 'Year'])
    # dfs_
    player_hist = player_hist.merge(champs.assign(Champ=True)[['Player', 'Year', 'Champ']], on=['Player', 'Year'], how='left').fillna(False)
    # player_hist = player_hist.merge(champs.assign(Champ='Yes')[['Player', 'Year', 'Champ']], on=['Player', 'Year'], how='left').fillna('No')
    # player_hist.loc[(player_hist['Year']==curr_year) & (player_hist['Champ']=='Yes'), 'Champ'] = 'Proj'
    # player_hist.tail(20)
    
    
    ## tried to use this for color=champ_condition .. can't get to work
    # champ_condition = {
    #     'condition': [
    #         {alt.datum.Champ: 'Yes', 'value': 'firebrick'},
    #         {alt.datum.Champ: 'Proj', 'value': 'Navy'}],
    #      'value': 'orange'}
         
        
    
    bars = alt.Chart()\
                .mark_bar()\
                .encode(
                    alt.X('Player:N', axis=alt.Axis(title='')),
                    alt.Y('Total_Win:Q', scale=alt.Scale(domain=[0, 50], zero=True)),
                    # color=alt.Color('Player:N', scale=alt.Scale(domain=dfs_['Player'].unique(),       range=list(plot_bg_clr_dct.values()))),
                    # color=champ_condition
                    color=alt.condition(
                        alt.datum.Champ == True, 
                        alt.value('firebrick'), 
                        # alt.value(list(plot_bg_clr_dct.values())),
                        alt.value(plot_bg_clr_dct['Mike']),
                        ),
                    )
                    

    text = bars.mark_text(align='center', baseline='bottom')\
                .encode(text='Total_Win:Q')
                
    ## Can't use "+" layer operator with faceted plots
    chart = alt.layer(bars, text, data=player_hist).facet(column=alt.Column('Year:O', header=alt.Header(title='')), title=alt.TitleParams(text='Wins by Year', anchor='middle'))#.resolve_scale(color='independent')


    st.altair_chart(chart)
    
    
    
    
    
    
    ## Ridgeline Plot - not using ATM
    # source = data.seattle_weather.url
    source = dfy
    step = 30
    overlap = 1

    # st.write(source.head(100))
    
    ridge = alt.Chart(source, height=step).transform_joinaggregate(
        mean_wins='mean(Total_Win)', groupby=['Player']
    ).transform_bin(
        ['bin_max', 'bin_min'], 'Total_Win'
    ).transform_aggregate(
        value='count()', groupby=['Player', 'mean_wins', 'bin_min', 'bin_max']
    ).transform_impute(
        impute='value', groupby=['Player', 'mean_wins'], key='bin_min', value=0
    ).mark_area(
        interpolate='monotone',
        fillOpacity=0.8,
        stroke='lightgray',
        strokeWidth=0.5
    ).encode(
        alt.X('bin_min:Q', bin='binned', title='Total Wins'),
        alt.Y(
            'value:Q',
            scale=alt.Scale(range=[step, -step * overlap]),
            axis=None
        ),
        alt.Fill(
            'mean_wins:Q',
            legend=None,
            scale=alt.Scale(domain=[source['Total_Win'].max(), source['Total_Win'].min()], scheme='redyellowblue')
        )
    ).facet(
        row=alt.Row(
            'Player:N',
            title=None,
            header=alt.Header(labelAngle=0, labelAlign='left')
        )
    ).properties(
        title='Win History by Player',
        bounds='flush'
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    ).configure_title(
        anchor='end'
    )
    # st.altair_chart(ridge)
    
    
     
     
    # ridge = alt.Chart(source, height=step).transform_joinaggregate(
    #     mean_wins='mean(Total_Win)', groupby=['Year']
    # ).transform_bin(
    #     ['bin_max', 'bin_min'], 'Total_Win'
    # ).transform_aggregate(
    #     value='count()', groupby=['Year', 'mean_wins', 'bin_min', 'bin_max']
    # ).transform_impute(
    #     impute='value', groupby=['Year', 'mean_wins'], key='bin_min', value=0
    # ).mark_area(
    #     interpolate='monotone',
    #     fillOpacity=0.8,
    #     stroke='lightgray',
    #     strokeWidth=0.5
    # ).encode(
    #     alt.X('bin_min:Q', bin='binned', title='Total Wins'),
    #     alt.Y(
    #         'value:Q',
    #         scale=alt.Scale(range=[step, -step * overlap]),
    #         axis=None
    #     ),
    #     alt.Fill(
    #         'mean_wins:Q',
    #         legend=None,
    #         scale=alt.Scale(domain=[1, 100], scheme='redyellowblue')
    #     )
    # ).facet(
    #     row=alt.Row(
    #         'Year:T',
    #         title=None,
    #         header=alt.Header(labelAngle=0, labelAlign='right', format='%Y')
    #     )
    # ).properties(
    #     title='Win History by Player',
    #     bounds='flush'
    # ).configure_facet(
    #     spacing=0
    # ).configure_view(
    #     stroke=None
    # ).configure_title(
    #     anchor='end'
    # )
    
    # st.altair_chart(ridge)
    
    
    
    
    
    

    st.write("""#""")
    st.write("""#### Personal Records""")    
    # st.write(body_dct['pr_txt'])
    st.write("""Last, here are the personal records for each player.  Blue highlight is for this season and shows who might have a chance at setting a new personal record for total wins.""")
    
    
    
    
    def show_player_hist_table(name):
        st.dataframe(style_frame(player_hist[player_hist['Player'] == name].drop(['Reg_Win', 'Playoff_Win'], axis=1), bg_clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=2021, bold_cols=['Total_Win']), width=700)
    

    def plot_wins_by_year(frame):
        points = alt.Chart(frame)\
                    .mark_line(strokeWidth=4, color='grey')\
                    .encode(
                        alt.X('Year:O', axis=alt.Axis(format='.0f', tickMinStep=1, labelFlush=True, grid=True)),
                        alt.Y('Total_Win:Q', scale=alt.Scale(zero=True)),
                        order='Year',
                        )\
                    .properties(
                        title='Wins by Year',
                        width=400,
                        height=200
                        )

        champ_pts = alt.Chart(frame)\
                    .mark_point(filled=True, stroke='black', strokeWidth=1, size=100, opacity=1)\
                    .encode(
                        alt.X('Year:O'),
                        alt.Y('Total_Win:Q', axis=alt.Axis(title=None)),
                        color=alt.Color('Champ', scale=alt.Scale(domain=[True, False], range=['firebrick', plot_bg_clr_dct[frame['Player'].unique().item()]]))
                        )
        
        text = points.mark_text(
                    align='center',
                    baseline='top',
                    dx=0,
                    dy=10
                )\
                .encode(
                    text='Total_Win'
                )
        
        st.altair_chart(points + champ_pts + text, use_container_width=False)



    left_column, right_column = st.columns([2, 1])
    
    with left_column:
        for name in dfy_['Player'].unique():
            show_player_hist_table(name)
            st.write("\n\n\n _")

    with right_column: 
        for name in dfy_['Player'].unique():
            plot_wins_by_year(player_hist[player_hist['Player'] == name])

import streamlit as st
import pandas as pd
import numpy as np
import time
from pathlib import Path
import re
import os
import sys
from typing import List, Tuple, Dict, Sequence, Optional
from bs4 import BeautifulSoup


# @st.cache
def load_and_prep_data():
    ROOT_PATH = Path('~/Dropbox/Data_Science/projects/github_projects/2021-11-10_nfl_picks_pool_streamlit/')
    ## Read from but never write to this file. Ref only.
    dfref = pd.read_excel(ROOT_PATH.joinpath('data', 'input', 'nfl_picks_pool_draft_history.xlsx'), sheet_name='draft_history')
    # dfref.columns = [c.title().replace(' ', '_') for c in dfref.columns]
    dfref.rename(columns=lambda col: col.title().replace(' ', '_'), inplace=True)
    df = dfref.copy()
    df.loc[df['Player'] == 'LEFTOVER', 'Player'] = 'Leftover'
    df = df[['Year', 'Round', 'Pick', 'Player', 'Team']]


    ## get regular ssn, post ssn, and total win/loss info
    dfreg = pd.read_csv('data/input/nfl_regular_ssn_standings_pool_years.csv').drop('Team', 1)
    dfpost = pd.read_csv('data/input/nfl_post_ssn_standings_pool_years.csv')
    dftot = pd.read_csv('data/input/nfl_regular_plus_post_ssn_standings_pool_years.csv')

    
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

    # df.to_csv(ROOT_PATH.joinpath('data', 'output', 'nfl_picks_pool_player_standings_history.csv'), index=False)
    return df

# @st.cache
def stats_by_year(df):
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
        frame.insert(0, f'{kind}_Win_Rk', frame[f'{kind}_Win'].rank(ascending=False, method='dense').astype(int))
        ints = [c for c in frame.columns if '%' not in c]
        frame[ints] = frame[ints].astype(int)
        return frame.sort_values(f'{kind}_Win_Rk', ascending=True).head(10)

    frames = []
    for kind in ['Reg', 'Playoff', 'Total']:
        hist = create_year_hist_frame(kind).reset_index()
        hist.columns = [c.replace(f"{kind}_", '') if 'Win_Rk' not in c else c for c in hist.columns]
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
    return frames


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
    frame = dfy.copy()
    frame = frame.loc[frame.groupby('Year')['Total_Win'].idxmax()][['Year', 'Player', 'Total_Win%', 'Total_Win', 'Total_Loss', 'Reg_Win', 'Playoff_Win']]\
        .sort_values('Year', ascending=False)
    int_cols = [c for c in frame.columns if any(['Win' in c, 'Loss' in c]) and '%' not in c]
    frame[int_cols] = frame[int_cols].astype(int)
    return frame
    


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
    clr_dct = {}
    frames = []
    for idx, period in enumerate([(1960, 1970), (1970, 1980), (1980, 1990), (1990, 2000), (2000, 2010), (2010, 2020), (2020, 2030)]):
        frames.append(years.reindex(range(period[0], period[1])).reset_index().fillna(0).astype(int).replace(0, '-'))
        clr_dct.update({y: clrs[idx] for y in range(period[0], period[1])})

    years = pd.concat(frames, 1)
    years = convert_frame_to_html(years)
    decade = convert_frame_to_html(decade.reset_index())

    def clr_decades(table):
        for year in clr_dct.keys():
            table = table.replace(f'<th>{year}', f'<th style="background-color:{clr_dct[year]}">{year}')
            table = table.replace(f'<td>{year}', f'<td style="background-color:{clr_dct[year]}">{year}')
        return table

    years = clr_decades(years)
    decade = clr_decades(decade)
    years = BeautifulSoup(years, features='lxml')
    decade = BeautifulSoup(decade, features='lxml')
    return decade, years



def get_curr_weekly_standings_text():
    return f"""Dan and reigning champion Alex are both having their best seasons ever (by a country mile for Dan), while Brandon, JP, and Mike are all having their -worst- seasons ever.
    """

def get_historical_nugget_text():
    decade, years = get_count_teams_over_n_wins(11)

    return f"""
        <BR><BR>
        Here's how many there have been in the SB era (pre-1978 pro-rated to 16-game seasons)
        {years}

        <BR><BR>
        Yes, that's right -- over 1/3 of the league won ELEVEN or more games this year.  As it turns out, this IS a record in the SB era.  So, if ever someone was going to break our Pool record, this was the likeliest year.

        <BR><BR>
        By Decade:<BR>
        {decade}

        <BR>
        (* 1970s using num of teams that would have had 11+ wins projected to 16-games)

        <BR><BR>
        As we see, it's definitely been climbing each decade since the 80s.  This alone can open up a lot of interesting questions about the state of the league itself during these periods.  Setting that aside for now, we seem to be in a top-heavy/bottom-heavy split for the league.

        <BR><BR>
        We would expect next year to return to the running mean, so it would be less likely for any single player to get all 4 teams to hit 11 wins next year than it was for this year.
        <BR><BR>
        """

def get_hist_intro_text():
    return f"""<BR><BR>
        But what about historical placements? Is anyone this year on an historical win pace?
        
        2021 -- will need to update history to also show for Win% b/c 17 games....
        <BR><BR>
        """

def get_reg_ssn_hist_text():
    return f"""<BR><BR>
        Let's take a look at the top 10 Regular Season finishes.<BR>

        <BR><BR>
        It's hard to draw too many hard conclusions here given the small amount of data we have for the Pool's history (4 years x 7 players = 28 entries + 4 Leftover entries = 32 total records).

        <BR><BR>
        Mike replicated his 2018 performance, Jordan repeated his from last year, and Jackson logged his 3rd entry in the top 10.
        <BR><BR>
        {hist_frames[0]}
        <BR><BR>
        """

def get_playoff_hist_text():
    return f"""
       How about the top 10 Playoff runs?

       <BR><BR>
       {hist_frames[1]}
       <BR><BR>
    """

def get_tot_hist_text():
    return f"""And what about the top 10 regular season and playoffs combined (for a single season) -- i.e. a player's total wins? <BR>

        <BR><BR>
        {hist_frames[2]}
        <BR><BR>
        """

def get_champs_hist_text():
    return """Past champions and their results, as well as projected champion for the current year (highlighted in blue).
        """

def get_career_text():
    return """Career standings...
        """

def get_personal_records_text():
    return """Last, here are the personal records for each player.  Blue highlight is for this season and shows who might have a chance at setting a new personal record for total wins."""


def message_body_wrapper(hist_frames):
    wk_txt = get_curr_weekly_standings_text()
    # nug_txt = get_historical_nugget_text()
    nug_txt = ''
    # hist_intro_txt = get_hist_intro_text()
    hist_intro_txt = ''
    # reghist_txt = get_reg_ssn_hist_text()
    reghist_txt = ''
    # pohist_txt = get_playoff_hist_text()
    pohist_txt = ''
    # tothist_txt = get_tot_hist_text()
    tothist_txt = ''
    champs_txt = get_champs_hist_text()
    career_txt = get_career_text()
    pr_txt = get_personal_records_text()
    return {'wk_txt': wk_txt, 'nug_txt': nug_txt, 'hist_intro_txt': hist_intro_txt, 'reghist_txt': reghist_txt, 'pohist_txt': pohist_txt, 'tothist_txt': tothist_txt, 'champs_txt': champs_txt, 'career_txt': career_txt, 'pr_txt': pr_txt}



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
    
    
    
    
    
    
    ROOT_PATH = Path('~/Dropbox/Data_Science/projects/github_projects/2021-11-10_nfl_picks_pool_streamlit')


    clr_dct = {
        'Alex': '#fff2cc',
        'Mike': '#cfe2f3',
        'JP': '#d9ead3',
        'Jordan': '#f4cccc',
        'Brandon': '#e6b8af',
        'Jackson': '#d9d2e9',
        'Dan': '#fce5cd',
        'LEFTOVER': '#d9d9d9',
        'Leftover': '#d9d9d9'
        }

    def enforce_bool(arg):
        '''convert 'True' or 'False' command-line arg from crontab into actual Python bool'''
        if isinstance(arg, bool):
            return arg
        if arg.lower() in ('yes', 'true', 'y', '1'):
            return True
        elif arg.lower() in ('no', 'false', 'n', '0'):
            return False
        else:
            raise TypeError(f'Boolean value expected. {type(arg)} received.')



    curr_year = 2021
    df = load_and_prep_data()
    dfy = stats_by_year(df)
    dfr = stats_by_round(df)
    dfc = stats_by_career(df)
    dfy_ = prep_year_data_for_website(dfy)
    dfr_ = prep_round_data_for_website(dfr)
    dfc_ = prep_career_data_for_website(dfc)
    dfpt = prep_player_teams_this_year(df, curr_year)
    hist_frames = prep_year_history(dfy, curr_year)
    player_hist = prep_player_history(dfy, curr_year)
    champs = prep_champ_history(dfy, curr_year)

    ## Read in sys arg via Crontab so guaranteed to always execute.
    ## Pass in False to send only to self test email addresses, True to send to Pool
    ## Use: %run <path>/nfl_picks_pool_email.py True/False in iPython for manual execution
    ## True = send to pool; False = send to JP accounts only
    # send_html_email(df, dfy_, dfr_, dfc_, dfpt, hist_frames, player_hist, champs, send_to_pool=enforce_bool(sys.argv[1]))

    # st.dataframe(dfpt, width=10000, height=500)
    # st.table(dfpt)
    # st.write(dfpt)
    

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



    
    the_date = time.strftime("%A, %d %b %Y", time.localtime())
    the_time = time.strftime("%H:%M CST", time.localtime())
    the_ssn = time.localtime().tm_year - 1 if time.localtime().tm_mon < 9 else time.localtime().tm_year
    the_wk = df.loc[df['Year'] == the_ssn, 'Reg_Games'].max()

    po_inc = '(playoffs included)' if 'Playoff Win' in str(dfy_) else ''

    # msg_dct = create_email_connection_info(df, send_to_pool)
    body_dct = message_body_wrapper(hist_frames)
    
    # if locals().get('blah', None):
    # st.write(f"""test me: {dfpt}""")
    # st.write(dfpt)
    def colorize_frame(cell, year, clr_dct):
        if cell == year:
            res = colorize_curr_year(cell, year)
        else:
            res = colorize_player_names_new(cell, clr_dct)
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
    
    
    def colorize_player_names_new(cell, clr_dct):
        # return f'text-align:center; color:black; background-color:{clr_dct.get(cell, "")};'
        # return f'text-align:center; color:black; background-color:{clr_dct.get(cell, "#FAFAFF")};'
        return f'text-align:center; color:black; background-color:{clr_dct.get(cell, "#EAEAEE")};'
        # return f'text-align:center; color:black; background-color:{clr_dct.get(cell, "#EDEDEE")};'
    
    
    def style_frame(frame, clr_dct, frmt_dct={}, bold_cols=[], clr_yr=None):
        return frame.reset_index(drop=True).style\
                .applymap(lambda cell: colorize_frame(cell, clr_yr, clr_dct))\
                .format(frmt_dct)\
                .set_properties(**{'font-weight': 'bold'}, subset=bold_cols)
                # .applymap(lambda cell: colorize_curr_year(cell, clr_yr))\
                # .applymap(lambda cell: colorize_player_names_new(cell, clr_dct))\
        
    
    dfpt.loc[0] = dfpt.columns
    dfpt.index = dfpt.index + 1
    dfpt = dfpt.sort_index()
    dfpt.index.name = 'Round'
    dfpt.reset_index(inplace=True)
    dfpt['Round'] = dfpt['Round'].shift(1).fillna(0).astype(int).astype(str)
    dfpt.loc[0, 'Round'] = 'Round'
    dfpt.columns = [''.join([' ']*i) for i in range(len(dfpt.columns))]
    
    
    dfpt = style_frame(dfpt, clr_dct)
    # st.write('st.dataframe')
    # st.dataframe(s)
    # st.write('st.table')
    # st.table(s)
    
    
    st.write(f"""
    Picks Pool dudes,  
    Here's your weekly update {po_inc} as of week {the_wk}!  
    Everyone's teams for {the_ssn}:
    """)
    
    st.dataframe(dfpt) ## sortable, honors center alignment and bold
    # st.table(dfpt)  ## better formatting for undesired index

    # dfy_[['Win%', 'Full_Ssn_Pace']] = dfy_[['Win%', 'Full_Ssn_Pace']].round(1)
    # st.write(dfy_)
    # dfy_.index.set_names('rank')
    # dfy_.set_index('Rank', inplace=True)
    st.dataframe(style_frame(dfy_, clr_dct, frmt_dct={'Win%': '{:.1f}', 'Full_Ssn_Pace': '{:.1f}'}))
    # st.table(style_frame(dfy_.reset_index(drop=True), clr_dct, frmt_dct={'Win%': '{:.1f}', 'Full_Ssn_Pace': '{:.1f}'}))
    # # st.table(dfy_.style.format({'Win%': '{:.1f}', 'Full_Ssn_Pace': '{:.1f}'}))
    # frame = dfy_.style.format({'Win%': '{:.1f}', 'Full_Ssn_Pace': '{:.1f}'})
    # st.table(style_frame(frame, clr_dct))

    
    
    st.write(body_dct['wk_txt'])
    # 
    # 
    st.write("""How did we do in our draft, by rounds? 
    Did we use our early draft picks wisely (does Round 1 have a higher win% than Round 2, etc.)?""")
    # 
    
    st.dataframe(style_frame(dfr_, clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win%']))
    
    
    st.write(body_dct['reghist_txt'])
    
    st.write(body_dct['pohist_txt'])
    
    st.write(body_dct['tothist_txt'])
    
    
    st.write(body_dct['champs_txt'])
    
    
    st.dataframe(style_frame(champs, clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=2021, bold_cols=['Total_Win']))
    
    st.write("Who in our pool has been the best over their careers (sorted by Wins)?")
    
    
    
    st.dataframe(style_frame(dfc_, clr_dct, frmt_dct={'Total Win%': '{:.1f}'}))
    
    
    
    
    
    st.write(body_dct['pr_txt'])
    
    
    st.dataframe(style_frame(player_hist[0], clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=2021, bold_cols=['Total_Win']))
    
    st.dataframe(style_frame(player_hist[1], clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=2021, bold_cols=['Total_Win']))
    
    st.dataframe(style_frame(player_hist[2], clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=2021, bold_cols=['Total_Win']))
    
    st.dataframe(style_frame(player_hist[3], clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=2021, bold_cols=['Total_Win']))
    
    st.dataframe(style_frame(player_hist[4], clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=2021, bold_cols=['Total_Win']))
    
    st.dataframe(style_frame(player_hist[5], clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=2021, bold_cols=['Total_Win']))
    
    st.dataframe(style_frame(player_hist[7], clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=2021, bold_cols=['Total_Win']))
    
    st.dataframe(style_frame(player_hist[6], clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=2021, bold_cols=['Total_Win']))
    

    # 
    # 
    # 
    # placeholder = st.empty()
    # 
    # # Replace the placeholder with some text:
    # placeholder.text("Hello")
    # 
    # # Replace the text with a chart:
    # placeholder.line_chart({"data": [1, 5, 2, 6]})
    # 
    # # Replace the chart with several elements:
    # with placeholder.container():
    #     st.write("This is one element")
    #     st.write("This is another")
    # 
    # # Clear all those elements:
    # placeholder.empty()
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # aa = """<html>
    #         <table>
    #         <thead>
    #         <tr style="text-align: center;">
    #         <th>Round</th>
    #         <th style="background-color:#fff2cc">Alex</th>
    #         <th style="background-color:#e6b8af">Brandon</th>
    #         <th style="background-color:#fce5cd">Dan</th>
    #         <th style="background-color:#d9ead3">JP</th>
    #         <th style="background-color:#d9d2e9">Jackson</th>
    #         <th style="background-color:#f4cccc">Jordan</th>
    #         <th style="background-color:#cfe2f3">Mike</th>
    #         <th style="background-color:#d9d9d9">Leftover</th>
    #         </tr>
    #         </thead>
    #         <tbody>
    #         <tr>
    #         <td>1</td>
    #         <td>Browns (5)</td>
    #         <td>Chiefs (5)</td>
    #         <td>Buccaneers (6)</td>
    #         <td style="color:#ff0000">Bills (5)</td>
    #         <td>Packers (7)</td>
    #         <td>Rams (7)</td>
    #         <td>Patriots (5)</td>
    #         <td>Jaguars (2)</td>
    #         </tr>
    #         <tr>
    #         <td>2</td>
    #         <td>Ravens (6)</td>
    #         <td>Colts (4)</td>
    #         <td>Cowboys (6)</td>
    #         <td>Chargers (5)</td>
    #         <td>49ers (3)</td>
    #         <td>Seahawks (3)</td>
    #         <td>Titans (7)</td>
    #         <td>Bengals (5)</td>
    #         </tr>
    #         <tr>
    #         <td>3</td>
    #         <td>Steelers (5)</td>
    #         <td>Falcons (4)</td>
    #         <td>Saints (5)</td>
    #         <td>Dolphins (2)</td>
    #         <td>Broncos (5)</td>
    #         <td>Vikings (3)</td>
    #         <td>Redskins (2)</td>
    #         <td>Lions (0)</td>
    #         </tr>
    #         <tr>
    #         <td>4</td>
    #         <td>Cardinals (8)</td>
    #         <td>Eagles (3)</td>
    #         <td>Raiders (5)</td>
    #         <td>Jets (2)</td>
    #         <td>Bears (3)</td>
    #         <td>Panthers (4)</td>
    #         <td>Giants (3)</td>
    #         <td>Texans (1)</td>
    #         </tr>
    #         </tbody>
    #         </table>
    #         </html>"""
    # 
    # 
    # ## Use this to render raw HTML <table>
    # # import streamlit.components.v1 as components
    # # components.html(aa)
    # 
    # 
    # # aa = pd.read_html(aa)[0]
    # # st.write(aa)
    # # st.table(aa)
    # # aa = aa.style.
    # 
    # # def style_negative(v, props=''):
    # #     return props if v < 0 else None
    # # s2 = df2.style.applymap(style_negative, props='color:red;')\
    # #               .applymap(lambda v: 'opacity: 20%;' if (v < 0.3) and (v > -0.3) else None)
    # # s2
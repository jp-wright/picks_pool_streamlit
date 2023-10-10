import pandas as pd
import numpy as np
import time
from pathlib import Path
import re
import os
import sys
from bs4 import BeautifulSoup
from typing import List, Tuple, Dict, Sequence, Optional
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def load_and_prep_data():
    ## Read from but never write to this file. Ref only.
    dfref = pd.read_excel(ROOT_PATH.joinpath('data', 'input', 'nfl_picks_pool_draft_history.xlsx'), sheet_name='draft_history')
    dfref.columns = [c.title().replace(' ', '_') for c in dfref.columns]
    df = dfref.copy()
    df.loc[df['Player'] == 'LEFTOVER', 'Player'] = 'Leftover'

    ## get regular ssn, post ssn, and total win/loss info
    dfreg = pd.read_csv('/Users/jpw/Dropbox/Data_Science/datasets/sports_data/NFL_databases/wiki_nfl_data/nfl_season_standings/all_nfl_regular_ssn_standings.csv').drop('Team', 1)
    dfpost = pd.read_csv('/Users/jpw/Dropbox/Data_Science/datasets/sports_data/NFL_databases/wiki_nfl_data/nfl_season_standings/all_nfl_post_ssn_standings.csv')
    dftot = pd.read_csv('/Users/jpw/Dropbox/Data_Science/datasets/sports_data/NFL_databases/wiki_nfl_data/nfl_season_standings/all_nfl_regular_plus_post_ssn_standings.csv')

    df = df[['Year', 'Round', 'Pick', 'Player', 'Team']]

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


def stats_by_year(df):
    ## Yearly Stats
    dfy = df.groupby(['Year', 'Player']).sum()
    idx = dfy.columns.get_loc('Reg_Games')
    dfy.insert(idx+1, 'Reg_Games_Left', (16*4) - dfy['Reg_Games'])

    for kind in ['Reg', 'Playoff', 'Total']:
        dfy[f"{kind}_Win%"] = dfy[f"{kind}_Win"].div(dfy[f"{kind}_Games"])

    dfy['16_Game_Pace'] = dfy['Reg_Win'] ## overwrite only if games remaining...
    dfy.loc[dfy['Reg_Games'] < 16*4, '16_Game_Pace'] = dfy.loc[dfy['Reg_Games'] < 16*4, 'Reg_Win%'] * (16 * 4)

    dfy['Playoff_Teams'] = df.groupby(['Year', 'Player'])['Playoff_Seed'].count()

    pct_cols = [c for c in dfy.columns if '%' in c]
    dfy[pct_cols] = (dfy[pct_cols] * 100).round(1)
    dfy.to_csv(ROOT_PATH.joinpath('data', 'output', 'picks_pool_stats_by_year.csv'))
    return dfy.reset_index()


def stats_by_round(df):
    dfr = df.groupby(['Year', 'Round']).sum()
    for kind in ['Reg', 'Playoff', 'Total']:
        dfr[f"{kind}_Win%"] = dfr[f"{kind}_Win"].div(dfr[f"{kind}_Games"])
        dfr[f"{kind}_Win%"] = (dfr[f"{kind}_Win%"] * 100).round(1)

    dfr['Playoff_Teams'] = df.groupby(['Year', 'Round'])['Playoff_Seed'].count()

    dfr.to_csv(ROOT_PATH.joinpath('data', 'output', 'picks_pool_stats_by_round.csv'))
    return dfr.reset_index()


def stats_by_career(df):
    dfc = df.groupby(['Player']).sum().drop(['Round', 'Pick', 'Reg_Win%', 'Playoff_Win%', 'Total_Win%'], 1)

    for kind in ['Reg', 'Playoff', 'Total']:
        dfc[f"{kind}_Win%"] = dfc[f"{kind}_Win"].div(dfc[f"{kind}_Games"])
        dfc[f"{kind}_Win%"] = (dfc[f"{kind}_Win%"] * 100).round(1)

    dfc['Playoff_Teams'] = df.groupby(['Player'])['Playoff_Seed'].count()

    dfc.to_csv(ROOT_PATH.joinpath('data', 'output', 'picks_pool_stats_by_career.csv'))
    return dfc.reset_index().sort_values(['Total_Win', 'Total_Win%'], ascending=False)


def add_css_style_to_html_table(table: str) -> str:
    style_sheet = """
    <style type="text/css">
    table{
        font-weight: 300;
        border-collapse: collapse;
        text-align: center;
        }
    th{
        background-color:  white;
        padding-right: 10px;
        padding-left: 10px;
        font-size: 14px;
        border-bottom: 2px solid black;
        text-align: center;
        }
    td{
        background-color:  #FAFAFF;
        padding-right: 10px;
        padding-left: 10px;
        border: 1px solid grey;
        text-align: center;
        }
    tr{
        text-align: center;
        }
    </style>
    <table>
    """
    # print('<table border="99" class="dataframe">' in table)
    return table.replace('<table border="99" class="dataframe">', style_sheet)


def colorize_player_names(table: str, mgr_cols: int=1) -> str:
    '''
    Highlight cells in HTML output table with manager names.
    For tables that have multiple columns of manager names, such as Winners & Losers, default is to highlight the first column (on the assumption this column is the POV of the table).
    To highlight more columns, use mgr_cols>1 to determine how many cols to highlight, in order of appearance.

    table : str
        the HTML table output of a Pandas DataFrame which is modified for highlighting
    mgr_cols : int (default=1)
        how many manager cols to highlight. Enable highlighting of multiple manager columns in a given table by using > 1.
    '''


    '''Alternative CSS approach...
        Could do this by adding an ID to CSS Style Sheet.
                <style type="text/css">
                #player_jordan{
                    background-color: clr_dct['jordan']
                    }
                </style>

                <td id="jordan">Jordan</td>

        And could put this in a loop to insert into the <style> string
    '''

    for name in clr_dct.keys():
        table = table.replace(f'<th>{name}', f'<th style="background-color:{clr_dct[name]}">{name}')

    rows = []
    mgrs = "|".join(clr_dct.keys())
    for row in table.split("<tr>"):
        names = re.findall(f"(?:<td>)\[|{mgrs}|\]\w+", row)
        if names:
            for name in names[:mgr_cols]:
                row = re.sub(f'<td>{name}', f'<td style="background-color:{clr_dct[name]}">{name}', row)
        rows.append(row)
    table = "<tr>".join(rows)
    return table


def colorize_player_names_OLD(table: str) -> str:
    '''DEFUNCT but keeping as an example in case needed again...
    Could do this by adding an ID to CSS Style Sheet.
            <style type="text/css">
            #player_jordan{
                background-color: clr_dct['jordan']
                }
            </style>

            <td id="jordan">Jordan</td>

    And could put this in a loop to insert into the <style> string
    '''

    for name in clr_dct.keys():
        table = table.replace(f'<th>{name}', f'<th style="background-color:{clr_dct[name]}">{name}')
        table = table.replace(f'<td>{name}', f'<td style="background-color:{clr_dct[name]}">{name}')
    return table


def convert_frame_to_html(frame: pd.DataFrame) -> str:
    ## df.to_html 'formatters' parameters applies to entire cols only
    table = frame.to_html(justify='center', index=False, border=99)
    table = colorize_player_names(table)
    table = add_css_style_to_html_table(table)
    return table


def highlight_curr_year_in_hist(table: str, years: List[int]) -> str:
    '''in history tables, highlight entries of specified years.  Maybe give a different BG color to whole row?  To just the index?  Maybe bold whole row?
    '''
    hl_dct = {}
    if isinstance(years, int): years = [years]
    if len(years) == 1:
        hl_dct.update({years[0]: '#3399ff'})
    # else:
    #     create dict of colors per year...

    for year in years:
        table = table.replace(f'<th>{year}', f'<th style="background-color:{hl_dct[year]}">{year}')
        table = table.replace(f'<td>{year}', f'<td style="background-color:{hl_dct[year]}">{year}')
    return table


def prep_year_data_for_email(dfy: pd.DataFrame):
    '''advanced formatting possible via df.style (requires jinja2).
    https://code.i-harness.com/en/q/df3234
    '''
    frame = dfy[dfy['Year'] == dfy['Year'].max()].drop(['Round', 'Pick'], axis=1)

    if frame['Playoff_Games'].sum() == 0:
        sort_col = 'Total_Win%'
        frame = frame.drop([c for c in frame.columns if all(['Playoff' in c, 'Teams' not in c]) or 'Total' in c], 1)
        if frame['Reg_Games_Left'].sum() == 0:
            frame.drop(['16_Game_Pace'], 1, inplace=True)
        else:
            frame = frame.round({'16_Game_Pace': 1}).sort_values('Reg_Win%', ascending=False)

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

    # ## df.to_html 'formatters' parameters applies to entire cols only
    # table = frame.to_html(justify='center', index=False, border=99)
    # table = colorize_player_names(table)
    # table = add_css_style_to_html_table(table)
    table = convert_frame_to_html(frame)
    return BeautifulSoup(table, features='lxml')


def prep_round_data_for_email(dfr: pd.DataFrame):
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
        frame.drop(['16_Game_Pace'], 1, errors='ignore', inplace=True)

    int_cols = [c for c in frame.columns if '16' not in c and 'Win%' not in c and 'Player' not in c]
    frame[int_cols] = frame[int_cols].astype(int)

    frame.insert(0, 'Rank', frame['Win%'].rank(ascending=False).astype('int'))
    frame.columns = [c.replace('_', ' ') for c in frame.columns]
    ## df.to_html 'formatters' parameters applies to entire cols only
    # table = frame.to_html(justify='center', index=False, border=99)
    # table = colorize_player_names(table)
    # table = add_css_style_to_html_table(table)
    table = convert_frame_to_html(frame)
    return BeautifulSoup(table, features='lxml')


def prep_career_data_for_email(dfc: pd.DataFrame):
    '''advanced formatting possible via df.style (requires jinja2).
    https://code.i-harness.com/en/q/df3234
    '''

    frame = dfc.loc[:, ['Player', 'Total_Win', 'Total_Games', 'Reg_Win', 'Playoff_Teams', 'Playoff_Win', 'Total_Win%']]
    frame[['Reg_Win', 'Playoff_Teams', 'Playoff_Win','Total_Win', 'Total_Games']] = frame[['Reg_Win', 'Playoff_Teams', 'Playoff_Win','Total_Win', 'Total_Games']].astype(int)

    frame.insert(0, 'Rank', frame['Total_Win'].rank(ascending=False).astype('int'))
    frame.columns = [c.replace('_', ' ') for c in frame.columns]

    ## df.to_html 'formatters' parameters applies to entire cols only
    # table = frame.to_html(justify='center', index=False, border=99)
    # table = colorize_player_names(table)
    # table = add_css_style_to_html_table(table)
    table = convert_frame_to_html(frame)
    return BeautifulSoup(table, features='lxml')


def prep_year_history(dfy: pd.DataFrame, highlight_year: int):
    '''Do Reg, Playoff, and Tot as separate tables?'''

    def create_year_hist_frame(kind: str):
        drops = {'Reg': ['Reg_Games_Left', 'Reg_Games'], 'Playoff': ['Playoff_Seed']}
        drop_me = drops.get(kind, []) + ['16_Game_Pace']
        frame = dfy.set_index(['Year', 'Player'])[[c for c in dfy.columns if kind in c]].drop(drop_me, 1, errors='ignore')
        frame.insert(0, f'{kind}_Win_Rk', frame[f'{kind}_Win'].rank(ascending=False, method='dense').astype(int))
        ints = [c for c in frame.columns if '%' not in c]
        frame[ints] = frame[ints].astype(int)
        return frame.sort_values(f'{kind}_Win_Rk', ascending=True).head(10)

    frames = []
    for kind in ['Reg', 'Playoff', 'Total']:
        hist = create_year_hist_frame(kind).reset_index()
        ## Figure out how to bold desired column -- use of .to_html() doesn't render it
        # hist[f"{kind}_Win"] = hist[f"{kind}_Win"].apply(lambda row: "<b>"+str(row)+"</b>")
        hist.columns = [c.replace(f"{kind}_", '') if 'Win_Rk' not in c else c for c in hist.columns]
        col_order = [f"{kind}_Win_Rk"] + [c for c in hist.columns if c != f"{kind}_Win_Rk"]
        hist = convert_frame_to_html(hist[col_order])
        hist = highlight_curr_year_in_hist(hist, highlight_year)
        frames.append(BeautifulSoup(hist, features='lxml'))
    return frames


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
        table = convert_frame_to_html(frame)
        table = highlight_curr_year_in_hist(table, highlight_year)
        frames.append(BeautifulSoup(table, features='lxml'))
    return frames


def prep_player_teams_this_year(df: pd.DataFrame, curr_year: int):
    df['Team'] = df['Team'] + " (" + df['Total_Win'].astype(int).astype(str) + ")"
    frame = df[df['Year'] == curr_year].sort_values(['Player', 'Pick'], ascending=[True, True])[['Player', 'Round', 'Team']]
    frame.loc[frame['Player']=='Leftover', 'Round'] = [1,2,3,4]
    frame = frame.set_index(['Round', 'Player']).unstack()
    frame.columns = frame.columns.droplevel(0)
    frame.columns.name = ''
    frame = frame[[c for c in frame.columns if c != 'Leftover'] + ['Leftover']]
    frame = convert_frame_to_html(frame.reset_index())
    frame = frame.replace('<th></th>', '') # some weird empty header col...
    return BeautifulSoup(frame, features='lxml')


def prep_champ_history(dfy: pd.DataFrame, highlight_year: int):
    frame = dfy.copy()
    frame = frame.loc[frame.groupby('Year')['Total_Win'].idxmax()][['Year', 'Player', 'Total_Win%', 'Total_Win', 'Total_Loss', 'Reg_Win', 'Playoff_Win']]\
        .sort_values('Year', ascending=False)
    int_cols = [c for c in frame.columns if any(['Win' in c, 'Loss' in c]) and '%' not in c]
    frame[int_cols] = frame[int_cols].astype(int)
    table = convert_frame_to_html(frame)
    table = highlight_curr_year_in_hist(table, highlight_year)
    return BeautifulSoup(table, features='lxml')
    

def get_count_teams_over_n_wins(nwins):
    aa = pd.read_excel("""/Users/jpw/Dropbox/Data_Science/datasets/sports_data/NFL_databases/Pythag Spreadsheets/JP_Pythag_Big_List.xlsx""")
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




def create_email_connection_info(df: pd.DataFrame, send_to_pool=False):
    email_dct = {
        'Alex': 'alex@wadswick.co.uk',
        'Mike': 'mpwright06@yahoo.com',
        'Jordan': 'jtd1991@gmail.com',
        'Brandon': 'bjbartholomew@gmail.com',
        'Jackson': 'jlewsutherland@gmail.com',
        'Dan': 'j.dan.rogers@gmail.com',
        'JP': "jpwright.nfl@gmail.com",
        'JP2': 'jonpaul.wright@icloud.com',
        }

    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"

    _SENDER_EMAIL = os.environ["JPNFLGMAILADDRESS"]
    _PASSWORD = os.environ["JPNFLGMAILPW"]

    ## IMPORTANT: If the email fails to send b/c of credential errors in Gmail, you 
    ## might have to allow "less secure apps" access in Gmail settings at
    ## https://myaccount.google.com/u/1/security

    if send_to_pool:
        recipients = list(email_dct.values())
    else:
        ## Test emails
        recipients = ["jpwright.nfl@gmail.com", 'jonpaul.wright@icloud.com']

    yr = df['Year'].max()
    wk = int(df.loc[df['Year'] == yr, 'Total_Games'].max())
    message = MIMEMultipart("alternative")
    message["Subject"] = f"NFL Picks Pool {yr} - Week {wk}"
    message["From"] = _SENDER_EMAIL
    message["To"] = ", ".join(recipients)

    return {'msg_obj': message, 'recipients': recipients, 'port': port, 'smtp_server': smtp_server, '_SENDER_EMAIL': _SENDER_EMAIL, '_PASSWORD': _PASSWORD}

## Create Email Message Body
def get_curr_weekly_standings_text():
    return f"""Dan and reigning champion Alex are both having their best seasons ever (by a country mile for Dan), while Brandon, JP, and Mike are all having their -worst- seasons ever (thanks Miami, you acquatic pieces of shit...).
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


def send_html_email(df, dfy_, dfr_, dfc_, dfpt, hist_frames, player_hist, champs, send_to_pool=False):
    ## BEST - https://realpython.com/python-send-email/#option-1-setting-up-a-gmail-account-for-development
    ## Older:
    ## 1. https://towardsdatascience.com/automate-email-with-python-1e755d9c6276
    ## 2. https://medium.com/better-programming/how-to-automate-your-emails-with-python-386b4e2d5395
    ## 3. https://towardsdatascience.com/email-automation-with-python-72c6da5eef52

    the_date = time.strftime("%A, %d %b %Y", time.localtime())
    the_time = time.strftime("%H:%M CST", time.localtime())
    the_ssn = time.localtime().tm_year - 1 if time.localtime().tm_mon < 9 else time.localtime().tm_year
    the_wk = df.loc[df['Year'] == the_ssn, 'Reg_Games'].max()

    po_inc = '(playoffs included)' if 'Playoff Win' in str(dfy_) else ''

    msg_dct = create_email_connection_info(df, send_to_pool)
    body_dct = message_body_wrapper(hist_frames)


    # Create the plain-text and HTML version of your message
    text = """\
    Skipping plain text version...
    """

    html = f"""\
    <html>
      <body>
        <p>NOTE: this email is meant to be viewed with a light background/theme.
        
           Picks Pool dudes,<br><BR>
           Here's your weekly update {po_inc} as of week {the_wk}! <BR>
           Everyone's teams for {the_ssn}:
           <BR><BR>
           {dfpt}

           <BR><BR>
           {dfy_}


           <BR><BR>
           {body_dct['wk_txt']}

           <BR><BR>
           How did we do in our draft, by rounds? <BR>
           Did we use our early draft picks wisely (does Round 1 have a higher win% than Round 2, etc.)?
           <BR><BR>
           {dfr_}

           {body_dct['nug_txt']}

           <BR>
           {body_dct['hist_intro_txt']}
           
           {body_dct['reghist_txt']}

           {body_dct['pohist_txt']}

           {body_dct['tothist_txt']}

           <BR>
           {body_dct['champs_txt']}
           
           <BR><BR>
           {champs}
           
           <BR>

           Who in our pool has been the best over their careers (sorted by Wins)? <BR>
           {body_dct['career_txt']}

           <BR><BR>
           {dfc_}



           <BR><BR>
           {body_dct['pr_txt']}

           <BR><BR>
           {player_hist[0]}
           <BR><BR>
           {player_hist[1]}
           <BR><BR>
           {player_hist[2]}
           <BR><BR>
           {player_hist[3]}
           <BR><BR>
           {player_hist[4]}
           <BR><BR>
           {player_hist[5]}
           <BR><BR>
           {player_hist[7]}
           <BR><BR>
           {player_hist[6]}

        </p>

    <BR>
    Best, <BR>
    Jonpaul Wright <BR><BR>

      This is an automated email.
      </body>
    </html>
    """

    # Turn these into plain/html MIMEText objects
    # part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")

    # Add HTML/plain-text parts to MIMEMultipart message
    # The email client will try to render the last "part" first
    message = msg_dct['msg_obj']
    # message.attach(part1)
    message.attach(part2)

    # Create secure connection with server and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(msg_dct['smtp_server'], msg_dct['port'], context=context) as server:
        server.login(msg_dct['_SENDER_EMAIL'], msg_dct['_PASSWORD'])
        server.sendmail(msg_dct['_SENDER_EMAIL'], msg_dct['recipients'], message.as_string())

    print_ppl = '\n'.join(msg_dct['recipients'])
    print(f"Email successfully sent to: \n{print_ppl}")












if __name__ == '__main__':
    ROOT_PATH = Path('/Users/jpw/Dropbox/Data_Science/projects/local_projects/2020-11-22_nfl_picks_pool')


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
    dfy_ = prep_year_data_for_email(dfy)
    dfr_ = prep_round_data_for_email(dfr)
    dfc_ = prep_career_data_for_email(dfc)
    dfpt = prep_player_teams_this_year(df, curr_year)
    hist_frames = prep_year_history(dfy, curr_year)
    player_hist = prep_player_history(dfy, curr_year)
    champs = prep_champ_history(dfy, curr_year)

    ## Read in sys arg via Crontab so guaranteed to always execute.
    ## Pass in False to send only to self test email addresses, True to send to Pool
    ## Use: %run <path>/nfl_picks_pool_email.py True/False in iPython for manual execution
    ## True = send to pool; False = send to JP accounts only
    send_html_email(df, dfy_, dfr_, dfc_, dfpt, hist_frames, player_hist, champs, send_to_pool=enforce_bool(sys.argv[1]))

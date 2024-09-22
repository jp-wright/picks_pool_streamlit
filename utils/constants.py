""" 
constants
"""
import datetime as dte
import os

ROOT_PATH = os.environ['DS_PROJ']
REPO_URL = os.environ['NFLPICKS_GITHUB_REPO']
LOCAL_PATH = os.path.join(ROOT_PATH, "local_projects")
REPO_LOCAL = os.environ['NFLPICKS_LOCAL_REPO']
LOCAL_PATH = os.environ['NFLPICKS_LOCAL_PROJ']


INT_COLS = ['Win', 'Loss', 'Tie', 'Games', 'Reg_Games_Left', 'Reg Games Left', 'Playoff_Teams', 'Total_Win', 'Total_Loss', 'Total_Tie', 'Total_Games', 'Playoff_Win', 'Playoff_Loss', 'Reg_Win', 'Reg_Loss', 'Reg_Tie']

FLOAT_COLS = ['Win%', 'Total_Win%', 'Playoff_Win%', 'Total Win%', 'Current_Proj_Wins', 'Wins_Over_Current_Pace', 'Full_Ssn_Proj_Wins', 'Wins_Over_Full_Pace']


def get_curr_season() -> int:
    """
    """
    today = dte.date.today()
    return today.year if today.month >= 9 else today.year - 1


SEASON_START = dte.date(2024, 9, 4)  ## enter this manually each season
CURR_SEASON = get_curr_season()

if SEASON_START.year != CURR_SEASON:
    raise Exception(f"Current year for this app is entered as '{SEASON_START.year}' which doesn't match current season year of {(CURR_SEASON)}.  Please enter current year in 'constants.py' ")


def get_curr_week() -> int:
    """
    isoweekday(): 1 = Monday ... 7 = Sunday
    Generally, the active portion of the NFL week is end of Thursday (4) through Monday (1).
    Season always begins on a Thursday (4).
    """
    week = int((dte.date.today() - SEASON_START).days/7)   ## floor

    ## active portion = +1 to week b/c week's games have begun
    if dte.date.today().isoweekday() in [5,6,7,1]: 
        week += 1 
    return int(week)


CURR_WEEK = get_curr_week()


import datetime
from typing import Optional
import os
import pandas as pd

ROOT_PATH = os.path.join(os.environ['DS_PROJ'], 'local_projects/2021-11-10_nfl_picks_pool_streamlit/data/output')
NFL_WEEK1 = datetime.date(2023, 9, 7)  ## using first Thursday, of season. Correct?

def get_curr_year() -> int:
    """
    """
    today = datetime.date.today()
    return today.year if today.month >= 9 else today.year - 1


# def get_curr_week() -> int:
#     dfssn = pd.read_csv(os.path.join(ROOT_PATH, 'history/all_seasons_history.csv'))
#     return dfssn.groupby('year')['tot_final_gp'].max().loc[get_curr_year()]


def get_curr_week() -> int:
    """requires entering NFL 'week 1' manually"""
    return (((datetime.date.today() - NFL_WEEK1) / 7) + datetime.timedelta(1)).days


def export_to_csv(frame: pd.DataFrame, fname: str, ovrwrt: bool=False, index: bool=False, subdir: Optional[str]=None, path: Optional[str]=None, **kwargs):
    
    if path is None: path = ROOT_PATH
    if subdir: path = os.path.join(path, subdir)
    save_path = os.path.join(path, fname)
    
    if ovrwrt or not os.path.exists(save_path):
        frame.to_csv(save_path, index=index, **kwargs)
        print(f"Saved: {save_path.split('/')[-1]}")
    else:
        print(f"{save_path.split('/')[-1]} exists and ovrwrt=False. Skipping.")
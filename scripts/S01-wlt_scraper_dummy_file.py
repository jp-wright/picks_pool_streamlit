#!/opt/anaconda3/envs/nflpickspool
import os
import subprocess
# import logging
# logging.basicConfig(level=logging.INFO)
from sports_modules.nfl_utilities import get_curr_year


PY_ENV = os.environ['NFLPICKS_PY_ENV']
# GIT_EXE = os.environ['GIT_EXE']
# ROOT_PATH = os.environ['DS_PROJ']
# LOCAL_PATH = os.path.join(ROOT_PATH, "local_projects")
# REPO_PATH = os.environ['NFLPICKS_LOCAL_REPO']
# REPO_URL = os.environ['NFLPICKS_GITHUB_REPO']
# CURR_YEAR = pickspool_funcs.get_curr_year()

# import datetime
# CURR_YEAR = datetime.date.today().year if datetime.date.today().month >= 9 else datetime.date.today().year - 1
# CURR_YEAR = get_curr_year()
# CURR_WEEK = (((datetime.date.today() - datetime.date(2023, 9, 7)) / 7) + datetime.timedelta(1)).days



def update_wlt_records():
    """Calls WLT scraper. """
    wlt_file = os.environ['NFLPICKS_WLT_SCRAPER']
    # os.system(f"{PY_ENV} {wlt_file}")
    subprocess.Popen([PY_ENV, wlt_file])

if __name__ == '__main__':
    update_wlt_records()
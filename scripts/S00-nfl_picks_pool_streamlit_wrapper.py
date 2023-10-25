#!/opt/anaconda3/envs/nflpickspool
### Wrapper to run all necessary scripts to generate NFL Picks Pool weekly email.
import os
import time
import sys
import subprocess
import pandas as pd
import re
import logging
from py_modules.python_cli_utilities import enforce_cli_bool
logging.basicConfig(level=logging.INFO)
from sports_modules.nfl_utilities import get_curr_year

PY_ENV = os.environ['NFLPICKS_PY_ENV']
GIT_EXE = os.environ['GIT_EXE']
ROOT_PATH = os.environ['DS_PROJ']
LOCAL_PATH = os.path.join(ROOT_PATH, "local_projects")
REPO_PATH = os.environ['NFLPICKS_LOCAL_REPO']
REPO_URL = os.environ['NFLPICKS_GITHUB_REPO']
CURR_YEAR = get_curr_year()

# import datetime
# CURR_YEAR = datetime.date.today().year if datetime.date.today().month >= 9 else datetime.date.today().year - 1
# CURR_WEEK = (((datetime.date.today() - datetime.date(2023, 9, 7)) / 7) + datetime.timedelta(1)).days


def update_wlt_records():
    """Calls WLT scraper. """
    wlt_file = os.environ['NFLPICKS_WLT_SCRAPER']
    # os.system(f"{PY_ENV} {wlt_file}")
    subprocess.Popen([PY_ENV, wlt_file])
    time.sleep(10)


def update_git_repo():
    """Updating this updates the Streamlit web app repo.  The Streamlit web app automatically updates by sitting on this repo."""
    frame = pd.read_csv(os.path.join(REPO_PATH, 'data/input/nfl_regular_plus_post_ssn_standings_pool_years.csv'))
    week = int(frame[frame['Year']==CURR_YEAR].assign(Week=lambda df_: df_['Win'].add(df_['Loss']))['Week'].max())

    # list to set directory and working tree
    git_dir = ['--git-dir=' + REPO_PATH + '/.git', '--work-tree=' + REPO_PATH]

    ## save status to output
    # subprocess.run([GIT_EXE] + git_dir + ['status']) ## basic status call

    ## comprehensive output status call for modified files
    PIPE = subprocess.PIPE
    process = subprocess.Popen([GIT_EXE, git_dir[0], git_dir[1], 'status'], stdout=PIPE, stderr=PIPE)
    stdoutput, stderroutput = [str(_) for _ in process.communicate()]
    modifieds = [x.replace('\\t', '') for x in str(stdoutput).split('\\n') if re.search('modified:', x)]


    if 'fatal' in str(stdoutput):
        logging.error(stdoutput)
    elif modifieds:
        # logging.info(modifieds)
        logging.info([f"{itm}\n" for itm in modifieds])
        print([f"{itm}\n" for itm in modifieds])
    else:
        logging.info("No git files modified")

    if modifieds:
        ## add changed files
        subprocess.run([GIT_EXE] + git_dir + ['add', '.'], cwd=REPO_PATH)
        time.sleep(3)

        ## commit
        subprocess.run([GIT_EXE] + git_dir + ['commit', '-m', f"update week {week} - {CURR_YEAR}"])
        time.sleep(3)

        ## push
        subprocess.run([GIT_EXE] + git_dir + ['push'])
    time.sleep(6)


def update_data_outputs():
    """Creates CSVs for website/email"""
    data_prepper = os.path.join(LOCAL_PATH, '2020-11-22_nfl_picks_pool/scripts/S02-nfl_picks_pool_data_prep.py')
    # os.system(f"{PY_ENV} {wlt_file}")
    subprocess.Popen([PY_ENV, data_prepper])
    time.sleep(10)


def compose_and_send_email(email_type: str='hybrid', send_to_pool: bool=False):
    """Send either an email that merely contains a link to the Streamlit site and says it's been updated or send an email that also contains some basic weekly charts for updating the Pool members
    """
    assert email_type in ['hybrid', 'streamlit_only'], f"email_type='{email_type}' not recognized. Choose from ['hybrid', 'streamlit_only']"

    email = {'streamlit_only': 'S03-nfl_picks_pool_email_streamlit_only.py', 
             'hybrid': 'S03-nfl_picks_pool_email_hybrid.py'}
    
    email_file = os.path.join(LOCAL_PATH, "2020-11-22_nfl_picks_pool/scripts/", email[email_type])
    # send_to_pool = sys.argv[1]  # True = send to pool, False = sent to test email only
    os.system(f"{PY_ENV} {email_file} {send_to_pool}")
    # subprocess.Popen([PY_ENV, email_file, send_to_pool])



def main():
    ## 1. Get updated results after most recent week's games
    update_wlt_records()

    ## 2. Update Git repo so Streamlit web app gets updated before sending email.
    update_git_repo()

    ## 3. Update data files for website/email
    update_data_outputs()

    ## 4. Compose and send email with updated results
    send_to_pool = False if len(sys.argv) == 1 else sys.argv[1]
    compose_and_send_email(email_type='hybrid', send_to_pool=enforce_cli_bool(send_to_pool))





if __name__ == '__main__':
    main()

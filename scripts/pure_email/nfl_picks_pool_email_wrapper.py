### Wrapper to run all necessary scripts to generate NFL Picks Pool weekly email.
import os
import time
import sys
import subprocess
import pandas as pd
import re
import logging
logging.basicConfig(level=logging.INFO)

PY_ENV = os.environ['NFLPICKS_PY_ENV']
ROOT_PATH = os.environ['DS_PROJ']
GIT_EXE = os.environ['GIT_EXE']
REPO_URL = os.environ['NFLPICKS_GITHUB_REPO']
LOCAL_PATH = os.path.join(ROOT_PATH, "local_projects")
REPO_PATH = os.path.join(ROOT_PATH, "github_projects/2021-11-10_nfl_picks_pool_streamlit")



## 1. Get updated results after most recent week's games
wlt_file = os.path.join(LOCAL_PATH, '2020-11-22_wiki_season_results_scraper/nfl_wiki_season_results_scraper.py')
# os.system(f"{PY_ENV} {wlt_file}")
subprocess.Popen([PY_ENV, wlt_file])
time.sleep(10)



# ## 2. Update Git repo so Streamlit web app gets updated before sending email.
# curr_year = time.localtime().tm_year - 1 if time.localtime().tm_mon < 9 else time.localtime().tm_year
# frame = pd.read_csv(os.path.join(REPO_PATH, 'data/input/nfl_regular_plus_post_ssn_standings_pool_years.csv'))
# week = int(frame[frame['Year']==2023].assign(Week=lambda df_: df_['Win'].add(df_['Loss']))['Week'].max())

# # list to set directory and working tree
# git_dir = ['--git-dir=' + REPO_PATH + '/.git', '--work-tree=' + REPO_PATH]

# ## save status to output
# # subprocess.run([GIT_EXE] + git_dir + ['status']) ## basic status call

# ## comprehensive output status call for modified files
# PIPE = subprocess.PIPE
# process = subprocess.Popen([GIT_EXE, git_dir[0], git_dir[1], 'status'], stdout=PIPE, stderr=PIPE)
# stdoutput, stderroutput = [str(_) for _ in process.communicate()]
# modifieds = [x.replace('\\t', '') for x in str(stdoutput).split('\\n') if re.search('modified:', x)]


# if 'fatal' in str(stdoutput):
#     logging.error(stdoutput)
# elif modifieds:
#     logging.info(modifieds)
# else:
#     logging.info("No files modified")

# ## add changed files
# subprocess.run([GIT_EXE] + git_dir + ['add', '.'], cwd=REPO_PATH)
# time.sleep(3)

# ## commit
# subprocess.run([GIT_EXE] + git_dir + ['commit', '-m', f"update week {week} - {curr_year}"])
# time.sleep(3)

# ## push
# subprocess.run([GIT_EXE] + git_dir + ['push'])
# time.sleep(10)



## 3. Compose and send email with updated results
email_file = os.path.join(LOCAL_PATH, "2020-11-22_nfl_picks_pool/scripts/pure_email/nfl_picks_pool_email_all_charts.py")
send_to_pool = sys.argv[1]  # True = send to pool, False = sent to test email only
os.system(f"{PY_ENV} {email_file} {send_to_pool}")
# subprocess.Popen([PY_ENV, email_file, send_to_pool])



## clean requirements.txt file generated using "conda list -e > requirements.txt"
## pip's "freeze" export lists URL destinations for some packages installed by conda
## this prevents version numbers being used via requirements.txt
## conda's export does include version numbers for all packages, so if I clean it
## I can use version specific package installs in the streamlit app

import os

os.chdir(os.path.dirname(os.path.abspath(__file__))))
p = "/Users/jpw/Dropbox/Data_Science/projects/github_projects/2021-11-10_nfl_picks_pool_streamlit/env/requirements_conda_condensed.txt"
# with open("../requirements.txt", "w+"") as f:
with open("../requirements.txt", "w+"") as f:
    
    
    
    dd = df[df.columns[0]].str.extract('(.*\=.*)(\=\D.*)')[0].str.replace('=', '==').dropna().reset_index(drop=True)
    dd.rename(dd.iloc[0]).drop(dd.index[0]).to_csv('/Users/jpw/Dropbox/Data_Science/projects/github_projects/2021-11-10_nfl_picks_pool_streamlit/requirements.txt', index=False)
    
    
    
    
    df.loc[-1] = df.columns[0]
    df = df.rename('temp').sort_index().reset_index(drop=True)
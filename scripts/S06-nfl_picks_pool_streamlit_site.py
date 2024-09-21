## Run in terminal with "streamlit run <this file>.py"

# import streamlit as st
# import streamlit.components.v1 as components
# import altair as alt
# from utils.data_prepper import DataPrepper
# from utils.palettes import bg_clr_dct, conf_clr_dct, plot_bg_clr_dct
# from utils.styler import style_frame
# import utils.plotter as pltr
# import utils.tabler as tblr
# from utils.page_layout import PageLayout
from utils.constants import get_curr_season

# import subprocess
# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# install('openpyxl')

# st.set_page_config(page_title="NFL Picks Pool", layout="wide", page_icon='🏈')




def main(page: str, year: int):
    if page == 'season':
        from utils.page_layout_season import PageLayoutSeason
        PageLayoutSeason(year)
    elif page == 'playoffs':
        from utils.page_layout_playoffs import PageLayoutPlayoffs
        PageLayoutPlayoffs(year)
    elif page == 'champs':
        from utils.page_layout_champs import PageLayoutChamps
        PageLayoutChamps(year)
    elif page == 'career':
        from utils.page_layout_career import PageLayoutCareer
        PageLayoutCareer(year)
    elif page == 'records':
        from utils.page_layout_records import PageLayoutRecords
        PageLayoutRecords(year)


if __name__ == '__main__':
    main('season', get_curr_season())
    
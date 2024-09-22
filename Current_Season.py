## Run in terminal with "streamlit run <this file>.py"

from utils.page_layout_season import PageLayoutSeason
from utils.constants import get_curr_season

# import subprocess
# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# install('openpyxl')


# def main(page: str, year: int):
#     if page == 'season':
#         from utils.page_layout_season import PageLayoutSeason
#         PageLayoutSeason(year)
#     elif page == 'playoffs':
#         from utils.page_layout_playoffs import PageLayoutPlayoffs
#         PageLayoutPlayoffs(year)
#     elif page == 'champs':
#         from utils.page_layout_champs import PageLayoutChamps
#         PageLayoutChamps(year)
#     elif page == 'career':
#         from utils.page_layout_career import PageLayoutCareer
#         PageLayoutCareer(year)
#     elif page == 'records':
#         from utils.page_layout_records import PageLayoutRecords
#         PageLayoutRecords(year)


if __name__ == '__main__':
    PageLayoutSeason(get_curr_season())
    
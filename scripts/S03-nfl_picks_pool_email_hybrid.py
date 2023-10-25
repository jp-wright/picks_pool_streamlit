#!/opt/anaconda3/envs/nflpickspool/bin/python
import pandas as pd
import os
import sys
import time 
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Sequence
from sports_modules.sports_emails import picks_pool_emails, test_emails
from py_modules.python_cli_utilities import enforce_cli_bool
# from .. utility import pickspool_funcs
# from ..utility.pickspool_funcs import get_curr_year, get_curr_week

from bs4 import BeautifulSoup

import datetime
import re

REPO_PATH = os.environ['NFLPICKS_LOCAL_REPO']
CURR_YEAR = datetime.date.today().year if datetime.date.today().month >= 9 else datetime.date.today().year - 1
CURR_WEEK = (((datetime.date.today() - datetime.date(2023, 9, 7)) / 7) + datetime.timedelta(1)).days
# CURR_YEAR = pickspool_funcs.get_curr_year()
# CURR_WEEK = pickspool_funcs.get_curr_week()


clr_dct = {
    'Alex': '#fff2cc',
    'Mike': '#cfe2f3',
    'JP': '#d9ead3',
    'Jordan': '#f4cccc',
    'Brandon': '#e6b8af',
    'Jackson': '#d9d2e9',
    'Dan': '#fce5cd',
    'Victoria': '#FFDAEC',
    # 'Victoria': '#FFB2E4',
    'LEFTOVER': '#d9d9d9',
    'Leftover': '#d9d9d9',
    }




def convert_frame_to_html(frame: pd.DataFrame) -> str:
    ## df.to_html 'formatters' parameters applies to entire cols only
    table = frame.to_html(justify='center', index=False, border=99)
    table = colorize_player_names(table)
    table = add_css_style_to_html_table(table)
    return table


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


        
# def colorize_curr_year(cell, year):
#     # if cell == year:
#     #     res = f"background-color: blue; color: white"
#     # else:
#     #     res = ''
#         # res = f"background-color: #FAFAFF; color: black"
#     # return res
#     # return f{"background-color: blue; color: white" if cell == year else '#FAFAFF'}"
#     return "background-color: blue; color: white"


# def colorize_player_names_new(cell, txt_clr_dct, bg_clr_dct):
#     # return f'text-align:center; color:black; background-color:{bg_clr_dct.get(cell, "")};'
#     return f'text-align:center; color:{txt_clr_dct.get(cell, "black")}; background-color:{bg_clr_dct.get(cell, "#FAFAFF")};'
#     # return f'text-align:center; color:black; background-color:{bg_clr_dct.get(cell, "#FAFAFF")};'
#     # return f'text-align:center; color:black; background-color:{bg_clr_dct.get(cell, "#EAEAEE")};'
#     # return f'text-align:center; color:black; background-color:{bg_clr_dct.get(cell, "#EDEDEE")};'


# def colorize_frame(cell, year, bg_clr_dct):
#     if cell == year:
#         res = colorize_curr_year(cell, year)
#     else:
#         res = colorize_player_names_new(cell, txt_clr_dct, bg_clr_dct)
#         # res = f"background-color: #FAFAFF; color: black"
#     return res


# def style_frame(frame, bg_clr_dct, frmt_dct={}, bold_cols=[], clr_yr=None):
#     return frame.reset_index(drop=True).style\
#             .applymap(lambda cell: colorize_frame(cell, clr_yr, bg_clr_dct))\
#             .format(frmt_dct)\
#             .set_properties(**{'font-weight': 'bold'}, subset=bold_cols)
#             # .applymap(lambda cell: colorize_curr_year(cell, clr_yr))\
#             # .applymap(lambda cell: colorize_player_names_new(cell, bg_clr_dct))\


# def create_soup_table(table):
#     return BeautifulSoup(table, features='lxml')


def process_html_table_for_email(fname: str, **kwargs) -> str:
    if not fname.endswith('.csv'): fname += '.csv'
    frame = pd.read_csv(os.path.join(REPO_PATH, f'data/output/email_tables/{fname}'))
    if 'manager_teams_data' in fname:
        frame.columns = frame.loc[0]
        frame = frame.drop(0, axis=0)
    table = convert_frame_to_html(frame)
    ###### table = style_frame(frame, clr_dct, **kwargs)  ## fails for HTML email
    ###### table = create_soup_table(table)               ## fails for HTML email
    return table


def compile_tables_for_email(fnames: Dict[str, dict]) -> Dict[str, str]:
    table_dct = {}
    for fname, params in fnames.items():
        table_dct[fname.replace('.csv', '')] = process_html_table_for_email(fname, **params)
    return table_dct




def create_email_object(send_to_pool: bool=False) -> Dict[str, str]:
        port = 465  # For SSL
        smtp_server = "smtp.gmail.com"

        _SENDER_EMAIL = os.environ["JPNFLGMAILADDRESS"]
        _PASSWORD = os.environ["JPNFLGMAILAPPPW"]
        # _PASSWORD = os.environ["JPNFLGMAILPW"] ## Gmail no longer allows login PW, must use App PW

        if send_to_pool:
            recipients = list(picks_pool_emails.values())
        else:
            ## Test emails
            recipients = list(test_emails.values())
            # recipients = ["jpwright.nfl@gmail.com", 'jonpaul.wright@icloud.com']
            

        loctime = time.localtime()
        CURR_YEAR = loctime.tm_year - 1 if loctime.tm_mon < 9 else loctime.tm_year
        # _ = pd.read_csv(os.path.join(REPO_PATH, 'data/input/nfl_regular_plus_post_ssn_standings_pool_years.csv'))
        # CURR_WEEK = int(_[_['Year']==2023].assign(Week=lambda df_: df_['Win'].add(df_['Loss']))['Week'].max())
        
        
        message = MIMEMultipart("related")
        # message = MIMEMultipart("alternative")
        # message["Subject"] = f"V-Town Fantasy Football {CURR_YEAR} - CHAMPIONSHIP WEEK"
        message["Subject"] = f"NFL Picks Pool {CURR_YEAR} - Week {CURR_WEEK}"
        message["From"] = _SENDER_EMAIL
        message["To"] = ", ".join(recipients)

        return {'msg_obj': message, 
                'recipients': recipients, 
                'port': port, 
                'smtp_server': smtp_server, 
                '_SENDER_EMAIL': _SENDER_EMAIL, 
                '_PASSWORD': _PASSWORD}


def create_email_html(tables):
    html = f"""
        <html>
        <body>
        Picks Pool Peeps! <BR>
        Here are the results after {CURR_WEEK} - {CURR_YEAR}
        
        <BR><BR><BR>
        {tables['manager_teams_data']}
        
        <BR><BR><BR>
        {tables['year_data_current']}
        
        <BR><BR>

        <p>Our Pool's website has been updated with this week's data:
        <BR>https://nfl-picks-pool.streamlit.app
        </p>

        <BR>
        Best, <BR>
        Jonpaul Wright <BR><BR>

        This is an automated email.
        </body>
        </html>
        """
    return html


def send_email_html(tables, send_to_pool: bool=False) -> None:
    '''
    ## New in 2022 for Gmail: https://stackoverflow.com/questions/73026671/how-do-i-now-since-june-2022-send-an-email-via-gmail-using-a-python-script
    ## Send inline images: https://gist.github.com/vjo/4119185 
    
    ## Older:
    ## BEST - https://realpython.com/python-send-email/#option-1-setting-up-a-gmail-account-for-development
    ## 1. https://towardsdatascience.com/automate-email-with-python-1e755d9c6276
    ## 2. https://medium.com/better-programming/how-to-automate-your-emails-with-python-386b4e2d5395
    ## 3. https://towardsdatascience.com/email-automation-with-python-72c6da5eef52
    '''

    msg_dct = create_email_object(send_to_pool)

    msgHTML = MIMEText(create_email_html(tables), "html")  # Turn this into html MIMEText object

    # Add HTML/plain-text parts to MIMEMultipart message
    # The email client will try to render the last "part" first
    message = msg_dct['msg_obj']
    message.attach(msgHTML)

    # Create secure connection with server and send email
    # context = ssl.create_default_context()
    # with smtplib.SMTP_SSL(msg_dct['smtp_server'], msg_dct['port'], context=context) as server:
    with smtplib.SMTP_SSL(msg_dct['smtp_server'], msg_dct['port']) as server:
        server.login(msg_dct['_SENDER_EMAIL'], msg_dct['_PASSWORD']) ## uses app-specific PW for Gmail in 2022
        server.sendmail(msg_dct['_SENDER_EMAIL'], msg_dct['recipients'], message.as_string())

    print_ppl = '\n'.join(msg_dct['recipients'])
    print(f"\nEmail successfully sent to: \n{print_ppl}")


def main():
    tables_and_formats = {'year_data_current.csv': dict(frmt_dct={'Win%': '{:.1f}', 
                                                                    'Full_Ssn_Pace': '{:.1f}', 
                                                                    'Win%': '{:.1f}'}), 
                        'manager_teams_data.csv': {}}


    tables = compile_tables_for_email(tables_and_formats)
    
    send_to_pool = False if len(sys.argv) == 1 else sys.argv[1]
    send_email_html(tables, send_to_pool=enforce_cli_bool(send_to_pool))

if __name__ == '__main__':
    main()
    # df = pd.read_csv(os.path.join(REPO_PATH, 'data/output/email_tables/manager_teams_data.csv'))
    # res = process_html_table_for_email('manager_teams_data')
    # create_email_html()
    # send_email_html(send_to_pool=enforce_cli_bool(False))
    # send_email_html(send_to_pool=enforce_cli_bool(sys.argv[1]))
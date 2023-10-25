#! /opt/anaconda3/envs/nflpickspool/bin/python
import pandas as pd
import os
import sys
import time 
from sports_modules.sports_emails import picks_pool_emails, test_emails
from py_modules.python_cli_utilities import enforce_cli_bool
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict

REPO_PATH = os.path.join(os.environ['DS_PROJ'], "github_projects/2021-11-10_nfl_picks_pool_streamlit")


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
        the_ssn = loctime.tm_year - 1 if loctime.tm_mon < 9 else loctime.tm_year
        frame = pd.read_csv(os.path.join(REPO_PATH, 'data/input/nfl_regular_plus_post_ssn_standings_pool_years.csv'))
        week = int(frame[frame['Year']==2023].assign(Week=lambda df_: df_['Win'].add(df_['Loss']))['Week'].max())
        
        message = MIMEMultipart("related")
        # message = MIMEMultipart("alternative")
        # message["Subject"] = f"V-Town Fantasy Football {the_ssn} - CHAMPIONSHIP WEEK"
        message["Subject"] = f"NFL Picks Pool {the_ssn} - Week {week}"
        message["From"] = _SENDER_EMAIL
        message["To"] = ", ".join(recipients)

        return {'msg_obj': message, 
                'recipients': recipients, 
                'port': port, 
                'smtp_server': smtp_server, 
                '_SENDER_EMAIL': _SENDER_EMAIL, 
                '_PASSWORD': _PASSWORD}


def send_html_email(send_to_pool: bool=False) -> None:
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

    html = """
        <html>
        <body>
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

    msgHTML = MIMEText(html, "html")  # Turn this into html MIMEText object

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



if __name__ == '__main__':
    send_html_email(send_to_pool=enforce_cli_bool(sys.argv[1]))
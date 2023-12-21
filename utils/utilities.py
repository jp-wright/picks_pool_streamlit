""" 
utilities
"""
import streamlit as st
import datetime
# import utils.image_refs as images
from typing import Optional
import requests
import functools
import logging
from utils.constants import SEASON_START

def local_css(file_name):
    """
    """
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


def load_lottieurl(url: str):
    """
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def gradient(grad_clr1, grad_clr2, title_clr, subtitle_clr, title, subtitle, subtitle_size: int=17):
    """
    """
    st.markdown(f'<h1 style="text-align:center;background-image: linear-gradient(to right,{grad_clr1}, {grad_clr2});font-size:60px;border-radius:2%;">'
                f'<span style="color:{title_clr};">{title}</span><br>'
                f'<span style="color:{subtitle_clr};font-size:{subtitle_size}px;">{subtitle}</span></h1>', 
                unsafe_allow_html=True)


def show_img(url: str, width: int=100, height: int=100, hover: Optional[str]=None, caption: Optional[str]=None, link: bool=False, spacer: Optional[int]=1):
    """
    """    
    img = f'''<img src="{url}" width={width} height={height}>'''
    if caption:
        img = img.replace('>', '') + f' alt={hover} title={hover}>'
    if link:
        img = f'<a href="{url}">' + img + '</a>'

    st.markdown(img, unsafe_allow_html=True)

    if caption:
        st.markdown(f"<font size=2>{caption}</font>", unsafe_allow_html=True)
    if spacer:
        st.markdown("<BR>" * spacer, unsafe_allow_html=True)


# def show_logo(name: str, width: int=100, height: int=100, spacer: Optional[int]=1):
#     """
#     """
#     st.markdown(f'''<img src="{images.logos[name]}" width={width} height={height}>''', unsafe_allow_html=True)
#     if spacer:
#         st.markdown("<BR>" * spacer, unsafe_allow_html=True)

def get_curr_year() -> int:
    """
    """
    today = datetime.date.today()
    return today.year if today.month >= 9 else today.year - 1


def get_curr_week() -> int:
    """
    isoweekday(): 1 = Monday ... 7 = Sunday
    Generally, the active portion of the NFL week is end of Thursday (4) through Monday (1).
    Season always begins on a Thursday (4).
    """
    week = int((datetime.date.today() - SEASON_START).days/7)   ## floor

    ## active portion = +1 to week b/c week's games have begun
    if datetime.date.today().isoweekday() in [5,6,7,1]: 
        week += 1 
    
    return int(week)

def func_metadata(func: object) -> object:
    """Print the function signature and return value.  The 'signature' line needs to be updated to work in a class."""
    @functools.wraps(func)
    def wrapper_func_metadata(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        # print(f"Calling {func.__name__}({signature})\n\n\n")
        logging.warning(f"Running: {func.__name__} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%m:%S')}")
        print(f"Running: {func.__name__} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%m:%S')}")
        res = func(*args, **kwargs)
        # print(f"{func.__name__!r} returned {res!r}")
        logging.warning(f"Completed: {func.__name__} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%m:%S')}\n")
        print(f"Completed: {func.__name__} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%m:%S')}\n")
        return res
    return wrapper_func_metadata


def output_logger(process, printout: bool=False, raise_err: bool=False):
    """process is subprocess.Popen(... , stdout=subprocess.PIPE, stderr=subprocess.PIPE)"""
    stdoutput, stderroutput = process.communicate()

    if len(stdoutput) > 0:
        logging.info(stdoutput)
        if printout: 
            print(str(stdoutput))
            
    if len(stderroutput) > 0:
        logging.error(stderroutput)
        if printout: 
            print(str(stderroutput))
        if raise_err:
            raise Exception(str(stderroutput))


""" 
utilities
"""
import datetime as dte
import functools
import logging
from pandas import DataFrame
from numpy import array as np_array, isin as np_isin
import subprocess
import os
from typing import Callable, Optional, List
from utils.constants import INT_COLS, FLOAT_COLS


def func_metadata(func: Callable) -> Callable:
    """Print the function signature and return value.  The 'signature' line needs to be updated to work in a class."""
    @functools.wraps(func)
    def wrapper_func_metadata(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        # print(f"Calling {func.__name__}({signature})\n\n\n")
        logging.warning(f"Running: \t{func.__name__} - {dte.datetime.now().strftime('%Y-%m-%d %H:%m:%S')}")
        print(f"Running: {func.__name__} - {dte.datetime.now().strftime('%Y-%m-%d %H:%m:%S')}")
        res = func(*args, **kwargs)
        # print(f"{func.__name__!r} returned {res!r}")
        logging.warning(f"Completed: \t{func.__name__} - {dte.datetime.now().strftime('%Y-%m-%d %H:%m:%S')}\n")
        print(f"Completed: {func.__name__} - {dte.datetime.now().strftime('%Y-%m-%d %H:%m:%S')}\n")
        return res
    return wrapper_func_metadata


def output_logger(process: subprocess.Popen, printout: bool=False, raise_err: bool=False):
    """process is subprocess.Popen(... , stdout=subprocess.PIPE, stderr=subprocess.PIPE)"""
    stdoutput, stderroutput = process.communicate()

    if len(stdoutput) > 0:
        logging.info(stdoutput)
        if printout: 
            s = str(stdoutput.decode('utf-8')).split('\\n')
            for itm in s:
                print(itm)
        
    if len(stderroutput) > 0:
        logging.error(stderroutput)
        if printout: 
            s = str(stderroutput.decode('utf-8')).split('\\n')
            for itm in s:
                print(itm)
        if raise_err:
            raise Exception(str(stderroutput))
        

def export_to_csv(frame: DataFrame, fname: str, ovrwrt: bool=False, index: bool=False, subdir: Optional[str]=None, path: Optional[str]=None, **kwargs):
    """
    """
        
    if subdir: path = os.path.join(path, subdir)
    save_path = os.path.join(path, fname)
    
    if ovrwrt or not os.path.exists(save_path):
        if not os.path.exists(path): 
            os.makedirs(path)
        frame.to_csv(save_path, index=index, **kwargs)
        print(f"Saved: {save_path.split('/')[-1]}")
    else:
        print(f"{save_path.split('/')[-1]} exists and ovrwrt=False. Skipping.")


def enforce_int_cols(frame: DataFrame, extra_cols: List[str]=[], log: bool=False):
    """
    asd
    """
    int_cols = np_array(INT_COLS + extra_cols)
    int_cols = int_cols[np_isin(int_cols, frame.columns)]
    
    for col in int_cols:
        try:
            frame[int_cols] = frame[int_cols].fillna(0).astype(int)
        except ValueError as e:
            print(e) ## I think this is captured by Streamlit's stdout
            if log:
                logging.error(col)
                logging.error(frame[col])
            

def enforce_float_cols(frame: DataFrame, extra_cols: List[str]=[], log: bool=False):
    """
    asd
    """
    float_cols = np_array(FLOAT_COLS + extra_cols)
    float_cols = float_cols[np_isin(float_cols, frame.columns)]
    
    for col in float_cols:
        try:
            frame[float_cols] = frame[float_cols].fillna(0).astype(float).round(1)
        except ValueError as e:
            print(e) ## I think this is captured by Streamlit's stdout
            if log:
                logging.error(col)
                logging.error(frame[col])
            
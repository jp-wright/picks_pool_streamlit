""" 
styling functions for Pandas DataFrames --> HTML
"""
from utils.palettes import bg_clr_dct, conf_clr_dct
from pandas import DataFrame
from typing import Union, Literal, List, Dict, Mapping
import re
# import logging
# logging.basicConfig(level=logging.INFO, filename='logs/s03_log.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

## I need to create dedicated functions for Streamlit and Email styling, as the process is different for each.

def style_frame(frame: DataFrame, kind: Literal['streamlit', 'email'], cell_clr_dct: Dict[str, str]=bg_clr_dct, frmt_dct: Dict[str, str]={}, bold_cols: List=[], clr_yr: int=2000) -> str:
    """
    asd
    """
    
    # logging.info(f"styler.py - style_frame - frame: \n{frame}")

    return frame.reset_index(drop=True).style\
            .applymap(lambda cell: colorize_frame(cell, clr_yr, cell_clr_dct, kind))\
            .format(frmt_dct)\
            .set_properties(**{'font-weight': 'bold'}, subset=bold_cols)
            # .applymap(lambda cell: colorize_curr_year(cell, clr_yr))\
            # .applymap(lambda cell: colorize_player_names(cell, cell_clr_dct))\


def colorize_frame(cell: Union[float, int, str], year: int, bg_clr_dct: Dict[str, str], kind: Literal['streamlit', 'email']):
    """
    asd
    """
    if cell == year:
        res = colorize_curr_year(cell, year)
    else:
        if kind == 'streamlit':
            res = colorize_player_names_streamlit(cell)
        elif kind == 'email':
            res = colorize_player_names_email(cell)
        # else:
        #     res = f"background-color: #FAFAFF; color: black"
    return res
    

def colorize_curr_year(cell: Union[float, int, str], year: int):
    """
    asd
    """
    # if cell == year:
    #     res = f"background-color: blue; color: white"
    # else:
    #     res = ''
        # res = f"background-color: #FAFAFF; color: black"
    # return res
    # return f{"background-color: blue; color: white" if cell == year else '#FAFAFF'}"
    return "background-color: blue; color: white"


def colorize_player_names_email(table: str, mgr_cols: int=1) -> str:
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
                    background-color: bg_clr_dct['jordan']
                    }
                </style>

                <td id="jordan">Jordan</td>

        And could put this in a loop to insert into the <style> string
    '''

    for name in bg_clr_dct.keys():
        table = table.replace(f'<th>{name}', f'<th style="background-color:{bg_clr_dct[name]}">{name}')

    rows = []
    mgrs = "|".join(bg_clr_dct.keys())
    for row in table.split("<tr>"):
        names = re.findall(f"(?:<td>)\[|{mgrs}|\]\w+", row)
        if names:
            for name in names[:mgr_cols]:
                row = re.sub(f'<td>{name}', f'<td style="background-color:{bg_clr_dct[name]}">{name}', row)
        rows.append(row)
    table = "<tr>".join(rows)
    return table


def colorize_player_names_streamlit(cell: Union[float, int, str], 
                              conf_clr_dct: Dict[str, str]=conf_clr_dct, 
                              bg_clr_dct: Dict[str, str]=bg_clr_dct):
    """
    NEW 2023
    """
    
    names = "|".join(bg_clr_dct.keys())
    cell = re.search(names, str(cell)).group() if re.search(names, str(cell)) else 'null'

    return f"""
            text-align:center; 
            color:{conf_clr_dct.get(cell, "black")}; 
            background-color:{bg_clr_dct.get(cell, "#FAFAFF")}; 
            """
    # return f'align:center; color:{conf_clr_dct.get(cell, "black")}; background-color:{bg_clr_dct.get(cell, "#FAFAFF")};'
    # return f'text-align:center; color:black; background-color:{bg_clr_dct.get(cell, "")};'
    # return f'text-align:center; color:black; background-color:{bg_clr_dct.get(cell, "#FAFAFF")};'
    # return f'text-align:center; color:black; background-color:{bg_clr_dct.get(cell, "#EAEAEE")};'
    # return f'text-align:center; color:black; background-color:{bg_clr_dct.get(cell, "#EDEDEE")};'


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
        border: 1px solid black;
        text-align: center;
        }
    td{
        background-color:  #FAFAFF;
        padding-right: 8px;
        padding-left: 8px;
        font-size: 13.5px;
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


def create_styled_frame(frame, names_regex, bg_clr_dct, use_row1_as_cols: bool=True):
    """for use in streamlit tables
    """
    if use_row1_as_cols:
        frame.columns = frame.iloc[0]
        frame = frame.drop(0)

    def get_name(col):
        return re.search(names_regex, str(col)).group() if re.search(names_regex, str(col)) else 'null'

    def selector_style(frame, col):
        return {
            'selector': f'th.col_heading.level0.col{frame.columns.get_loc(col)}',
            'props': [
                ('background-color', bg_clr_dct.get(get_name(col), 'white')),
                ('color', 'black'),
                # ('font-family', 'Arial, sans-serif'),
                ('font-size', '16px')
                ]
            # },
            # {
            #     'selector': 'td, th',
            #     'props': [
            #         ('border', '2px solid #4CAF50')
            #     ]
            # }
        }
    
    def style_headers(frame): 
        res = frame.style.set_table_styles(
                    [selector_style(frame, col) for col in frame.columns]
                    )
        return res
    
    return style_headers(frame).hide()



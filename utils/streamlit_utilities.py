""" 
streamlit page utilities
"""
import streamlit as st
from typing import Optional
from utils.palettes import blue_bath1




def local_css(file_name):
    """
    """
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


# def load_lottieurl(url: str):
#     """
#     """
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()


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

def frmt_cols(int_cols: list, float_cols: list) -> dict:
    """
    """
    frmts = {k: '{:.0f}' for k in int_cols}
    frmts.update({k: '{:.1f}' for k in float_cols})
    return frmts


def intro(year: int):
    gradient(blue_bath1[1], blue_bath1[3], blue_bath1[5], '#fcfbfb', f"🏈 NFL Picks Pool", "A Swimming Pool of Interceptions", 27)

    st.markdown(f"""<h4 align=center>Global Pool from Colorado ("Hey!"), Texas (Howdy!), California (Like, HeLloOo!), England ('allo!), Japan (Konnichiwa!), Germany (Guten Tag!), and Sweden (Allo!)</h4><BR>""", unsafe_allow_html=True)
    st.markdown(f"<h3 align=center>{year} Review</h3>", unsafe_allow_html=True)
    



# def manually_remove_weird_row_in_table(table: str):
#     import re
#     # dct = {}
#     remove = """<tr> <th class="index_name level0" >Round</th> <th class="blank col0" >&nbsp;</th> <th class="blank col1" >&nbsp;</th> <th class="blank col2" >&nbsp;</th> <th class="blank col3" >&nbsp;</th> <th class="blank col4" >&nbsp;</th> <th class="blank col5" >&nbsp;</th> <th class="blank col6" >&nbsp;</th> <th class="blank col7" >&nbsp;</th> </tr>"""

#     # logging.info(f"\n\nNEW! Table: {table}")
#     # logging.info(rf"\n\nNEW! Table: {table}")
#     # remove = """.*<tr> <th class="index_name level0" >Round.*"""
#     # remove = r"""> <th class="index_name level0" >Round"""
#     # pat = re.compile(remove)
#     remove = """.*<tr>&nbsp;<th class="index_name level0" >.*|.*<tr>/s<th class="index_name level0" >.*|.*<tr>/b<th class="index_name level0" >.*"""
#     # assert re.search(remove, rf'{table}'), f"Table row not found: {remove}"
#     assert re.search(remove, f'{table}'), f"Table row not found: {remove}"

#     # table = table.replace(remove, '')

#     replacer = """<th class="index_name level0" >0</th>"""
#     # table = re.sub(remove, '', table, flags=re.M | re.I)
#     return re.sub(replacer, """<th class="index_name level0" >Round</th>""", table)


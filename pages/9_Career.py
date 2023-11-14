import altair as alt
# import pandas as pd
import streamlit as st
# import base64
# from utils.constant import *
from utils.utilities import *
from utils.palettes import *
# from utils.image_refs import *
from utils.data_prepper import DataPrepper


DP = DataPrepper()



class PageLayout():
    def __init__(self):
        st.set_page_config(page_title="NFL Picks Pool", layout="wide", page_icon='🏈', initial_sidebar_state="expanded")
        local_css("style/style.css")
        # st.sidebar.markdown(info['Photo'], unsafe_allow_html=True)
        self.intro()

    def intro(self):
        gradient(blue_bath1[1], blue_bath1[3], blue_bath1[5], '#fcfbfb', f"🏈 NFL Picks Pool", "A Swimming Pool of Interceptions", 27)

        st.markdown(f"""<h6 align=center>We've gone global: Colorado, Texas, California, England, Japan, and Sweden</h6><BR>""", unsafe_allow_html=True)
        st.markdown(f"<h5 align=center>Standings {DP.po_inc} as of Week {int(DP.the_wk)} - {DP.the_ssn}!</h5>", unsafe_allow_html=True)
        
    def careers(self):
        st.write("""#""")
        st.write("""#### Career Performance""")
        st.write("Who in our pool has been the best over their careers (sorted by Wins)?")
        
        st.dataframe(DP.style_frame(DP.dfc_, DP.bg_clr_dct, frmt_dct={'Total Win%': '{:.1f}'}))
        
        st.write("""...Victoria hasn't even won as many games as the Leftovers.  Sad! 😜""")
        
        st.write("""#""")
        # dfs_ = player_hist.sort_values('Year', ascending=True).groupby(['Player', 'Year']).sum().groupby('Player').cumsum().reset_index().sort_values(['Player', 'Year'])
        # dfs_
        if 'Champ' in DP.player_hist.columns:
            player_hist = DP.player_hist 
        else:
            player_hist = DP.player_hist.merge(DP.champs.assign(Champ=True)[['Player', 'Year', 'Champ']], on=['Player', 'Year'], how='left').fillna(False)
        # player_hist = player_hist.merge(champs.assign(Champ='Yes')[['Player', 'Year', 'Champ']], on=['Player', 'Year'], how='left').fillna('No')
        # player_hist.loc[(player_hist['Year']==curr_year) & (player_hist['Champ']=='Yes'), 'Champ'] = 'Proj'
        # player_hist.tail(20)
        
        
        ## tried to use this for color=champ_condition .. can't get to work
        # champ_condition = {
        #     'condition': [
        #         {alt.datum.Champ: 'Yes', 'value': 'firebrick'},
        #         {alt.datum.Champ: 'Proj', 'value': 'Navy'}],
        #      'value': 'orange'}
            
            
        
        bars = alt.Chart()\
                    .mark_bar()\
                    .encode(
                        alt.X('Player:N', axis=alt.Axis(title='')),
                        alt.Y('Total_Win:Q', scale=alt.Scale(domain=[0, 50], zero=True)),
                        # color=alt.Color('Player:N', scale=alt.Scale(domain=dfs_['Player'].unique(),       range=list(DP.plot_bg_clr_dct.values()))),
                        # color=champ_condition
                        color=alt.condition(
                            alt.datum.Champ == True, 
                            alt.value('firebrick'), 
                            # alt.value(list(DP.plot_bg_clr_dct.values())),
                            alt.value(DP.plot_bg_clr_dct['Mike']),
                            ),
                        )
                        

        text = bars.mark_text(align='center', baseline='bottom')\
                    .encode(text='Total_Win:Q')
                    
        ## Can't use "+" layer operator with faceted plots
        chart = alt.layer(bars, text, data=player_hist).facet(column=alt.Column('Year:O', header=alt.Header(title='')), title=alt.TitleParams(text='Wins by Year', anchor='middle'))#.resolve_scale(color='independent')

        st.altair_chart(chart)


if __name__ == '__main__':
    PageLayout()
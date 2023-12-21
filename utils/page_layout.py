""" .
"""

from typing import Optional
from math import ceil
import datetime as dte
import streamlit as st
import altair as alt
# import pandas as pd
from pandas import DataFrame
# import functools
import logging
logging.basicConfig(level=logging.INFO, filename='logs/page_layout.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
from utils.utilities import gradient, local_css, get_curr_year, get_curr_week, func_metadata
from utils.palettes import *
from utils.data_prepper import DataPrepper
import utils.styler as sty
import utils.plotter as plt

SEASON_START = dte.date(2023, 9, 7)  ## enter this manually each season

class PageLayout(DataPrepper):
    """Layout class for all pages
    """
    def __init__(self, page: str, year: Optional[int]=None):
        print("Reloading Page...")
        if year is None: year = get_curr_year()
        super().__init__(year)
        st.set_page_config(page_title="NFL Picks Pool", layout="wide", page_icon='🏈', initial_sidebar_state="expanded")
        local_css("style/style.css")
        self.year = year
        self.week = get_curr_week()   
        # self.week = int((dte.date.today() - SEASON_START).days/7)   ## floor
        # self.week = ceil((dte.date.today() - SEASON_START).days/7)
        
        
        # st.sidebar.markdown(info['Photo'], unsafe_allow_html=True)
        if page == 'season':
            # st.table()
            self.intro()
            if self.year == get_curr_year():
                self.WoW_metrics()
            self.manager_ranking()
            self.manager_by_round()
            st.markdown("***")
            self.draft_overview_chart()
            self.best_worst_picks()
            self.best_draft_rounds()
            # self.project_info()
            # self.conclusion()
            # self.gallery()

        elif page == 'best_seasons':
            self.best_seasons_header()
            self.top_10_total_wins()
            self.top_10_playoffs()
            self.top_10_reg_ssn()
        elif page == 'champs':
            self.champions()
            self.champs_by_year_bar_chart()
        elif page == 'career':
            self.career_wins()
            self.personal_records()
        elif page == 'pool_records':
            pass

    def frmt_cols(self):
        frmts = {k: '{:.0f}' for k in self.int_cols}
        frmts.update({k: '{:.1f}' for k in self.float_cols})
        return frmts

    def intro(self):
        gradient(blue_bath1[1], blue_bath1[3], blue_bath1[5], '#fcfbfb', f"🏈 NFL Picks Pool", "A Swimming Pool of Interceptions", 27)

        st.markdown(f"""<h6 align=center>We've gone global: Colorado, Texas, California, England, Japan, and Sweden</h6><BR>""", unsafe_allow_html=True)

        # hdr = 'Weekly Update!' if self.year == get_curr_year() else 'in Review!'
        hdr = f'- Week {self.week}' if all([self.year == get_curr_year(), self.week < 21]) else 'in Review!'
        st.markdown(f"<h2 align=center>{self.year} {hdr}</h2>", unsafe_allow_html=True)

    @func_metadata
    def WoW_metrics(self):
        """
        """
        st.write("""<BR><h6 align=center>Week over Week Changes 📈</h6>""", unsafe_allow_html=True)

        def show_metric(frame, idx):
            if frame.iloc[idx]['WoW_Wins'] <= 1:
                clr = 'inverse'
            elif frame.iloc[idx]['WoW_Wins'] < 3:
                clr = 'off'
            else:
                clr = 'normal'

            st.metric(f":blue[{idx+1}. {frame.iloc[idx]['Player']}]", f"{int(frame.iloc[idx]['Total_Win'])} wins", f"{int(frame.iloc[idx]['WoW_Wins'])} last week", delta_color=clr)
            

        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                show_metric(self.wow, 0)
            with col2:
                show_metric(self.wow, 1)
            with col3:
                show_metric(self.wow, 2)
            with col4:
                show_metric(self.wow, 3)
            with col1:
                show_metric(self.wow, 4)
            with col2:
                show_metric(self.wow, 5)
            with col3:
                show_metric(self.wow, 6)
            with col4:
                show_metric(self.wow, 7)

    @func_metadata
    def manager_ranking(self):
        """
        """
        frame = self.dfy_
        st.write("""<BR><h6 align=center>Manager Ranking #️⃣1️⃣</h6>""", unsafe_allow_html=True)
        bold = 'Total Win' if 'Total Win' in frame.columns else 'Win'
        
        with st.container():
            _, col2, _ = st.columns([.05, .9, .05])
            with col2:
                # st.dataframe(sty.style_frame(frame, bg_clr_dct, frmt_dct=self.frmt_cols(), bold_cols=[bold]), use_container_width=True, hide_index=True)
                
                ## using st.table() honors intracellular formatting...  I actually like this more, but it's hard to center (yes..) and isn't sortable AND can't get rid of the damned index yet
                st.table(sty.style_frame(frame, bg_clr_dct, frmt_dct=self.frmt_cols(), bold_cols=[bold]))

    @func_metadata                
    def manager_by_round(self):
        """Use st.dataframe() here to allow for the colored column 'headers' (really row 1, not column header row, but it's the workaround for now...)
        """
        st.write("""<BR><h6 align=center>Manager by Round (wins) </h6>""", unsafe_allow_html=True)
        with st.container():
            _, col2, _ = st.columns([.05, .9, .05])
            with col2:
                st.dataframe(sty.style_frame(self.dfpt, self.bg_clr_dct), hide_index=True, use_container_width=True)

    def playoff_teams(self):
        """
        """
        if not self.dfpo.empty:
            st.write("""<BR><h6 align=center>Playoff Teams Tracker</h6>""", unsafe_allow_html=True)
            st.dataframe(sty.style_frame(self.dfpo, self.bg_clr_dct, frmt_dct={'Playoff_Win': '{:.0f}', 'Playoff_Loss': '{:.0f}'}), width=765, height=620, hide_index=True)

    def draft_overview_chart(self):
        """
        """
        st.write(f"""<BR><h4 align=center>The {self.year} Draft</h4>""", unsafe_allow_html=True)
        st.write("""<p align=center>TIP: Click any player's dot to see only their picks. Shift-Click dots to add more players; double-click to reset.</p>""", unsafe_allow_html=True)
        plt.plot_draft_overview_altair(self.df, self.year)

    def best_worst_picks(self):
        """
        """
        def picks_by_round(frame: DataFrame, best_worst: str, rd: int): 
            """
            """
            st.markdown(f"""<div align=center style="color:#76756e">Round {rd}</div>""", unsafe_allow_html=True)
            max_min = 'max' if best_worst.lower() == 'best' else 'min'
            idx = frame.groupby('Round')['Total_Win'].transform(max_min) == frame['Total_Win']
            
            # st.dataframe(sty.style_frame(frame[(idx) & (frame['Round']==rd)], bg_clr_dct, frmt_dct={'Total_Win': '{:.0f}'}), width=495, hide_index=True, use_container_width=True)
            st.table(sty.style_frame(frame[(idx) & (frame['Round']==rd)].rename(columns={'Total_Win': 'Win'}), bg_clr_dct, frmt_dct={'Win': '{:.0f}'}, bold_cols=['Win']))


        st.write('<h4 align=center> Best and Worst Picks By Round </h4>', unsafe_allow_html=True)
        with st.container():
            left_col, right_col = st.columns([1, 1])
            with left_col:
                st.write("""<h6 align=center>✅ Best Picks</h6>""", unsafe_allow_html=True)
            
            with right_col:
                st.write("""<h6 align=center>❌ Worst Picks</h6>""", unsafe_allow_html=True)

        for rd in range(1, 5):
            with st.container():
                left_col, right_col = st.columns([1, 1])
                with left_col:
                    picks_by_round(self.dfd, 'Best', rd)
                with right_col:
                    picks_by_round(self.dfd, 'Worst', rd)

    def best_draft_rounds(self):
        st.write(""" # """)
        st.write(f"""<BR><h5 align=center>Best Draft Rounds</h5>""", unsafe_allow_html=True)
        st.write("""<div align=center>Did we use our early draft picks wisely (does Round 1 have a higher win% than Round 2, etc.)?</div>""", unsafe_allow_html=True)
        
        drop_cols = ['Rank', 'Year'] if self.dfr_['Playoff Teams'].sum() > 0 else ['Rank', 'Year', 'Playoff Teams']
        frame = self.dfr_.drop(drop_cols, axis=1)[['Round', 'Win%', 'Win', 'Loss', 'Tie', 'Games']]
        with st.container():
            _, col2, _ = st.columns([.35, 1, .35])
            with col2:
                st.table(sty.style_frame(frame, bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win%']))
        # with st.container():
        #     _, col2 = st.columns([.45, 1])
        #     with col2:
        #         st.dataframe(sty.style_frame(frame, bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win%']))
            
    def best_seasons_header(self):
        st.write("""<h1 align=center>Best Seasons in Pool History</h1>""", unsafe_allow_html=True)

    def top_10_reg_ssn(self):
        st.write("""<BR><h6 align=center>Top 10 Regular Seasons</h6>""", unsafe_allow_html=True)
        with st.container():
            _, col2, _ = st.columns([.25, 1, .25])
            with col2:
                # st.dataframe(sty.style_frame(self.hist_frames[0].sort_values(['Win%_Rk', 'Win', 'Year'], ascending=[True, False, True]), bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win%']), width=620, hide_index=True)

                st.table(sty.style_frame(self.hist_frames[0].sort_values(['Win%_Rk', 'Win', 'Year'], ascending=[True, False, True]).rename(columns={'Win%_Rk': 'Rk'}).head(10), self.bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win%']))
        
    def top_10_playoffs(self):
        
        st.write("""<BR><h6 align=center>Top 10 Playoff Runs</h6>""", unsafe_allow_html=True)
        with st.container():
            _, col2, _ = st.columns([.25, 1, .25])
            with col2:
                # st.dataframe(sty.style_frame(self.hist_frames[1].sort_values(['Win_Rk', 'Win%', 'Year'], ascending=[True, False, True]), bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win']), width=620, hide_index=True)

                ## table vs dataframe formatting
                st.table(sty.style_frame(self.hist_frames[1].sort_values(['Win_Rk', 'Win%', 'Year'], ascending=[True, False, True]).rename(columns={'Win_Rk': 'Rk'}).head(10), self.bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win']))

    def top_10_total_wins(self):

        st.write("""<BR><h6 align=center>Top 10 Total Season Wins</h6>""", unsafe_allow_html=True)
        with st.container():
            _, col2, _ = st.columns([.25, 1, .25])
            with col2:
                # st.dataframe(sty.style_frame(self.hist_frames[2].sort_values(['Win_Rk', 'Win%', 'Year'], ascending=[True, False, True]), bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win']), width=620, hide_index=True)

                ## table vs dataframe formatting
                st.table(sty.style_frame(self.hist_frames[2].drop('Win%_Rk', axis=1).sort_values(['Win', 'Win%', 'Year'], ascending=[False, False, True]).head(10), self.bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win']))

    def champions(self):
        st.write("""<h1 align=center>Pool Champions</h1>""", unsafe_allow_html=True)
        st.write("""<div align=center>Past champions and their results, as well as projected champion for the current year (<text style="color:{year_highlight}"><b>highlighted in grey</b></text>).</div>""", unsafe_allow_html=True)
        st.write("""<div align=center>I'm not sure how to parse the added week 18 in the regular season except to use win percent as opposed to wins.</div>""", unsafe_allow_html=True)
        
        with st.container():
            _, col2, _ = st.columns([.25, 1, .25])
            with col2:
                # st.dataframe(sty.style_frame(self.champs, bg_clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=self.curr_year, bold_cols=['Total_Win']), hide_index=True) 

                st.table(sty.style_frame(self.champs, bg_clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=self.curr_year, bold_cols=['Total_Win']))
        
    def career_wins(self):
        st.write("""<h1 align=center>Career Performance</h1>""", unsafe_allow_html=True)
        st.write("<div align=center>Who in our pool has been the best over their careers (sorted by Wins)?</div>", unsafe_allow_html=True)
        
        with st.container():
            _, col2, _ = st.columns([.25, 1, .25])
            with col2:
                # st.dataframe(sty.style_frame(self.dfc_, self.bg_clr_dct, frmt_dct={'Total Win%': '{:.1f}'}), hide_index=True)
                st.table(sty.style_frame(self.dfc_, self.bg_clr_dct, frmt_dct={'Total Win%': '{:.1f}'}, bold_cols=['Total Win']))
        
        st.write("""<div align=center>...Victoria hasn't even won as many games as the Leftovers.  Sad! 😜</div>""", unsafe_allow_html=True)
        
    def champs_by_year_bar_chart(self):
        st.write("""<h4 align=center>Champion Comparison per Year [NEEDS WORK]</h4>""", unsafe_allow_html=True)
        if 'Champ' in self.player_hist.columns:
            player_hist = self.player_hist 
        else:
            player_hist = self.player_hist.merge(self.champs.assign(Champ=True)[['Player', 'Year', 'Champ']], on=['Player', 'Year'], how='left').fillna(False)
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
                        # color=alt.Color('Player:N', scale=alt.Scale(domain=dfs_['Player'].unique(),       range=list(self.plot_bg_clr_dct.values()))),
                        # color=champ_condition
                        color=alt.condition(
                            alt.datum.Champ == True, 
                            alt.value('firebrick'), 
                            # alt.value(list(self.plot_bg_clr_dct.values())),
                            alt.value(self.plot_bg_clr_dct['Mike']),
                            ),
                        )
                        

        text = bars.mark_text(align='center', baseline='bottom')\
                    .encode(text='Total_Win:Q')
                    
        ## Can't use "+" layer operator with faceted plots
        chart = alt.layer(bars, text, data=player_hist).facet(column=alt.Column('Year:O', header=alt.Header(title='')), title=alt.TitleParams(text='Wins by Year', anchor='middle'))#.resolve_scale(color='independent')

        st.altair_chart(chart)
    
    def show_player_hist_table(self, name):
        st.write(f"""<div align=center>Career Review</div>""", unsafe_allow_html=True)
        # st.write(f"""<div align=center>Career Review - <text style="color:{plot_bg_clr_dct[name]}"><b>{name}</b></text></div>""", unsafe_allow_html=True)
        st.dataframe(sty.style_frame(self.player_hist[self.player_hist['Player'] == name].drop(['Reg_Win', 'Playoff_Win'], axis=1).rename(columns={c: c.replace('Total_', '') for c in self.player_hist.columns}), bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, clr_yr=self.curr_year, bold_cols=['Win%']), width=700, hide_index=True)

    def personal_records(self):
        st.write("""<h1 align=center>Pool Records [WIP]</h1>""", unsafe_allow_html=True)
        st.write(f"""<div align=center>Each manager's career history, sorted by highest winning seasons at top.  <BR><text style="color:{year_highlight}"><b>Grey highlight</b></text> is for this season and shows at a glance where the current season ranks in each manger's career.</div>""", unsafe_allow_html=True)
        
        for name in self.dfy_['Player'].unique():
            with st.container():
                st.markdown('***')
                st.write(f"""<div align=center><text style="color:{plot_bg_clr_dct[name]}; font-size:34px"><b>{name}</b></text></div>""", unsafe_allow_html=True)
                col1, _, col3 = st.columns([1, .2,  1])
                with col1:
                    self.show_player_hist_table(name)
                with col3: 
                    plt.plot_wins_by_year(self.player_hist[self.player_hist['Player'] == name])
                    st.write("\n\n\n _")


 




    # def project_info(self):
    #     st.markdown('<BR>', unsafe_allow_html=True)
    #     # st.header(":blue[Project Information]")
    #     # st.header('💻  :blue[Platforms Used]')
    #     st.header('💻  Platforms Used')

    #     with st.container():
    #         # st.subheader('💻  Platforms Used')
    #         col1, col2, col3 = st.columns([1, 1, 1])
    #         with col1:
    #             show_logo('python', width=120, height=70)
    #         with col2:
    #             show_logo('pandas', width=120, height=70)
    #         with col3:
    #             show_logo('r_data', width=120, height=80)
    #         with col1:
    #             show_logo('bash', width=120, height=55)
    #         with col2:
    #             show_logo('numpy', width=120, height=80)
    #         with col3:
    #             show_logo('elastic_search', width=120, height=60)
    #         with col1:
    #             show_logo('rest_api', width=120, height=60)
            


    #     with st.container():
    #         st.markdown('<BR>', unsafe_allow_html=True)
    #         # st.header('⚒️ :blue[Skills]')
    #         st.header('⚒️ Skills')

    #         md_domain = f"""
    #         ##### <font color={subheading_blue}>Domain Research</font>
    #         A multitude of economic and demographic terms were learned in order to explain to users some of the more complex signals that were developed.  Understanding how this data was tracked and released was also necessary to accurately devise information-rich signals which would offer significant value to clients.
    #         """

    #         md_gather = f"""
    #         ##### <font color={subheading_blue}>Data Gathering</font>
    #         A dizzying array of geo-based data sources were used and their cadence of ingestion depended on how often they were released, as some were weekly, others monthly, and still others quarterly.  This was not in the scope of Jonpaul's role.
    #         """

    #         md_storage = f"""
    #         ##### <font color={subheading_blue}>Data Storage</font>
    #         Data was stored in ElasticSearch.  This was not in the scope of Jonpaul's role.
    #         """

    #         md_clean = f"""
    #         ##### <font color={subheading_blue}>Data Cleaning & Preparation</font>
    #         Most input data was from published sources and tended to be relatively clean. Jonpaul ported data cleaning and processing scripts from R to Python as the Signals Repository was a Python-based pipeline.
    #         """

    #         md_eng = f"""
    #         ##### <font color={subheading_blue}>Feature Engineering</font>
    #         Jonpaul authored scripts which consisted of basic (raw) data and more complex, customized signals.  Key signals were engineered by allowing a user to define a specific area (down to the block of a street) for which they wanted to see any number of changes in interactions over time.  Multiple raw signals were used to compmrise information-rich and model-ready signals that clients used in their own data science pipelines.  
    #         """

    #         md_model = f"""
    #         ##### <font color={subheading_blue}>Model Building</font>
    #         This is a signals repository and did not contain models.
    #         """

    #         md_res = f"""
    #         ##### <font color={subheading_blue}>Threshold Selection and Result Visualization</font>
    #         Demo visualizations displayed the concepts of key signals but the goal of this repository was to provide clients with data to enhance their own modeling endeavors.
    #         """



    #         st.markdown(md_domain, unsafe_allow_html=True)
    #         st.markdown(md_gather, unsafe_allow_html=True)
    #         st.markdown(md_storage, unsafe_allow_html=True)
    #         st.markdown(md_clean, unsafe_allow_html=True)
    #         st.markdown(md_eng, unsafe_allow_html=True)
    #         st.markdown(md_model, unsafe_allow_html=True)
    #         st.markdown(md_res, unsafe_allow_html=True)


    # def conclusion(self):
    #     st.markdown('<BR>', unsafe_allow_html=True)
    #     with st.container():
    #         st.header('✅ Conclusion')

    #         md_conc = """
    #         The Signals Repository was (is) highly successful and has clients across Europe in addition to America, now.  At bi-annual events hosted by KPMG, clients have given presentations on the ways in which the Signals Repository has bolstered their own modeling efforts.  Overall, the platform matured to the point of being a notable revenue generator for KPMG due to the real and accessible value it added to clients' data science workflows.
    #         """
    #         st.markdown(md_conc)


    # def gallery(self):
    #     st.markdown("<BR>", unsafe_allow_html=True)
    #     with st.container():
    #         st.header('🖼 Gallery')
    # 
    #         col1, col2 = st.columns([1, 1])
    #         with col1:
    #             show_img(signals['us_geo_growth'], width=450, height=450, hover='', caption='Example of the type of geo-based activity that signals offered.')
    #         with col2:
    #             show_img(signals['kpmg_signals'], width=450, height=450, hover='', caption='')


if __name__ == '__main__':
    pass
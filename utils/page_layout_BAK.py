import streamlit as st
import altair as alt
from pandas import DataFrame
from utils.streamlit_utilities import gradient, local_css
from utils.palettes import *
from utils.streamlit_data_processing import DataPrepper
import utils.styler as stylr
import utils.plotter as pltr



class PageLayout(DataPrepper):
    """Layout class for all pages
    """
    def __init__(self, page: str, year: int):
        super().__init__()
        # st.set_page_config(page_title="NFL Picks Pool", layout="wide", page_icon='🏈', initial_sidebar_state="expanded")
        local_css("style/style.css")
        self.year = year
        self.names = bg_clr_dct.keys()
        self.names_regex = "|".join(self.names)
        
        # st.sidebar.markdown(info['Photo'], unsafe_allow_html=True)
        if page == 'season':
            # st.table()
            self.intro()
            # self.WoW_metrics()
            # self.manager_ranking()
            self.manager_wins_by_round_NEW_STYLE()
            self.manager_wins_by_round()
            self.manager_projwins_by_round()
            st.markdown("***")
            self.draft_overview_chart()
            self.best_worst_picks()
            self.wins_by_round()
            # self.project_info()
            # self.conclusion()
            # self.gallery()

        elif page == 'playoffs':
            pass
        elif page == 'champs':
            pass
        elif page == 'career':
            pass
        elif page == 'records':
            pass


    def frmt_cols(self):
        frmts = {k: '{:.0f}' for k in self.int_cols}
        frmts.update({k: '{:.1f}' for k in self.float_cols})
        return frmts

    def intro(self):
        gradient(blue_bath1[1], blue_bath1[3], blue_bath1[5], '#fcfbfb', f"🏈 NFL Picks Pool", "A Swimming Pool of Interceptions", 27)

        st.markdown(f"""<h4 align=center>Global Pool from Colorado (), Texas (Howdy!), California (Like, HeLloOo!), England ('allo!), Japan (konnichiwa!), Germany (guten tag!), and Sweden (Allo!)</h4><BR>""", unsafe_allow_html=True)
        st.markdown(f"<h3 align=center>{self.year} Review</h3>", unsafe_allow_html=True)
        
    def WoW_metrics(self):
        """
        """
        st.write("""<BR><h4 align=center>Week over Week Changes 📈</h4>""", unsafe_allow_html=True)

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

    def manager_ranking(self):
        """
        """
        st.write("""<BR><h4 align=center>Manager Ranking #️⃣1️⃣</h4>""", unsafe_allow_html=True)
        bold = 'Total Win' if 'Total Win' in self.dfy_.columns else 'Win'


        with st.container():
            _, col2, _ = st.columns([.05, .9, .05])
            with col2:
                st.dataframe(stylr.style_frame(self.dfy_, bg_clr_dct=bg_clr_dct, frmt_dct=self.frmt_cols(), bold_cols=[bold], kind='streamlit'), use_container_width=True, hide_index=True)
                # st.dataframe(stylr.style_frame(self.dfy_, bg_clr_dct, frmt_dct={}, bold_cols=[bold]), use_container_width=True, hide_index=True)
                
            ## using st.table() honors intracellular formatting...  I actually like this more, but it's hard to center (yes..) and isn't sortable AND can't get rid of the damned index yet
            # with col2:
            #     frame = stylr.style_frame(self.dfy_.set_index('Rank').drop('Total Win%' if self.dfy_['Total Win%'].sum() == 0 else '', errors='ignore', axis=1), bg_clr_dct=bg_clr_dct, frmt_dct={'Win%': '{:.1f}', 'Full_Ssn_Pace': '{:.1f}', col: '{:.1f}'}, bold_cols=['Win'])
            #     st.table(frame)

    def manager_wins_by_round(self):
        """
        """
        st.write("""<BR><h4 align=center>Manager Wins by Round</h4>""", unsafe_allow_html=True)
        with st.container():
            _, col2, _ = st.columns([.05, .9, .05])
            with col2:
                st.dataframe(stylr.style_frame(self.dfpt, bg_clr_dct=bg_clr_dct, kind='streamlit'), hide_index=True, use_container_width=True)
    
    def manager_wins_by_round_NEW_STYLE(self):
        """
        """
        st.write("""<BR><h4 align=center>Manager Wins by Round NEW</h4>""", unsafe_allow_html=True)
        with st.container():
            _, col2, _ = st.columns([.05, .9, .05])
            with col2:
                tmp = self.dfpt.copy()
                tmp.columns = tmp.iloc[0]
                # st.dataframe(stylr.style_frame(tmp, bg_clr_dct=bg_clr_dct, kind='streamlit'), hide_index=True, use_container_width=True)
                
                # st.table(stylr.style_frame(tmp, bg_clr_dct=bg_clr_dct, kind='streamlit'))
                # st.dataframe(stylr.style_frame(self.dfpt, bg_clr_dct=bg_clr_dct, kind='streamlit'), hide_index=True, use_container_width=True)

                
# <th id="T_0629a_level0_col1" class="col_heading level0 col1">Alex (1)</th>
# class="col_heading level0 col1"
#         #T_0629a_level0_col1
        # tmp.columns = tmp.columns.map(lambda x: f'<th id="{x}" class="col_heading level0 col1">{x}</th>')
        import re
        
        def get_name(col):
            return re.search(self.names_regex, str(col)).group() if re.search(self.names_regex, str(col)) else 'null'


        def selector_style(col):
            return {'selector': f'th.col_heading.level0.col{tmp.columns.get_loc(col)}',
                'props': [
                    ('background-color', bg_clr_dct.get(get_name(col), 'white')),
                    ('color', 'black'),
                    ('font-family', 'Arial, sans-serif'),
                    ('font-size', '16px')
                    ]
            }

        tmp.drop(0, inplace=True)
        # st.write(tmp.set_index('Round'))
        # st.write(tmp.set_index('Round').T)
        # tmp = tmp.set_index('Round')
        # header_styles = [selector_style(col) for col in tmp.columns]

        def style_dataframe(df): 
            res = df.set_index('Round').style.set_table_styles(
                [selector_style(col) for col in tmp.columns]
            # res = df.reset_index(drop=True).style.set_table_styles(
            #     [selector_style(col) for col in tmp.columns]

                # [
                #     {
                #     'selector': 'th.col_heading.level0.col1',
                #     'props': [
                #         ('background-color', '#4CAF50'),
                #         ('color', 'white'),
                #         ('font-family', 'Arial, sans-serif'),
                #         ('font-size', '16px')
                #     ]
                # }, 
                # {
                #     'selector': 'td, th',
                #     'props': [
                #         ('border', '2px solid #4CAF50')
                #     ]
                # }
                # ]
            )
            return res
        
        import pandas as pd

        styled_df = style_dataframe(tmp)
        # st.write(type(styled_df))
        # st.write(pd.DataFrame.from(styled_df))

        st.table(styled_df)
        # st.write(styled_df, hide_index=True, unsafe_allow_html=True)
        # st.write(styled_df, hide_index=True)
        st.write(styled_df.to_html(), unsafe_allow_html=True, hide_index=True)
        # st.dataframe(pd.read_html(styled_df.to_html()), hide_index=True)




    def manager_projwins_by_round(self):
        """
        """
        st.write("""<BR><h4 align=center>Manager Full Season Projected Wins by Round</h4>""", unsafe_allow_html=True)
        st.write("""<p align=center>Projections from regression model.</p>""", unsafe_allow_html=True)
        with st.container():
            _, col2, _ = st.columns([.05, .9, .05])
            with col2:
                st.dataframe(self.dfproj, hide_index=True, use_container_width=True)
                # st.dataframe(stylr.style_frame(self.dfproj, bg_clr_dct=bg_clr_dct, kind='streamlit'), hide_index=True, use_container_width=True)

    def playoff_teams(self):
        """
        """
        if not self.dfpo.empty:
            st.write("""<BR><h4 align=center>Playoff Teams Tracker</h4>""", unsafe_allow_html=True)
            st.dataframe(stylr.style_frame(self.dfpo, bg_clr_dct=bg_clr_dct, frmt_dct={'Playoff_Win': '{:.0f}', 'Playoff_Loss': '{:.0f}'}, kind='streamlit'), width=765, height=620)

    def draft_overview_chart(self):
        """
        """
        st.write(f"""<BR><h4 align=center>The {self.curr_year} Draft</h4>""", unsafe_allow_html=True)
        st.write("""<p align=center>TIP: Click any player's dot to see only their picks. Shift-Click dots to add more players; double-click to reset.</p>""", unsafe_allow_html=True)
        # self.df['Total_Win'] = np.random.randint(1,18, size=self.df.shape[0])  ## testing for chart
        pltr.plot_draft_overview_altair(self.df, year_range=[self.curr_year])

    def best_worst_picks(self):
        """
        """
        st.write('  #')
        with st.container():
            left_col, right_col = st.columns([1, 1])
            with left_col:
                st.write("""<h4 align=center>✅ Best picks by round:</h4>""", unsafe_allow_html=True)
            
            with right_col:
                st.write("""<h4 align=center>❌ Worst picks by round:</h4>""", unsafe_allow_html=True)

        for rd in range(1, 5):
            with st.container():
                left_col, right_col = st.columns([1, 1])
                with left_col:
                    # st.write("""**Best picks by round:**""")
                    self.display_picks_by_round(self.dfd, 'Best', rd)
                
                with right_col:
                    # st.write("""**Worst picks by round:**""")
                    self.display_picks_by_round(self.dfd, 'Worst', rd)

    def wins_by_round(self):
        st.write(""" # """)
        st.write(f"""<BR><h5 align=center>Best Draft Rounds</h5>""", unsafe_allow_html=True)
        st.write("""<div align=center>Did we use our early draft picks wisely (does Round 1 have a higher win% than Round 2, etc.)?</div>""", unsafe_allow_html=True)
        
        drop_cols = ['Rank', 'Year'] if self.dfr_['Playoff Teams'].sum() > 0 else ['Rank', 'Year', 'Playoff Teams']
        frame = self.dfr_.drop(drop_cols, axis=1)[['Round', 'Win%', 'Win', 'Loss', 'Tie', 'Games']]
        with st.container():
            col1, col2 = st.columns([.45, 1])
            with col2:
                st.dataframe(stylr.style_frame(frame, bg_clr_dct=bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win%'], kind='streamlit'))
            
    def top_10_reg_ssn(self):
        st.write(f"""
            Let's take a look at the top 10 Regular Season finishes.
            """)
        st.dataframe(stylr.style_frame(self.hist_frames[0].sort_values(['Win%_Rk', 'Win', 'Year'], ascending=[True, False, True]), bg_clr_dct=bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win%'], kind='streamlit'), width=620, height=550)
        
    def top_10_playoffs(self):
        # st.write(body_dct['pohist_txt'])
        st.write(f"""
        How about the top 10 Playoff runs?
        """)

        st.dataframe(stylr.style_frame(self.hist_frames[1].sort_values(['Win_Rk', 'Win%', 'Year'], ascending=[True, False, True]), bg_clr_dct=bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win']), width=620, height=550)
        
    def top_10_total_wins(self):
        # st.write(body_dct['tothist_txt'])
        st.write(f"""And what about the top 10 regular season and playoffs combined (for a single season) -- i.e. a player's total wins? 
            """)
        
        st.dataframe(stylr.style_frame(self.hist_frames[2].sort_values(['Win%_Rk', 'Win%', 'Year'], ascending=[True, False, True]), bg_clr_dct=bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win']), width=620, height=550)        
            
    def champions(self):
        st.write("""#### Champions""")
        st.write("""Past champions and their results, as well as projected champion for the current year (highlighted in blue).
            """)
        st.write("""I'm not sure how to parse the added week 18 in the regular season except to use win percent as opposed to wins.  
        """)
        
        st.dataframe(stylr.style_frame(self.champs, bg_clr_dct=bg_clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=self.curr_year, bold_cols=['Total_Win'], kind='streamlit'))  
        
    def careers(self):
        st.write("""#""")
        st.write("""#### Career Performance""")
        st.write("Who in our pool has been the best over their careers (sorted by Wins)?")
        
        st.dataframe(stylr.style_frame(self.dfc_, bg_clr_dct=bg_clr_dct, frmt_dct={'Total Win%': '{:.1f}'}, kind='streamlit'))
        
        st.write("""...Victoria hasn't even won as many games as the Leftovers.  Sad! 😜""")
        
        st.write("""#""")
        # dfs_ = player_hist.sort_values('Year', ascending=True).groupby(['Player', 'Year']).sum().groupby('Player').cumsum().reset_index().sort_values(['Player', 'Year'])
        # dfs_
        if 'Champ' in self.player_hist.columns:
            player_hist = self.player_hist 
        else:
            player_hist = self.player_hist.merge(self.champs.assign(Champ=True)[['Player', 'Year', 'Champ']], on=['Player', 'Year'], how='left').fillna(False)
        # player_hist = player_hist.merge(champs.assign(Champ='Yes')[['Player', 'Year', 'Champ']], on=['Player', 'Year'], how='left').fillna('No')
        # player_hist.loc[(player_hist['Year']==curr_year) & (player_hist['Champ']=='Yes'), 'Champ'] = 'Proj'
        # player_hist.tail(20)
        
        pltr.plot_wide_wins_by_year_bar_chart()
    
    def sweet_ridge_plot(self):
        # source = data.seattle_weather.url
        source = self.dfy
        step = 30
        overlap = 1
        # st.write(source.head(100))
    
        ridge = alt.Chart(source, height=step).transform_joinaggregate(
            mean_wins='mean(Total_Win)', groupby=['Player']
        ).transform_bin(
            ['bin_max', 'bin_min'], 'Total_Win'
        ).transform_aggregate(
            value='count()', groupby=['Player', 'mean_wins', 'bin_min', 'bin_max']
        ).transform_impute(
            impute='value', groupby=['Player', 'mean_wins'], key='bin_min', value=0
        ).mark_area(
            interpolate='monotone',
            fillOpacity=0.8,
            stroke='lightgray',
            strokeWidth=0.5
        ).encode(
            alt.X('bin_min:Q', bin='binned', title='Total Wins'),
            alt.Y(
                'value:Q',
                scale=alt.Scale(range=[step, -step * overlap]),
                axis=None
            ),
            alt.Fill(
                'mean_wins:Q',
                legend=None,
                scale=alt.Scale(domain=[source['Total_Win'].max(), source['Total_Win'].min()], scheme='redyellowblue')
            )
        ).facet(
            row=alt.Row(
                'Player:N',
                title=None,
                header=alt.Header(labelAngle=0, labelAlign='left')
            )
        ).properties(
            title='Win History by Player',
            bounds='flush'
        ).configure_facet(
            spacing=0
        ).configure_view(
            stroke=None
        ).configure_title(
            anchor='end'
        )
        st.altair_chart(ridge)
    
    def personal_records(self):
        st.write("""#""")
        st.write("""#### Personal Records""")    
        # st.write(body_dct['pr_txt'])
        st.write("""Last, here are the personal records for each player, sorted by most at top.  \nBlue highlight is for this season and shows who might have a chance at setting a new personal record for total wins.""")
        
        
        # print(self.player_hist)

        left_column, right_column = st.columns([2, 1])
        
        # # st.write(self.player_hist)
        # with left_column:
        #     for name in self.dfy_['Player'].unique():
        #         self.show_player_hist_table(name)
        #         st.write("\n\n\n _")

        # with right_column: 
        #     for name in self.dfy_['Player'].unique():
        #         self.plot_wins_by_year(self.player_hist[self.player_hist['Player'] == name])


        for name in self.dfy_['Player'].unique():
            with left_column:
                self.show_player_hist_table(name)
            with right_column: 
                self.plot_wins_by_year(self.player_hist[self.player_hist['Player'] == name])
                st.write("\n\n\n _")


    def display_picks_by_round(self, frame: DataFrame, best_worst: str, rd: int): 
        """
        """
        # idx_max = frame.groupby('Round')['Total_Win'].transform('max') == frame['Total_Win']
        # idx_min = frame.groupby('Round')['Total_Win'].transform('min') == frame['Total_Win']
        
        
        
        # components.html(f'<div style="text-align: center"> Round {rd} </div>')
        st.write(f""" <div align=center>Round {rd}</div>""", unsafe_allow_html=True)
        # idx = idx_max if best_worst.lower() == 'best' else idx_min
        max_min = 'max' if best_worst.lower() == 'best' else 'min'
        idx = frame.groupby('Round')['Total_Win'].transform(max_min) == frame['Total_Win']
        
        st.dataframe(stylr.style_frame(frame[(idx) & (frame['Round']==rd)], bg_clr_dct=bg_clr_dct, frmt_dct={'Total_Win': '{:.0f}'}, kind='streamlit'), width=495, hide_index=True)
        # st.dataframe(stylr.style_frame(frame[idx].query("""Round==@rd"""), bg_clr_dct, frmt_dct={'Total_Win': '{:.0f}'}), width=495)
    
        # for rd_res in [(rd, best_worst) for rd in range(1,5)]:
        #     rd, res = rd_res[0], rd_res[1]
        #     # components.html(f'<div style="text-align: center"> Round {rd} </div>')
        #     st.write(f""" Round {rd}""")
        #     idx = idx_max if res == 'Best' else idx_min
        #     st.dataframe(stylr.style_frame(frame[idx].query("""Round==@rd"""), bg_clr_dct, frmt_dct={'Total_Win': '{:.0f}'}), width=495)






if __name__ == '__main__':
    pass
import streamlit as st
from pandas import DataFrame
from utils.streamlit_utilities import gradient, local_css, frmt_cols
from utils.palettes import bg_clr_dct, blue_bath1
from utils.streamlit_data_processing import DataProcessor
from utils.constants import CURR_SEASON, CURR_WEEK
from utils.utilities import enforce_int_cols, enforce_float_cols
import utils.styler as stylr
import utils.plotter as pltr



class PageLayoutSeason(DataProcessor):
    """Layout class 
    """
    def __init__(self, year: int):
        super().__init__(year)
        st.set_page_config(page_title="NFL Picks Pool", layout="wide", page_icon='🏈', initial_sidebar_state="expanded")
        # st.set_page_config(page_title="NFL Picks Pool", layout="wide", page_icon='🏈')
        # local_css("style/style.css")
        # st.sidebar.markdown(info['Photo'], unsafe_allow_html=True)
        self.year = year
        self.names = bg_clr_dct.keys()
        self.names_regex = "|".join(self.names)
        
        self.frmts = {'Wins': '{:.0f}', 
                        'Win': '{:.0f}',
                        'Losses': '{:.0f}',
                        'Loss': '{:.0f}',
                        'Ties': '{:.0f}',
                        'Games': '{:.0f}',
                        'Win%': '{:.1f}',
                        'Games Left': '{:.0f}',
                        'Playoff Win': '{:.0f}',
                        'Playoff Loss': '{:.0f}',
                        f'Proj. Wins by Wk{CURR_WEEK}': '{:.1f}',
                        'Ssn Proj. Wins': '{:.1f}',
                        f'Wins over Wk{CURR_WEEK} Proj.': '{:.1f}',
                        'Wins over Ssn Proj.': '{:.1f}',
                        'Rank': '{:.0f}',
                        }
        
        self.intro()
        self.WoW_metrics()
        self.manager_ranking()
        self.manager_wins_by_round()
        self.manager_projwins_by_round()
        st.markdown("***")
        self.draft_overview_chart()
        self.best_worst_picks_tables()
        self.pool_wins_by_round()


    def intro(self):
        gradient(blue_bath1[1], blue_bath1[3], blue_bath1[5], '#fcfbfb', f"🏈 NFL Picks Pool", "A Swimming Pool of Interceptions", 27)

        # st.markdown(f"""<h4 align=center>Global Pool from <br>Colorado ("hey"), Texas (Howdy!), California (Like, HeLloOo!), England ('allo!), Japan (konnichiwa!), Germany (guten tag!), and Sweden (Allo!)</h4><BR>""", unsafe_allow_html=True)
        st.markdown(f"<h3 align=center>{self.year} Review</h3>", unsafe_allow_html=True)

    def WoW_metrics(self):
        """
        """
        ## Limited to current week of current season.
        if self.year == CURR_SEASON:  
            st.write("""<BR><h4 align=center>Week over Week Changes 📈</h4>""", unsafe_allow_html=True)

            def show_metric(frame, idx):
                if frame.iloc[idx]['WoW_Wins'] <= 1:
                    clr = 'inverse'
                elif frame.iloc[idx]['WoW_Wins'] < 3:
                    clr = 'off'
                else:
                    clr = 'normal'

                st.metric(f":blue[{idx+1}. {frame.iloc[idx]['Player']}]", f"{int(frame.iloc[idx]['Total_Wins'])} wins", f"{int(frame.iloc[idx]['WoW_Wins'])} this week", delta_color=clr)
                

            with st.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    show_metric(self.wow_curr, 0)
                with col2:
                    show_metric(self.wow_curr, 1)
                with col3:
                    show_metric(self.wow_curr, 2)
                with col4:
                    show_metric(self.wow_curr, 3)
                with col1:
                    show_metric(self.wow_curr, 4)
                with col2:
                    show_metric(self.wow_curr, 5)
                with col3:
                    show_metric(self.wow_curr, 6)
                with col4:
                    show_metric(self.wow_curr, 7)

    def manager_ranking(self):
        """
        """
        st.write("""<BR><h4 align=center>Manager Ranking #️⃣1️⃣</h4>""", unsafe_allow_html=True)

        
        def format_frame(frame):
            renames = {'Total_Win': 'Wins', 
                        'Total_Loss': 'Losses',
                        'Total_Games': 'Games',
                        'Total_Win%': 'Win%',
                        'Reg_Games_Left': 'Games Left',
                        'Total_Tie': 'Ties',
                        'Current_Proj_Wins': f'Proj. Wins by Wk{CURR_WEEK}',
                        'Wins_Over_Current_Proj': f'Wins over Wk{CURR_WEEK} Proj.',
                        'Full_Ssn_Proj_Wins': 'Ssn Proj. Wins',
                        'Wins_Over_Ssn_Proj': 'Wins over Ssn Proj.',
                        }

            cols = ['Rank', 'Year', 'Player'] + list(renames.values())
            if self.year == CURR_SEASON:
                if CURR_WEEK > 18:
                    cols += ['Playoff Teams', 'Playoff Wins', 'Playoff Losses']
                else:
                    cols.remove('Wins over Ssn Proj.')
            else:
                cols.remove(f'Wins over Wk{CURR_WEEK} Proj.')
                cols.remove(f'Proj. Wins by Wk{CURR_WEEK}')

            frame = frame.rename(columns=renames)[cols]
            if frame['Ties'].sum() == 0:
                frame = frame.drop('Ties', axis=1)
            return frame

        frame = format_frame(self.df_years_site)

        with st.container():
            _, col2 = st.columns([.05, 1])
            with col2:
                st.dataframe(stylr.style_frame(frame, 
                                                cell_clr_dct=bg_clr_dct, 
                                                frmt_dct=self.frmts, 
                                                kind='streamlit'),
                            hide_index=True)

    def manager_wins_by_round(self):
        """
        """
        st.write("""<BR><h4 align=center>Manager Wins by Round</h4>""", unsafe_allow_html=True)
        styled_df = stylr.create_styled_frame(self.df_mgr_tms.copy(), self.names_regex, bg_clr_dct)
        
        with st.container():
            _, col2, _ = st.columns([.05, .9, .05])
            with col2:
                st.write(styled_df.to_html(), unsafe_allow_html=True, hide_index=True)
                
    def manager_projwins_by_round(self):
        """
        """
        st.write("""<BR><h4 align=center>Manager Full-Season Projected Wins by Round</h4>""", unsafe_allow_html=True)
        st.write("""<p align=center>Projections from regression model.</p>""", unsafe_allow_html=True)
        styled_df = stylr.create_styled_frame(self.df_mgr_tms_proj, self.names_regex, bg_clr_dct)
        with st.container():
            _, col2, _ = st.columns([.05, .9, .05])
            with col2:
                st.write(styled_df.to_html(), unsafe_allow_html=True, hide_index=True)

    def playoff_teams(self):
        """
        """
        if not self.df_playoffs.empty:
            st.write("""<BR><h4 align=center>Playoff Teams Tracker</h4>""", unsafe_allow_html=True)
            st.dataframe(stylr.style_frame(self.df_playoffs, cell_clr_dct=bg_clr_dct, frmt_dct={'Playoff_Win': '{:.0f}', 'Playoff_Loss': '{:.0f}'}, kind='streamlit'), width=765, height=620)

    def draft_overview_chart(self):
        """
        """
        st.write(f"""<BR><h4 align=center>The {self.year} Draft</h4>""", unsafe_allow_html=True)
        st.write("""<p align=center>Click any player's dot to see only their picks.  &nbsp &nbsp Shift-Click dots to add more players. &nbsp &nbsp Click blank space to reset.</p>""", unsafe_allow_html=True)
        # self.df['Total_Win'] = np.random.randint(1,18, size=self.df.shape[0])  ## testing for chart
        pltr.plot_draft_overview_altair(self.df, year_range=[self.year])

    def best_worst_picks_tables(self):
        """
        """

        def picks_by_round_table(frame: DataFrame, best_worst: str, rd: int): 
            # idx_max = frame.groupby('Round')['Total_Win'].transform('max') == frame['Total_Win']
            # idx_min = frame.groupby('Round')['Total_Win'].transform('min') == frame['Total_Win']
            
            st.write(f""" <div align=center>Round {rd}</div>""", unsafe_allow_html=True)
            max_min = 'max' if best_worst.lower() == 'best' else 'min'
            idx = frame.groupby('Round')['Total_Win'].transform(max_min) == frame['Total_Win']
            
            st.dataframe(stylr.style_frame(frame[(idx) & (frame['Round']==rd)]\
                                        .rename(columns={'Total_Win': 'Wins', 
                                                            'Full_Ssn_Proj_Wins': 'Ssn Proj. Wins'}), 
                                        cell_clr_dct=bg_clr_dct, 
                                        frmt_dct={'Wins': '{:.0f}', 'Ssn Proj. Wins': '{:.0f}'}, 
                                        kind='streamlit'),
                            width=495, 
                            hide_index=True)
            

        st.write('  #')
        with st.container():
            left_col, right_col = st.columns([1, 1])
            with left_col:
                st.write("""<h4 align=center>✅ Best picks by round:</h4>""", unsafe_allow_html=True)
            
            with right_col:
                st.write("""<h4 align=center>❌ Worst picks by round:</h4>""", unsafe_allow_html=True)

        for rd in range(1, 5):
            # st.write(f""" <h5 align=center> Round {rd} </h5>""", unsafe_allow_html=True)
            # st.markdown(f""" *** """)
            with st.container():
                left_col, right_col = st.columns([1, 1])
                with left_col:
                    # st.write("""**Best picks by round:**""")
                    picks_by_round_table(self.df_best_worst_rd, 'Best', rd)
                
                with right_col:
                    # st.write("""**Worst picks by round:**""")
                    picks_by_round_table(self.df_best_worst_rd, 'Worst', rd)

    def pool_wins_by_round(self):
        st.write(""" # """)
        st.write(f"""<BR><h5 align=center>Best Draft Rounds</h5>""", unsafe_allow_html=True)
        st.write("""<div align=center>Did we use our early draft picks wisely (does Round 1 have a higher win% than Round 2, etc.)?</div>""", unsafe_allow_html=True)
        # st.write("""<div align=center>Click on the legend to hide/show rounds.</div>""", unsafe_allow_html=True)
        if self.year <= 2021:
            st.write("""<div align=center>Round 99 = the four "Leftover" teams we didn't draft.</div>""", unsafe_allow_html=True)
        

        ## Unsure why but the formatting output on this table is just very stubborn. Taken a lot more nudging to get it to look right.
        frame = self.df_pool_rounds_site
        cols = ['Round', 'Total_Win%', 'Total_Win', 'Total_Loss', 'Total_Tie', 'Total_Games']

        if frame['Playoff_Teams'].sum() > 0:
            cols += ['Playoff_Teams', 'Playoff_Win', 'Playoff_Loss']
        
        if frame['Total_Tie'].sum() == 0:
            cols.remove('Total_Tie')
        
        frame = enforce_int_cols(frame)
        frame = enforce_float_cols(frame)
        frame = frame[cols]\
                .assign(Rank=lambda f: f['Total_Win%'].astype(float).rank(method='dense', ascending=False))\
                .sort_values('Rank')\
                .set_index('Rank')\
                .reset_index()
        frame.columns = frame.columns\
                        .str.replace('Total', '', regex=True)\
                        .str.replace('_', ' ', regex=True)\
                        .str.strip()
    
        with st.container():
            _, col2 = st.columns([.35, 1])
            with col2:
                st.dataframe(stylr.style_frame(frame, cell_clr_dct=bg_clr_dct, frmt_dct=self.frmts, kind='streamlit'), hide_index=True)
                  
    def top_10_reg_ssn(self):
        st.write(f"""
            Let's take a look at the top 10 Regular Season finishes.
            """)
        st.dataframe(stylr.style_frame(self.df_top10[0].sort_values(['Win%_Rk', 'Win', 'Year'], ascending=[True, False, True]), cell_clr_dct=bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win%'], kind='streamlit'), width=620, height=550)
        
    def top_10_playoffs(self):
        # st.write(body_dct['pohist_txt'])
        st.write(f"""
        How about the top 10 Playoff runs?
        """)

        st.dataframe(stylr.style_frame(self.df_top10[1].sort_values(['Win_Rk', 'Win%', 'Year'], ascending=[True, False, True]), cell_clr_dct=bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win']), width=620, height=550)
        
    def top_10_total_wins(self):
        # st.write(body_dct['tothist_txt'])
        st.write(f"""And what about the top 10 regular season and playoffs combined (for a single season) -- i.e. a player's total wins? 
            """)
        
        st.dataframe(stylr.style_frame(self.df_top10[2].sort_values(['Win%_Rk', 'Win%', 'Year'], ascending=[True, False, True]), cell_clr_dct=bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win']), width=620, height=550)        
            
    def champions(self):
        st.write("""#### Champions""")
        st.write("""Past champions and their results, as well as projected champion for the current year (highlighted in blue).
            """)
        st.write("""I'm not sure how to parse the added week 18 in the regular season except to use win percent as opposed to wins.  
        """)
        
        st.dataframe(stylr.style_frame(self.champs, cell_clr_dct=bg_clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=self.curr_year, bold_cols=['Total_Win'], kind='streamlit'))  
        
    def careers(self):
        st.write("""#""")
        st.write("""#### Career Performance""")
        st.write("Who in our pool has been the best over their careers (sorted by Wins)?")
        
        st.dataframe(stylr.style_frame(self.df_careers, cell_clr_dct=bg_clr_dct, frmt_dct={'Total Win%': '{:.1f}'}, kind='streamlit'))
        
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
    
    def personal_records(self):
        st.write("""#""")
        st.write("""#### Personal Records""")    
        # st.write(body_dct['pr_txt'])
        st.write("""Last, here are the personal records for each player, sorted by most at top.  \nBlue highlight is for this season and shows who might have a chance at setting a new personal record for total wins.""")
        
        
        # print(self.player_hist)

        left_column, right_column = st.columns([2, 1])
        
        # # st.write(self.player_hist)
        # with left_column:
        #     for name in self.df_years_site['Player'].unique():
        #         self.show_player_hist_table(name)
        #         st.write("\n\n\n _")

        # with right_column: 
        #     for name in self.df_years_site['Player'].unique():
        #         self.plot_wins_by_year(self.player_hist[self.player_hist['Player'] == name])


        for name in self.df_years_site['Player'].unique():
            with left_column:
                self.show_player_hist_table(name)
            with right_column: 
                self.plot_wins_by_year(self.player_hist[self.player_hist['Player'] == name])
                st.write("\n\n\n _")





if __name__ == '__main__':
    pass
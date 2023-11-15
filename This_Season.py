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
        
        # st.table()
        self.intro()
        self.WoW_metrics()
        self.manager_ranking()
        self.manager_by_round()
        st.markdown("***")
        self.draft_overview_chart()
        self.best_worst_picks()
        self.wins_by_round()
        # self.project_info()
        # self.conclusion()
        # self.gallery()




    def intro(self):
        gradient(blue_bath1[1], blue_bath1[3], blue_bath1[5], '#fcfbfb', f"🏈 NFL Picks Pool", "A Swimming Pool of Interceptions", 27)

        st.markdown(f"""<h6 align=center>We've gone global: Colorado, Texas, California, England, Japan, and Sweden</h6><BR>""", unsafe_allow_html=True)
        st.markdown(f"<h3 align=center>Standings {DP.po_inc} as of Week {int(DP.the_wk)} - {DP.the_ssn}!</h3>", unsafe_allow_html=True)
        
    def manager_ranking(self):
        """
        """
        st.write("""<BR><h6 align=center>Manager Ranking #️⃣1️⃣</h6>""", unsafe_allow_html=True)
        col = 'Total Win%' if 'Total Win%' in DP.dfy_ else 'Win%'
        with st.container():
            _, col2, _ = st.columns([.05, .9, .05])
            with col2:
                # st.dataframe(DP.style_frame(DP.dfy_.drop('Total Win%' if DP.dfy_['Total Win%'].sum() == 0 else '', errors='ignore', axis=1), DP.bg_clr_dct, frmt_dct={'Win%': '{:.1f}', 'Full_Ssn_Pace': '{:.1f}', col: '{:.1f}', 'bold_cols':['Win']}), width=900, hide_index=True)
                st.dataframe(DP.style_frame(DP.dfy_.drop('Total Win%' if DP.dfy_['Total Win%'].sum() == 0 else '', errors='ignore', axis=1), DP.bg_clr_dct, frmt_dct={'Win%': '{:.1f}', 'Full_Ssn_Pace': '{:.1f}', col: '{:.1f}'}, bold_cols=['Win']), use_container_width=True, hide_index=True)
            
            ## using st.table() honors intracellular formatting...
            # with col2:
            #     frame = DP.style_frame(DP.dfy_.set_index('Rank').drop('Total Win%' if DP.dfy_['Total Win%'].sum() == 0 else '', errors='ignore', axis=1), DP.bg_clr_dct, frmt_dct={'Win%': '{:.1f}', 'Full_Ssn_Pace': '{:.1f}', col: '{:.1f}'}, bold_cols=['Win'])

            #     st.table(frame)

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

            # plot_bg_clr_dct

            st.metric(f":blue[{idx+1}. {frame.iloc[idx]['Player']}]", f"{int(frame.iloc[idx]['Total_Win'])} wins", f"{int(frame.iloc[idx]['WoW_Wins'])} last week", delta_color=clr)
            

        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                show_metric(DP.wow, 0)
            with col2:
                show_metric(DP.wow, 1)
            with col3:
                show_metric(DP.wow, 2)
            with col4:
                show_metric(DP.wow, 3)
            with col1:
                show_metric(DP.wow, 4)
            with col2:
                show_metric(DP.wow, 5)
            with col3:
                show_metric(DP.wow, 6)
            with col4:
                show_metric(DP.wow, 7)




    def manager_by_round(self):
        """
        """
        st.write("""<BR><h6 align=center>Manager by Round (wins) </h6>""", unsafe_allow_html=True)
        with st.container():
            _, col2, _ = st.columns([.05, .9, .05])
            with col2:
                st.dataframe(DP.style_frame(DP.dfpt, DP.bg_clr_dct), hide_index=True, use_container_width=True)
        # st.dataframe(DP.style_frame(DP.dfpt, DP.bg_clr_dct), hide_index=True, use_container_width=True)
        # frame = DP.dfpt
        # frame.columns = frame.iloc[0]
        # frame = frame.drop(0, axis='index')
        # st.table(DP.style_frame(frame, DP.bg_clr_dct))

    def playoff_teams(self):
        """
        """
        if not DP.dfpo.empty:
            st.write("""<BR><h6 align=center>Playoff Teams Tracker</h6>""", unsafe_allow_html=True)
            st.dataframe(DP.style_frame(DP.dfpo, DP.bg_clr_dct, frmt_dct={'Playoff_Win': '{:.0f}', 'Playoff_Loss': '{:.0f}'}), width=765, height=620)

    def draft_overview_chart(self):
        """
        """
        st.write(f"""<BR><h4 align=center>The {DP.curr_year} Draft</h4>""", unsafe_allow_html=True)
        st.write("""<p align=center>TIP: Click any player's dot to see only their picks. Shift-Click dots to add more players; double-click to reset.</p>""", unsafe_allow_html=True)
        # DP.df['Total_Win'] = np.random.randint(1,18, size=DP.df.shape[0])  ## testing for chart
        DP.plot_draft_overview_altair(DP.df, year_range=[DP.curr_year])

    def best_worst_picks(self):
        """
        """
        st.write('  #')
        with st.container():
            left_col, right_col = st.columns([1, 1])
            with left_col:
                st.write("""<h6 align=center>✅ Best picks by round:</h6>""", unsafe_allow_html=True)
            
            with right_col:
                st.write("""<h6 align=center>❌ Worst picks by round:</h6>""", unsafe_allow_html=True)

        for rd in range(1, 5):
            with st.container():
                left_col, right_col = st.columns([1, 1])
                with left_col:
                    # st.write("""**Best picks by round:**""")
                    DP.picks_by_round(DP.dfd, 'Best', rd)
                
                with right_col:
                    # st.write("""**Worst picks by round:**""")
                    DP.picks_by_round(DP.dfd, 'Worst', rd)

    def wins_by_round(self):
        st.write(""" # """)
        st.write(f"""<BR><h5 align=center>Best Draft Rounds</h5>""", unsafe_allow_html=True)
        st.write("""<div align=center>Did we use our early draft picks wisely (does Round 1 have a higher win% than Round 2, etc.)?</div>""", unsafe_allow_html=True)
        
        drop_cols = ['Rank', 'Year'] if DP.dfr_['Playoff Teams'].sum() > 0 else ['Rank', 'Year', 'Playoff Teams']
        frame = DP.dfr_.drop(drop_cols, axis=1)[['Round', 'Win%', 'Win', 'Loss', 'Tie', 'Games']]
        with st.container():
            col1, col2 = st.columns([.45, 1])
            with col2:
                st.dataframe(DP.style_frame(frame, DP.bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win%']))
            

    def top_10_reg_ssn(self):
        st.write(f"""
            Let's take a look at the top 10 Regular Season finishes.
            """)
        st.dataframe(DP.style_frame(DP.hist_frames[0].sort_values(['Win%_Rk', 'Win', 'Year'], ascending=[True, False, True]), DP.bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win%']), width=620, height=550)
        
    # def top_10_playoffs(self):
    #     # st.write(body_dct['pohist_txt'])
    #     st.write(f"""
    #     How about the top 10 Playoff runs?
    #     """)

    #     st.dataframe(DP.style_frame(DP.hist_frames[1].sort_values(['Win_Rk', 'Win%', 'Year'], ascending=[True, False, True]), DP.bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win']), width=620, height=550)
        
    # def top_10_total_wins(self):
    #     # st.write(body_dct['tothist_txt'])
    #     st.write(f"""And what about the top 10 regular season and playoffs combined (for a single season) -- i.e. a player's total wins? 
    #         """)
        
    #     st.dataframe(DP.style_frame(DP.hist_frames[2].sort_values(['Win%_Rk', 'Win%', 'Year'], ascending=[True, False, True]), DP.bg_clr_dct, frmt_dct={'Win%': '{:.1f}'}, bold_cols=['Win']), width=620, height=550)        
            
    def champions(self):
        st.write("""#### Champions""")
        st.write("""Past champions and their results, as well as projected champion for the current year (highlighted in blue).
            """)
        st.write("""I'm not sure how to parse the added week 18 in the regular season except to use win percent as opposed to wins.  
        """)
        
        st.dataframe(DP.style_frame(DP.champs, DP.bg_clr_dct, frmt_dct={'Total_Win%': '{:.1f}'}, clr_yr=DP.curr_year, bold_cols=['Total_Win']))  
        
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
    
    def sweet_ridge_plot(self):
        # source = data.seattle_weather.url
        source = DP.dfy
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
        
        
        # print(DP.player_hist)

        left_column, right_column = st.columns([2, 1])
        
        # # st.write(DP.player_hist)
        # with left_column:
        #     for name in DP.dfy_['Player'].unique():
        #         DP.show_player_hist_table(name)
        #         st.write("\n\n\n _")

        # with right_column: 
        #     for name in DP.dfy_['Player'].unique():
        #         DP.plot_wins_by_year(DP.player_hist[DP.player_hist['Player'] == name])


        for name in DP.dfy_['Player'].unique():
            with left_column:
                DP.show_player_hist_table(name)
            with right_column: 
                DP.plot_wins_by_year(DP.player_hist[DP.player_hist['Player'] == name])
                st.write("\n\n\n _")







    def project_info(self):
        st.markdown('<BR>', unsafe_allow_html=True)
        # st.header(":blue[Project Information]")
        # st.header('💻  :blue[Platforms Used]')
        st.header('💻  Platforms Used')

        with st.container():
            # st.subheader('💻  Platforms Used')
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                show_logo('python', width=120, height=70)
            with col2:
                show_logo('pandas', width=120, height=70)
            with col3:
                show_logo('r_data', width=120, height=80)
            with col1:
                show_logo('bash', width=120, height=55)
            with col2:
                show_logo('numpy', width=120, height=80)
            with col3:
                show_logo('elastic_search', width=120, height=60)
            with col1:
                show_logo('rest_api', width=120, height=60)
            


        with st.container():
            st.markdown('<BR>', unsafe_allow_html=True)
            # st.header('⚒️ :blue[Skills]')
            st.header('⚒️ Skills')

            md_domain = f"""
            ##### <font color={subheading_blue}>Domain Research</font>
            A multitude of economic and demographic terms were learned in order to explain to users some of the more complex signals that were developed.  Understanding how this data was tracked and released was also necessary to accurately devise information-rich signals which would offer significant value to clients.
            """

            md_gather = f"""
            ##### <font color={subheading_blue}>Data Gathering</font>
            A dizzying array of geo-based data sources were used and their cadence of ingestion depended on how often they were released, as some were weekly, others monthly, and still others quarterly.  This was not in the scope of Jonpaul's role.
            """

            md_storage = f"""
            ##### <font color={subheading_blue}>Data Storage</font>
            Data was stored in ElasticSearch.  This was not in the scope of Jonpaul's role.
            """

            md_clean = f"""
            ##### <font color={subheading_blue}>Data Cleaning & Preparation</font>
            Most input data was from published sources and tended to be relatively clean. Jonpaul ported data cleaning and processing scripts from R to Python as the Signals Repository was a Python-based pipeline.
            """

            md_eng = f"""
            ##### <font color={subheading_blue}>Feature Engineering</font>
            Jonpaul authored scripts which consisted of basic (raw) data and more complex, customized signals.  Key signals were engineered by allowing a user to define a specific area (down to the block of a street) for which they wanted to see any number of changes in interactions over time.  Multiple raw signals were used to compmrise information-rich and model-ready signals that clients used in their own data science pipelines.  
            """

            md_model = f"""
            ##### <font color={subheading_blue}>Model Building</font>
            This is a signals repository and did not contain models.
            """

            md_res = f"""
            ##### <font color={subheading_blue}>Threshold Selection and Result Visualization</font>
            Demo visualizations displayed the concepts of key signals but the goal of this repository was to provide clients with data to enhance their own modeling endeavors.
            """



            st.markdown(md_domain, unsafe_allow_html=True)
            st.markdown(md_gather, unsafe_allow_html=True)
            st.markdown(md_storage, unsafe_allow_html=True)
            st.markdown(md_clean, unsafe_allow_html=True)
            st.markdown(md_eng, unsafe_allow_html=True)
            st.markdown(md_model, unsafe_allow_html=True)
            st.markdown(md_res, unsafe_allow_html=True)


    def conclusion(self):
        st.markdown('<BR>', unsafe_allow_html=True)
        with st.container():
            st.header('✅ Conclusion')

            md_conc = """
            The Signals Repository was (is) highly successful and has clients across Europe in addition to America, now.  At bi-annual events hosted by KPMG, clients have given presentations on the ways in which the Signals Repository has bolstered their own modeling efforts.  Overall, the platform matured to the point of being a notable revenue generator for KPMG due to the real and accessible value it added to clients' data science workflows.
            """
            st.markdown(md_conc)


    def gallery(self):
        st.markdown("<BR>", unsafe_allow_html=True)
        with st.container():
            st.header('🖼 Gallery')

            col1, col2 = st.columns([1, 1])
            with col1:
                show_img(signals['us_geo_growth'], width=450, height=450, hover='', caption='Example of the type of geo-based activity that signals offered.')
            with col2:
                show_img(signals['kpmg_signals'], width=450, height=450, hover='', caption='')
           



if __name__ == '__main__':
    PageLayout()
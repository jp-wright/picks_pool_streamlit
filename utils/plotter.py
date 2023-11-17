"""
Plotting Funcs
"""
import streamlit as st
import altair as alt
from pandas import DataFrame, read_csv
from utils.palettes import *
from typing import List, Tuple, Dict, Sequence, Optional, Union
from utils.utilities import get_curr_year

# ssn = get_curr_year()
# games = read_csv('data/output/nfl_picks_pool_player_standings_history.csv', dtype={'Year': int})
# the_wk = int(games.loc[games['Year']==ssn, 'Reg_Games'].max())
# if the_wk > 12: 
#     the_wk += 1



def plot_draft_overview_altair(frame: DataFrame, year: int) -> None:
    """
    In theory can plot multiple years but I don't think it's good viz at all to do so.
    """
    the_wk = int(frame.loc[frame['Year']==year, 'Reg_Games'].max())
    if the_wk > 12: the_wk += 1
    source = frame[frame['Year']==year]

    points = alt.Chart()\
                .mark_point(strokeWidth=1, filled=True, stroke='black', size=185)\
                .encode(
                    alt.X('Pick:O', axis=alt.Axis(format='.0f', tickMinStep=1, labelAngle=0, labelFlush=True, grid=True)),
                    alt.Y('Total_Win:Q', scale=alt.Scale(zero=True), 
                            axis=alt.Axis(values=list(range(the_wk + 1)))),
                    tooltip="Player:N"
                    )

    text_wins = points.mark_text(
                align='center',
                baseline='top',
                dx=0,
                dy=12
            )\
            .encode(
                text='Total_Win'
            )
            
    text_tm = points.mark_text(
                align='center',
                baseline='bottom',
                dx=0,
                dy=-10
            )\
            .encode(
                text='Team'
            )



    rule1 = alt.Chart().mark_rule(color='black')\
            .encode(
                x=alt.X('rd2:O', title='pick'),
                size=alt.value(2),
            )
    rule2 = alt.Chart().mark_rule(color='black')\
            .encode(
                x=alt.X('rd3:O', title=''),
                size=alt.value(2),
                # title=''
            )
    rule3 = alt.Chart().mark_rule(color='black')\
            .encode(
                x=alt.X('rd4:O', title=''),
                size=alt.value(2),
            )

    ## Prior to 2023 we only had 7 mgrs so we used Leftovers for the remaining 4 teams
    size = 2 if year < 2023 else 0
        
    rule4 = alt.Chart().mark_rule(color='black')\
            .encode(
                x=alt.X('leftover:O', title=''),
                size=alt.value(size),
                # name=''
            )

        ## color changing marks via radio buttons
        # input_checkbox = alt.binding_checkbox()
    # checkbox_selection = alt.selection_single(bind=input_checkbox, name="Big Budget Films")

    # size_checkbox_condition = alt.condition(checkbox_selection,
    #                                         alt.SizeValue(25),
    #                                         alt.Size('Hundred_Million_Production:Q')
    #                      
                    # )
    # selection = alt.selection_multi(fields=['name'])
    # color = alt.condition(selection, alt.Color('name:N'), alt.value('lightgray'))
    # make_selector = alt.Chart(make).mark_rect().encode(y='name', color=color).add_selection(selection)
    # fuel_chart = alt.Chart(fuel).mark_line().encode(x='index', y=alt.Y('fuel', scale=alt.Scale(domain=[0, 10])), color='name').transform_filter(selection)
                                        
                                        
    player_selection = alt.selection_multi(fields=['Player'])

    domain_ = list(plot_bg_clr_dct.keys())
    range_ = list(plot_bg_clr_dct.values())
    opacity_ = alt.condition(player_selection, alt.value(1.0), alt.value(.4))
    
    player_color_condition = alt.condition(player_selection,
                                alt.Color('Player:N', scale=alt.Scale(domain=domain_, range=range_)),
                                alt.value('lightgray')
                            )

    highlight_players = points.add_selection(player_selection)\
                            .encode(
                                color=player_color_condition,
                                opacity=opacity_
                                )#\
                            # .properties(title=f"{', '.join([str(i) for i in year_range])} Picks by Player")
    
    player_selector = alt.Chart(source).mark_rect()\
            .encode(x='Player', color=player_color_condition)\
            .add_selection(player_selection)
    
    
    
    
    # ## color changing marks via radio buttons - WORKS
    # player_radio = alt.binding_radio(options=frame['Player'].unique())
    # player_selection = alt.selection_single(fields=['Player'], bind=player_radio, name=".")
    # 
    # domain_ = list(plot_bg_clr_dct.keys())
    # range_ = list(plot_bg_clr_dct.values())
    # opacity_ = alt.condition(player_selection, alt.value(1.0), alt.value(.4))
    # 
    # player_color_condition = alt.condition(player_selection,
    #                             alt.Color('Player:N', 
    #                                 scale=alt.Scale(domain=domain_, range=range_)),
    #                             alt.value('lightgray')
    #                         )
    # 
    # highlight_players = points.add_selection(player_selection)\
    #                         .encode(
    #                             color=player_color_condition,
    #                             opacity=opacity_
    #                             )\
    #                         .properties(title=f"{curr_year} Picks by Player")
    # 
    # 
    
    
    
    # ## PLAYOFFS ? color changing marks via radio buttons
    # po_radio = alt.binding_radio(options=['Playoffs'])
    # po_select = alt.selection_single(fields=['Playoffs'], bind=po_radio, name="po!")
    # 
    # # domain_ = list(plot_bg_clr_dct.keys())
    # # range_ = list(plot_bg_clr_dct.values())
    # opacity_ = alt.condition(po_select, alt.value(1.0), alt.value(.4))
    # 
    # po_color_condition = alt.condition(po_select,
    #                             alt.Color('Playoffs:N', 
    #                                 scale=alt.Scale(domain=domain_, range=range_)),
    #                             alt.value('lightgray')
    #                         )
    # 
    # highlight_po = points.add_selection(po_select)\
    #                         .encode(
    #                             color=po_color_condition,
    #                             opacity=opacity_
    #                             )\
    #                         .properties(title=f"{curr_year} PO")
    
    
    
    if year < 2023:
        dividers = dict(rd2="7.5",          ## use pick halfway b/w rounds to draw vert line
                    rd3="14.5",
                    rd4="21.5",
                    leftover="28.5",
                    )
    else:
        dividers = dict(rd2="8.5",          ## use pick halfway b/w rounds to draw vert line
                    rd3="16.5",
                    rd4="24.5",
                    )

    res = alt.layer(rule1, rule2, rule3, rule4, text_wins, text_tm, highlight_players,
        data=source, width=1250).transform_calculate(**dividers)
        
    st.altair_chart(res) 
    # st.altair_chart(player_selector)


def plot_wins_by_year(frame: DataFrame):
    # print(frame)
    points = alt.Chart(frame)\
                .mark_line(strokeWidth=4, color='grey')\
                .encode(
                    alt.X('Year:O', axis=alt.Axis(format='.0f', tickMinStep=1, labelFlush=True, grid=True)),
                    alt.Y('Total_Win:Q', scale=alt.Scale(zero=True)),
                    order='Year',
                    )\
                .properties(
                    title='Wins by Year',
                    width=400,
                    height=200
                    )

    champ_pts = alt.Chart(frame)\
                .mark_point(filled=True, stroke='black', strokeWidth=1, size=100, opacity=1)\
                .encode(
                    alt.X('Year:O'),
                    alt.Y('Total_Win:Q', axis=alt.Axis(title=None)),
                    color=alt.Color('Champ', scale=alt.Scale(domain=[True, False], range=['azure', plot_bg_clr_dct[frame['Player'].unique().item()]]))
                    )
    
    text = points.mark_text(
                align='center',
                baseline='top',
                dx=0,
                dy=10
            )\
            .encode(
                text='Total_Win'
            )
    
    st.altair_chart(points + champ_pts + text, use_container_width=False)


def plot_ridge_altair(source: DataFrame, step: Union[float, int], overlap: Union[float, int]):
    ## very cool but not currently in use b/c not sure what to plot with it
    ridge = alt.Chart(source, height=step).transform_joinaggregate(
        mean_wins='mean(Total_Win)', groupby=['Year']
    ).transform_bin(
        ['bin_max', 'bin_min'], 'Total_Win'
    ).transform_aggregate(
        value='count()', groupby=['Year', 'mean_wins', 'bin_min', 'bin_max']
    ).transform_impute(
        impute='value', groupby=['Year', 'mean_wins'], key='bin_min', value=0
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
            scale=alt.Scale(domain=[1, 100], scheme='redyellowblue')
        )
    ).facet(
        row=alt.Row(
            'Year:T',
            title=None,
            header=alt.Header(labelAngle=0, labelAlign='right', format='%Y')
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










if __name__ == '__main__':
    pass
    # D = DataPrepper()
    # D.prep_WoW_metrics(2023)
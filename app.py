import matplotlib
matplotlib.use('Agg')
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from dash import Dash, html, dcc, Input, Output, State, no_update, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.colors as colors
import pandas as pd
from plotting_functions import shot_chart_plotly, shot_chart_zones, convert_seaborn_to_img


app = Dash(__name__, suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.BOOTSTRAP])

# *********************************************************************************************************

shotdata = pd.read_csv('./data/nba_shots_2012_to_2023.csv')
shots_comparison = pd.read_csv("./data/shotchart_comparison_2012_to_2023.csv")
shooting_corr = pd.read_csv("./data/shooting_correlation_matrix.csv")
shopting_corr_dict={
    "fga_diff_above_the_break_3": "FGA Differrence Above The Break 3",
    "fga_diff_left_corner_3": "FGA Differrence Left Corner 3",
    "fga_diff_mid-range": "FGA Differrence Mid Range",
    "fga_diff_in_the_paint_(non-ra)": "FGA Differrence Paint",
    "fga_diff_right_corner_3": "FGA Differrence Right Corner 3",
    "fga_diff_restricted_area": "FGA Differrence Restricted Area",
    "fg_pct_diff_above_the_break_3": "FG% Differrence Above The Break 3",
    "fg_pct_diff_left_corner_3": "FG% Differrence Left Corner 3",
    "fg_pct_diff_mid-range": "FG% Differrence Mid Range",
    "fg_pct_diff_in_the_paint_(non-ra)": "FG% Differrence Paint",
    "fg_pct_diff_right_corner_3": "FG% Differrence Right Corner 3",
    "fg_pct_diff_restricted_area": "FG% Differrence Restricted Area"
}
#shooting_averages = pd.read_csv('./data/shooting_averages_2012_to_2023.csv')
stats_numerical = pd.read_csv("./data/team_stats_numerical_2012_to_2023.csv")
#stats_multiindex = pd.read_pickle("./data/team_stats_multiindex_2012_to_2023.pkl")
#stats_corr_off = pd.read_pickle("./data/team_stats_correlation_matrix_off_rating.pkl")
#stats_corr_def = pd.read_pickle("./data/team_stats_correlation_matrix_def_rating.pkl")
stats_corr_off_def = pd.read_csv("./data/corr_df_off_def.csv")
stats_corr_overall = stats_corr_off_def.groupby(["feature"]).agg({"OFF_RATING_corr":"mean", "DEF_RATING_corr":"mean"}).reset_index(drop=False)
stats_corr_overall = stats_corr_overall.rename(columns={"OFF_RATING_corr":"OFF_RATING_corr_mean", "DEF_RATING_corr":"DEF_RATING_corr_mean"})
#stats_rank = pd.read_csv("./data/team_stats_rank_2012_to_2023.csv")

#cols_to_drop = ["GP_RANK", "W_RANK", "L_RANK", "W_PCT_RANK", "MIN_RANK"]
#team_stats_rank = stats_rank.drop(cols_to_drop, axis=1)
#own_rank_cols = [col for col in team_stats_rank.columns[7:-1] if "RANK" in col and "OPP" not in col]
#opp_rank_cols = [col for col in team_stats_rank.columns[7:-1] if "RANK" in col and "OPP" in col]

pca_data = pd.read_csv("./data/pca_data_2012_to_2023.csv")
pca_coefs = pd.read_csv("./data/pca_coefs_2012_to_2023.csv")

teams = stats_numerical[['TEAM_ID', 'TEAM_NAME']].drop_duplicates()
seasons = stats_numerical['SEASON'].unique()
#stat_groups = ['Base', 'Advanced','Misc', 'Four Factors', 'Scoring', 'Opponent', 'Defense']
# get lakers, boston, new york, chicago team_ids
lakers_id = teams[teams['TEAM_NAME'] == 'Los Angeles Lakers']['TEAM_ID'].values[0]
boston_id = teams[teams['TEAM_NAME'] == 'Boston Celtics']['TEAM_ID'].values[0]
new_york_id = teams[teams['TEAM_NAME'] == 'New York Knicks']['TEAM_ID'].values[0]
chicago_id = teams[teams['TEAM_NAME'] == 'Chicago Bulls']['TEAM_ID'].values[0]
starting_team_ids = [lakers_id, boston_id, new_york_id, chicago_id]

# *********************************************************************************************************
modal_shotchart = html.Div(
    [
        dbc.Button("Info", id="modal-shotchart_info"),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Explanaition of the abbreviations on this page")),
            dbc.ModalBody(
                [html.Div("FG%: Field Goal Percentage"),
                html.Div("FGA: Field Goal Attempt"),
                ]
            ),

        ],
        id="modal-shotchart",
        size="xl",  # "sm", "lg", "xl"
        backdrop=True,  # True, False or Static for modal to not be closed by clicking on backdrop
        scrollable=True,  # False or True if modal has a lot of text
        centered=True,  # True, False
        fade=True  # True, False
        )
    ]
)

modal_stats = html.Div(
    [
        dbc.Button("Info", id="modal-stats_info"),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Explanaition of the abbreviations on this page")),
            dbc.ModalBody(
                [html.Div("AST: Assists"),
                html.Div("AST%: Percentage of team's field goals that are assisted"),
                html.Div("AST_RATIO: Assists per 100 possessions"),
                html.Div("AST_TO: Assist to Turnover Ratio"),
                html.Div("BLK: Blocks"),
                html.Div("BLKA: Blocks against"),
                html.Div("DREB: Defensive Rebounds"),
                html.Div("DREB_PCT: Percentage of available defensive rebounds a team grabbed while on the floor"),
                html.Div("EFG_PCT: Effective Field Goal Percentage"),
                html.Div("E_PACE: Estimated Pace Factor"),
                html.Div("FG3A: Three Point Field Goals Attempted"),
                html.Div("FG3M: Three Point Field Goals Made"),
                html.Div("FG3_PCT: Three Point Field Goal Percentage"),
                html.Div("FGA: Field Goals Attempted"),
                html.Div("FGM: Field Goals Made"),
                html.Div("FG_PCT: Field Goal Percentage"),
                html.Div("FTA: Free Throws Attempted"),
                html.Div("FTA_RATE: Free Throw Attempt Rate (FTA/FGA)"),
                html.Div("FTM: Free Throws Made"),
                html.Div("FT_PCT: Free Throw Percentage"),
                html.Div("OPP_AST: Opponent Assists"),
                html.Div("OPP_BLK: Opponent Blocks"),
                html.Div("OPP_BLKA: Opponent Blocks Against"),
                html.Div("OPP_DREB: Opponent Defensive Rebounds"),
                html.Div("OPP_EFG_PCT: Opponent Effective Field Goal Percentage"),
                html.Div("OPP_FG3A: Opponent Three Point Field Goals Attempted"),
                html.Div("OPP_FG3M: Opponent Three Point Field Goals Made"),
                html.Div("OPP_FG3_PCT: Opponent Three Point Field Goal Percentage"),
                html.Div("OPP_FGA: Opponent Field Goals Attempted"),
                html.Div("OPP_FGM: Opponent Field Goals Made"),
                html.Div("OPP_FG_PCT: Opponent Field Goal Percentage"),
                html.Div("OPP_FTA: Opponent Free Throws Attempted"),
                html.Div("OPP_FTA_RATE: Opponent Free Throw Attempt Rate (FTA/FGA)"),
                html.Div("OPP_FTM: Opponent Free Throws Made"),
                html.Div("OPP_FT_PCT: Opponent Free Throw Percentage"),
                html.Div("OPP_OREB: Opponent Offensive Rebounds"),
                html.Div("OPP_OREB_PCT: The percentage of available offensive rebounds the opponent obtains."),
                html.Div("OPP_PF: Opponent Personal Fouls"),
                html.Div("OPP_PFD: Opponent Personal Fouls Drawn"),
                html.Div("OPP_PTS: Opponent Points"),
                html.Div("OPP_PTS_2ND_CHANCE: Opponent Points from Second Chance (After Offensive Rebound)"),
                html.Div("OPP_PTS_FB: Opponent Points from Fast Break"),
                html.Div("OPP_PTS_OFF_TOV: Opponent Points off Turnovers"),
                html.Div("OPP_PTS_PAINT: Opponent Points in the Paint"),
                html.Div("OPP_REB: Opponent Rebounds"),
                html.Div("OPP_STL: Opponent Steals"),
                html.Div("OPP_TOV: Opponent Turnovers"),
                html.Div("OPP_TOV_PCT: The number of turnovers an opponent averages per 100 of their own possessions"),
                html.Div("OREB: Offensive Rebounds"),
                html.Div("OREB_PCT: The percentage of available offensive rebounds a team grabbed while on the floor"),
                html.Div("PACE: Pace Factor"),
                html.Div("PACE_PER40: Pace per 40 minutes"),
                html.Div("PCT_AST_2PM: Percentage of 2-point field goals that are assisted"),
                html.Div("PCT_AST_3PM: Percentage of 3-point field goals that are assisted"),
                html.Div("PCT_AST_FGM: Percentage of field goals that are assisted"),
                html.Div("PCT_FGA_2PT: Percentage of field goal attempts that are 2-point field goals"),
                html.Div("PCT_FGA_3PT: Percentage of field goal attempts that are 3-point field goals"),
                html.Div("PCT_PTS_2PT: Percentage of points that are 2-point field goals"),
                html.Div("PCT_PTS_2PT_MR: Percentage of points that are 2-point field goals made from mid-range"),
                html.Div("PCT_PTS_3PT: Percentage of points that are 3-point field goals"),
                html.Div("PCT_PTS_FB: Percentage of points that are from fast breaks"),
                html.Div("PCT_PTS_FT: Percentage of points that are from free throws"),
                html.Div("PCT_PTS_OFF_TOV: Percentage of points that are off turnovers"),
                html.Div("PCT_PTS_PAINT: Percentage of points that are in the paint"),
                html.Div("PCT_UAST_2PM: Percentage of 2-point field goals that are unassisted"),
                html.Div("PCT_UAST_3PM: Percentage of 3-point field goals that are unassisted"),
                html.Div("PCT_UAST_FGM: Percentage of field goals that are unassisted"),
                html.Div("PF: Personal Fouls"),
                html.Div("PFD: Personal Fouls Drawn"),
                html.Div("PIE: Player Impact Estimate"),
                html.Div("PLUS_MINUS: Plus-Minus"),
                html.Div("POSS: Possessions"),
                html.Div("PTS: Points"),
                html.Div("PTS_2ND_CHANCE: Points from Second Chance (After Offensive Rebound)"),
                html.Div("PTS_FB: Points from Fast Break"),
                html.Div("PTS_OFF_TOV: Points off Turnovers"),
                html.Div("PTS_PAINT: Points in the Paint"),
                html.Div("REB: Rebounds"),
                html.Div("REB_PCT: The percentage of available rebounds a team grabbed while on the floor"),
                html.Div("STL: Steals"),
                html.Div("TM_TOV_PCT: The number of turnovers a team averages per 100 possessions"),
                html.Div("TOV: Turnovers"),
                html.Div("TS_PCT: True Shooting Percentage"),
                ]
            ),

        ],
        id="modal-stats",
        size="xl",  # "sm", "lg", "xl"
        backdrop=True,  # True, False or Static for modal to not be closed by clicking on backdrop
        scrollable=True,  # False or True if modal has a lot of text
        centered=True,  # True, False
        fade=True  # True, False
        )
    ]
)

nav_bar = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink("Team shooting map",href='/',id='shot-map-nav',active='exact')),
        dbc.NavItem(dbc.NavLink('Stats',href='/stats',id='stats-nav',active='exact')),
        dbc.NavItem(dbc.NavLink('PCA',href='/pca',id='pca-nav',active='exact')),
    ],
    pills=True,
    justified=True
)

shotmap_container = dbc.Container([
    html.H1("How did the optimal shooting map change over the years?"),
    dbc.Card([

        dbc.CardBody([
            dbc.Row([
                html.H4("Select a team and season"),
            ]),
            dbc.Row([
                dbc.Col([modal_shotchart]),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='team-dropdown',
                        options=[{'label': team['TEAM_NAME'], 'value': team['TEAM_ID']} for team in teams.to_dict('records')],
                        value=teams['TEAM_ID'].iloc[0],  # Default value
                        placeholder="Select a team"
                    )
                ],width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='season-dropdown',
                        options=[{'label': season, 'value': season} for season in seasons],
                        value=seasons[0],  # Default value
                        placeholder="Select a season"
                    )
                ],width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    #dcc.Graph(id='shot-map'),
                    html.Img(id='shot-map-efficiency')
                ],width=6),
                dbc.Col([
                    html.Img(id='shot-map-volume')
                ],width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='shot-map-kde')
                ],width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='shooting-corr-bar-chart')
                ],width=12),
            ]),
            
        ])
    ]),
])

stats_container = dbc.Container([
    html.H1("What statistics were key for good offense/defense?"),
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                modal_stats
            ]),
            dbc.Row([
                dash_table.DataTable(
                    id='stats-table',
                    data=stats_corr_overall.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in stats_corr_overall.columns],
                    fixed_rows={'headers': True, 'data': 0},
                    #style_cell='whiteSpace: normal',
                    #virtualization=True,
                    editable=False,
                    #filter_action="native",
                    #sort_action="native",
                    #sort_mode="multi",
                    row_selectable="multi",
                    row_deletable=False,
                    #selected_rows=[],
                    #style_cell_conditional=[
                    #    {'if': {'column_id': 'feature'}, 'width': '30%'},
                    #    {'if': {'column_id': 'OFF_RATING_corr_mean'}, 'width': '35%'},
                    #    {'if': {'column_id': 'DEF_RATING_corr_mean'}, 'width': '35%'}
                    #],
                )
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='offense-corr-line-chart')
                ]),
                dbc.Col([
                    dcc.Graph(id='defense-corr-line-chart')
                ]),
            ]),
        ])
    ]),
])

pca_container = dbc.Container([
    html.H1("PCA results of statistics"),
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                html.H4("Select a team and season"),
            ]),
            dbc.Row([
                modal_stats
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='team-dropdown-pca',
                        options=[{'label': team['TEAM_NAME'], 'value': team['TEAM_ID']} for team in teams.to_dict('records')],
                        value=starting_team_ids,  # Default value
                        multi=True,
                        placeholder="Select a team"
                    )
                ],width=8),
                dbc.Col([
                    dcc.Dropdown(
                        id='season-dropdown-pca',
                        options=[{'label': season, 'value': season} for season in seasons],
                        value=seasons[0],  # Default value
                        placeholder="Select a season"
                    )
                ],width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='pca-graph')
                ],width=6),
                dbc.Col([
                    dcc.Graph(id='off-def-graph')
                ],width=6),
            ]),
            
        ])
    ]),
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                html.H4("Select x-axis for bar chart"),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='x-axis-dropdown',
                        options=[{'label': 'Teams', 'value': 'teams'}, {'label': 'Features', 'value': 'features'}],
                        value='teams',  # Default value
                        placeholder="Select x-axis"
                    )
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='pca_stats-bar-chart')
                ]),
            ]),
        ])
    ])
])

content = html.Div(id='page_content',children=[],style={'padding':'2rem'})
app.layout = html.Div([
    dcc.Location(id='url'),
    nav_bar,
    content
    #dbc.Row([dbc.Col(image_card_shot_perform, width=4), dbc.Col(graph_card_shot_perform, width=8)], justify="around")
])

# *********************************************************************************************************

@app.callback(
    Output('page_content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/':
        return shotmap_container
    elif pathname == '/stats':
        return stats_container
    elif pathname == '/pca':
        return pca_container
    else:
        return dbc.Card(
            [
                html.H1('404: Not found', className='text-danger'),
                html.Hr(),
                html.P(f'The pathname {pathname} was not recognised...')
            ]
        )


def create_diff_dict_and_colorscale(df, y, by, team_id, season):
    """
    Create FG% difference dictionary and a colorscale for all teams in a given season.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing shot zone data.
    season : str
        The SEASON for which to calculate FG% differences and create the colorscale.

    Returns
    -------
    fg_diff_dict : dict
        A dictionary where keys are shot zone names and values are FG% differences.
    colorscale : list
        A list of tuples defining the colorscale based on FG% difference values.
    vmin, vmax : float
        The minimum and maximum FG% difference values used for scaling.
    """
    # Filter the data for the specific season
    season_data = df[df['SEASON'] == season]

    # Calculate the min and max FG% difference across all teams
    vmin = season_data[y].min()
    vmax = season_data[y].max()
    max_abs = max(abs(vmin), abs(vmax))
    vmin = -max_abs
    vmax = max_abs

    # Filter the data for the specific team and season
    team_season_data = df[(df['TEAM_ID'] == team_id) & (df['SEASON'] == season)]

    # Group by 'by' column and get the mean y
    fg_diff_dict = team_season_data.groupby(by)[y].mean().to_dict()

    return fg_diff_dict, vmin, vmax

# *********************************************************************************************************

@app.callback(
    Output('shot-map-efficiency', 'src'),
    Output('shot-map-volume', 'src'),
    Output('shot-map-kde', 'figure'),
    Input('team-dropdown', 'value'),
    Input('season-dropdown', 'value')
)
def update_shot_map(team_id, season):
    #
    fg_diff_dict,vmin,vmax = create_diff_dict_and_colorscale(shots_comparison,"FG_PCT_diff","SHOT_ZONE_BASIC", team_id, season)
    team_name = teams[teams['TEAM_ID'] == team_id]['TEAM_NAME'].values[0]
    
    fig,ax = plt.subplots(figsize=(6,5))
    ax = shot_chart_zones(fg_diff_dict,vmin=vmin,vmax=vmax, title=f"{team_name} FG% Difference\nin {season}", cbar_label="Field Goal % Difference", ax=ax, cmap="RdBu_r",flip_court=True)
    fig_efficiency = convert_seaborn_to_img(fig)
    
    fg_diff_dict,vmin,vmax = create_diff_dict_and_colorscale(shots_comparison,"FGA_diff","SHOT_ZONE_BASIC", team_id, season)
    fig,ax = plt.subplots(figsize=(6,5))
    ax = shot_chart_zones(fg_diff_dict,vmin=vmin,vmax=vmax, title=f"{team_name} FGA Difference\nin {season}", cbar_label="Field Goal Attempt Difference", ax=ax, cmap="RdBu_r",flip_court=True)
    fig_volume = convert_seaborn_to_img(fig)

    team_shot_data_season = shotdata[(shotdata['TEAM_ID'] == team_id) & (shotdata['SEASON'] == season)]
    fig_kde = shot_chart_plotly(data=team_shot_data_season, x='LOC_X', y='LOC_Y', kind="scatter")#, title=f"{team_name} Shot Chart in {season}")
    
    return fig_efficiency, fig_volume, fig_kde

@app.callback(
    Output('shooting-corr-bar-chart', 'figure'),
    Input('season-dropdown', 'value')
)
def update_shooting_corr_bar_chart(season):
    shooting_corr_season = shooting_corr.loc[shooting_corr["season"] == season]
    # delete columns starting with fgm
    shooting_corr_season = shooting_corr_season.loc[:,~shooting_corr_season.columns.str.startswith("fgm")]
    shooting_corr_season = shooting_corr_season.iloc[:,:-1]
    shooting_corr_season = shooting_corr_season.T.reset_index(drop=False)
    shooting_corr_season.columns = ["stat", "correlation"]
    # Map the annotations using the dictionary
    shooting_corr_season['stat'] = shooting_corr_season['stat'].map(shopting_corr_dict)

    #fig = px.bar(shooting_corr_season, x="stat", y="correlation", text='annotation', title=f"Correlation with Offensive Rating in {season}")
    #fig.update_traces(textposition='outside')
    
    fig = px.bar(shooting_corr_season, x="stat", y="correlation", title=f"FGA and FG% differences\ncorrelation with Offensive Rating in {season}")
    fig.update_layout(yaxis_title="Correlation Coefficient", xaxis_title="Statistic")
    return fig

@app.callback(
    Output('pca-graph', 'figure'),
    Output('off-def-graph', 'figure'),
    Output('pca_stats-bar-chart', 'figure'),
    Input('team-dropdown-pca', 'value'),
    Input('season-dropdown-pca', 'value'),
    Input('x-axis-dropdown', 'value')
)
def update_pca_graph(team_ids, season, bar_x_axis):
    season_data = pca_data[pca_data['season'] == season]
    team_data = season_data[season_data['team_id'].isin(team_ids)]

    fig1 = px.scatter(team_data, x='PC1', y='PC2', color="net_rating", text="team_name", hover_name='team_name', hover_data=['team_id', 'season', 'net_rating'])
    fig1.update_layout(title=f"PCA of team statistics in {season}")
    fig1.update_traces(textposition='top center')
    fig2 = px.scatter(team_data, x='off_rating', y='def_rating', color="net_rating", text="team_name", hover_name='team_name', hover_data=['team_id', 'season', 'net_rating'])
    fig2.update_layout(title=f"Offensive vs Defensive Rating in {season}")
    fig2.update_traces(textposition='top center')

    pca_season = pca_coefs[pca_coefs['season'] == season]
    pca_season = pca_season.groupby("season").mean().reset_index(drop=False)
    top5index = np.argsort(pca_season.iloc[:,1:].abs())[0][-5:]
    top_5_features = pca_season.columns[1:][top5index].tolist()
    #top_5_features_rank = [f"{col}_RANK" for col in top_5_features]
    
    stats_season = stats_numerical[stats_numerical['SEASON'] == season]
    stats_team = stats_season.loc[stats_season['TEAM_ID'].isin(team_ids),["TEAM_NAME"]+top_5_features]

    df_plot = stats_team.melt(id_vars=["TEAM_NAME"], var_name="feature", value_name="value")
    if bar_x_axis == "teams":
        fig3 = px.bar(df_plot, x='TEAM_NAME', y='value', color='feature', barmode='group', title=f'Teams statistics in top 5 PCA features in {season}')
    else:
        fig3 = px.bar(df_plot, x='feature', y='value', color='TEAM_NAME', barmode='group', title=f'Teams statistics in top 5 PCA features in {season}')

    return fig1, fig2,fig3

@app.callback(
    Output('offense-corr-line-chart', 'figure'),
    Output('defense-corr-line-chart', 'figure'),
    Input('stats-table', 'selected_rows')
)
def update_corr_line_chart(selected_rows):
    if selected_rows is None or len(selected_rows) == 0:
        return no_update, no_update
    selected_features = stats_corr_overall.iloc[selected_rows]['feature'].tolist()
    plot_df = stats_corr_off_def[stats_corr_off_def['feature'].isin(selected_features)]

    fig_off = px.line(plot_df, x='season', y='OFF_RATING_corr', color='feature', title='Offensive Rating Correlation')
    fig_def = px.line(plot_df, x='season', y='DEF_RATING_corr', color='feature', title='Defensive Rating Correlation')
    return fig_off, fig_def

@app.callback(
    Output('modal-shotchart', 'is_open'),
    Input('modal-shotchart_info', 'n_clicks'),
    State('modal-shotchart', 'is_open')
)
def toggle_modal_shotchart(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    Output('modal-stats', 'is_open'),
    Input('modal-stats_info', 'n_clicks'),
    State('modal-stats', 'is_open')
)
def toggle_modal_stats(n1, is_open):
    if n1:
        return not is_open
    return is_open
# *********************************************************************************************************
if __name__ == '__main__':
    app.run_server(debug=True)
# ==============================================================
# NHL Player Salary Dashboard – Business Polished Version
# ==============================================================

import os
import pandas as pd
import plotly.express as px
import dash
from dash import Dash, dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# --------------------------------------------------------------
# 1. App Setup
# --------------------------------------------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SANDSTONE, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "🏒 NHL Salary Analytics Dashboard"
server = app.server

DATA_PATH = r"C:\Users\tlcon\OneDrive\Documents\GitHub\NHL_Data\data"

# --------------------------------------------------------------
# 2. Global Style Config
# --------------------------------------------------------------

import plotly.io as pio

pio.templates["business_hover"] = pio.templates["plotly_white"]

# Global hover styling
pio.templates["business_hover"].layout.hoverlabel = dict(
    bgcolor="rgba(255, 255, 255, 0.95)",
    bordercolor="#1f3c5b",
    font=dict(
        family="Segoe UI, Roboto, Open Sans, sans-serif",
        size=13,
        color="#1f3c5b"
    ),
    align="left"
)

pio.templates["business_hover"].layout.hovermode = "closest"
pio.templates["business_hover"].layout.paper_bgcolor = "rgba(0,0,0,0)"
pio.templates["business_hover"].layout.plot_bgcolor = "rgba(0,0,0,0)"

# Set this as default
pio.templates.default = "business_hover"

CARD_STYLE = {
    "textAlign": "center",
    "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
    "borderRadius": "10px",
    "backgroundColor": "#ffffff",
    "padding": "18px",
}
CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Segoe UI, sans-serif", size=13, color="#333"),
    title_font=dict(size=20, family="Segoe UI, sans-serif", color="#1f3c5b"),
    margin=dict(l=60, r=40, t=70, b=60),
)

# --------------------------------------------------------------
# 3. Load Data
# --------------------------------------------------------------
def load_data():
    try:
        summary_df = pd.read_csv(os.path.join(DATA_PATH, "model_summary.csv"))
        pos_results_df = pd.read_csv(os.path.join(DATA_PATH, "position_r2.csv"))
        player_preds_df = pd.read_csv(os.path.join(DATA_PATH, "player_predictions.csv"))

        pos_map = {0: "C", 1: "D", 3: "LW", 4: "RW"}
        pos_results_df["Position"] = pos_results_df["Pos_encoded"].map(pos_map)

    except FileNotFoundError:
        summary_df = pd.DataFrame(columns=["Model", "R2", "RMSE"])
        pos_results_df = pd.DataFrame(columns=["Pos_encoded", "R2", "RMSE", "Position"])
        player_preds_df = pd.DataFrame(columns=["Player_Name", "Team", "Pos", "AAV_M", "Predicted_AAV_M", "Residual_M"])

    return summary_df, pos_results_df, player_preds_df


summary_df, pos_results_df, player_preds_df = load_data()

# --------------------------------------------------------------
# 4. KPI Cards
# --------------------------------------------------------------
def kpi_card(title, value, icon, color):
    return dbc.Card(
        dbc.CardBody([
            html.I(className=f"bi bi-{icon}", style={"fontSize": "1.8rem", "color": color}),
            html.H6(title.upper(), className="mt-2 text-muted"),
            html.H3(value, className="fw-bold", style={"color": color}),
        ]),
        style=CARD_STYLE
    )

overall_r2 = summary_df.loc[summary_df["Model"].str.contains("Overall", na=False), "R2"].max() if not summary_df.empty else 0
veteran_r2 = summary_df.loc[summary_df["Model"].str.contains("Veteran", na=False), "R2"].max() if not summary_df.empty else 0
rookie_r2 = summary_df.loc[summary_df["Model"].str.contains("Rookie", na=False), "R2"].max() if not summary_df.empty else 0

# --------------------------------------------------------------
# 5. Overview Tab Charts
# --------------------------------------------------------------
r2_bar = px.bar(
    summary_df,
    x="Model",
    y="R2",
    color="Model",
    title="Model R² Comparison: Overall vs Rookie vs Veteran",
    color_discrete_sequence=px.colors.qualitative.Safe
)
pos_bar = px.bar(
    pos_results_df.sort_values("R2", ascending=True),
    x="R2",
    y="Position",
    orientation="h",
    title="R² by Player Position",
    color="R2",
    color_continuous_scale="viridis"
)
for fig in [r2_bar, pos_bar]:
    fig.update_layout(**CHART_LAYOUT)
    fig.update_traces(marker_line_color="white", marker_line_width=0.5)

# --------------------------------------------------------------
# 6. Overview Tab Layout
# --------------------------------------------------------------
overview_tab = dbc.Container([
    html.H1("🏒 NHL Player Salary Modeling Dashboard",
            className="fw-bold text-center my-4",
            style={"color": "#1f3c5b"}),

    dbc.Row([
        dbc.Col(kpi_card("Overall Model R²", f"{overall_r2:.3f}", "graph-up", "#2E8B57"), md=4),
        dbc.Col(kpi_card("Veteran Model R²", f"{veteran_r2:.3f}", "person-badge", "#007BFF"), md=4),
        dbc.Col(kpi_card("Rookie Model R²", f"{rookie_r2:.3f}", "star", "#DAA520"), md=4),
    ], className="mb-4"),

    html.P(
        "This dashboard summarizes NHL player salary modeling performance and contract efficiency across player groups and positions.",
        className="text-center text-muted mb-5", style={"fontSize": "1rem"}
    ),

    dbc.Row([
        dbc.Col(dcc.Graph(figure=r2_bar), md=6),
        dbc.Col(dcc.Graph(figure=pos_bar), md=6)
    ]),

    html.Hr(style={"borderTop": "2px solid #dee2e6", "width": "80%", "margin": "2rem auto"}),

    html.H4("📊 Model Summary Table", className="fw-bold text-center mb-3", style={"color": "#1f3c5b"}),
    dash_table.DataTable(
        data=summary_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in summary_df.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#f1f1f1", "fontWeight": "bold"},
        style_cell={"textAlign": "center", "fontSize": "13px"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}]
    ),

    html.Br(),
    html.P("All data generated by NHL_Data_project.ipynb pipeline",
           className="text-muted text-center mb-4", style={"fontSize": "0.9rem"}),

], fluid=True, style={"backgroundColor": "#f8f9fa", "padding": "2rem", "borderRadius": "12px"})


# --------------------------------------------------------------
# 7. Player Predictions Tab (Interactive)
# --------------------------------------------------------------
if not player_preds_df.empty:
    r2_by_pos = (
        player_preds_df.groupby("Pos")[["AAV_M", "Predicted_AAV_M"]]
        .apply(lambda df: df.corr().iloc[0, 1] ** 2)
        .reset_index(name="R2")
    )
else:
    r2_by_pos = pd.DataFrame(columns=["Pos", "R2"])

position_options = [{"label": pos, "value": pos} for pos in sorted(player_preds_df["Pos"].dropna().unique())]

player_tab = dbc.Container([
    html.H2("📈 Player Predictions: Actual vs Predicted AAV", className="fw-bold text-center my-4", style={"color": "#1f3c5b"}),

    dbc.Row([
        dbc.Col([
            html.Label("Select Position:", style={"fontWeight": "bold", "fontSize": "1rem"}),
            dcc.Dropdown(
                id="position-filter",
                options=position_options,
                value=None,
                placeholder="All Positions",
                clearable=True,
                style={"marginBottom": "1rem"}
            )
        ], md=4),
        dbc.Col([
            html.Div(id="r2-display", className="text-center", style={"fontSize": "1.2rem", "marginTop": "1rem"})
        ], md=8)
    ]),

    dcc.Graph(id="predictions-scatter"),
], fluid=True)

# Age normalization
#player_preds_df["Age"] = player_preds_df["Age"].fillna(0).astype(int)

# Do NOT overwrite pipeline Unique_ID — only create if missing
if "Unique_ID" not in player_preds_df.columns:
    # Add a fallback for missing Age
    player_preds_df['Age'] = player_preds_df.get('Age', '')  # Use empty string if Age is missing
    
    player_preds_df["Unique_ID"] = (
        player_preds_df["Player_Name"].str.replace(" ", "")
        + "_" + player_preds_df["Team"].str.replace(" ", "")
        + "_" + player_preds_df["Pos"]
        + "_" + player_preds_df["Age"].astype(str)
    )



@app.callback(
    [Output("predictions-scatter", "figure"),
     Output("r2-display", "children")],
    [Input("position-filter", "value")]
)


def update_scatter(selected_pos):
    #print("\n===== DEBUG: player_preds_df SHAPE & COLUMNS =====")
    #print(player_preds_df.shape)
    #print(player_preds_df.columns.tolist())

    # Show only Elias Pettersson rows
    ep_debug = player_preds_df[player_preds_df["Player_Name"] == "Elias Pettersson"]
    #print("\n===== DEBUG: All Elias Pettersson rows BEFORE FILTER =====")
    #print(ep_debug.to_string(index=False))

    if selected_pos:
        filtered_df = player_preds_df[player_preds_df["Pos"] == selected_pos]
        ep_filter = filtered_df[filtered_df["Player_Name"] == "Elias Pettersson"]
        #print(f"\n===== DEBUG: Elias Pettersson rows AFTER FILTER ({selected_pos}) =====")
        #print(ep_filter.to_string(index=False))
    else:
        filtered_df = player_preds_df.copy()

    # Show numeric comparison details
    #print("\n===== DEBUG: Key Salary / Prediction / Residual Fields =====")
    #print(player_preds_df.loc[player_preds_df["Player_Name"] == "Elias Pettersson",
                              #["Player_Name","Pos","Age","Team",
                               #"AAV_M","Predicted_AAV_M","Residual_M","Unique_ID"]
                            #].to_string(index=False))

    if player_preds_df.empty:
        fig = px.scatter(title="No data available")
        fig.update_layout(**CHART_LAYOUT)
        return fig, ""
    filtered_df = player_preds_df.copy()
    if selected_pos:
        filtered_df = filtered_df[filtered_df["Pos"] == selected_pos]
    if len(filtered_df) > 1:
        corr = filtered_df[["AAV_M", "Predicted_AAV_M"]].corr().iloc[0, 1]
        r2_value = corr ** 2 if not pd.isna(corr) else 0
    else:
        r2_value = 0
    fig = px.scatter(
    filtered_df,
    x="AAV_M",
    y="Predicted_AAV_M",
    color="Pos",
    hover_name="Unique_ID",
    hover_data={
        "Player_Name": True,
        "Team": True,
        "Pos": True,
        "Age": True,
        "AAV_M": ":.2f",
        "Predicted_AAV_M": ":.2f",
        "Residual_M": ":.2f"
    },
    title="Actual vs Predicted Player Salaries (Millions)",
    labels={"AAV_M": "Actual Salary (M)", "Predicted_AAV_M": "Predicted Salary (M)"}
)

    fig.update_layout(**CHART_LAYOUT)
    fig.add_shape(
        type="line",
        x0=filtered_df["AAV_M"].min(),
        y0=filtered_df["AAV_M"].min(),
        x1=filtered_df["AAV_M"].max(),
        y1=filtered_df["AAV_M"].max(),
        line=dict(color="red", dash="dash")
    )
    r2_text = f"Predictive R² for {'All Positions' if not selected_pos else selected_pos}: **{r2_value:.3f}**"
    return fig, r2_text



# --------------------------------------------------------------
# 8. Underpaid Players Tab (Interactive)
# --------------------------------------------------------------
try:
    player_df = pd.read_csv(os.path.join(DATA_PATH, "player_predictions.csv"))
except FileNotFoundError:
    player_df = pd.DataFrame(columns=["Player_Name", "Team", "Pos", "AAV_M", "Predicted_AAV_M", "Residual_M", "Points_per_game"])

def categorize_ppg(row):
    if row["Pos"] in ["C", "LW", "RW", "F"]:
        if row["Points_per_game"] >= 1.0:
            return "Elite"
        elif row["Points_per_game"] >= 0.7:
            return "Very Good"
        elif row["Points_per_game"] >= 0.5:
            return "Solid Role"
        else:
            return "Depth Role"
    else:
        if row["Points_per_game"] >= 0.8:
            return "Elite"
        elif row["Points_per_game"] >= 0.5:
            return "Very Good"
        elif row["Points_per_game"] >= 0.3:
            return "Solid Role"
        else:
            return "Depth Role"

if not player_df.empty and "PPG_Category" not in player_df.columns:
    player_df["PPG_Category"] = player_df.apply(categorize_ppg, axis=1)
elif player_df.empty:
    player_df["PPG_Category"] = pd.Series(dtype="object")

#player_df["Age"] = player_df["Age"].fillna(0).astype(int)

player_df = player_df.drop_duplicates(
    subset=["Player_Name", "Team", "Pos", "Age", "AAV_M", "Predicted_AAV_M", "Residual_M"]
)


player_df["Residual_M_Clipped"] = player_df["Residual_M"].clip(-5, 5)

category_colors = {"Elite": "#2ca02c", "Very Good": "#1f77b4", "Solid Role": "#ff7f0e", "Depth Role": "#d62728"}
position_options_underpaid = [{"label": pos, "value": pos} for pos in sorted(player_df["Pos"].dropna().unique())]

tab3 = dbc.Container([
    html.H2("💰 Top 25 Underpaid Players by Position", className="fw-bold text-center my-4", style={"color": "#1f3c5b"}),
    html.P("Select a position to view the 25 most underpaid players based on residuals (Actual – Predicted AAV), grouped by scoring tier.",
           className="text-center text-muted mb-4", style={"fontSize": "1rem"}),

    dbc.Row([
        dbc.Col([
            html.Label("Select Position:", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="underpaid-position-filter",
                options=position_options_underpaid,
                value=position_options_underpaid[0]["value"] if position_options_underpaid else None,
                clearable=False,
                style={"marginBottom": "1rem"}
            )
        ], md=4)
    ], justify="center"),

    dcc.Graph(id="underpaid-bar-graph"),
], fluid=True)

@app.callback(
    Output("underpaid-bar-graph", "figure"),
    [Input("underpaid-position-filter", "value")]
)
def update_underpaid_chart(selected_pos):
    if player_df.empty or not selected_pos:
        fig = px.bar(title="No data available")
        fig.update_layout(**CHART_LAYOUT)
        return fig
    pos_df = (
        player_df[player_df["Pos"] == selected_pos]
        .sort_values("Residual_M", ascending=True)
        .head(25)
    )
    fig_height = 32 * len(pos_df) + 140
    fig = px.bar(
        pos_df,
        x="Residual_M_Clipped",
        y="Player_Name",
        color="PPG_Category",
        text=pos_df["Residual_M_Clipped"].round(2).astype(str) + "M",
        orientation="h",
        color_discrete_map=category_colors,
        title=f"Top 25 Underpaid {selected_pos}s",
        labels={"Residual_M_Clipped": "Underpaid (Millions)", "PPG_Category": "Points per Game Tier"},
        hover_data={
            "Team": True,
            "AAV_M": ":.2f",
            "Predicted_AAV_M": ":.2f",
            "Residual_M": ":.2f",
            "Points_per_game": ":.2f",
            "PPG_Category": True
        },
    )
    fig.update_layout(**CHART_LAYOUT, height=fig_height)
    fig.update_traces(textposition="outside")
    return fig

# --------------------------------------------------------------
# 9. Trade Analyzer Tab – Compare Predicted Value Between Teams
# --------------------------------------------------------------

# Safety: ensure Unique_ID exists (if you haven't already earlier)
if "Unique_ID" not in player_df.columns:
    player_df["Unique_ID"] = (
        player_df["Player_Name"].astype(str)
        + "_" + player_df["Team"].astype(str)
        + "_" + player_df["Pos"].astype(str)
    )

# Team dropdown options
trade_team_options = [
    {"label": t, "value": t}
    for t in sorted(player_df["Team"].dropna().unique())
]

trade_tab = dbc.Container([
    html.H2("🔁 Trade Analyzer", className="fw-bold text-center my-4",
            style={"color": "#1f3c5b"}),

    html.P(
        "Select two teams and choose players going each way. "
        "The stacked bars show the total predicted salary value (Predicted AAV).",
        className="text-center text-muted mb-4",
        style={"fontSize": "0.95rem"}
    ),

    # --- Team selectors + player selectors ---
    dbc.Row([
        # TEAM 1 PANEL
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Team 1 Receives", className="mb-0")),
                dbc.CardBody([
                    html.Label("Team 1", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="trade-team1",
                        options=trade_team_options,
                        placeholder="Select Team 1",
                        clearable=True,
                        style={"marginBottom": "1rem"}
                    ),

                    html.Label("Players to Team 1", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="trade-players1",
                        options=[],
                        value=[],
                        multi=True,
                        placeholder="Select players from chosen team",
                        style={"marginBottom": "1rem"}
                    ),

                    dash_table.DataTable(
                        id="trade-table1",
                        columns=[
                            {"name": "Player", "id": "Player_Display"},
                            {"name": "Pos", "id": "Pos"},
                            {"name": "Predicted AAV (M)", "id": "Predicted_AAV_M",
                             "type": "numeric", "format": {"specifier": ".2f"}},
                        ],
                        data=[],
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "fontSize": 12},
                        style_header={
                            "backgroundColor": "#f1f1f1",
                            "fontWeight": "bold"
                        },
                        style_data_conditional=[
                            {"if": {"row_index": "odd"},
                             "backgroundColor": "#fafafa"}
                        ],
                        page_size=10
                    ),
                ])
            ], style={"boxShadow": "0 4px 8px rgba(0,0,0,0.05)",
                      "borderRadius": "10px"})
        ], md=6),

        # TEAM 2 PANEL
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Team 2 Receives", className="mb-0")),
                dbc.CardBody([
                    html.Label("Team 2", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="trade-team2",
                        options=trade_team_options,
                        placeholder="Select Team 2",
                        clearable=True,
                        style={"marginBottom": "1rem"}
                    ),

                    html.Label("Players to Team 2", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="trade-players2",
                        options=[],
                        value=[],
                        multi=True,
                        placeholder="Select players from chosen team",
                        style={"marginBottom": "1rem"}
                    ),

                    dash_table.DataTable(
                        id="trade-table2",
                        columns=[
                            {"name": "Player", "id": "Player_Display"},
                            {"name": "Pos", "id": "Pos"},
                            {"name": "Predicted AAV (M)", "id": "Predicted_AAV_M",
                             "type": "numeric", "format": {"specifier": ".2f"}},
                        ],
                        data=[],
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "fontSize": 12},
                        style_header={
                            "backgroundColor": "#f1f1f1",
                            "fontWeight": "bold"
                        },
                        style_data_conditional=[
                            {"if": {"row_index": "odd"},
                             "backgroundColor": "#fafafa"}
                        ],
                        page_size=10
                    ),
                ])
            ], style={"boxShadow": "0 4px 8px rgba(0,0,0,0.05)",
                      "borderRadius": "10px"})
        ], md=6),
    ], className="mb-4"),

    html.Hr(),

    # --- Totals + stacked comparison chart ---
    html.H3("Total Predicted Value (Millions)", className="fw-bold text-center mb-3",
            style={"color": "#1f3c5b"}),

    dcc.Graph(id="trade-bar-chart"),

    dbc.Row([
        dbc.Col(html.Div(id="trade-total-1", className="text-center fw-bold"), md=4),
        dbc.Col(html.Div(id="trade-total-2", className="text-center fw-bold"), md=4),
        dbc.Col(html.Div(id="trade-diff", className="text-center fw-bold"), md=4),
    ], className="mt-3 mb-4"),
], fluid=True)

# --------------------------------------------------------------
# Trade Analyzer Callbacks
# --------------------------------------------------------------

def _build_player_options_for_team(team_value):
    """Helper: return dropdown options + default values for a given team."""
    if not team_value:
        return [], []

    team_players = player_df[player_df["Team"] == team_value].copy()

    # Build display column if not already present
    if "Player_Display" not in team_players.columns:
        team_players["Player_Display"] = (
            team_players["Player_Name"] + " (" +
            team_players["Pos"] + ", " +
            team_players["Team"] + ")"
        )

    options = [
        {
            "label": row["Player_Display"],
            "value": row["Unique_ID"]
        }
        for _, row in team_players.iterrows()
    ]
    return options, []  # default no pre-selected players


@app.callback(
    [
        Output("trade-players1", "options"),
        Output("trade-players1", "value"),
        Output("trade-players2", "options"),
        Output("trade-players2", "value"),
    ],
    [
        Input("trade-team1", "value"),
        Input("trade-team2", "value"),
    ]
)
def update_trade_player_dropdowns(team1, team2):
    opts1, vals1 = _build_player_options_for_team(team1)
    opts2, vals2 = _build_player_options_for_team(team2)
    return opts1, vals1, opts2, vals2


@app.callback(
    [
        Output("trade-bar-chart", "figure"),
        Output("trade-total-1", "children"),
        Output("trade-total-2", "children"),
        Output("trade-diff", "children"),
        Output("trade-table1", "data"),
        Output("trade-table2", "data"),
    ],
    [
        Input("trade-players1", "value"),
        Input("trade-players2", "value"),
        Input("trade-team1", "value"),
        Input("trade-team2", "value"),
    ]
)
def update_trade_view(players1_ids, players2_ids, team1, team2):

    # Normalize inputs
    players1_ids = players1_ids or []
    players2_ids = players2_ids or []

    # Filter players
    df1 = player_df[player_df["Unique_ID"].isin(players1_ids)].copy()
    df2 = player_df[player_df["Unique_ID"].isin(players2_ids)].copy()

    # Ensure Player_Display exists
    for df in (df1, df2):
        if not df.empty and "Player_Display" not in df.columns:
            df["Player_Display"] = (
                df["Player_Name"] + " (" +
                df["Pos"] + ", " +
                df["Team"] + ")"
            )

    # Compute totals (always floats)
    total1 = float(df1["Predicted_AAV_M"].sum()) if not df1.empty else 0.0
    total2 = float(df2["Predicted_AAV_M"].sum()) if not df2.empty else 0.0
    diff = total1 - total2

    # Build stacked bar data safely
    bar_df_list = []
    if not df1.empty:
        t1name = team1 or "Team 1"
        temp1 = df1[["Player_Name", "Predicted_AAV_M"]].copy()
        temp1["Side"] = t1name
        bar_df_list.append(temp1)

    if not df2.empty:
        t2name = team2 or "Team 2"
        temp2 = df2[["Player_Name", "Predicted_AAV_M"]].copy()
        temp2["Side"] = t2name
        bar_df_list.append(temp2)

    # ALWAYS build a figure
    if bar_df_list:
        bar_df = pd.concat(bar_df_list, ignore_index=True)
        fig = px.bar(
            bar_df,
            x="Side",
            y="Predicted_AAV_M",
            color="Player_Name",
            barmode="stack",
            title="Stacked Predicted Salary Value by Side",
            labels={"Predicted_AAV_M": "Predicted AAV (M)"}
        )
    else:
        # SAFE fallback figure
        fig = px.bar(title="Select players to compare value")

    fig.update_layout(**CHART_LAYOUT)

    # Safe text outputs
    t1text = f"{team1 or 'Team 1'} Total: {total1:.2f} M"
    t2text = f"{team2 or 'Team 2'} Total: {total2:.2f} M"
    difftext = f"Difference (Team 1 - Team 2): {diff:+.2f} M"

    # Tables ALWAYS exist (empty okay)
    table1_data = df1[["Player_Display", "Pos", "Predicted_AAV_M"]].to_dict("records") if not df1.empty else []
    table2_data = df2[["Player_Display", "Pos", "Predicted_AAV_M"]].to_dict("records") if not df2.empty else []

    # RETURN ALL 6 OUTPUTS ALWAYS
    return fig, t1text, t2text, difftext, table1_data, table2_data





# --------------------------------------------------------------
# 9. Team Salary Efficiency Overview Tab + Standings Bubble View
# --------------------------------------------------------------
try:
    team_df = pd.read_csv(os.path.join(DATA_PATH, "player_predictions.csv"))
except FileNotFoundError:
    team_df = pd.DataFrame(columns=["Team", "AAV_M", "Predicted_AAV_M", "Residual_M", "Points_per_game"])

# ---- Fix incorrect / inconsistent team names before grouping ----
team_name_fixes = {
    "Winnepeg Jets": "Winnipeg Jets",
    "Las Vegas Golden Knights": "Vegas Golden Knights",
    "Loas Angles Kings": "Los Angeles Kings",
    "Seattle Kracken": "Seattle Kraken",
    "Toronto Maple Leafers": "Toronto Maple Leafs",
    "New York islanders": "New York Islanders",
    "Utah Hockey Club (Mammoth)": "Utah Hockey Club",
}

team_df["Team"] = team_df["Team"].replace(team_name_fixes)

if not team_df.empty:
    team_summary = (
        team_df.groupby("Team", as_index=False)
        .agg(
            Total_Spend_M=("AAV_M", "sum"),
            Overpay_M=("Residual_M", "sum"),
            Avg_Points_Per_Game=("Points_per_game", "mean")
        )
        .sort_values("Overpay_M", ascending=True)
    )
    team_summary["Efficiency_Ratio"] = (
        (team_summary["Total_Spend_M"] - team_summary["Overpay_M"]) / team_summary["Total_Spend_M"]
    )

    spend_data = team_summary.sort_values("Total_Spend_M", ascending=False)
    spend_fig = px.bar(
        spend_data,
        x="Team",
        y="Total_Spend_M",
        title="💰 Total Team Spending (Sallary Sum, Millions)",
        labels={"Total_Spend_M": "Team Salary (Millions)"},
        color="Total_Spend_M",
        color_continuous_scale="Blues",
        text=spend_data["Total_Spend_M"].round(1).astype(str) + "M"
    )
    overpay_fig = px.bar(
        team_summary.sort_values("Overpay_M", ascending=True),
        x="Team",
        y="Overpay_M",
        title="⚖️ Team Contract Efficiency (Negative = Underpaid Roster)",
        color="Overpay_M",
        labels={"Overpay_M": "Overpay / Underpay (Millions)"},
        color_continuous_scale="RdYlGn_r",
        text=team_summary["Overpay_M"].round(2).astype(str) + "M"
    )
    for fig in [spend_fig, overpay_fig]:
        fig.update_layout(**CHART_LAYOUT, height=550)
        fig.update_traces(textposition="outside", cliponaxis=False)
else:
    team_summary = pd.DataFrame()
    spend_fig = px.bar(title="No team data available")
    spend_fig.update_layout(**CHART_LAYOUT)
    overpay_fig = px.bar(title="No team data available")
    overpay_fig.update_layout(**CHART_LAYOUT)

# ---- NEW: Load team standings and build merged view for bubble chart ----
try:
    team_perf_df = pd.read_csv(os.path.join(DATA_PATH, "team", "team_standings_2025.csv"))
except FileNotFoundError:
    team_perf_df = pd.DataFrame()

if not team_summary.empty and not team_perf_df.empty:
    #print("\n===== DEBUG: UNIQUE TEAM VALUES FROM PLAYER DATA =====")
    #print(team_summary["Team"].unique())
    #print("Count:", len(team_summary["Team"].unique()))

    #print("\n===== DEBUG: UNIQUE TEAM ABBREVIATIONS FROM STANDINGS =====")
    #print(team_perf_df["abbrev"].unique())
    #print("Count:", len(team_perf_df["abbrev"].unique()))

    #print("\n===== DEBUG: FULL STANDINGS TEAM NAMES =====")
    #print(team_perf_df["team"].unique())

    # Try merge and inspect
    debug_merge = team_summary.merge(
        team_perf_df,
        left_on="Team",
        right_on="abbrev",
        how="left"
    )

    #print("\n===== DEBUG: MERGED SAMPLE (HEAD) =====")
    #print(debug_merge.head())

    #print("\n===== DEBUG: NULL VALUES BY COLUMN IN MERGE =====")
    #print(debug_merge.isna().sum())

    merged_team = team_summary.merge(
        team_perf_df,
        left_on="Team",      # salary data uses abbreviation in Team column
        right_on="team",   # standings uses abbrev column
        how="left"
    )
    merged_team["goalDifferential"] = merged_team["goalDifferential"].fillna(0)
else:
    merged_team = pd.DataFrame()

if not merged_team.empty:
    conference_values = sorted(merged_team["conferenceName"].dropna().unique())
    conference_options = (
        [{"label": "All Conferences", "value": "All"}] +
        [{"label": conf, "value": conf} for conf in conference_values]
    )
else:
    conference_options = [{"label": "All Conferences", "value": "All"}]

team_tab = dbc.Container([
    html.H2("🏒 Team Salary Efficiency Overview", className="fw-bold text-center my-4", style={"color": "#1f3c5b"}),
    html.P(
        "Total spending shows each team’s salary investment. The efficiency chart shows overpaid (positive) or underpaid (negative) rosters.",
        className="text-center text-muted mb-4",
        style={"fontSize": "1rem"}
    ),

    dbc.Row([dbc.Col(dcc.Graph(figure=spend_fig), md=12)]),
    html.Hr(),
    dbc.Row([dbc.Col(dcc.Graph(figure=overpay_fig), md=12)]),

    html.Hr(style={"marginTop": "2rem", "marginBottom": "2rem"}),

    html.H3("📊 Payroll vs On-Ice Results", className="fw-bold text-center mb-3", style={"color": "#1f3c5b"}),
    html.P(
        "Bubble chart compares total spend vs points. Bubble size reflects goal differential. "
        "Use the conference filter to focus East, West, or view all teams.",
        className="text-center text-muted mb-3",
        style={"fontSize": "0.95rem"}
    ),

    dbc.Row([
        dbc.Col([
            html.Label("Filter by Conference:", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="conference-filter",
                options=conference_options,
                value="All",
                clearable=False,
                style={"marginBottom": "1rem"}
            )
        ], md=4)
    ], justify="center"),

    dcc.Dropdown(
    id="playoff-toggle",
    options=[
        {"label": "All Teams", "value": "All"},
        {"label": "Playoff Teams Only", "value": "Playoff"}
    ],
    value="All",
    clearable=False,
    style={"marginBottom": "1rem", "width": "50%"}
),


    dbc.Row([
        dbc.Col(dcc.Graph(id="cap-eff-graph"), md=12)
    ]),
], fluid=True)


# Callback for team efficiency bubble chart
# Callback for bubble chart
@app.callback(
    Output("cap-eff-graph", "figure"),
    [Input("conference-filter", "value"),
     Input("playoff-toggle", "value")]
)
def update_cap_efficiency_chart(selected_conf, selected_group):
    if merged_team.empty:
        fig = px.scatter(title="No team + standings data available")
        fig.update_layout(**CHART_LAYOUT)
        return fig

    df = merged_team.copy()

    # --- PLAYOFF TEAM LIST ---
    playoff_teams = [
        "Winnipeg Jets", "Dallas Stars", "Vegas Golden Knights", "Edmonton Oilers",
        "St. Louis Blues", "Colorado Avalanche", "Minnesota Wild", "Los Angeles Kings",
        "Florida Panthers", "Toronto Maple Leafs", "Carolina Hurricanes",
        "Washington Capitals", "Tampa Bay Lightning", "Ottawa Senators",
        "New Jersey Devils", "Montréal Canadiens"
    ]

    # Filter to playoff teams if selected
    if selected_group == "Playoff":
        df = df[df["team"].isin(playoff_teams)]

    # Conference filter
    if selected_conf and selected_conf != "All":
        df = df[df["conferenceName"] == selected_conf]

    if df.empty:
        fig = px.scatter(title="No data for selected filters")
        fig.update_layout(**CHART_LAYOUT)
        return fig

    # Bubble size from magnitude of inefficiency
    df["BubbleSize"] = df["Overpay_M"].abs().replace(0, 0.25)

    # Color from direction of inefficiency
    df["SpendOutcome"] = df["Overpay_M"].apply(lambda x: "Underpaying" if x < 0 else "Overpaying")

    fig = px.scatter(
        df,
        x="Total_Spend_M",
        y="points",
        size="BubbleSize",
        labels={"Total_Spend_M": "Total Salary (Millions)"},
        color="SpendOutcome",
        hover_name="team",  # Full team name on hover
        text=None,  # Remove direct text labels
        color_discrete_map={"Underpaying": "#2ca02c", "Overpaying": "#d62728"},
        hover_data={
            "Total_Spend_M": ":.1f",  # Total payroll to 1 decimal
            "points": ":.0f",          # Team points as whole number
            "Overpay_M": ":.2f",       # Overpay amount
            "SpendOutcome": True,      # Show spending outcome
            "goalDifferential": ":.0f" # Goal differential 
        }
    )

    # Keep existing threshold lines and annotations
    fig.add_vline(
        x=78, line_dash="dot", line_color="gray", line_width=2,
        annotation_text="Lower Playoff Spend Threshold ($78M)",
        annotation_position="bottom left"
    )

    fig.add_vline(
        x=97, line_dash="dot", line_color="gray", line_width=2,
        annotation_text="Upper Playoff Spend Threshold ($97M)",
        annotation_position="bottom right"
    )

    fig.add_shape(
        type="line",
        x0=78, x1=97, y0=91, y1=91,
        line=dict(color="gray", width=2, dash="dot")
    )

    fig.add_annotation(
        x=(78+97)/2, y=91,
        text="Playoff Performance Threshold (91 Points)",
        showarrow=False,
        font=dict(color="gray", size=12)
    )

    fig.update_layout(**CHART_LAYOUT)
    return fig

# --------------------------------------------------------------
# 10. Sumary Tab
# --------------------------------------------------------------

def prepare_summary_data(player_df, merged_team, selected_team=None):
    """
    Returns top 5 overpaid and top 5 underpaid players.
    If selected_team is provided, results are filtered to that team.
    """

    player_df['Residual_M'] = pd.to_numeric(player_df['Residual_M'], errors='coerce')


    # Filter to selected team if provided
    if selected_team:
        df = player_df[player_df["Team"] == selected_team].copy()
    else:
        df = player_df.copy()

    df['Residual_M'] = pd.to_numeric(df['Residual_M'], errors='coerce')

    # Top 5 Overpaid
    top_overpaid = (
        df[df["Residual_M"] > 0]
        .nlargest(5, "Residual_M")[["Player_Name", "Team","Pos", "AAV_M", "Predicted_AAV_M", "Residual_M"]]
    )

    # Top 5 Underpaid
    top_underpaid = (
        df[df["Residual_M"] < 0]
        .nsmallest(5, "Residual_M")[["Player_Name", "Team", "Pos","AAV_M", "Predicted_AAV_M", "Residual_M"]]
    )

    # Playoff team salary stats remain unchanged
    playoff_teams = [
        "Winnipeg Jets", "Dallas Stars", "Vegas Golden Knights", "Edmonton Oilers",
        "St. Louis Blues", "Colorado Avalanche", "Minnesota Wild", "Los Angeles Kings",
        "Florida Panthers", "Toronto Maple Leafs", "Carolina Hurricanes",
        "Washington Capitals", "Tampa Bay Lightning", "Ottawa Senators",
        "New Jersey Devils", "Montréal Canadiens"
    ]

    playoff_team_salaries = merged_team[merged_team['team'].isin(playoff_teams)]['Total_Spend_M']
    
    team_salary_stats = {
        'Min Playoff Salary': playoff_team_salaries.min(),
        'Max Playoff Salary': playoff_team_salaries.max(),
        'Avg Playoff Salary': playoff_team_salaries.mean(),
        'Playoff Teams Count': len(playoff_teams)
    }

    return top_overpaid, top_underpaid, team_salary_stats

# Prepare the summary data
top_overpaid, top_underpaid, team_salary_stats = prepare_summary_data(player_df, merged_team)

# ------- DISPLAY FORMATTING FOR SUMMARY TABLES -------

display_overpaid = top_overpaid.rename(columns={
    "Player_Name": "Player",
    "Team": "Team",
    "Pos": "Position",
    "AAV_M": "Yearly Salary (M)",
    "Predicted_AAV_M": "Predicted Salary (M)",
    "Residual_M": "Difference (M)"
})

display_underpaid = top_underpaid.rename(columns={
    "Player_Name": "Player",
    "Team": "Team",
    "Pos": "Position",
    "AAV_M": "Yearly Salary (M)",
    "Predicted_AAV_M": "Predicted Salary (M)",
    "Residual_M": "Difference (M)"
})

# Round financial values to 2 decimals for display
for col in ["Yearly Salary (M)", "Predicted Salary (M)", "Difference (M)"]:
    display_overpaid[col] = display_overpaid[col].round(2)
    display_underpaid[col] = display_underpaid[col].round(2)


# Create Summary Tab
summary_tab = dbc.Container([
    html.H1("NHL Salary Analytics - Executive Summary", 
            className="fw-bold text-center my-4", 
            style={"color": "#1f3c5b"}),


    # Model Performance Section
    dbc.Row([
        dbc.Col([
            html.H3("Modeling Insights", className="text-center"),
            dbc.Card(
                dbc.CardBody([
                    html.P(f"Overall Model R²: {overall_r2:.3f}", className="card-text"),
                    html.P(f"Veteran Model R²: {veteran_r2:.3f}", className="card-text"),
                    html.P(f"Rookie Model R²: {rookie_r2:.3f}", className="card-text"),
                    html.P("Model explains 66% of salary variance across NHL players", 
                           className="card-text text-muted")
                ]),
                className="mb-4"
            )
        ], width=6),
        
        # Player Prediction Section
        dbc.Col([
            html.H3("Player Prediction Overview", className="text-center"),
            dbc.Card(
                dbc.CardBody([
                    html.P("Predicts player salaries based on:", className="card-text"),
                    html.Ul([
                        html.Li("Performance metrics"),
                        html.Li("Player position"),
                        html.Li("Game experience")
                    ]),
                    html.P("Identifies potential over/underpaid players", 
                           className="card-text text-muted")
                ]),
                className="mb-4"
            )
        ], width=6)
    ]),

    # Team Salary Analysis
    dbc.Row([
        dbc.Col([
            html.H3("Playoff Team Salary Insights", className="text-center"),
            dbc.Card(
                dbc.CardBody([
                    html.P(f"Minimum Playoff Team Salary: ${team_salary_stats['Min Playoff Salary']:.2f}M", className="card-text"),
                    html.P(f"Maximum Playoff Team Salary: ${team_salary_stats['Max Playoff Salary']:.2f}M", className="card-text"),
                    html.P(f"Average Playoff Team Salary: ${team_salary_stats['Avg Playoff Salary']:.2f}M", className="card-text"),
                    html.P(f"Playoff Teams: {team_salary_stats['Playoff Teams Count']}", className="card-text"),
                ]),
                className="mb-4"
            )
        ], width=12)
    ]),

    # Top Overpaid/Underpaid Players
    html.Div([
        html.Label("Select Team:", style={"fontWeight": "bold", "fontSize": "1rem"}),
        dcc.Dropdown(
            id="summary-team-filter",
            options=[{"label": team, "value": team} for team in sorted(player_df["Team"].unique())],
            value=None,
            placeholder="Select a Team",
            clearable=True,
            style={"marginBottom": "1.5rem", "width": "60%"}
        )
    ], style={"textAlign": "center"}),

    dbc.Row([
        dbc.Col([
            html.H3("Top Overpaid Players", className="text-center"),
            dash_table.DataTable(
    id="display_overpaid",
    data=display_overpaid.to_dict('records'),
    columns=[{"name": i, "id": i, "presentation": "markdown"} for i in display_overpaid.columns],
    style_table={
        'overflowX': 'auto',
        'minWidth': '100%',
        'border': 'none'
    },
    style_header={
        'backgroundColor': '#1f3c5b',
        'fontWeight': 'bold',
        'color': 'white',
        'whiteSpace': 'normal',
        'height': 'auto',
        'textAlign': 'center',
        'padding': '8px',
        'border': 'none'
    },
    style_cell={
        'textAlign': 'left',
        'padding': '6px 10px',
        'fontSize': '14px',
        'whiteSpace': 'normal',
        'height': 'auto',
        'border': 'none'
    },
    style_data={
        'border': 'none'
    },
    style_data_conditional=[
        {'if': {'row_index': 'odd'}, 'backgroundColor': '#f7f9fc'},
    ],
)


        ], width=6),
        dbc.Col([
            html.H3("Top Underpaid Players", className="text-center"),
            dash_table.DataTable(
    id="display_underpaid",
    data=display_underpaid.to_dict('records'),
    columns=[{"name": i, "id": i, "presentation": "markdown"} for i in display_underpaid.columns],
    style_table={
        'overflowX': 'auto',
        'minWidth': '100%',
        'border': 'none'
    },
    style_header={
        'backgroundColor': '#1f3c5b',
        'fontWeight': 'bold',
        'color': 'white',
        'whiteSpace': 'normal',
        'height': 'auto',
        'textAlign': 'center',
        'padding': '8px',
        'border': 'none'
    },
    style_cell={
        'textAlign': 'left',
        'padding': '6px 10px',
        'fontSize': '14px',
        'whiteSpace': 'normal',
        'height': 'auto',
        'border': 'none'
    },
    style_data={
        'border': 'none'
    },
    style_data_conditional=[
        {'if': {'row_index': 'odd'}, 'backgroundColor': '#f7f9fc'},
    ],
)

        ], width=6)
    ]),

], fluid=True)


@app.callback(
    [Output("display_overpaid", "data"),
     Output("display_underpaid", "data")],
    [Input("summary-team-filter", "value")]
)
def update_summary_tables(selected_team):
    top_overpaid, top_underpaid, _ = prepare_summary_data(player_df, merged_team, selected_team)

    top_overpaid = top_overpaid.rename(columns={
    "Player_Name": "Player",
    "Team": "Team",
    "Pos": "Position",
    "AAV_M": "Yearly Salary (M)",
    "Predicted_AAV_M": "Predicted Salary (M)",
    "Residual_M": "Difference (M)"
    }).round(2)

    top_underpaid = top_underpaid.rename(columns={
    "Player_Name": "Player",
    "Team": "Team",
    "Pos": "Position",
    "AAV_M": "Yearly Salary (M)",
    "Predicted_AAV_M": "Predicted Salary (M)",
    "Residual_M": "Difference (M)"
    }).round(2)


    return top_overpaid.to_dict("records"), top_underpaid.to_dict("records")



# --------------------------------------------------------------
# 10. App Layout
# --------------------------------------------------------------
app.layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab(label="Summary", children=[summary_tab], activeTabClassName="fw-bold"),
        dbc.Tab(label="Model Overview", children=[overview_tab], activeTabClassName="fw-bold"),
        dbc.Tab(label="Player Predictions", children=[player_tab], activeTabClassName="fw-bold"),
        dbc.Tab(label="Underpaid Players", children=[tab3], activeTabClassName="fw-bold"),
        dbc.Tab(label="Team Analysis", children=[team_tab], activeTabClassName="fw-bold"),
        dbc.Tab(label="Trade Analyzer", children=[trade_tab], activeTabClassName="fw-bold"),
    ], className="mb-3"),

    html.Footer([
        html.Hr(),
        html.P("Data from NHL 2024–25 | Dashboard by Tim Conser",
               className="text-center text-muted",
               style={"fontSize": "0.9rem", "marginTop": "1rem"})
    ])
], fluid=True, style={"backgroundColor": "#f8f9fa", "padding": "2rem"})


# --------------------------------------------------------------
# 11. Run App
# --------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)


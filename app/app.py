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

@app.callback(
    [Output("predictions-scatter", "figure"),
     Output("r2-display", "children")],
    [Input("position-filter", "value")]
)
def update_scatter(selected_pos):
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
        hover_data=["Player_Name", "Team"],
        title="Actual vs Predicted Player Salaries (Millions)",
        labels={"AAV_M": "Actual AAV (M)", "Predicted_AAV_M": "Predicted AAV (M)"}
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

player_df = (
    player_df.sort_values("Residual_M", ascending=False)
    .groupby(["Player_Name", "Pos"], as_index=False)
    .first()
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
        text=pos_df["AAV_M"].round(2).astype(str) + "M",
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
        title="💰 Total Team Spending (AAV Sum, Millions)",
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
    print("\n===== DEBUG: UNIQUE TEAM VALUES FROM PLAYER DATA =====")
    print(team_summary["Team"].unique())
    print("Count:", len(team_summary["Team"].unique()))

    print("\n===== DEBUG: UNIQUE TEAM ABBREVIATIONS FROM STANDINGS =====")
    print(team_perf_df["abbrev"].unique())
    print("Count:", len(team_perf_df["abbrev"].unique()))

    print("\n===== DEBUG: FULL STANDINGS TEAM NAMES =====")
    print(team_perf_df["team"].unique())

    # Try merge and inspect
    debug_merge = team_summary.merge(
        team_perf_df,
        left_on="Team",
        right_on="abbrev",
        how="left"
    )

    print("\n===== DEBUG: MERGED SAMPLE (HEAD) =====")
    print(debug_merge.head())

    print("\n===== DEBUG: NULL VALUES BY COLUMN IN MERGE =====")
    print(debug_merge.isna().sum())

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

    dbc.Row([
        dbc.Col(dcc.Graph(id="cap-eff-graph"), md=12)
    ]),
], fluid=True)

# Callback for bubble chart
@app.callback(
    Output("cap-eff-graph", "figure"),
    [Input("conference-filter", "value")]
)
def update_cap_efficiency_chart(selected_conf):
    if merged_team.empty:
        fig = px.scatter(title="No team + standings data available")
        fig.update_layout(**CHART_LAYOUT)
        return fig

    df = merged_team.copy()
    if selected_conf and selected_conf != "All":
        df = df[df["conferenceName"] == selected_conf]

    if df.empty:
        fig = px.scatter(title="No data for selected conference")
        fig.update_layout(**CHART_LAYOUT)
        return fig

    df["goalDifferential"] = df["goalDifferential"].fillna(0)

    # Compute efficiency ratio based on your predictive salary modeling
    df["Efficiency_Ratio"] = (df["Total_Spend_M"] - df["Overpay_M"]) / df["Total_Spend_M"]

    # Bubble size = Points ROI on spending (Points per $1M spent)
    df["BubbleSize"] = (df["points"] / df["Total_Spend_M"])

    # FIX: replace NaN or zero values with a minimum marker size
    df["BubbleSize"] = df["BubbleSize"].replace([None, float("nan")], 0).astype(float)
    df["BubbleSize"] = df["BubbleSize"].apply(lambda x: 1 if x <= 0 else x)

    # Positive vs negative goal differential coloring
    df["GoalTrend"] = df["goalDifferential"].apply(lambda x: "Positive" if x >= 0 else "Negative")

    fig = px.scatter(
        df,
        x="Efficiency_Ratio",
        y="points",
        size="BubbleSize",
        color="GoalTrend",
        hover_name="team",
        text="Team",
        title="📈 Efficiency Ratio vs Points — Bubble Size = Goal Differential",
        labels={
            "Efficiency_Ratio": "Team Efficiency Ratio",
            "points": "Total Points",
            "GoalTrend": "Goal Differential",
        },
        color_discrete_map={"Positive": "#2ca02c", "Negative": "#d62728"}
    )

    fig.update_traces(textposition="top center")
    fig.update_layout(**CHART_LAYOUT)
    return fig





# --------------------------------------------------------------
# 10. App Layout
# --------------------------------------------------------------
app.layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab(label="Model Overview", children=[overview_tab], activeTabClassName="fw-bold"),
        dbc.Tab(label="Player Predictions", children=[player_tab], activeTabClassName="fw-bold"),
        dbc.Tab(label="Underpaid Players", children=[tab3], activeTabClassName="fw-bold"),
        dbc.Tab(label="Team Analysis", children=[team_tab], activeTabClassName="fw-bold"),
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

print(player_preds_df["Team"].unique())


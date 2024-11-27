import dash
from dash import dcc, html
import pandas as pd


# Load datasets
wide_df = pd.read_pickle("data_under_experiment\\pickle\wide\Wide_Dataframe_22112024_18_05.pickle")
long_df = pd.read_pickle("data_under_experiment\\pickle\wide\Long_Dataframe_22112024_18_05.pickle")
eval_df = pd.read_pickle("saved_dataframes\\pickel\\Evaluation_Wide_Dataset_23_11_2024 17_48_23112024_17_48.pickle")

app = dash.Dash(__name__)

app.layout = html.Div(
    style={"backgroundColor": "lightgrey", "padding": "20px"},
    children=[
    html.H1(
        "Study Project: Estimating Olive Yield by Climate Factors in Spain",
        style={"textAlign": "center", "color": "black"},),
    dcc.Dropdown(
        id="dataset-dropdown",
        options=[
            {"label": "Wide Dataset", "value": "wide"},
            {"label": "Long Dataset", "value": "long"},
            {"label": "Evaluation Dataset", "value": "evaluation"}
        ],
        placeholder="Select a dataset",
        style={"width": "50%", "margin": "auto"},
    ),
    html.Div(id="selected-dataset", style={"marginTop": "20px"})
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
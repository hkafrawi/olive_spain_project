import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from data_transformer import DataTransformer as dt
import pandas as pd


# Load datasets

wide_df = dt.prepare_dataset_for_regression(
    data=pd.read_pickle("data_under_experiment\\pickle\wide\Wide_Dataframe_22112024_18_05.pickle"), descr="wide")
long_df = dt.prepare_dataset_for_regression(
    data=pd.read_pickle("data_under_experiment\\pickle\wide\Long_Dataframe_22112024_18_05.pickle"), descr= "long")
eval_df = dt.prepare_dataset_for_regression(
    data=pd.read_pickle("saved_dataframes\\pickel\\Evaluation_Wide_Dataset_23_11_2024 17_48_23112024_17_48.pickle"), descr="eval")

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

@app.callback(
    Output("selected-dataset", "children"),
    Input("dataset-dropdown", "value")
)
def update_table(selected_dataset):
    if not selected_dataset:
        return html.Div("Please select a dataset from the dropdown.")

    # Map selection to dataset
    if selected_dataset == "wide":
        df = wide_df
    elif selected_dataset == "long":
        df = long_df
    elif selected_dataset == "evaluation":
        df = eval_df
    else:
        return html.Div("Invalid selection.")

    # Create a DataTable
    return dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in df.columns],
        data=df.head(10).to_dict("records"),  # Display only the first 10 rows for brevity
        style_table={"overflowX": "auto"},  # Handle horizontal scrolling for wide tables
        style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
        style_cell={"textAlign": "left", "padding": "10px"},
    )


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from data_transformer import DataTransformer as dt
import plotly.express as px
import pandas as pd


# Load datasets

wide_df = dt.prepare_dataset_for_regression(
    data=pd.read_pickle("data_under_experiment\\pickle\wide\Wide_Dataframe_22112024_18_05.pickle"), descr="wide")
long_df = dt.prepare_dataset_for_regression(
    data=pd.read_pickle("data_under_experiment\\pickle\wide\Long_Dataframe_22112024_18_05.pickle"), descr= "long")
eval_df = dt.prepare_dataset_for_regression(
    data=pd.read_pickle("saved_dataframes\\pickel\\Evaluation_Wide_Dataset_23_11_2024 17_48_23112024_17_48.pickle"), descr="eval")

datasets = {
    "wide": wide_df,
    "long": long_df,
    "evaluation": eval_df
}

app = dash.Dash(__name__)

default_dataset = "wide"
default_features = [col for col in wide_df.columns if col not in ["Year", "Province"]]

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
        value=default_dataset,
        placeholder="Select a dataset",
        style={"width": "50%", "margin": "auto"},
    ),
    html.Div(id="selected-dataset", style={"marginTop": "20px"}),
    html.Div([
        html.H3("Boxplot Filter"),
        dcc.Dropdown(
            id="boxplot-filter",
            options=[{"label": col, "value": col} for col in default_features],
            value=default_features,
            multi=True,
            placeholder="Select features for the boxplot",
        ),
    ]),
    dcc.Graph(id="boxplot", style={"backgroundColor": "lightgrey"}),
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

@app.callback(
    Output("boxplot-filter", "options"),
    [Input("dataset-dropdown", "value")]
)
def update_filter_options(selected_dataset):
    """Update filter options based on the selected dataset."""
    dataset = datasets[selected_dataset]
    options = [{"label": col, "value": col} for col in dataset.columns if col not in ["Year", "Province", "Model", "Params"]]
    return options


@app.callback(
    Output("boxplot", "figure"),
    [Input("dataset-dropdown", "value"),
     Input("boxplot-filter", "value")]
)
def update_boxplot(selected_dataset, selected_features):
    """Update boxplot based on selected dataset and features."""
    dataset = datasets[selected_dataset]

    if not selected_features:
        selected_features = [col for col in dataset.columns if col not in ["Year", "Province", "Model", "Params"]]

    # Filter dataset to selected features
    filtered_data = dataset[selected_features]

    # Create the boxplot
    fig = px.box(
        filtered_data,
        title=f"Boxplot for {selected_dataset.capitalize()} Dataset",
    )
    fig.update_layout(
        title={"x": 0.5},
        paper_bgcolor="lightgrey",
        plot_bgcolor="white",
        xaxis_title="Features",
        yaxis_title="Values",
    )
    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
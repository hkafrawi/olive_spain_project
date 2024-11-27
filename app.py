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
    "long": long_df
}

app = dash.Dash(__name__)

default_dataset = "wide"
default_features = [col for col in wide_df.columns if col not in ["Year", "Province"]]
default_feature = default_features[0] if default_features else None

app.layout = html.Div(
    style={"backgroundColor": "lightgrey", "padding": "20px"},
    children=[
        # Header
        html.H1(
            "Study Project: Estimating Olive Yield by Climate Factors in Spain",
            style={"textAlign": "center", "color": "black"},
        ),

        # Dataset Selector
        dcc.Dropdown(
            id="dataset-dropdown",
            options=[
                {"label": "Wide Dataset", "value": "wide"},
                {"label": "Long Dataset", "value": "long"}
            ],
            value=default_dataset,
            placeholder="Select a dataset",
            style={"width": "50%", "margin": "auto"},
        ),

        # Top layout: Table View and Boxplot
        html.Div([
            html.Div([
                html.H3("Table View"),
                dash_table.DataTable(
                    id="table-view",
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left"},
                ),
            ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),

            html.Div([
                html.H3("Boxplot"),
                dcc.Dropdown(
                    id="boxplot-filter",
                    multi=True,
                    placeholder="Select features for the boxplot"
                ),
                dcc.Graph(
                    id="boxplot",
                    style={"backgroundColor": "lightgrey", "height": "500px"}
                )
            ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),
        ], style={"display": "flex", "justifyContent": "space-between"}),

        # Bottom layout: Scatter Plot
        html.Div([
            html.Div([
                html.Label("X-axis:"),
                dcc.Dropdown(id="scatter-x-axis"),
            ], style={"width": "48%", "display": "inline-block", "padding": "10px"}),

            html.Div([
                html.Label("Y-axis:"),
                dcc.Dropdown(id="scatter-y-axis"),
            ], style={"width": "48%", "display": "inline-block", "padding": "10px"}),

            dcc.Graph(
                id="scatter-plot",
                style={"backgroundColor": "lightgrey", "height": "500px"}
            )
        ], style={"marginTop": "30px"}),
         html.Div([
            html.H3("Correlation Heatmap"),
            dcc.Dropdown(
                id="heatmap-feature-selector",
                multi=True,
                placeholder="Select features for the heatmap",
                style={"width": "70%"},
            ),
            html.Button(
                "Generate Heatmap",
                id="generate-heatmap-button",
                style={"margin": "10px"}
            ),
            dcc.Graph(
                id="heatmap-graph",
                style={"backgroundColor": "lightgrey", "height": "500px"}
            ),
        ], style={"padding": "20px", "marginTop": "20px", "border": "1px solid black"})
    ]
)

# Callbacks
@app.callback(
    Output("table-view", "data"),
    [Input("dataset-dropdown", "value")]
)
def update_table(selected_dataset):
    """Update the table view with the first 10 rows of the selected dataset."""
    dataset = datasets[selected_dataset]
    return dataset.head(10).to_dict("records")

@app.callback(
    Output("boxplot-filter", "options"),
    [Input("dataset-dropdown", "value")]
)
def update_boxplot_filter(selected_dataset):
    """Update boxplot filter options based on the selected dataset."""
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

@app.callback(
    [Output("scatter-x-axis", "options"),
     Output("scatter-y-axis", "options")],
    [Input("dataset-dropdown", "value")]
)
def update_scatter_dropdowns(selected_dataset):
    """Update scatter plot dropdown options based on the selected dataset."""
    dataset = datasets[selected_dataset]
    options = [{"label": col, "value": col} for col in dataset.columns if col not in ["Year", "Province"]]
    return options, options

@app.callback(
    Output("scatter-plot", "figure"),
    [Input("dataset-dropdown", "value"),
     Input("scatter-x-axis", "value"),
     Input("scatter-y-axis", "value")]
)
def update_scatter_plot(selected_dataset, x_axis, y_axis):
    """Update scatter plot based on selected dataset and axes."""
    dataset = datasets[selected_dataset]

    # Validate selected columns
    if not x_axis or not y_axis or x_axis not in dataset.columns or y_axis not in dataset.columns:
        return px.scatter(title="Please select valid variables for X and Y axes")

    # Create scatter plot
    fig = px.scatter(
        dataset, x=x_axis, y=y_axis,
        title=f"Scatter Plot: {x_axis} vs {y_axis}"
    )
    fig.update_layout(
        title={"x": 0.5},
        paper_bgcolor="lightgrey",
        plot_bgcolor="white",
        xaxis_title=x_axis,
        yaxis_title=y_axis,
    )
    return fig

# Callback to populate dropdown options for numeric columns
@app.callback(
    Output("heatmap-feature-selector", "options"),
    Input("dataset-dropdown", "value")
)
def update_heatmap_feature_selector(selected_dataset):
    """Update dropdown with numeric columns for correlation heatmap."""
    dataset = datasets[selected_dataset]
    numeric_cols = dataset.select_dtypes(include=["number"]).columns.tolist()
    for col in dataset.columns:
        if col not in numeric_cols:
            try:
                dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
                if pd.api.types.is_numeric_dtype(dataset[col]):
                    numeric_cols.append(col)
            except ValueError:
                continue
    options = [{"label": col, "value": col} for col in numeric_cols]
    return options

# Callback to generate heatmap
@app.callback(
    Output("heatmap-graph", "figure"),
    [Input("generate-heatmap-button", "n_clicks"),
     Input("dataset-dropdown", "value"),
     Input("heatmap-feature-selector", "value")]
)
def generate_heatmap(n_clicks, selected_dataset, selected_features):
    """Generate and display correlation heatmap."""
    if not n_clicks:
        return px.imshow(
            [[None]], 
            labels={"color": "Correlation"},
            title="Select features and click 'Generate Heatmap'",
            template="simple_white"
        )

    # Validate selected features
    dataset = datasets[selected_dataset]
    if not selected_features or not set(selected_features).issubset(dataset.columns):
        return px.imshow(
            [[None]], 
            labels={"color": "Correlation"},
            title="Invalid or no features selected",
            template="simple_white"
        )

    # Compute correlation matrix
    corr_matrix = dataset[selected_features].corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        labels=dict(color="Correlation"),
        title="Correlation Heatmap",
    )
    fig.update_layout(
        paper_bgcolor="lightgrey",
        plot_bgcolor="white",
    )
    return fig
# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
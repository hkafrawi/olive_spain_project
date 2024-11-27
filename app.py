import dash
from dash import html

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Study Project: Estimating Olive Yield by Climate Factors in Spain"),
    html.P("Use the dropdown to select a dataset.")
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import pandas as pd
import json

# Load data
df = pd.read_csv('Internet Usage by Individuals.csv')
json_file = 'Countries GeoJSON Mini Project.geojson'
with open(json_file, 'r') as f:
    geojson = json.load(f)

df_countries = df[df['Code'].notnull()]
df_categories = df[df['Code'].isnull()]

# Initialize app
app = Dash(__name__)
app.title = 'Internet Usage Dashboard'

# App layout
app.layout = html.Div([
    html.H1('Internet Usage by Country (1990 - Present)'),
    html.Hr(),
    html.H2('Line Plot of Internet Usage for Selected Countries and Categories'),
    html.Hr(),
    html.Div([
        dcc.Dropdown(
            id='country-dropdown',
            options=[{'label': c, 'value': c} for c in sorted(df_countries['Entity'].unique())],
            multi=True,
            placeholder='Select countries',
            style={'width': '400px'}
        ),
        dcc.Dropdown(
            id='continent-dropdown',
            options=[{'label': c, 'value': c} for c in sorted(df_categories['Entity'].unique())],
            multi=True,
            placeholder='Select categories (continents, etc.)',
            style={'width': '400px'}
        ),
    ], style={'display': 'flex', 'gap': '10px'}),
    dcc.Graph(id='line-plot'),
    html.Hr(),
    html.H2('Choropleth Map of Internet Usage'),
    html.Hr(),
    html.Div([
        dcc.Graph(id='map-plot'),
        html.H4("Drag to select different years"),
        html.Div(id='year-display', style={'textAlign': 'center', 'fontSize': 20}),
        dcc.Slider(
            id='year-slider',
            min=df['Year'].min(),
            max=df['Year'].max(),
            step=1,
            value=df['Year'].min(),
            marks={str(year): str(year) for year in range(df['Year'].min(), df['Year'].max()+1, 10)}
        ),
        html.Div(id='slider-output', style={'marginTop': 20}),
    ])
])

@app.callback(
    Output('line-plot', 'figure'),
    [Input('country-dropdown', 'value'),
     Input('continent-dropdown', 'value')]
)
def update_line_plot(selected_countries, selected_continents):
    if not selected_countries:
        selected_countries = []
    if not selected_continents:
        selected_continents = []
    if not selected_countries and not selected_continents:
        selected_countries = sorted(df_countries['Entity'].unique())
        selected_continents = sorted(df_categories['Entity'].unique())
    filtered = df[df['Entity'].isin(selected_countries+selected_continents)]
    fig = px.line(
        filtered,
        x='Year',
        y='Individuals using the Internet (% of population)',
        color='Entity',
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig.update_layout(width=1200, height=800)
    return fig

@app.callback(
    Output('map-plot', 'figure'),
    Input('year-slider', 'value')
)
def update_map(year):
    filtered = df_countries[df_countries['Year'] == year]
    fig = px.choropleth(
        filtered,
        locations='Code',
        geojson=geojson,
        featureidkey='properties.ISO_A3',
        color='Individuals using the Internet (% of population)',
        hover_name='Entity',
        color_continuous_scale='YlGnBu',
        projection='natural earth'
    )
    fig.update_layout(width=1200, height=800, coloraxis_colorbar=dict(title='Internet Usage (%)'))
    fig.update_geos(fitbounds="locations", visible=False)
    return fig

@app.callback(
    Output('year-display', 'children'),
    Input('year-slider', 'value')
)
def display_selected_year(year):
    return f"Year: {year}"

@app.callback(
    Output('slider-output', 'children'),
    Input('year-slider', 'value')
)
def update_output(value):
    return f"Selected year: {value}"

if __name__ == "__main__":
    app.run_server(debug=True, host='127.0.0.1')

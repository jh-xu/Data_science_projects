import streamlit as st
import pandas as pd
import plotly.express as px
import json
st.set_page_config(layout="wide")

# Load data (replace with your actual file paths or data loading logic)
df = pd.read_csv('Internet Usage by Individuals.csv')
json_file = 'Countries GeoJSON Mini Project.geojson'
with open(json_file, 'r') as f:
    geojson = json.load(f)

# Split data
df_countries = df[df['Code'].notnull()]
df_categories = df[df['Code'].isnull()]

# Streamlit layout
st.title('Internet Usage by Country (1960 - Present)')
st.markdown("---")
st.header("Line Plot of Internet Usage for Selected Countries and Categories")
st.markdown("---")

# Country and continent selection
cols = st.columns(2)
with cols[0]:
    selected_countries = st.multiselect(
        "Select countries",
        sorted(df_countries['Entity'].unique())
    )
with cols[1]:
    selected_continents = st.multiselect(
        "Select categories (continents, etc.)",
        sorted(df_categories['Entity'].unique())
    )

# Filter and plot line chart
if not selected_countries and not selected_continents:
    selected_countries = sorted(df_countries['Entity'].unique())
    selected_continents = sorted(df_categories['Entity'].unique())

filtered_df = df[df['Entity'].isin(selected_countries + selected_continents)]
line_fig = px.line(
    filtered_df,
    x='Year',
    y='Individuals using the Internet (% of population)',
    color='Entity',
    title='Internet Usage Over Time'
)
line_fig.update_layout(
    width=1200,  # in pixels
    height=800,
    xaxis_title='Year',
    yaxis_title='Internet Usage (% of population)',
    legend_title='Country/Category'
)
st.plotly_chart(line_fig, use_container_width=True)

# Map section
st.markdown("---")
st.header("Choropleth Map of Internet Usage")
st.markdown("---")

# Year slider
year = st.slider(
    'Drag to select different years',
    min_value=int(df['Year'].min()),
    max_value=int(df['Year'].max()),
    step=1,
    value=int(df['Year'].min())
)
st.markdown(f"### Selected Year: {year}")

# Map plot
filtered_map = df_countries[df_countries['Year'] == year]
map_fig = px.choropleth(
    filtered_map,
    locations='Code',
    geojson=geojson,
    featureidkey='properties.ISO_A3',
    color='Individuals using the Internet (% of population)',
    hover_name='Entity',
    color_continuous_scale='YlGnBu',
    projection='natural earth',
    title=f"Internet Usage in {year}"
)
# Set colorbar title
map_fig.update_layout(
    width=1400,  # in pixels
    height=1000,
    coloraxis_colorbar=dict(
        title='Internet Usage (%)'
    )
)

map_fig.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(map_fig, use_container_width=True)

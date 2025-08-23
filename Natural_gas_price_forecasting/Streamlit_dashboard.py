import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

st.set_page_config(page_title="LNG Price Forecast & Netback", layout="wide")

# --- Constants and Utility Functions ---
UNIT_CONVERSION = 0.29307107  # €/MWh to $/MMBtu

INDEX_REF_DICT = {
    "TTF": "Prediction_TTF",
    "PVB": "Prediction_PVB",
    "HH": "Prediction_HH($MMBtu)"
}
COMPONENTS = ["Total Regas", "Fuel_loss from US", "Route Cost from US"] # Cost breakdown components plot
COMPONENTS_ALL = ["Total Regas", "Fuel_loss from US", "Route Cost from US", "Total Freight From US", "Cargo Cost US origin",
                  "Benchmark_Price", "Netback", "Net Profit"]  # All components including netback and cargo cost

def get_benchmark_price(row):
    ref_price = row.get('reference benchmark price', None)
    if pd.isna(ref_price) or ref_price not in INDEX_REF_DICT:
        return np.nan
    row_col = INDEX_REF_DICT[ref_price]
    return row[row_col]

@st.cache_data
def load_data():
    df = pd.read_csv("destination_netback_us_origin.csv")
    df = df.dropna(subset=["Period", "Country", "Terminal"])
    df["Period"] = pd.to_datetime(df["Period"])
    df["Month"] = df["Period"].dt.strftime("%b %Y")
    # Interpolate missing values for prices!!!!
    # Currently interpolate is wrong because of the stacking of repeated periods
    st.warning("Interpolating missing values for the forecasted prices if needed!")
    df["Prediction_TTF"] = df["Prediction_TTF(\u20ac/MWh)"].interpolate() * UNIT_CONVERSION
    df["Prediction_PVB"] = df["Prediction_PVB(\u20ac/MWh)"].interpolate() * UNIT_CONVERSION
    df["Benchmark_Price"] = df.apply(get_benchmark_price, axis=1)
    df["Netback"] = df["Benchmark_Price"] + df["Total Regas"] + df["Total Freight From US"]
    df["Net Profit"] = df["Netback"] - df["Cargo Cost US origin"]
    df["Period"] = df["Period"].dt.strftime("%b %Y")  # For display and filtering
    return df

@st.cache_data
def load_forecast_data():
    forecast = pd.read_csv("forecast.csv")
    forecast["Date"] = pd.to_datetime(forecast["Date"]).dt.strftime("%b %Y")
    return forecast

# --- Load and Prepare Data ---
df = load_data()
df_k = load_forecast_data()

# --- UI: Filters ---
st.title("Liquefied Natural Gas Netback Calculation")
st.subheader("Filter by Country, Terminal, and Month")

cols = st.columns(3)
countries = df["Country"].dropna().unique()
country = cols[0].selectbox("Select Country", countries)

terminals = df[df["Country"] == country]["Terminal"].dropna().unique()
terminal = cols[1].selectbox("Select Terminal", terminals)

months = df[(df["Country"] == country) & (df["Terminal"] == terminal)]["Period"].dropna().unique()
month = cols[2].selectbox("Select Period of Delivery", months)

filtered = df[(df["Country"] == country) & (df["Terminal"] == terminal) & (df["Period"] == month)]
if filtered.empty:
    st.warning("No data for this selection.")
    st.stop()
data = filtered.iloc[0]
benchmark_col = INDEX_REF_DICT[data["reference benchmark price"]]

# --- Editable Inputs ---
st.markdown("---")
st.info("Adjust the values below to see how they affect the netback and profit calculations.")

inputs = {}
col_right, col_left, _ = st.columns([2, 1, 1])
with col_left:
    inputs["Adjust on BM price"] = st.number_input("Adjusted on BM price", 
                     value=0.0, step=0.01, key="Adjust on BM price", 
                     help="Adjust the benchmark price reference")
    inputs["Adjust on Regas Fee"] = st.number_input("Adjust on Regas Fee", 
                     value=0.0, step=0.01, key="Adjust on Regas Fee", 
                     help="Adjust the regasification fee")
    inputs["Adjust on Fuel Loss"] = st.number_input("Adjust on Fuel Loss", 
                     value=0.0, step=0.01, key="Adjust on Fuel Loss", 
                     help="Adjust the fuel loss from US origin")
    inputs["Adjust on Freight Cost"] = st.number_input("Adjust on Freight Cost", 
                     value=0.0, step=0.01, key="Adjust on Freight Cost", 
                     help="Adjust the freight cost from US origin")
    inputs["Adjust on Cargo Cost"] = st.number_input("Adjust on Cargo Cost", 
                     value=0.0, step=0.01, key="Adjust on Cargo Cost", 
                     help="Adjust the cargo cost based on Henry Hub price")
with col_right:
    # --- Calculate Revenue and Costs ---
    benchmark_price_ref = data['reference benchmark price']
    benchmark_price = data["Benchmark_Price"]
    benchmark_price_adjust = inputs["Adjust on BM price"]
    revenue = benchmark_price + benchmark_price_adjust

    regas_fee_adj = inputs["Adjust on Regas Fee"]
    regas_fee = data["Total Regas"] + regas_fee_adj

    fuel_loss_adj = inputs["Adjust on Fuel Loss"]
    fuel_loss = data["Fuel_loss from US"] + fuel_loss_adj

    freight_adj = inputs["Adjust on Freight Cost"]
    freight = data["Total Freight From US"] + freight_adj

    total_freight = fuel_loss + freight

    benchmark_hh = data["Prediction_HH($MMBtu)"]
    adjust_on_cargo_price = inputs["Adjust on Cargo Cost"]
    cargo_cost = benchmark_hh + adjust_on_cargo_price

    netback = revenue + regas_fee + total_freight
    net_profit = netback - cargo_cost

    # --- Revenue and Cost Breakdown Table ---
    table_data = [
        ["Terminal Name", terminal],
        ["Revenue", f"{revenue:.2f}"],
        ["Benchmark Price Reference", benchmark_price_ref],
        ["Benchmark Price", f"{benchmark_price:.2f}"],
        ["Adj. on price", f"{benchmark_price_adjust:.2f}"],
        ["Total Regas", f"{regas_fee:.2f}"],
        ["Regas fee", f"{data['Total Regas']:.2f}"],
        ["Adj. on regas fee", f"{regas_fee_adj:.2f}"],
        ["Total Freight", f"{total_freight:.2f}"],
        ["Fuel Loss", f"{fuel_loss:.2f}"],
        ["Freight Cost", f"{freight:.2f}"],
        ["Adj. on Fuel Loss", f"{fuel_loss_adj:.2f}"],
        ["Adj. on Freight Cost", f"{freight_adj:.2f}"],
        ["Netback", f"{netback:.2f}"],
        ["Cargo Cost", f"{cargo_cost:.2f}"],
        ["Benchmark price (HH)", f"{benchmark_hh:.2f}"],
        ["Adj. on cargo prices", f"{adjust_on_cargo_price:.2f}"],
        ["Net profit", f"{net_profit:.2f}"],
    ]
    df_table = pd.DataFrame(table_data, columns=["Item", "Value"])

    def highlight_key_rows(row):
        bold_rows = {"Terminal Name", "Revenue", "Total Regas", "Total Freight", "Netback", "Cargo Cost"}
        if row["Item"] in bold_rows:
            return ['font-weight: bold'] * len(row)
        else:
            return [''] * len(row)

    def highlight_profit(row):
        return ['font-weight: bold; background-color: #fff9c4']*len(row) if row["Item"] == "Net profit" else [''] * len(row)

    styler = (
        df_table.style
        .apply(highlight_key_rows, axis=1)
        .apply(highlight_profit, axis=1)
    )
    st.dataframe(
        styler,
        use_container_width=False,
        hide_index=True,
        width=500,
        height=38 * len(df_table) - 10
    )

# --- Netback by Terminal in Selected Month ---
st.markdown("---")
if 'month1_select' not in st.session_state:
    st.session_state.month1_select = month

st.subheader(f"📊 Netback by Terminal | {st.session_state.month1_select}")
month1 = st.selectbox("Select Period of Delivery", df["Period"].unique(), key="month1_select")
fig = px.bar(
    df[df["Period"] == month1], x="Terminal", y="Netback",
    labels={"Terminal": "Terminal Name", "Netback": "Netback ($/MMBtu)"},
    hover_data={"Netback": ':.3f'},
)
st.plotly_chart(fig, use_container_width=True)

# --- Cost Chart for Terminal (all months) ---
st.markdown("---")
if 'terminal1' not in st.session_state:
    st.session_state.terminal1 = terminal

st.subheader(f"📈 Cost Chart by month | {st.session_state.terminal1}")
terminal1 = st.selectbox("Select Terminal", df["Terminal"].unique(), key="terminal1")

fig = go.Figure()
for component in COMPONENTS:
    fig.add_trace(
        go.Bar(
            name=component,
            x=df[df['Terminal'] == terminal1]["Period"],
            y=df[df['Terminal'] == terminal1][component],
            hovertemplate=f"%{{y:.3f}} $/MMBtu",
        )
    )
# add sum of all components
fig.add_trace(
    go.Scatter(
        name="Total",
        x=df[df['Terminal'] == terminal1]["Period"],
        y=df[df['Terminal'] == terminal1][COMPONENTS].sum(axis=1),
        hovertemplate=f"%{{y:.3f}} $/MMBtu",
        marker_color='rgba(255, 0, 0, 0.5)',
        marker_size=10,
        mode='lines+markers',
    )
)
fig.update_layout(
    barmode='relative',
    xaxis_title="Month",
    yaxis_title="$/MMBtu",
    showlegend=True,
    height=400,
    hovermode="x unified" # for hover interaction
)
fig.update_layout(colorway=px.colors.qualitative.Plotly)
event = st.plotly_chart(fig, use_container_width=True, on_select="ignore")

# --- Price Forecast Trends ---
st.markdown("---")
st.subheader("Benchmark Price Forecast Trends")
price_df = df[["Period", "Prediction_TTF", "Prediction_PVB", "Prediction_HH($MMBtu)"]].dropna().drop_duplicates("Period")

# fig2 = px.line(price_df, x="Period", y=["Prediction_TTF", "Prediction_PVB", "Prediction_HH($MMBtu)"],
#                labels={"Period": "Period of Delivery", "value": "Price ($/MMBtu)", "variable": "Index"})
fig2 = px.line(df_k, x="Date", y=["JKM (Asia)", "TTF (Europe)"],# "HH (US)"],
               labels={"Date": "Period of Delivery", "value": "Price ($/MMBtu)", "variable": "Index"},
               )
# Alternative: both price_df and df_k: but wired Jul 2025 position
# fig2 = px.line(price_df, x="Period", y=["Prediction_TTF", "Prediction_PVB", "Prediction_HH($MMBtu)"],
#                labels={"Period": "Period of Delivery", "value": "Price ($/MMBtu)", "variable": "Index"},
#                )
# for col, color in zip(["Asian LNG", "European TTF", "US Henry Hub"], ["orange", "green", "blue"]):
#     fig2.add_scatter(
#         x=df_k["Date"],
#         y=df_k[col],
#         mode='lines',
#         name=f'K-Forecast {col}',
#         #line=dict(color=color)
#     )

fig2.update_layout(
    xaxis_title="Period of Delivery",
    yaxis_title="Price ($/MMBtu)",
    height=400,
    showlegend=True,
    hovermode="x unified"
)
st.plotly_chart(fig2, use_container_width=True)

# --- Netback Heatmap (Scatter Plot) ---
st.markdown("---")
st.subheader("📍 Netback Heatmap (Scatter Plot)")
st.info("Select points in the plot to compare cost breakdowns below (`<Shift> + click` to select multiple).")
netback_min = df["Netback"].min()
netback_max = df["Netback"].max()

st.toggle("Swap X and Y axes", key="swap_axes")
if st.session_state.get("swap_axes", False):
    xaxis, yaxis = "Terminal", "Period"
    height, width = 550, 850
else:
    xaxis, yaxis = "Period", "Terminal"
    height, width = 750, 700

fig4 = px.scatter(
    df.fillna(0), 
    x=xaxis,
    y=yaxis,
    color="Netback", 
    size="Netback",
    size_max=20, 
    range_color=[netback_min, netback_max],
    opacity=1.0,
    hover_name="Terminal", 
    hover_data={
        "Netback": ':.2f',
        "Total Regas": ':.2f',
        "Total Freight From US": ':.2f',
        "Period": True,
        "Terminal": False  # Already shown in hover_name
    },
    labels={
        "Period": "Period of Delivery", 
        "Terminal": "Terminal Name", 
        "Netback": "Netback ($/MMBtu)",
        "Total Regas": "Total Regas",
        "Total Freight From US": "Total Freight from US"
    }
)

fig4.data[0].update(selected=dict(marker=dict(size=15)),
                    unselected=dict(marker=dict(opacity=1))
                    ) # Though no selection exists, a opacity is applied to all points; with 'opacity=1.0' fix this in px.scatter fix it.


fig4.update_layout(
    xaxis_title="Period of Delivery" if xaxis == "Period" else "Terminal Name",
    yaxis_title="Terminal Name" if yaxis == "Terminal" else "Period of Delivery",
    height=height,
    width=width,
    hovermode="closest",
    showlegend=False,
)
fig4.update_layout(height=height, width=width)
selected_months_terminals = st.plotly_chart(fig4, use_container_width=False, on_select="rerun") # The point selected in the json is sorted already by x and y

# --- Compare Cost Breakdown for Selected Month-Terminal(s) ---
# Plotly feature/bug: after box/lasso selection, 'shift+point selection' resets the selection to that point but box/lasso frame remains.
st.subheader("🔍 Compare Cost Breakdown for Selected Month-Terminal(s)")

if 'compare_df' not in st.session_state:
    st.session_state.compare_df = pd.DataFrame()
if 'selected_points' not in st.session_state:
    st.session_state.selected_points = []

selected_points = selected_months_terminals['selection']['points']
if not selected_points:
    st.session_state.selected_points = []
    st.session_state.compare_df = pd.DataFrame()
    st.warning("No points selected. Please select points in the scatter plot to compare cost breakdowns.")
    st.stop()
else:
    new_points = [(pt['x'], pt['y']) for pt in selected_points if (pt['x'], pt['y']) not in st.session_state.selected_points]
    remove_points = [pt for pt in st.session_state.selected_points if pt not in [(pt['x'], pt['y']) for pt in selected_points]]

if new_points:
    st.session_state.selected_points.extend(new_points)
    new_rows = df[df.apply(lambda row: (row[xaxis], row[yaxis]) in new_points, axis=1)].copy()
    st.session_state.compare_df = pd.concat([st.session_state.compare_df, new_rows], ignore_index=True)

if remove_points:
    st.session_state.selected_points = [pair for pair in st.session_state.selected_points if pair not in remove_points]
    st.session_state.compare_df = st.session_state.compare_df[~st.session_state.compare_df.apply(lambda row: (row[xaxis], row[yaxis]) in remove_points, axis=1)]

def on_change_compare_df():
    edited_rows = st.session_state['compare_cost_breakdown_data_editor']['edited_rows']
    editor_df = st.session_state.compare_df.reset_index(drop=True)
    for index, row in edited_rows.items():
        for col, value in row.items():
            if col in editor_df.columns:
                editor_df.at[index, col] = value
    editor_df['Netback'] = editor_df["Benchmark_Price"] + editor_df["Total Regas"] + editor_df["Total Freight From US"]
    editor_df['Net Profit'] = editor_df['Netback'] - editor_df['Cargo Cost US origin']
    st.session_state['compare_df'] = editor_df

st.info("Adjust the values to recalculate the netback")
st.data_editor(
    st.session_state.compare_df[["Period", "Terminal"] + COMPONENTS_ALL],
    use_container_width=True,
    height=40 * len(st.session_state.compare_df)+40,  # Adjust height based on number of rows
    hide_index=False,
    disabled=["Period", "Terminal", "Fuel_loss from US", "Route Cost from US", "Netback", "Net Profit"],
    column_order=["Period", "Terminal"] + COMPONENTS_ALL,
    column_config={
        "Period": st.column_config.DatetimeColumn("Period", help="Period of delivery", format='MMM YYYY'),
        "Terminal": st.column_config.TextColumn("Terminal", help="Terminal Name"),
        **{col: st.column_config.NumberColumn(col) for col in COMPONENTS_ALL},
        "Netback": st.column_config.NumberColumn(
            "Netback",
            help="Calculated Netback",
        )
    },
    num_rows='fixed',
    key="compare_cost_breakdown_data_editor",
    on_change=on_change_compare_df,
    )

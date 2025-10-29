#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# --- 1. Load and Preprocess Data ---
# Load the CSV file
try:
    df = pd.read_csv('ROI Wind Forecast Sample.csv')
except FileNotFoundError:
    print("Error: 'ROI Wind Forecast Sample.csv' not found.")
    print("Please make sure the script is in the same directory as the CSV file.")
    exit()

# Convert Timestamp to datetime objects
# Adjust format if your 'Timestamp' is different. Sample '1/1/2022 0:00' suggests 'day/month/Year'
try:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M')
except ValueError:
    # Fallback for different common formats
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    except Exception as e:
        print(f"Error parsing timestamps: {e}")
        print("Please check the 'Timestamp' column format in your CSV.")
        exit()


# Create 'Wind Level' categories based on 33rd and 66th percentiles of 'ROI Actual'
low_threshold = df['ROI Actual'].quantile(0.33)
high_threshold = df['ROI Actual'].quantile(0.66)

def categorize_wind(actual):
    if actual <= low_threshold:
        return 'Low'
    elif actual <= high_threshold:
        return 'Medium'
    else:
        return 'High'

df['Wind Level'] = df['ROI Actual'].apply(categorize_wind)

# Create 'Day Type' (Weekday/Weekend)
df['Day Type'] = df['Timestamp'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

# Create 'Month'
df['Month'] = df['Timestamp'].dt.month_name()

# Create 'Hour' (as an integer 0-23 for easy grouping)
df['Hour'] = df['Timestamp'].dt.hour

# Create 'Year'
df['Year'] = df['Timestamp'].dt.year

# Prepare filter options
wind_level_options = [
    {'label': 'All', 'value': 'All'},
    {'label': 'Low', 'value': 'Low'},
    {'label': 'Medium', 'value': 'Medium'},
    {'label': 'High', 'value': 'High'}
]

day_type_options = [
    {'label': 'All', 'value': 'All'},
    {'label': 'Weekday', 'value': 'Weekday'},
    {'label': 'Weekend', 'value': 'Weekend'}
]

# Sort month options in calendar order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
all_months = df['Month'].unique()
sorted_months = sorted(all_months, key=lambda m: month_order.index(m) if m in month_order else -1)
month_options = [{'label': m, 'value': m} for m in sorted_months]

# --- 2. Initialize Dash App ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server # Add this line for Gunicorn
app.title = "Wind Forecast Dashboard"

# --- 3. Define App Layout ---
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif'}, children=[
    html.H1(
        children='Wind Forecast Accuracy Dashboard',
        style={'textAlign': 'center', 'color': '#333'}
    ),

    html.Div(children='Investigate forecast accuracy based on wind level, day type, date, and month.', style={
        'textAlign': 'center', 'color': '#555', 'marginBottom': '30px'
    }),

    # Filter Controls Row
    html.Div(className='row', style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '8px'}, children=[
        # Wind Level Filter
        html.Div(style={'flex': 1}, children=[
            html.Label('Wind Level', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='wind-level-filter',
                options=wind_level_options,
                value='All',
                clearable=False
            )
        ]),

        # Day Type Filter
        html.Div(style={'flex': 1}, children=[
            html.Label('Day Type', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='day-type-filter',
                options=day_type_options,
                value='All',
                clearable=False
            )
        ]),

        # Month Filter
        html.Div(style={'flex': 2}, children=[
            html.Label('Months', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='month-filter',
                options=month_options,
                value=list(all_months), # Default to all months selected
                multi=True,
                placeholder="Select months..."
            )
        ]),

        # Date Filter
        html.Div(style={'flex': 2}, children=[
            html.Label('Date Range', style={'fontWeight': 'bold', 'display': 'block'}),
            dcc.DatePickerRange(
                id='date-filter',
                min_date_allowed=df['Timestamp'].min().date(),
                max_date_allowed=df['Timestamp'].max().date(),
                start_date=df['Timestamp'].min().date(),
                end_date=df['Timestamp'].max().date(),
                display_format='DD/MM/YYYY',
                style={'width': '100%'}
            )
        ]),
    ]),

    # Main Content Area
    html.Div(className='row', children=[
        # Line Plot
        dcc.Graph(id='forecast-plot'),
    ]),

    # New Summary Stats Section
    html.Div(id='summary-stats-container', style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '8px', 'marginTop': '20px', 'marginBottom': '20px'}),

    html.Hr(),

    html.H3("Hourly Ratio (Actuals / Forecast)", style={'textAlign': 'center', 'marginTop': '30px'}),
    
    # Data Table
    html.Div(style={'padding': '0 20px'}, children=[
        dash_table.DataTable(
            id='hourly-ratio-table',
            columns=[
                {'name': 'Hour of Day', 'id': 'Hour'},
                {'name': 'Actuals / Forecast Ratio', 'id': 'Ratio (Actual/Forecast)'}
            ],
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={
                'backgroundColor': '#f0f0f0',
                'fontWeight': 'bold',
                'border': '1px solid #ddd'
            },
            style_data={
                'border': '1px solid #ddd'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#fdfdfd'
                }
            ]
        )
    ])
])

# --- 4. Define Callback Logic ---
@app.callback(
    [Output('forecast-plot', 'figure'),
     Output('hourly-ratio-table', 'data'),
     Output('summary-stats-container', 'children')],
    [Input('wind-level-filter', 'value'),
     Input('day-type-filter', 'value'),
     Input('date-filter', 'start_date'),
     Input('date-filter', 'end_date'),
     Input('month-filter', 'value')]
)
def update_dashboard(wind_level, day_type, start_date, end_date, months):
    # Start with the full dataframe
    dff = df.copy()

    # Apply Wind Level filter
    if wind_level != 'All':
        dff = dff[dff['Wind Level'] == wind_level]

    # Apply Day Type filter
    if day_type != 'All':
        dff = dff[dff['Day Type'] == day_type]

    # Apply Date filter
    # Convert start/end dates to datetime for comparison
    if start_date:
        dff = dff[dff['Timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        # Add 1 day to end_date to make it inclusive of the whole day
        end_date_inclusive = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        dff = dff[dff['Timestamp'] < end_date_inclusive]

    # Apply Month filter
    if months: # Check if list is not empty
        dff = dff[dff['Month'].isin(months)]
    else:
        # If no months are selected, return empty state
        dff = pd.DataFrame(columns=df.columns)

    # --- Perform Aggregations ---
    if not dff.empty:
        # Group by the hour (0-23)
        hourly_agg = dff.groupby('Hour').agg(
            Avg_Forecast=('ROI Forecast', 'mean'),
            Avg_Actual=('ROI Actual', 'mean')
        ).reset_index()

        # Calculate ratio, handling potential division by zero
        hourly_agg['Ratio (Actual/Forecast)'] = (hourly_agg['Avg_Actual'] / hourly_agg['Avg_Forecast']).replace([np.inf, -np.inf], np.nan)
        # Fill NaNs (from 0/0) or Infs (from x/0) with 0 or another placeholder
        hourly_agg['Ratio (Actual/Forecast)'] = hourly_agg['Ratio (Actual/Forecast)'].fillna(0)
    else:
        # Create empty dataframe if no data matches filters
        hourly_agg = pd.DataFrame(columns=['Hour', 'Avg_Forecast', 'Avg_Actual', 'Ratio (Actual/Forecast)'])

    # --- NEW: Calculate Summary Stats ---
    summary_children = [] # A list to hold the new HTML components
    if not dff.empty:
        # 1. Calculate Overall Summary
        overall_total_forecast = dff['ROI Forecast'].sum()
        overall_total_actual = dff['ROI Actual'].sum()
        overall_ratio = (overall_total_actual / overall_total_forecast) if overall_total_forecast != 0 else 0
        overall_ratio = round(overall_ratio, 3)

        summary_children.append(html.H4(
            f'Overall Summary Ratio (Actuals / Forecast) for Filtered Data: {overall_ratio}', 
            style={'textAlign': 'center', 'color': '#333', 'marginBottom': '20px'}
        ))

        # 2. Calculate Table Stats
        # Total Ratio by Year
        year_total_stats = dff.groupby('Year').agg(
            Total_Forecast=('ROI Forecast', 'sum'),
            Total_Actual=('ROI Actual', 'sum')
        ).reset_index()
        year_total_stats['Total'] = (year_total_stats['Total_Actual'] / year_total_stats['Total_Forecast'].replace(0, np.nan)).fillna(0).round(3)

        # Weekday Ratio by Year
        weekday_stats = dff[dff['Day Type'] == 'Weekday'].groupby('Year').agg(
            Total_Forecast=('ROI Forecast', 'sum'),
            Total_Actual=('ROI Actual', 'sum')
        ).reset_index()
        weekday_stats['Weekday'] = (weekday_stats['Total_Actual'] / weekday_stats['Total_Forecast'].replace(0, np.nan)).fillna(0).round(3)

        # Weekend Ratio by Year
        weekend_stats = dff[dff['Day Type'] == 'Weekend'].groupby('Year').agg(
            Total_Forecast=('ROI Forecast', 'sum'),
            Total_Actual=('ROI Actual', 'sum')
        ).reset_index()
        weekend_stats['Weekend'] = (weekend_stats['Total_Actual'] / weekend_stats['Total_Forecast'].replace(0, np.nan)).fillna(0).round(3)

        # Merge stats into one table
        # Start with a df of all unique years in the filtered data
        all_years_df = pd.DataFrame({'Year': dff['Year'].unique()})
        all_years_df = all_years_df.sort_values(by='Year').reset_index(drop=True)

        summary_df = pd.merge(all_years_df, year_total_stats[['Year', 'Total']], on='Year', how='left')
        summary_df = pd.merge(summary_df, weekday_stats[['Year', 'Weekday']], on='Year', how='left')
        summary_df = pd.merge(summary_df, weekend_stats[['Year', 'Weekend']], on='Year', how='left')

        # Fill NaNs with 0 (for years where Weekday/Weekend might not have data)
        summary_df = summary_df.fillna(0)

        # 3. Create Table Components
        summary_table_cols = [
            {'name': 'Year', 'id': 'Year'},
            {'name': 'Total Ratio', 'id': 'Total'},
            {'name': 'Weekday Ratio', 'id': 'Weekday'},
            {'name': 'Weekend Ratio', 'id': 'Weekend'},
        ]
        summary_table_data = summary_df.to_dict('records')

        summary_children.append(dash_table.DataTable(
            id='summary-stats-table',
            columns=summary_table_cols,
            data=summary_table_data,
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={
                'backgroundColor': '#f0f0f0',
                'fontWeight': 'bold',
                'border': '1px solid #ddd'
            },
            style_data={
                'border': '1px solid #ddd'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#fdfdfd'
                }
            ]
        ))
    
    else:
        # Handle empty data case
        summary_children = [html.P("No data available for selected filters.", style={'textAlign': 'center'})]


    # --- Create Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_agg['Hour'],
        y=hourly_agg['Avg_Forecast'],
        mode='lines+markers',
        name='Average Forecast',
        line=dict(width=3, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=hourly_agg['Hour'],
        y=hourly_agg['Avg_Actual'],
        mode='lines+markers',
        name='Average Actual',
        line=dict(width=3)
    ))
    
    # Set plot title based on filters
    filter_summary = f"Wind: {wind_level}, Day: {day_type}"
    fig.update_layout(
        title=f'Average Forecast vs. Actual by Hour<br><sup>Filtered by: {filter_summary}</sup>',
        xaxis_title='Hour of Day',
        yaxis_title='Wind Output',
        xaxis=dict(
            tickmode='array', 
            tickvals=list(range(24)),
            gridcolor='#e0e0e0'
        ),
        yaxis=dict(gridcolor='#e0e0e0'),
        hovermode="x unified",
        legend_title_text='Series',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='#333',
        transition_duration=500
    )

    # --- Create Table Data ---
    # Rounding for display
    hourly_agg['Ratio (Actual/Forecast)'] = hourly_agg['Ratio (Actual/Forecast)'].round(3)
    # Ensure all 24 hours are present for the table, filling missing with 0 or NaN
    all_hours = pd.DataFrame({'Hour': range(24)})
    table_df = pd.merge(all_hours, hourly_agg[['Hour', 'Ratio (Actual/Forecast)']], on='Hour', how='left').fillna(0)
    
    table_data = table_df.to_dict('records')

    return fig, table_data, summary_children

# --- 5. Run the App ---
if __name__ == '__main__':
    # Changed debug=True to debug=False for production
    # Added host='0.0.0.0' and port=8080, common for deployment
    app.run(debug=False, host='0.0.0.0', port=8080)



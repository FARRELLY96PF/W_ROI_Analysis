#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime

# --- 1. Load and Preprocess Data ---

# Load the new CSV file
try:
    # Use the new file name
    df = pd.read_csv('ROI_Wind_Forecast.csv')
except FileNotFoundError:
    print("Error: 'ROI_Wind_ForecastV1.csv' not found.")
    print("Please make sure the script is in the same directory as the CSV file.")
    exit()

# Convert Timestamp to datetime objects
try:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M')
except ValueError:
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    except Exception as e:
        print(f"Error parsing timestamps: {e}")
        print("Please check the 'Timestamp' column format in your CSV.")
        exit()

# --- Preprocessing for BOTH Tabs ---

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

# Create 'Year'
df['Year'] = df['Timestamp'].dt.year

# Create 'Hour'
df['Hour'] = df['Timestamp'].dt.hour

# Create 'Time' object for PnL time filtering
df['Time'] = df['Timestamp'].dt.time

# --- PnL Calculation Columns ---

# Define IDA trading times (inclusive)
ida2_start = datetime.time(11, 0)
ida2_end = datetime.time(22, 30)
ida3_start = datetime.time(17, 0)
ida3_end = datetime.time(22, 30)

# Create time masks (for pre-calculation)
ida2_mask_global = (df['Time'] >= ida2_start) & (df['Time'] <= ida2_end)
ida3_mask_global = (df['Time'] >= ida3_start) & (df['Time'] <= ida3_end)

# Fill NaNs in price/forecast columns to avoid errors in calculations
price_cols = ['Day Ahead Price', 'IDA1 Price', 'IDA2 Price', 'IDA3 Price', 'Balancing Price']
# Add the new DA Forecast HH column
forecast_cols = ['DA Forecast', 'IDA1 Forecast', 'IDA2 Forecast', 'IDA3 Forecast', 'ROI Actual', 'DA Forecast HH']
# df[price_cols] = df[price_cols].fillna(0)
# df[forecast_cols] = df[forecast_cols].fillna(0)


# Calculate PnL components (we do this once for efficiency)
# All PnL calculations are divided by 2.0 to convert MW to MWh for 30-min periods

# 1. Forecasting Cost (DA PnL)
df['DA_PnL'] = ((df['DA Forecast'] - df['ROI Actual']) * (df['Day Ahead Price'] - df['Balancing Price'])) / 2.0

# 2. IDA1 PnL (Split)
# 2a. Granularity PnL
# Granularity exposure = DA Hourly - DA Half-Hourly
df['Granularity_Exposure'] = df['DA Forecast HH'] - df['DA Forecast']
# Fillna for periods where DA Forecast HH is not available
df['Granularity_Exposure'] = df['Granularity_Exposure'].fillna(0)
df['IDA1_Granularity_PnL'] = (df['Granularity_Exposure'] * (df['IDA1 Price'] - df['Balancing Price'])) / 2.0

# 2b. Change in Forecast PnL
# Change exposure = (DA Hourly - IDA1) - Granularity Exposure
df['IDA1_Change_Exposure'] = (df['IDA1 Forecast'] - df['DA Forecast']) - df['Granularity_Exposure']
df['IDA1_Change_PnL'] = (df['IDA1_Change_Exposure'] * (df['IDA1 Price'] - df['Balancing Price'])) / 2.0

# 3. IDA2 PnL (only for valid hours)
df['IDA2_PnL'] = np.where(
    ida2_mask_global,
    ((df['IDA2 Forecast'] - df['IDA1 Forecast']) * (df['IDA2 Price'] - df['Balancing Price'])) / 2.0,
    0  # PnL is 0 outside trading hours
)

# 4. IDA3 PnL (only for valid hours)
df['IDA3_PnL'] = np.where(
    ida3_mask_global,
    ((df['IDA3 Forecast'] - df['IDA2 Forecast']) * (df['IDA3 Price'] - df['Balancing Price'])) / 2.0,
    0  # PnL is 0 outside trading hours
)

# --- 2. Initialize Dash App ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server  # For Gunicorn
app.title = "Wind Forecast Dashboard"

# --- 3. Define App Layout ---
app.layout = html.Div([
    
    html.H1("Wind Dashboard: Accuracy & Intraday PnL", style={'textAlign': 'center'}),

    # --- GLOBAL FILTERS ---
    html.Div(className='container', style={'backgroundColor': '#f9f9f9', 'padding': '20px', 'borderRadius': '10px'}, children=[
        html.H4("Global Dashboard Filters", style={'textAlign': 'center'}),
        html.Div(className='row', children=[
            # Date Range Filter
            html.Div(className='six columns', children=[
                html.Label('Select Date Range:'),
                dcc.DatePickerRange(
                    id='date-range-picker',
                    start_date=df['Timestamp'].min().date(),
                    end_date=df['Timestamp'].max().date(),
                    display_format='DD/MM/YYYY',
                    style={'width': '100%'}
                )
            ]),
            # Wind Level Filter
            html.Div(className='six columns', children=[
                html.Label('Select Wind Level:'),
                dcc.Dropdown(
                    id='wind-level-dropdown',
                    options=[
                        {'label': 'All', 'value': 'All'},
                        {'label': 'Low', 'value': 'Low'},
                        {'label': 'Medium', 'value': 'Medium'},
                        {'label': 'High', 'value': 'High'}
                    ],
                    value='All'
                )
            ])
        ]),
        html.Div(className='row', style={'marginTop': '15px'}, children=[
            # Day Type Filter
            html.Div(className='six columns', children=[
                html.Label('Select Day Type:'),
                dcc.Dropdown(
                    id='day-type-dropdown',
                    options=[
                        {'label': 'All', 'value': 'All'},
                        {'label': 'Weekday', 'value': 'Weekday'},
                        {'label': 'Weekend', 'value': 'Weekend'}
                    ],
                    value='All'
                )
            ]),
            # Month Filter
            html.Div(className='six columns', children=[
                html.Label('Select Months:'),
                dcc.Dropdown(
                    id='month-dropdown',
                    options=[{'label': month, 'value': month} for month in df['Month'].unique()],
                    value=df['Month'].unique().tolist(),
                    multi=True
                )
            ])
        ])
    ]),
    
    # --- TABS CONTAINER ---
    dcc.Tabs(id="dashboard-tabs", children=[
        
        # --- TAB 1: Forecast Accuracy ---
        dcc.Tab(label='Forecast Accuracy', children=[
            html.Div([
                # Graph
                dcc.Graph(id='forecast-plot'),

                # Summary Statistics
                html.Div(id='summary-stats-container', style={'padding': '20px'}),

                # Hourly Ratio Table
                html.H4("Hourly Ratio (Actual / Forecast)", style={'textAlign': 'center', 'marginTop': '30px'}),
                dash_table.DataTable(
                    id='hourly-ratio-table',
                    columns=[{'name': 'Hour', 'id': 'Hour'}, {'name': 'Ratio (Actual/Forecast)', 'id': 'Ratio (Actual/Forecast)'}],
                    style_cell={'textAlign': 'center', 'fontFamily': 'Arial'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ]
                )
            ], style={'padding': '20px'})
        ]),
        
        # --- TAB 2: Intraday PnL Analysis ---
        dcc.Tab(label='Intraday PnL Analysis', children=[
            html.Div([
                # Forecast Selection Checklist
                html.Div(className='container', style={'padding': '10px'}, children=[
                    html.Label('Select Forecasts to Analyze (for Plot):'),
                    dcc.Checklist(
                        id='forecast-checklist',
                        options=[
                            {'label': 'Day Ahead (DA)', 'value': 'DA'},
                            {'label': 'IDA1', 'value': 'IDA1'},
                            {'label': 'IDA2', 'value': 'IDA2'},
                            {'label': 'IDA3', 'value': 'IDA3'}
                        ],
                        value=['DA'], # Default to DA
                        inline=True,
                        style={'fontFamily': 'Arial'}
                    )
                ]),
                
                # PnL Forecast Plot
                dcc.Graph(id='pnl-forecast-plot'),
                
                # PnL Summary
                html.Div(
                    id='pnl-summary', 
                    style={
                        'padding': '20px', 
                        'backgroundColor': '#f9f9f9', 
                        'borderRadius': '10px', 
                        'marginTop': '20px'
                    }
                ),

                # Balancing Risk Input
                html.Div(className='container', style={'padding': '10px', 'marginTop': '20px'}, children=[
                    html.Label('Balancing Risk Premium Multiplier:'),
                    dcc.Input(
                        id='balancing-risk-multiplier',
                        type='number',
                        value=4.5,
                        style={'width': '100px', 'marginRight': '10px'}
                    ),
                    html.Span(" (default: 4.5)")
                ]),

                # DA Revenue Summary (Moved from table)
                html.H5(
                    id='da-revenue-summary', 
                    style={'textAlign': 'center', 'marginTop': '30px', 'fontWeight': 'bold', 'fontSize': '1.2em'}
                ),

                # Scenario PnL Table
                html.H4("Scenario PnL Analysis Table", style={'textAlign': 'center', 'marginTop': '10px'}),
                dash_table.DataTable(
                    id='scenario-pnl-table',
                    style_cell={'textAlign': 'left', 'fontFamily': 'Arial'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ]
                )
                
            ], style={'padding': '20px'})
        ]),
    ])
])

# --- 4. Callback for TAB 1 (Accuracy) ---
@app.callback(
    [Output('forecast-plot', 'figure'),
     Output('summary-stats-container', 'children'),
     Output('hourly-ratio-table', 'data')],
    [Input('wind-level-dropdown', 'value'),
     Input('day-type-dropdown', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('month-dropdown', 'value')]
)
def update_accuracy_tab(wind_level, day_type, start_date, end_date, selected_months):
    
    # Filter by date range
    dff = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]
    
    # Filter by selected months
    dff = dff[dff['Month'].isin(selected_months)]

    # Filter by Wind Level
    if wind_level != 'All':
        dff = dff[dff['Wind Level'] == wind_level]

    # Filter by Day Type
    if day_type != 'All':
        dff = dff[dff['Day Type'] == day_type]
        
    if dff.empty:
        # Return empty state if no data
        return go.Figure(layout_title_text='No data for selected filters.'), [], []

    # --- Create Plot ---
    # Group by hour and calculate mean
    # Use DA Forecast (the original ROI Forecast)
    hourly_agg = dff.groupby('Hour')[['DA Forecast', 'ROI Actual']].mean().reset_index()
    hourly_agg.rename(columns={'DA Forecast': 'Avg_Forecast', 'ROI Actual': 'Avg_Actual'}, inplace=True)

    fig = go.Figure()

    # Add Forecast trace
    fig.add_trace(go.Scatter(
        x=hourly_agg['Hour'],
        y=hourly_agg['Avg_Forecast'],
        mode='lines+markers',
        name='Average Forecast (DA)'
    ))
    # Add Actual trace
    fig.add_trace(go.Scatter(
        x=hourly_agg['Hour'],
        y=hourly_agg['Avg_Actual'],
        mode='lines+markers',
        name='Average Actual',
        line=dict(width=3)
    ))
    
    filter_summary = f"Wind: {wind_level}, Day: {day_type}"
    fig.update_layout(
        title=f'Average Forecast vs. Actual by Hour<br><sup>Filtered by: {filter_summary}</sup>',
        xaxis_title='Hour of Day',
        yaxis_title='Wind Output (MW)',
        xaxis=dict(tickmode='array', tickvals=list(range(24)), gridcolor='#e0e0e0'),
        yaxis=dict(gridcolor='#e0e0e0'),
        hovermode="x unified",
        legend_title_text='Series',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='#333',
        transition_duration=300
    )

    # --- Create Summary Stats ---
    
    # FIX for DeprecationWarning and divide-by-zero errors
    def safe_ratio(x):
        actual_sum = x['ROI Actual'].sum()
        # Use DA Forecast
        forecast_sum = x['DA Forecast'].sum()
        if forecast_sum == 0:
            return 0  # Avoid division by zero
        return actual_sum / forecast_sum

    overall_ratio = safe_ratio(dff)
    summary_children = [
        html.H4("Summary Statistics for Filtered Data", style={'textAlign': 'center'}),
        html.H5(f"Overall Ratio (Total Actual / Total Forecast): {overall_ratio: .4f}")
    ]

    all_years = sorted(dff['Year'].unique())
    all_years_df = pd.DataFrame({'Year': all_years})

    # Apply the safe_ratio function with include_groups=False to silence the warning
    year_total_stats = dff.groupby('Year').apply(safe_ratio, include_groups=False).reset_index(name='Total')
    weekday_stats = dff[dff['Day Type'] == 'Weekday'].groupby('Year').apply(safe_ratio, include_groups=False).reset_index(name='Weekday')
    weekend_stats = dff[dff['Day Type'] == 'Weekend'].groupby('Year').apply(safe_ratio, include_groups=False).reset_index(name='Weekend')

    summary_df = pd.merge(all_years_df, year_total_stats[['Year', 'Total']], on='Year', how='left')
    summary_df = pd.merge(summary_df, weekday_stats[['Year', 'Weekday']], on='Year', how='left')
    summary_df = pd.merge(summary_df, weekend_stats[['Year', 'Weekend']], on='Year', how='left')
    summary_df = summary_df.fillna(0).round(4)
    
    summary_children.append(
        dash_table.DataTable(
            data=summary_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in summary_df.columns],
            style_cell={'textAlign': 'center', 'fontFamily': 'Arial'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )
    )

    # --- Create Table Data ---
    hourly_agg['Ratio (Actual/Forecast)'] = hourly_agg['Avg_Actual'] / hourly_agg['Avg_Forecast']
    # Handle potential divide-by-zero in table if Avg_Forecast is 0
    hourly_agg['Ratio (Actual/Forecast)'] = hourly_agg['Ratio (Actual/Forecast)'].replace([np.inf, -np.inf], 0).round(3)
    
    all_hours = pd.DataFrame({'Hour': range(24)})
    table_df = pd.merge(all_hours, hourly_agg[['Hour', 'Ratio (Actual/Forecast)']], on='Hour', how='left').fillna(0)
    
    table_data = table_df.to_dict('records')

    return fig, summary_children, table_data

# --- 5. Callback for TAB 2 (PnL Analysis) ---
@app.callback(
    [Output('pnl-forecast-plot', 'figure'),
     Output('pnl-summary', 'children'),
     Output('da-revenue-summary', 'children'),      # New Output
     Output('scenario-pnl-table', 'data'),      
     Output('scenario-pnl-table', 'columns')], 
    [Input('wind-level-dropdown', 'value'),
     Input('day-type-dropdown', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('month-dropdown', 'value'),
     Input('forecast-checklist', 'value'),
     Input('balancing-risk-multiplier', 'value')] 
)
def update_pnl_tab(wind_level, day_type, start_date, end_date, selected_months, selected_forecasts, balancing_risk_multiplier):
    
    # Filter by date range
    dff = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]
    
    # Filter by selected months
    dff = dff[dff['Month'].isin(selected_months)]

    # Filter by Wind Level
    if wind_level != 'All':
        dff = dff[dff['Wind Level'] == wind_level]

    # Filter by Day Type
    if day_type != 'All':
        dff = dff[dff['Day Type'] == day_type]
        
    # --- Validate Multiplier ---
    try:
        multiplier = float(balancing_risk_multiplier)
    except (ValueError, TypeError):
        multiplier = 4.5 # Default value if input is invalid
        
    if dff.empty:
        empty_fig = go.Figure(layout_title_text='No data for selected filters.')
        empty_summary = [html.H4("No data for selected filters.")]
        return empty_fig, empty_summary, '', [], [] # Added empty string for new output

    # --- Create PnL Plot ---
    fig = go.Figure()
    
    if selected_forecasts:
        # Always add Actual
        hourly_agg_actual = dff.groupby('Hour')['ROI Actual'].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=hourly_agg_actual['Hour'],
            y=hourly_agg_actual['ROI Actual'],
            mode='lines',
            name='Average Actual',
            line=dict(width=3, color='black', dash='dot')
        ))

        # Add DA Forecast
        if 'DA' in selected_forecasts:
            hourly_agg_da = dff.groupby('Hour')['DA Forecast'].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=hourly_agg_da['Hour'],
                y=hourly_agg_da['DA Forecast'],
                mode='lines', name='Avg DA Forecast'
            ))

        # Add IDA1 Forecast
        if 'IDA1' in selected_forecasts:
            hourly_agg_ida1 = dff.groupby('Hour')['IDA1 Forecast'].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=hourly_agg_ida1['Hour'],
                y=hourly_agg_ida1['IDA1 Forecast'],
                mode='lines', name='Avg IDA1 Forecast'
            ))

        # FIX for UserWarning: Re-create time masks based on the *filtered* dataframe (dff)
        ida2_mask_filtered = (dff['Time'] >= ida2_start) & (dff['Time'] <= ida2_end)
        ida3_mask_filtered = (dff['Time'] >= ida3_start) & (dff['Time'] <= ida3_end)

        # Add IDA2 Forecast (for trading hours only)
        if 'IDA2' in selected_forecasts:
            dff_ida2 = dff[ida2_mask_filtered] # Use the filtered mask
            if not dff_ida2.empty:
                hourly_agg_ida2 = dff_ida2.groupby('Hour')['IDA2 Forecast'].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=hourly_agg_ida2['Hour'],
                    y=hourly_agg_ida2['IDA2 Forecast'],
                    mode='lines', name='Avg IDA2 Forecast (Trading Hours)'
                ))

        # Add IDA3 Forecast (for trading hours only)
        if 'IDA3' in selected_forecasts:
            dff_ida3 = dff[ida3_mask_filtered] # Use the filtered mask
            if not dff_ida3.empty:
                hourly_agg_ida3 = dff_ida3.groupby('Hour')['IDA3 Forecast'].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=hourly_agg_ida3['Hour'],
                    y=hourly_agg_ida3['IDA3 Forecast'],
                    mode='lines', name='Avg IDA3 Forecast (Trading Hours)'
                ))
    else:
        fig.update_layout(title='Please select a forecast to display on the plot.')


    fig.update_layout(
        title='Average Forecasts vs. Actual by Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Wind Output (MW)',
        xaxis=dict(tickmode='array', tickvals=list(range(24)), gridcolor='#e0e0e0'),
        yaxis=dict(gridcolor='#e0e0e0'),
        hovermode="x unified",
        legend_title_text='Series',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='#333',
        transition_duration=300
    )

    # --- Calculate PnL Summary (Top Section) ---
    total_pnl = 0
    pnl_summary_children = [html.H4("PnL Summary for Filtered Data", style={'textAlign': 'center'})]

    if selected_forecasts:
        if 'DA' in selected_forecasts:
            # DA_PnL column is already divided by 2
            da_pnl_total = dff['DA_PnL'].sum()
            total_pnl += da_pnl_total
            pnl_summary_children.append(html.P(f"DA PnL (vs Actual): {da_pnl_total:,.2f} €"))

        if 'IDA1' in selected_forecasts:
            # These columns are already divided by 2
            ida1_gran_pnl_total = dff['IDA1_Granularity_PnL'].sum()
            ida1_chng_pnl_total = dff['IDA1_Change_PnL'].sum()
            total_pnl += (ida1_gran_pnl_total + ida1_chng_pnl_total)
            pnl_summary_children.append(html.P(f"IDA1 Granularity PnL: {ida1_gran_pnl_total:,.2f} €"))
            # Rename "IDA1 Change PnL"
            pnl_summary_children.append(html.P(f"IDA1 Forecast Change PnL: {ida1_chng_pnl_total:,.2f} €"))

        if 'IDA2' in selected_forecasts:
            # This column is already divided by 2
            ida2_pnl_total = dff['IDA2_PnL'].sum() 
            total_pnl += ida2_pnl_total
            pnl_summary_children.append(html.P(f"IDA2 PnL (IDA1 vs IDA2 Trade): {ida2_pnl_total:,.2f} €"))

        if 'IDA3' in selected_forecasts:
            # This column is already divided by 2
            ida3_pnl_total = dff['IDA3_PnL'].sum()
            total_pnl += ida3_pnl_total
            pnl_summary_children.append(html.P(f"IDA3 PnL (IDA2 vs IDA3 Trade): {ida3_pnl_total:,.2f} €"))
        
        pnl_summary_children.append(html.Hr())
        pnl_summary_children.append(html.H5(f"Total Pnl (from selected): {total_pnl:,.2f} €", style={'fontWeight': 'bold'}))
    else:
        pnl_summary_children.append(html.P("Please select a forecast to see the PnL breakdown."))

    # --- Calculate Scenario PnL Table ---
    
    # Calculate base values from the filtered dataframe
    # DA Revenue is now divided by 2
    da_revenue = ((dff['DA Forecast'] * dff['Day Ahead Price']).sum()) / 2.0
    
    # These PnL values are already divided by 2 from the pre-calculation
    forecasting_cost = dff['DA_PnL'].sum() # This is the DA PnL
    ida1_gran_pnl = dff['IDA1_Granularity_PnL'].sum()
    ida1_chng_pnl = dff['IDA1_Change_PnL'].sum()
    ida2_pnl = dff['IDA2_PnL'].sum() # Already respects trading hours
    ida3_pnl = dff['IDA3_PnL'].sum() # Already respects trading hours
    
    # ONLY Balancing Risk Premium is divided by 2
    balancing_risk = (multiplier * dff['ROI Actual'].sum()) / 2.0

    # Calculate Total PnL for each scenario based on your new formula
    # Total PnL = Sum of all PnL components
    s1_total_pnl = forecasting_cost
    s2_total_pnl = forecasting_cost + ida1_gran_pnl + ida1_chng_pnl
    s3_total_pnl = forecasting_cost + ida1_gran_pnl + ida1_chng_pnl + ida2_pnl
    s4_total_pnl = forecasting_cost + ida1_gran_pnl + ida1_chng_pnl + ida2_pnl + ida3_pnl

    # Format values for the table
    def format_currency(value):
        return f"{value:,.2f} €"

    # Create the string for the DA Revenue summary line
    da_revenue_str = f"Total DA Revenue (Filtered): {da_revenue:,.2f} €"

    scenario_data = [
        {
            'Scenario': 'Scenario 1',
            # DA Revenue and Blank Column removed
            'DA Forecasting Cost': format_currency(forecasting_cost),
            'IDA1 Granularity PnL': '—',
            'IDA1 Forecast Change PnL': '—', # Renamed
            'IDA2 PnL': '—',
            'IDA3 PnL': '—',
            'Total PnL': format_currency(s1_total_pnl),
            'Balancing Risk Premium': format_currency(balancing_risk)
        },
        {
            'Scenario': 'Scenario 2',
            'DA Forecasting Cost': format_currency(forecasting_cost),
            'IDA1 Granularity PnL': format_currency(ida1_gran_pnl),
            'IDA1 Forecast Change PnL': format_currency(ida1_chng_pnl), # Renamed
            'IDA2 PnL': '—',
            'IDA3 PnL': '—',
            'Total PnL': format_currency(s2_total_pnl),
            'Balancing Risk Premium': format_currency(balancing_risk)
        },
        {
            'Scenario': 'Scenario 3',
            'DA Forecasting Cost': format_currency(forecasting_cost),
            'IDA1 Granularity PnL': format_currency(ida1_gran_pnl),
            'IDA1 Forecast Change PnL': format_currency(ida1_chng_pnl), # Renamed
            'IDA2 PnL': format_currency(ida2_pnl),
            'IDA3 PnL': '—',
            'Total PnL': format_currency(s3_total_pnl),
            'Balancing Risk Premium': format_currency(balancing_risk)
        },
        {
            'Scenario': 'Scenario 4',
            'DA Forecasting Cost': format_currency(forecasting_cost),
            'IDA1 Granularity PnL': format_currency(ida1_gran_pnl),
            'IDA1 Forecast Change PnL': format_currency(ida1_chng_pnl), # Renamed
            'IDA2 PnL': format_currency(ida2_pnl),
            'IDA3 PnL': format_currency(ida3_pnl),
            'Total PnL': format_currency(s4_total_pnl),
            'Balancing Risk Premium': format_currency(balancing_risk)
        }
    ]

    scenario_columns = [
        {'name': 'Scenario', 'id': 'Scenario'},
        # DA Revenue and Blank Column removed
        {'name': 'DA Forecasting Cost', 'id': 'DA Forecasting Cost'}, 
        {'name': 'IDA1 Granularity PnL', 'id': 'IDA1 Granularity PnL'}, 
        {'name': 'IDA1 Forecast Change PnL', 'id': 'IDA1 Forecast Change PnL'}, # Renamed
        {'name': 'IDA2 PnL', 'id': 'IDA2 PnL'},
        {'name': 'IDA3 PnL', 'id': 'IDA3 PnL'},
        {'name': 'Total PnL', 'id': 'Total PnL'},
        {'name': 'Balancing Risk Premium', 'id': 'Balancing Risk Premium'}
    ]

    return fig, pnl_summary_children, da_revenue_str, scenario_data, scenario_columns

# --- 5. Run the App ---
if __name__ == '__main__':
    # Changed debug=True to debug=False for production
    # Added host='0.0.0.0' and port=8080, common for deployment
    app.run(debug=False, host='0.0.0.0', port=8080)


# In[ ]:





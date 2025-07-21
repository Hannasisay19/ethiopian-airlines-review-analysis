import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

file_path = (r'datasets\labeled_data\ethiopian_airlines_overall_and_topic_sentiment.csv')
df = pd.read_csv(file_path)

df['year'] = df['year'].astype(str)
df['month'] = pd.Categorical(
    df['month'],
    categories=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
    ordered=True
)
df['year_month'] = df['year'] + '-' + df['month'].astype(str)
df['year_month_dt'] = pd.to_datetime(df['year'] + '-' + df['month'].astype(str), format='%Y-%b')

topic_options = [
    {'label': 'Cabin Crew', 'value': 'cabin_crew_sentiment'},
    {'label': 'Flight Delay', 'value': 'flight_delay_sentiment'},
    {'label': 'Luggage Handling', 'value': 'luggage_handling_sentiment'},
    {'label': 'Food Service', 'value': 'food_service_sentiment'},
    {'label': 'Seat Comfort', 'value': 'seat_comfort_sentiment'},
    {'label': 'Restroom Quality', 'value': 'restroom_quality_sentiment'},
    {'label': 'Airport Check', 'value': 'airport_check_sentiment'},
    {'label': 'Customer Service', 'value': 'customer_service_sentiment'},
    {'label': 'Value for Money', 'value': 'value_for_money_sentiment'},
    {'label': 'Inflight Entertainment', 'value': 'inflight_entertainment_sentiment'}
]

year_options = [{'label': 'All Years', 'value': 'ALL'}] + [
    {'label': y, 'value': y} for y in sorted(df['year'].unique())
]

source_options = [{'label': 'All Sources', 'value': 'ALL'}] + [
    {'label': s, 'value': s} for s in sorted(df['source'].unique())
]

color_discrete_map = {
    "Positive": "#1f77b4",
    "Negative": "#d62728",
    "Neutral": "#2ca02c",
    "neutral": "#2ca02c"
}

def make_topic_title(base, topic_value):
    topic_str = topic_value.replace('_sentiment', '').replace('_', ' ').title()
    return f"{base}: {topic_str}"

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H3("Topic & Overall Sentiment Dashboard"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Year"),
            dcc.Dropdown(id='year-dropdown', options=year_options, value='ALL', clearable=False),
        ], width=3),
        dbc.Col([
            html.Label("Select Topic"),
            dcc.Dropdown(id='topic-dropdown', options=topic_options, value='cabin_crew_sentiment', clearable=False),
        ], width=4),
        dbc.Col([
            html.Label("Select Source"),
            dcc.Dropdown(id='source-dropdown', options=source_options, value='ALL', clearable=False),
        ], width=3),
    ], className='mb-4'),

    # Pie chart: Full width
    dbc.Row([
        dbc.Col(dcc.Graph(id='topic-pie'), width=12),
    ]),

    # Topic charts row
    dbc.Row([
        dbc.Col([
            html.H6(id='topic-bar-title'),
            dcc.Graph(id='topic-bar')
        ], width=5),  # Changed from 6 to 5
        dbc.Col([
            html.H6(id='topic-trend-title'),
            dcc.Graph(id='topic-trend')
        ], width=7),  # Changed from 6 to 7
    ]),

    # Overall charts row
    dbc.Row([
        dbc.Col([
            html.H6("Overall Sentiment Bar Chart"),
            dcc.Graph(id='overall-bar')
        ], width=5),  # Changed from 6 to 5
        dbc.Col([
            html.H6("Overall Sentiment Trend Over Time"),
            dcc.Graph(id='overall-trend')
        ], width=7),  # Changed from 6 to 7
    ]),

    dash_table.DataTable(
        id='reviews-table',
        columns=[
            {'name': 'Year', 'id': 'year'},
            {'name': 'Month', 'id': 'month'},
            {'name': 'Source', 'id': 'source'},
            {'name': 'Review Title', 'id': 'review_title'},
            {'name': 'Overall Sentiment', 'id': 'overall_sentiment'},
            {'name': 'Topic Sentiment', 'id': 'topic_sentiment'}
        ],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'}
    )
], fluid=True)

@app.callback(
    [
        Output('topic-pie', 'figure'),
        Output('topic-bar', 'figure'),
        Output('topic-trend', 'figure'),
        Output('overall-bar', 'figure'),
        Output('overall-trend', 'figure'),
        Output('reviews-table', 'data'),
        Output('topic-bar-title', 'children'),
        Output('topic-trend-title', 'children'),
    ],
    [
        Input('year-dropdown', 'value'),
        Input('topic-dropdown', 'value'),
        Input('source-dropdown', 'value')
    ]
)
def update_dashboard(selected_year, selected_topic, selected_source):
    filtered_df = df.copy()
    if selected_year != 'ALL':
        filtered_df = filtered_df[filtered_df['year'] == selected_year]
    if selected_source != 'ALL':
        filtered_df = filtered_df[filtered_df['source'] == selected_source]

    # Pie Chart
    topic_sentiment_counts = filtered_df[selected_topic].value_counts().reset_index()
    topic_sentiment_counts.columns = ['Sentiment', 'Count']
    pie_fig = px.pie(
        topic_sentiment_counts,
        names='Sentiment',
        values='Count',
        title=f"Sentiment Distribution: {selected_topic.replace('_', ' ').title()} "
              f"({'All Years' if selected_year == 'ALL' else selected_year}, "
              f"{selected_source if selected_source != 'ALL' else 'All Sources'})",
        color='Sentiment',
        color_discrete_map=color_discrete_map
    )

    sentiment_order = ['Positive', 'Negative', 'Neutral']

    # Topic Bar Chart
    topic_bar_df = pd.DataFrame({'Sentiment': sentiment_order})
    topic_bar_df = topic_bar_df.merge(topic_sentiment_counts, on='Sentiment', how='left').fillna(0)
    topic_bar_df['Count'] = topic_bar_df['Count'].astype(int)
    topic_bar_title = make_topic_title("Topic Sentiment Bar Chart", selected_topic)
    topic_bar_fig = px.bar(
        topic_bar_df,
        x='Sentiment',
        y='Count',
        color='Sentiment',
        color_discrete_map=color_discrete_map,
        text='Count',
        category_orders={'Sentiment': sentiment_order}
    )
    topic_bar_fig.update_traces(marker_line_width=0, textposition='inside')
    topic_bar_fig.update_yaxes(range=[0, max(5, topic_bar_df['Count'].max() * 1.15)], title='Count')
    topic_bar_fig.update_xaxes(title='Sentiment')
    topic_bar_fig.update_layout(
        showlegend=False, 
        height=320, 
        width=340,   # Make it narrower
        margin={'l':40, 'r':20, 't':80, 'b':60}
    )

    # Topic Trend Line
    trend = filtered_df.groupby(['year_month_dt', selected_topic]).size().reset_index(name='count')
    trend.columns = ['year_month_dt', 'Sentiment', 'count']
    topic_trend_title = make_topic_title("Topic Sentiment Trend Over Time", selected_topic)
    trend_fig = px.line(
        trend,
        x='year_month_dt',
        y='count',
        color='Sentiment',
        labels={'year_month_dt': 'Year-Month', 'count': 'Count'},
        color_discrete_map=color_discrete_map
    )
    trend_fig.update_layout(
        title=topic_trend_title, 
        height=350, 
        width=None,   # Use full space in its column
        margin={'l':40,'r':20,'t':40,'b':60}
    )

    # Overall Bar Chart
    overall_counts = filtered_df['overall_sentiment'].value_counts().reset_index()
    overall_counts.columns = ['Sentiment', 'Count']
    overall_bar_df = pd.DataFrame({'Sentiment': sentiment_order})
    overall_bar_df = overall_bar_df.merge(overall_counts, on='Sentiment', how='left').fillna(0)
    overall_bar_df['Count'] = overall_bar_df['Count'].astype(int)
    overall_bar_fig = px.bar(
        overall_bar_df,
        x='Sentiment',
        y='Count',
        color='Sentiment',
        color_discrete_map=color_discrete_map,
        text='Count',
        category_orders={'Sentiment': sentiment_order}
    )
    overall_bar_fig.update_traces(marker_line_width=0, textposition='inside')
    overall_bar_fig.update_yaxes(range=[0, max(5, overall_bar_df['Count'].max() * 1.15)], title='Count')
    overall_bar_fig.update_xaxes(title='Sentiment')
    overall_bar_fig.update_layout(
        showlegend=False, 
        height=320, 
        width=340,   # Make it narrower
        margin={'l':40, 'r':20, 't':80, 'b':60}
    )

    # Overall Trend Line
    overall_trend = filtered_df.groupby(['year_month_dt', 'overall_sentiment']).size().reset_index(name='count')
    overall_trend.columns = ['year_month_dt', 'Sentiment', 'count']
    overall_trend_fig = px.line(
        overall_trend,
        x='year_month_dt',
        y='count',
        color='Sentiment',
        labels={'year_month_dt': 'Year-Month', 'count': 'Count'},
        color_discrete_map=color_discrete_map
    )
    overall_trend_fig.update_layout(
        title="Overall Sentiment Trend Over Time", 
        height=350, 
        width=None,   # Use full space in its column
        margin={'l':40,'r':20,'t':40,'b':60}
    )

    # Table
    table_data = filtered_df[['year', 'month', 'source', 'review_title', 'overall_sentiment', selected_topic]].rename(
        columns={selected_topic: 'topic_sentiment'}).to_dict('records')

    return (pie_fig, topic_bar_fig, trend_fig, overall_bar_fig, overall_trend_fig, table_data,
            topic_bar_title, topic_trend_title)

if __name__ == "__main__":
    app.run(debug=True, port=8057)

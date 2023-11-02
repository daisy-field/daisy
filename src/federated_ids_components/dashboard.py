"""
    A simple server that should collect the evaluation metrics from the clients and process them

    Author: Seraphin Zunzer
    Modified: 09.05.22
"""
# FIXME (everything)
import logging
import threading

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Output, Input

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class Dashboard(threading.Thread):
    """
    Class for creating the Dashboard in dash to plot the evaluation metrics.
    Static values are used for a simple working interface. These variables should be updated with the live data,
    then the canges will be visible in the dashboard.

    """

    _app = dash.Dash(__name__, update_title=None, external_stylesheets=[dbc.themes.SLATE])
    _app.title = "Federated Learning"

    x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    fpr = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    tpr = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    acc = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    client_list = [["client_2", 58666, "up", 0, 0, 0, 0], ["client_5", 58999, "up", 0, 0, 0, 0]]
    train_c5 = False
    train_c2 = False

    def __init__(self):
        threading.Thread.__init__(self)
        self._app.layout = html.Div(
            [

                html.Br(),
                dbc.Card(
                    dbc.CardBody([
                        html.H1("Federated Intrusion Detection System for the Edge", style={'margin': "0.5cm"}),
                    ]), style={'margin': "0.5cm"}),
                html.Br(),
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Average Performance:", className="card-title"),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="ACC_graph", animate=True, config={"displayModeBar": False},
                                          style={'display': 'inline-block'}),
                            ], width=4),
                            dbc.Col([
                                dcc.Graph(id="TPR_graph", animate=True, config={"displayModeBar": False},
                                          style={'display': 'inline-block'}),

                            ], width=4),
                            dbc.Col([
                                dcc.Graph(id="FPR_graph", animate=True, config={"displayModeBar": False},
                                          style={'display': 'inline-block'}),
                            ], width=4),
                            dcc.Interval(id="graph-update", interval=1000, n_intervals=0),
                        ], align='center'),
                    ]), style={'margin': "0.5cm"}),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            dbc.Card(
                                dbc.CardBody([
                                    html.Div(id='client_2_output')
                                ]))
                        ]),
                        dbc.Row([
                            dbc.Card(
                                dbc.CardBody([
                                    html.Div(id='client_5_output')
                                ])),
                        ], style={'margin-top': '5px'})
                    ], width=8),
                    dbc.Col([
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Data Distribution in Training:", className="card-title",
                                        style={"height": "25px", "display": "flex", "justify-content": "center"}),
                                dcc.Graph(id="pie-graph", config={"displayModeBar": False}, ),

                            ])),
                    ], width=4),

                ], style={'margin': "0.5cm"})
            ])

        if self._app is not None and hasattr(self, "callbacks"):
            self.callbacks(self._app)

    def callbacks(self, _app):
        """
        Function to define interface callbacks.

        :param _app:
        :return:
        """
        max_display = 30

        @_app.callback(
            Output('pie-graph', 'figure'),
            [Input('graph-update', "n_intervals")])
        def piechart(n):
            """
            Callback to show pie chart.

            :param n:
            :return:
            """
            df = pd.DataFrame(
                {"Traffic": ["Client 2 - Normal", "Client 2 - Anomalies", "Client 5 - Normal", "Client 5 - Anomalies"],
                 "Samples": [0.25, 0.25, 0.25, 0.25]})
            fig = px.pie(df, values="Samples", labels=["Client 2 - Normal", "Client 2 - Anomalies", "Client 5 - Normal",
                                                       "Client 5 - Anomalies"],
                         names=["Client 2 - Normal", "Client 2 - Anomalies", "Client 5 - Normal",
                                "Client 5 - Anomalies"], hole=.5,
                         color_discrete_map=['#1f77b4', '#b41f2d', '#185a88', '#881822'])
            fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)')
            fig.update_layout(
                legend=dict(
                    font=dict(
                        size=12,
                        color="white",
                    ),

                )
            )

            return fig

        @_app.callback(Output('client_2_output', 'children'),
                       [Input('graph-update', 'n_intervals')])
        def client_2(n):
            """
            Callback to show Client 2 table on dashboard.

            :param n: number of callback, automatic parameter set by interval component
            :return:
            """
            for i in self.client_list:
                if "2" in i[0]:
                    table_header = [html.Thead(html.Tr(
                        [html.Th("True-Positives"), html.Th("False-Negatives"), html.Th("False-Positives"),
                         html.Th("True-Negatives")]))]
                    row1 = html.Tr([html.Td(i[3]), html.Td(i[4]), html.Td(i[5]), html.Td(i[6])])
                    table_body = [html.Tbody([row1])]
                    if i[2] == "up":
                        state = html.Div([
                            html.H4(f"Client 2 Performance: {'🔄' if self.train_c2 else ' '}", className="card-title",
                                    style={"height": "25px", "display": "flex", "justify-content": "center"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "grey",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "grey",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "green",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(f"Available on Port: {i[1]}  - 🔒 Traffic Encrypted",
                                      style={"height": "25px", "display": "block"}),
                            html.Br(), ])
                    elif i[2] == "no data":
                        state = html.Div([
                            html.H4("Client 2 Performance:", className="card-title",
                                    style={"height": "25px", "display": "flex", "justify-content": "center"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "grey",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "yellow",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "grey",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(f"No Data available. Connected on Port: {i[1]}",
                                      style={"height": "25px", "display": "block"}),
                            html.Br(), ])
                    else:
                        state = html.Div([
                            html.H4("Client 2 Performance:", className="card-title",
                                    style={"height": "25px", "display": "flex", "justify-content": "center"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "red",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "grey",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "grey",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(f"Not connected", style={"height": "25px", "display": "block"}),
                            html.Br(), ])
                    return html.Div([
                        state,
                        dbc.Table(table_header + table_body, bordered=True)
                    ])

            table_header = [html.Thead(html.Tr(
                [html.Th("True-Positives"), html.Th("False-Negatives"), html.Th("False-Positives"),
                 html.Th("True-Negatives")]))]
            row1 = html.Tr([html.Td("-"), html.Td("-"), html.Td("-"), html.Td("-")])
            table_body = [html.Tbody([row1])]
            state = html.Div([
                html.H4("Client 2 Performance:", className="card-title",
                        style={"height": "25px", "display": "flex", "justify-content": "center"}),
                html.Span(style={"height": "25px", "width": "25px", "background-color": "red", "border-radius": "50%",
                                 "display": "inline-block", 'margin': "5px"}),
                html.Span(style={"height": "25px", "width": "25px", "background-color": "grey", "border-radius": "50%",
                                 "display": "inline-block", 'margin': "5px"}),
                html.Span(style={"height": "25px", "width": "25px", "background-color": "grey", "border-radius": "50%",
                                 "display": "inline-block", 'margin': "5px"}),
                html.Span(f"Not connected", style={"height": "25px", "display": "block"}),
                html.Br(), ])
            return html.Div([
                state,
                dbc.Table(table_header + table_body, bordered=True)
            ])

        @_app.callback(Output('client_5_output', 'children'),
                       [Input('graph-update', 'n_intervals')])
        def client_5(n):
            """
            Callback to show Client 5 table on dashboard.

            :param n: number of callback, automatic parameter set by interval component
            :return:
            """

            for i in self.client_list:
                print(i)
                if "5" in i[0]:
                    table_header = [html.Thead(html.Tr(
                        [html.Th("True-Positives"), html.Th("False-Negatives"), html.Th("False-Positives"),
                         html.Th("True-Negatives")]))]
                    row1 = html.Tr([html.Td(i[3]), html.Td(i[4]), html.Td(i[5]), html.Td(i[6])])
                    table_body = [html.Tbody([row1])]
                    if i[2] == "up":
                        state = html.Div([
                            html.H4(f"Client 5 Performance: {'🔄' if self.train_c5 else ' '}", className="card-title",
                                    style={"height": "25px", "display": "flex", "justify-content": "center"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "grey",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "grey",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "green",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(f"Available on Port: {i[1]}  - 🔒 Traffic Encrypted",
                                      style={"height": "25px", "display": "block"}),
                            html.Br(), ])
                    elif i[2] == "no data":
                        state = html.Div([
                            html.H4("Client 5 Performance:", className="card-title",
                                    style={"height": "25px", "display": "flex", "justify-content": "center"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "grey",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "yellow",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "grey",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(f"No Data available. Connected on Port: {i[1]}",
                                      style={"height": "25px", "display": "block"}),
                            html.Br(), ])
                    else:
                        state = html.Div([
                            html.H4("Client 5 Performance:", className="card-title",
                                    style={"height": "25px", "display": "flex", "justify-content": "center"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "red",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "grey",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(style={"height": "25px", "width": "25px", "background-color": "grey",
                                             "border-radius": "50%", "display": "inline-block", 'margin': "5px"}),
                            html.Span(f"Not connected", style={"height": "25px", "display": "block"}),
                            html.Br(), ])
                    return html.Div([
                        state,
                        dbc.Table(table_header + table_body, bordered=True)
                    ])

            table_header = [html.Thead(html.Tr(
                [html.Th("True-Positives"), html.Th("False-Negatives"), html.Th("False-Positives"),
                 html.Th("True-Negatives")]))]
            row1 = html.Tr([html.Td("-"), html.Td("-"), html.Td("-"), html.Td("-")])
            table_body = [html.Tbody([row1])]
            state = html.Div([
                html.H4("Client 5 Performance:", className="card-title",
                        style={"height": "25px", "display": "flex", "justify-content": "center"}),
                html.Span(style={"height": "25px", "width": "25px", "background-color": "red", "border-radius": "50%",
                                 "display": "inline-block", 'margin': "5px"}),
                html.Span(style={"height": "25px", "width": "25px", "background-color": "grey", "border-radius": "50%",
                                 "display": "inline-block", 'margin': "5px"}),
                html.Span(style={"height": "25px", "width": "25px", "background-color": "grey", "border-radius": "50%",
                                 "display": "inline-block", 'margin': "5px"}),
                html.Span(f"Not connected", style={"height": "25px", "display": "block"}),
                html.Br(), ])
            return html.Div([
                state,
                dbc.Table(table_header + table_body, bordered=True)
            ])

        @_app.callback(
            Output("FPR_graph", "figure"), [Input("graph-update", "n_intervals")],
        )
        def update_fpr_scatter(n):
            """
            Callback to show scatter plot for false positive rate.
            :param n:
            :return:
            """
            data = plotly.graph_objs.Scatter(
                x=self.x_axis[-max_display:], y=self.fpr[-max_display:], name='FPR!', mode="lines+markers"
            )
            return {
                "data": [data],
                "layout": go.Layout(
                    xaxis=dict(range=[min(self.x_axis[-max_display:]), max(self.x_axis[-max_display:])]),
                    yaxis=dict(range=[0, 1]),
                    template='plotly_dark',
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    title="False-Positive Rate"
                ),
            }

        @_app.callback(
            Output("TPR_graph", "figure"), [Input("graph-update", "n_intervals")],
        )
        def update_tpr_scatter(n):
            """
            Callbacks to show scatter plot for true positive rate.
            :param n:
            :return:
            """
            data = plotly.graph_objs.Scatter(
                x=self.x_axis[-max_display:], y=self.tpr[-max_display:], name='FPR!', mode="lines+markers"
            )
            return {
                "data": [data],
                "layout": go.Layout(
                    xaxis=dict(range=[min(self.x_axis[-max_display:]), max(self.x_axis[-max_display:])]),
                    yaxis=dict(range=[0, 1]),
                    template='plotly_dark',
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    title="True-Positive Rate",
                ),
            }

        @_app.callback(
            Output("ACC_graph", "figure"), [Input("graph-update", "n_intervals")],
        )
        def update_acc_scatter(n):
            """
            Callback to show scatter plot for accuracy.
            :param n:
            :return:
            """
            data = plotly.graph_objs.Scatter(
                x=self.x_axis[-max_display:], y=self.acc[-max_display:], name='FPR!', mode="lines+markers"
            )
            return {
                "data": [data],
                "layout": go.Layout(
                    xaxis=dict(range=[min(self.x_axis[-max_display:]), max(self.x_axis[-max_display:])]),
                    yaxis=dict(range=[0, 1]),
                    template='plotly_dark',
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    title="Accuracy"

                ),
            }

    def run(self):
        print("Starting Evaluation Server")
        self._app.run_server(port=8050)


class testDashboard(threading.Thread):
    _app = dash.Dash(__name__, update_title=None, external_stylesheets=[dbc.themes.SLATE])
    _app.title = "Federated Learning"

    _app.layout = html.Div([
        html.H1(children='WELCOME TO THE FEDERATED IDS EVALUATION SERVER')
    ])

    def run(self):
        print("Starting Evaluation Server")
        self._app.run_server(port=8050)

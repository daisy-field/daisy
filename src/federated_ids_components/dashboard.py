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
from dash.dependencies import Output, Input, State
from dash_bootstrap_templates import ThemeSwitchAIO, load_figure_template

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class Dashboard(threading.Thread):
    """
    Class for creating the Dashboard in dash to plot the evaluation metrics.
    Static values are used for a simple working interface. These variables should be updated with the live data,
    then the canges will be visible in the dashboard.

    """

    _app = dash.Dash(__name__, update_title=None)
    _app.title = "Federated Learning"

    x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    fpr = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    tpr = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    acc_df = {'addr1':[0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], 'acc2': [0.8, 0.3, 0.1, 0.5, 0.7, 0.5, 0.2, 0.4, 0.1]}

    _dark_template = "plotly_dark"
    _light_template = "ggplot2"

    def __init__(self):
        threading.Thread.__init__(self)
        self._app.layout = html.Div(
            [
                html.Br(),
                html.Div([ThemeSwitchAIO(aio_id="theme", themes=[dbc.themes.COSMO, dbc.themes.DARKLY])],
                    style={'float': "right", 'margin-right':"0.5cm"}),
                html.Br(),
                dbc.Card(
                    dbc.CardBody([
                            html.H1("Federated Intrusion Detection System for the Edge", style={'margin': "0.5cm"}),
                    ]), style={'margin': "0.5cm"}),

                html.Br(),
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="chart_acc", style={'display': 'inline-block', 'width':'100%'}),
                            ]),
                            dbc.Col([
                                dcc.Graph(id="chart_rec", style={'display': 'inline-block', 'width':'100%'}),
                            ]),
                        ], align='center'),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="chart_prec", style={'display': 'inline-block', 'width':'100%'}),
                            ]),
                            dbc.Col([
                                dcc.Graph(id="chart_f1", style={'display': 'inline-block', 'width':'100%'}),
                            ])
                        ], align='center'),
                        #dbc.Row([
                        #    dbc.Col([
                        #        dcc.Graph(id="chart_npv", style={'display': 'inline-block'}),
                        #    ], width=4),
                        #    dbc.Col([
                        #        dcc.Graph(id="chart_fnr", style={'display': 'inline-block'}),
                        #    ], width=4),
                        #], align='center'),
                        #dbc.Row([
                        #    dbc.Col([
                        #        dcc.Graph(id="chart_fpr", style={'display': 'inline-block'}),
                        #    ], width=4),
                        #    dbc.Col([
                        #        dcc.Graph(id="chart_f1", style={'display': 'inline-block'}),
                        #    ], width=4),
                        #
                        #], align='center'),
                        dcc.Interval(id="graph-update", interval=1000, n_intervals=0),
                    ]), style={'margin': "0.5cm"}),

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
            Output("chart_acc", "figure"),
            [Input("graph-update", "n_intervals"),
            Input(ThemeSwitchAIO.ids.switch("theme"), "value")],
        )
        def update_acc(n, toggle):
            """
            Callbacks to show scatter plot for true positive rate.
            :param n:
            :return:
            """
            fig = go.Figure()
            for i in self.acc_df:
                fig.add_trace(go.Scatter(y=self.acc_df[i],
                                     mode='lines',
                                     name=i))
            fig.update_layout(
                    title='Accuracy',
                    template= self._light_template if toggle else self._dark_template,
                    plot_bgcolor = 'rgba(0, 0, 0, 0)',
                    paper_bgcolor = 'rgba(0, 0, 0, 0)',
                    font = dict(size=18),
            )

            return fig

        @_app.callback(
            Output("chart_rec", "figure"),
            [Input("graph-update", "n_intervals"),
             Input(ThemeSwitchAIO.ids.switch("theme"), "value")],
        )
        def update_rec(n, toggle):
            """
            Callbacks to show scatter plot for true positive rate.
            :param n:
            :return:
            """
            print(toggle)
            fig = go.Figure()
            for i in self.acc_df:
                fig.add_trace(go.Scatter(y=self.acc_df[i],
                                         mode='lines',
                                         name=i))
            fig.update_layout(
                title='Recall',
                template=self._light_template if toggle else self._dark_template,
                plot_bgcolor = 'rgba(0, 0, 0, 0)',
                paper_bgcolor = 'rgba(0, 0, 0, 0)',
                font=dict(size=18),
            )

            return fig

        @_app.callback(
            Output("chart_prec", "figure"),
            [Input("graph-update", "n_intervals"),
             Input(ThemeSwitchAIO.ids.switch("theme"), "value")],
        )
        def update_prec(n, toggle):
            """
            Callbacks to show scatter plot for true positive rate.
            :param n:
            :return:
            """
            print(toggle)
            fig = go.Figure()
            for i in self.acc_df:
                fig.add_trace(go.Scatter(y=self.acc_df[i],
                                         mode='lines',
                                         name=i))
            fig.update_layout(
                title='Precision',
                template=self._light_template if toggle else self._dark_template,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(size=18),
            )
            return fig

        @_app.callback(
            Output("chart_f1", "figure"),
            [Input("graph-update", "n_intervals"),
             Input(ThemeSwitchAIO.ids.switch("theme"), "value")],
        )
        def update_f1(n, toggle):
            """
            Callbacks to show scatter plot for true positive rate.
            :param n:
            :return:
            """
            print(toggle)
            fig = go.Figure()
            for i in self.acc_df:
                fig.add_trace(go.Scatter(y=self.acc_df[i],
                                         mode='lines',
                                         name=i))
            fig.update_layout(
                title='F1-Score',
                template=self._light_template if toggle else self._dark_template,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(size=18),
            )
            return fig

        # @_app.callback(
        #     Output("chart_npv", "figure"),
        #     [Input("graph-update", "n_intervals"),
        #      Input(ThemeSwitchAIO.ids.switch("theme"), "value")],
        # )
        # def update_acc(n, toggle):
        #     """
        #     Callbacks to show scatter plot for true positive rate.
        #     :param n:
        #     :return:
        #     """
        #     print(toggle)
        #     fig = go.Figure()
        #     for i in self.acc_df:
        #         fig.add_trace(go.Scatter(y=self.acc_df[i],
        #                                  mode='lines',
        #                                  name=i))
        #     fig.update_layout(
        #         title='Net Present Value',
        #         template=self._light_template if toggle else self._dark_template,
        #         plot_bgcolor='rgba(0, 0, 0, 0)',
        #         paper_bgcolor='rgba(0, 0, 0, 0)')
        #     return fig
        #
        # @_app.callback(
        #     Output("chart_fnr", "figure"),
        #     [Input("graph-update", "n_intervals"),
        #      Input(ThemeSwitchAIO.ids.switch("theme"), "value")],
        # )
        # def update_acc(n, toggle):
        #     """
        #     Callbacks to show scatter plot for true positive rate.
        #     :param n:
        #     :return:
        #     """
        #     print(toggle)
        #     fig = go.Figure()
        #     for i in self.acc_df:
        #         fig.add_trace(go.Scatter(y=self.acc_df[i],
        #                                  mode='lines',
        #                                  name=i))
        #     fig.update_layout(
        #         title='False-Negative-Rate',
        #         template=self._light_template if toggle else self._dark_template,
        #         plot_bgcolor='rgba(0, 0, 0, 0)',
        #         paper_bgcolor='rgba(0, 0, 0, 0)')
        #     return fig
        #
        # @_app.callback(
        #     Output("chart_fpr", "figure"),
        #     [Input("graph-update", "n_intervals"),
        #      Input(ThemeSwitchAIO.ids.switch("theme"), "value")],
        # )
        # def update_acc(n, toggle):
        #     """
        #     Callbacks to show scatter plot for true positive rate.
        #     :param n:
        #     :return:
        #     """
        #     print(toggle)
        #     fig = go.Figure()
        #     for i in self.acc_df:
        #         fig.add_trace(go.Scatter(y=self.acc_df[i],
        #                                  mode='lines',
        #                                  name=i))
        #     fig.update_layout(
        #         title='False-Positive-Rate',
        #         template=self._light_template if toggle else self._dark_template,
        #         plot_bgcolor='rgba(0, 0, 0, 0)',
        #         paper_bgcolor='rgba(0, 0, 0, 0)')
        #     return fig
        #
        # @_app.callback(
        #     Output("chart_f1", "figure"),
        #     [Input("graph-update", "n_intervals"),
        #      Input(ThemeSwitchAIO.ids.switch("theme"), "value")],
        # )
        # def update_acc(n, toggle):
        #     """
        #     Callbacks to show scatter plot for true positive rate.
        #     :param n:
        #     :return:
        #     """
        #     print(toggle)
        #     fig = go.Figure()
        #     for i in self.acc_df:
        #         fig.add_trace(go.Scatter(y=self.acc_df[i],
        #                                  mode='lines',
        #                                  name=i))
        #     fig.update_layout(
        #         title='F1-Score',
        #         template=self._light_template if toggle else self._dark_template,
        #         plot_bgcolor='rgba(0, 0, 0, 0)',
        #         paper_bgcolor='rgba(0, 0, 0, 0)')
        #     return fig

    def run(self):
        print("Starting Evaluation Server")
        self._app.run_server(port=8050)


class testDashboard(threading.Thread):
    _app = dash.Dash(__name__, update_title=None, external_stylesheets=[dbc.themes.FLATLY])
    _app.title = "Federated Learning"

    _app.layout = html.Div([
        html.H1(children='WELCOME TO THE FEDERATED IDS EVALUATION SERVER')
    ])

    def run(self):
        print("Starting Evaluation Server")
        self._app.run_server(port=8050)

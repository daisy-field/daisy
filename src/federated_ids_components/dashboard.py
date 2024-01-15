"""
    A simple server that should collect the evaluation metrics from the clients and process them

    Author: Seraphin Zunzer
    Modified: 09.05.22

    TODO: CLEANUP IN WAY SIMILAR TO AGGREGATOR.PY
"""
import logging

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
from dash_bootstrap_templates import ThemeSwitchAIO, load_figure_template

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class Dashboard:
    """
    Class for creating the Dashboard in dash to plot the evaluation metrics.
    Static values are used for a simple working interface. These variables should be updated with the live data,
    then the canges will be visible in the dashboard.

    """

    _app = dash.Dash(__name__, update_title=None)
    _app.title = "Federated Learning"
    _dark_template = "slate"
    _light_template = "journal"  # lux"
    load_figure_template([_light_template, _dark_template])

    def __init__(self, evaluator, window_size):
        """ Initialize Dashboard

        :param evaluator: Evaluator object receiving the metrics
        :param window_size: Number of values to plot for each client, None if all
        """

        self._evaluator = evaluator
        self._window_size = window_size
        self._app.layout = html.Div(
            [
                html.Br(),
                html.Div([ThemeSwitchAIO(aio_id="theme", themes=[dbc.themes.COSMO, dbc.themes.DARKLY])],
                         style={'float': "right", 'margin-right': "0.5cm"}),
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
                                dcc.Graph(id="chart_acc", style={'display': 'inline-block', 'width': '100%'}),
                            ]),
                            dbc.Col([
                                dcc.Graph(id="chart_rec", style={'display': 'inline-block', 'width': '100%'}),
                            ]),
                        ], align='center'),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="chart_prec", style={'display': 'inline-block', 'width': '100%'}),
                            ]),
                            dbc.Col([
                                dcc.Graph(id="chart_f1", style={'display': 'inline-block', 'width': '100%'}),
                            ])
                        ], align='center'),
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

        @_app.callback(
            Output("chart_acc", "figure"),
            [Input("graph-update", "n_intervals"),
             Input(ThemeSwitchAIO.ids.switch("theme"), "value")],
            State("chart_acc", "figure")
        )
        def update_acc(n, toggle, figure):
            """
            Callbacks to show scatter plot for true positive rate.
            :param n:
            :return:
            """
            x_max, x_min = self.update_range(figure, len(self._evaluator._logged_metrics['x']))
            fig = go.Figure()
            for i in self._evaluator._logged_metrics['accuracy']:
                fig.add_trace(go.Scatter(x=self._evaluator._logged_metrics['x'],
                                         y=self._evaluator._logged_metrics['accuracy'][i],
                                         mode='lines',
                                         name=i))
            fig.update_layout(
                title='Accuracy',
                template=self._light_template if toggle else self._dark_template,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                yaxis_range=[0, 1],
                font=dict(size=18),
                uirevision=True,
                xaxis=dict(rangeslider=dict(visible=True),
                           range=[x_min, x_max],
                           tickvals=[x_min + 1, x_max],
                           tickfont=dict(size=14))
            )
            return fig

        @_app.callback(
            Output("chart_rec", "figure"),
            [Input("graph-update", "n_intervals"),
             Input(ThemeSwitchAIO.ids.switch("theme"), "value")],
            State("chart_rec", "figure")
        )
        def update_rec(n, toggle, figure):
            """
            Callbacks to show scatter plot for true positive rate.
            :param n:
            :return:
            """
            x_max, x_min = self.update_range(figure, len(self._evaluator._logged_metrics['x']))
            fig = go.Figure()
            for i in self._evaluator._logged_metrics['recall']:
                fig.add_trace(go.Scatter(x=self._evaluator._logged_metrics['x'],
                                         y=self._evaluator._logged_metrics['recall'][i],
                                         mode='lines',
                                         name=i))
            fig.update_layout(
                title='Recall',
                template=self._light_template if toggle else self._dark_template,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                yaxis_range=[0, 1],
                font=dict(size=18),
                uirevision=True,
                xaxis=dict(rangeslider=dict(visible=True),
                           range=[x_min, x_max],
                           tickvals=[x_min + 1, x_max],
                           tickfont=dict(size=14))
            )
            return fig

        @_app.callback(
            Output("chart_prec", "figure"),
            [Input("graph-update", "n_intervals"),
             Input(ThemeSwitchAIO.ids.switch("theme"), "value")],
            State("chart_prec", "figure")
        )
        def update_prec(n, toggle, figure):
            """
            Callbacks to show scatter plot for true positive rate.
            :param n:
            :return:
            """
            x_max, x_min = self.update_range(figure, len(self._evaluator._logged_metrics['x']))
            fig = go.Figure()
            for i in self._evaluator._logged_metrics['precision']:
                fig.add_trace(go.Scatter(x=self._evaluator._logged_metrics['x'],
                                         y=self._evaluator._logged_metrics['precision'][i],
                                         mode='lines',
                                         name=i))
            fig.update_layout(
                title='Precision',
                template=self._light_template if toggle else self._dark_template,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                yaxis_range=[0, 1],
                font=dict(size=18),
                uirevision=True,
                xaxis=dict(rangeslider=dict(visible=True),
                           range=[x_min, x_max],
                           tickvals=[x_min + 1, x_max],
                           tickfont=dict(size=14))
            )
            return fig

        @_app.callback(
            Output("chart_f1", "figure"),
            [Input("graph-update", "n_intervals"),
             Input(ThemeSwitchAIO.ids.switch("theme"), "value")],
            State("chart_f1", "figure")
        )
        def update_f1(n, toggle, figure):
            """
            Callbacks to show scatter plot for true positive rate.
            :param n:
            :return:
            """
            x_max, x_min = self.update_range(figure, len(self._evaluator._logged_metrics['x']))
            fig = go.Figure()
            for i in self._evaluator._logged_metrics['f1']:
                fig.add_trace(go.Scatter(x=self._evaluator._logged_metrics['x'],
                                         y=self._evaluator._logged_metrics['f1'][i],
                                         mode='lines',
                                         name=i))
            fig.update_layout(
                title='F1-Score',
                template=self._light_template if toggle else self._dark_template,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                yaxis_range=[0, 1],
                font=dict(size=18),
                uirevision=True,
                xaxis=dict(rangeslider=dict(visible=True),
                           range=[x_min, x_max],
                           tickvals=[x_min + 1, x_max],
                           tickfont=dict(size=14))
            )
            return fig

    def update_range(self, figure, len_x):
        try:
            x_min = figure['layout']['xaxis']['range'][0]
            x_max = figure['layout']['xaxis']['range'][1]
            if len_x - x_max <= 2:
                range = x_max - x_min
                x_max = len_x
                if x_min > 1:
                    x_min = x_max - range
        except:
            x_min = 0
            x_max = 10

        return x_max, x_min

    def run(self):
        self._app.run_server(port=8050)

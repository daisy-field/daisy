# Copyright (C) 2028 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import plotly.graph_objs as go

# Create your views here.
import plotly.offline as opy
from api.models import Aggregation
from dash_bootstrap_templates import load_figure_template
from django.http import HttpResponseRedirect
from django.shortcuts import render


# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


def index(request):
    # all = Accuracy.objects.all()
    # k =[]
    # for i in all:
    #    k.append(i.accuracy)
    # print(k)

    theme = request.session.get("is_dark_theme")

    accuracy = opy.plot(accuracy_plot(theme), auto_open=False, output_type="div")
    recall = opy.plot(recall_plot(theme), auto_open=False, output_type="div")
    precision = opy.plot(recall_plot(theme), auto_open=False, output_type="div")
    f1 = opy.plot(recall_plot(theme), auto_open=False, output_type="div")

    return render(
        request,
        "index.html",
        {
            "f1": f1,
            "precision": precision,
            "accuracy": accuracy,
            "recall": recall,
            "dark_theme": theme,
            "agg_status": getattr(Aggregation.objects.last(), "agg_status"),
            "agg_count": getattr(Aggregation.objects.last(), "agg_count"),
            "agg_time": getattr(Aggregation.objects.last(), "agg_time"),
            "eval_status": "Operational",
            "eval_count": "9",
            "eval_time": "02.03.2024",
        },
    )


_logged_metrics = {
    "accuracy": {
        "node_addr_1": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "node_addr_2": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    },
    "f1": {
        "node_addr_1": [0.3, 0.4, 0.1, 0.5, 0.6, 0.9],
        "node_addr_2": [0.1, 0.9, 0.2, 0.3, 0.5, 0.6],
    },
    "precision": {
        "node_addr_1": [0.2, 0.2, 0.3, 0.5, 0.7, 0.9],
        "node_addr_2": [0.2, 0.4, 0.5, 0.1, 0.5, 0.6],
    },
    "recall": {
        "node_addr_1": [0.7, 0.6, 0.1, 0.4, 0.9, 0.4],
        "node_addr_2": [0.5, 0.2, 0.6, 0.7, 0.9, 0.6],
    },
    "x": ["07:41:19", "07:41:20", "07:41:21", "07:41:22", "07:41:23", "07:41:24"],
}

_dark_template = "bootstrap_dark"  # "slate"
_light_template = "bootstrap"  # pulse"
load_figure_template([_light_template, _dark_template])


def accuracy_plot(dark_theme):
    k = []
    x = []
    c = 0
    # for i in Accuracy.objects.all():
    #    x.append(c)
    #    c += 1
    #    k.append(i.accuracy)

    # x_max, x_min = update_range(figure, len(self._evaluator._logged_metrics['x']))
    fig = go.Figure()
    # for i in _logged_metrics['accuracy']:
    fig.add_trace(
        go.Scatter(
            x=x,  # _logged_metrics['x'],
            y=k,  # logged_metrics['accuracy'][i],
            mode="lines",
        )
    )
    fig.update_layout(
        template=_dark_template if dark_theme else _light_template,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        yaxis_range=[0, 1],
        font=dict(size=10),
        uirevision=True,
        # yaxis={
        #    'showgrid': False},
        xaxis=dict(
            rangeslider=dict(visible=True),
            # range=[x_min, x_max],
            # tickvals=[x_min + 1, x_max],
            tickfont=dict(size=10),
            # showgrid= False,
        ),
    )
    return fig


def recall_plot(dark_theme):
    # x_max, x_min = update_range(figure, len(self._evaluator._logged_metrics['x']))
    fig = go.Figure()
    for i in _logged_metrics["accuracy"]:
        fig.add_trace(
            go.Scatter(
                x=_logged_metrics["x"],
                y=_logged_metrics["accuracy"][i],
                mode="lines",
                name=i,
            )
        )
    fig.update_layout(
        template=_dark_template if dark_theme else _light_template,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        yaxis_range=[0, 1],
        font=dict(size=10),
        uirevision=True,
        xaxis=dict(
            rangeslider=dict(visible=True),
            # range=[x_min, x_max],
            # tickvals=[x_min + 1, x_max],
            tickfont=dict(size=10),
        ),
    )
    return fig


def change_theme(request):
    if "is_dark_theme" in request.session:
        print("Light Theme")
        request.session["is_dark_theme"] = not request.session.get("is_dark_theme")
    else:
        print("Dark Theme")
        request.session["is_dark_theme"] = True
    return HttpResponseRedirect(request.META.get("HTTP_REFERER", "/"))


def alerts(request):
    theme = request.session.get("is_dark_theme")
    return render(request, "alerts.html", {"dark_theme": theme})


def aggregate(request):
    theme = request.session.get("is_dark_theme")
    return render(request, "aggregation.html", {"dark_theme": theme})


def evaluate(request):
    theme = request.session.get("is_dark_theme")
    return render(request, "evaluation.html", {"dark_theme": theme})


def nodes(request):
    theme = request.session.get("is_dark_theme")
    return render(request, "nodes.html", {"dark_theme": theme})


def terms(request):
    theme = request.session.get("is_dark_theme")
    return render(request, "tc.html", {"dark_theme": theme})


def privacy(request):
    theme = request.session.get("is_dark_theme")
    return render(request, "pp.html", {"dark_theme": theme})


def accuracy(request):
    theme = request.session.get("is_dark_theme")
    return render(request, "accuracy.html", {"dark_theme": theme})


def f1(request):
    theme = request.session.get("is_dark_theme")
    return render(request, "f1.html", {"dark_theme": theme})


def recall(request):
    theme = request.session.get("is_dark_theme")
    return render(request, "recall.html", {"dark_theme": theme})


def precision(request):
    theme = request.session.get("is_dark_theme")
    return render(request, "precision.html", {"dark_theme": theme})

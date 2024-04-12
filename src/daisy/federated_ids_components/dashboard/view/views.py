# Copyright (C) 2024 DAI-Labor and others
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

    theme = request.session.get("is_dark_theme")

    try:
        agg_status = getattr(Aggregation.objects.last(), "agg_status")
        agg_count = getattr(Aggregation.objects.last(), "agg_count")
        agg_time = getattr(Aggregation.objects.last(), "agg_time")
    except AttributeError:
        agg_status = "None"
        agg_count = 0
        agg_time = "None"

    return render(
        request,
        "index.html",
        {
            "dark_theme": theme,
            "agg_status": agg_status,
            "agg_count": agg_count,
            "agg_time": agg_time,
            "eval_status": "Operational",
            "eval_count": "9",
            "eval_time": "02.03.2024",
        },
    )


_dark_template = "bootstrap_dark"  # "slate"
_light_template = "bootstrap"  # pulse"
load_figure_template([_light_template, _dark_template])



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

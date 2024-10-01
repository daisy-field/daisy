# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from collections import defaultdict

from api.models import Aggregation, Prediction, Evaluation, Alerts, Metrics_long

# from dash_bootstrap_templates import load_figure_template
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect, get_object_or_404

import json


def index(request):
    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    try:
        agg_status = getattr(Aggregation.objects.last(), "agg_status")
        agg_count = getattr(Aggregation.objects.last(), "agg_count")
        agg_time = getattr(Aggregation.objects.last(), "agg_time")
    except AttributeError:
        agg_status = "None"
        agg_count = 0
        agg_time = "None"

    try:
        prediction_status = getattr(Prediction.objects.last(), "pred_status")
        prediction_count = getattr(Prediction.objects.last(), "pred_count")
        prediction_time = getattr(Prediction.objects.last(), "pred_time")
    except AttributeError:
        prediction_status = "None"
        prediction_count = 0
        prediction_time = "None"

    try:
        evaluation_status = getattr(Evaluation.objects.last(), "eval_status")
        evaluation_count = getattr(Evaluation.objects.last(), "eval_count")
        evaluation_time = getattr(Evaluation.objects.last(), "eval_time")
    except AttributeError:
        evaluation_status = "None"
        evaluation_count = 0
        evaluation_time = "None"

    return render(
        request,
        "index.html",
        {
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "agg_status": agg_status,
            "agg_count": agg_count,
            "agg_time": agg_time,
            "prediction_status": prediction_status,
            "prediction_count": prediction_count,
            "prediction_time": prediction_time,
            "evaluation_status": evaluation_status,
            "evaluation_count": evaluation_count,
            "evaluation_time": evaluation_time,
        },
    )


_dark_template = "bootstrap_dark"  # "slate"
_light_template = "bootstrap"  # pulse"
# load_figure_template([_light_template, _dark_template])


def change_theme(request):
    if "is_dark_theme" in request.session:
        request.session["is_dark_theme"] = not request.session.get("is_dark_theme")
    else:
        request.session["is_dark_theme"] = True
    return HttpResponseRedirect(request.META.get("HTTP_REFERER", "/"))


def change_smoothing(request):
    if "smoothing" in request.session:
        request.session["smoothing"] = not request.session.get("smoothing")
    else:
        request.session["smoothing"] = True
    return HttpResponseRedirect(request.META.get("HTTP_REFERER", "/"))


def change_interpolation(request):
    if "interpolation" in request.session:
        request.session["interpolation"] = not request.session.get("interpolation")
    else:
        request.session["interpolation"] = True
    return HttpResponseRedirect(request.META.get("HTTP_REFERER", "/"))


def alerts(request):
    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    alarm_alerts = Alerts.objects.filter(category="alert").filter(active=True)
    warning_alerts = Alerts.objects.filter(category="warning").filter(active=True)
    info_alerts = Alerts.objects.filter(category="info").filter(active=True)
    history = Alerts.objects.filter(active=False)

    return render(
        request,
        "alerts.html",
        {
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "alarms": alarm_alerts,
            "warnings": warning_alerts,
            "infos": info_alerts,
            "history": history,
        },
    )


def resolve(request, alert_id):
    alert = get_object_or_404(Alerts, id=alert_id)
    alert.active = False
    alert.save()
    return redirect("alerts")


def resolveAll(request):
    alerts = Alerts.objects.filter(category="alert")
    for i in alerts:
        i.active = False
        i.save()
    return redirect("alerts")


def delete(request, alert_id):
    alert = get_object_or_404(Alerts, id=alert_id)
    alert.delete()
    return redirect("alerts")


def deleteAll(request):
    alerts = Alerts.objects.filter(active=False)
    for i in alerts:
        i.delete()
    return redirect("alerts")


def restore(request, alert_id):
    alert = get_object_or_404(Alerts, id=alert_id)
    alert.active = True
    alert.save()
    return redirect("alerts")


def aggregate(request):
    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "model_aggregation.html",
        {"dark_theme": theme, "smoothing": smoothing, "interpolation": interpolation},
    )


def predict(request):
    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "prediction_aggregation.html",
        {"dark_theme": theme, "smoothing": smoothing, "interpolation": interpolation},
    )


def evaluate(request):
    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "evaluation_aggregation.html",
        {"dark_theme": theme, "smoothing": smoothing, "interpolation": interpolation},
    )


def nodes(request):
    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "nodes.html",
        {"dark_theme": theme, "smoothing": smoothing, "interpolation": interpolation},
    )


def terms(request):
    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "tc.html",
        {"dark_theme": theme, "smoothing": smoothing, "interpolation": interpolation},
    )


def privacy(request):
    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "pp.html",
        {"dark_theme": theme, "smoothing": smoothing, "interpolation": interpolation},
    )


def accuracy(request):
    data = Metrics_long.objects.all().values()  # or filter() as needed

    # Convert queryset to list of dictionaries (JSON-like structure)
    data_list = list(data)

    metrics = Metrics_long.objects.all().order_by("timestamp")
    unique_timestamps = sorted(metrics.values_list("timestamp", flat=True).distinct())
    unique_timestamps = [
        timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in unique_timestamps
    ]  # Format timestamps as strings

    print(unique_timestamps)

    node_data = defaultdict(
        lambda: [None] * len(unique_timestamps)
    )  # Initializes lists with 'None'
    for metric in metrics:
        timestamp_index = unique_timestamps.index(
            metric.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        )
        node_data[metric.address][timestamp_index] = metric.accuracy
    print(node_data)
    node_data = dict(node_data)

    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "accuracy.html",
        {
            "data": data_list,
            "unique_timestamps": unique_timestamps,
            "node_data": json.dumps(node_data),
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "metric_text": "Accuracy",
            "metric_name": "accuracy",
        },
    )


def f1(request):
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    theme = request.session.get("is_dark_theme")
    return render(
        request,
        "metrics.html",
        {
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "metric_text": "F1-Score",
            "metric_name": "f1",
        },
    )


def recall(request):
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    theme = request.session.get("is_dark_theme")
    return render(
        request,
        "metrics.html",
        {
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "metric_text": "Recall",
            "metric_name": "recall",
        },
    )


def precision(request):
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    theme = request.session.get("is_dark_theme")
    return render(
        request,
        "metrics.html",
        {
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "metric_text": "Precision",
            "metric_name": "precision",
        },
    )

# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from collections import defaultdict

from api.models import (
    Aggregation,
    Prediction,
    Evaluation,
    Alerts,
    Metrics_long,
    Metrics_export,
    Metrics,
    num_export_samples,
    num_longterm_samples,
    num_shortterm_samples,
)

from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse

import json
import csv


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
        "aggregation.html",
        {
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "server_count": "agg_count",
            "server_time": "agg_time",
            "server_text": "Aggregation",
            "server_url": "http://127.0.0.1:8000/aggregation/",
        },
    )


def predict(request):
    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "aggregation.html",
        {
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "server_count": "pred_count",
            "server_time": "pred_time",
            "server_text": "Prediction",
            "server_url": "http://127.0.0.1:8000/prediction/",
        },
    )


def evaluate(request):
    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "aggregation.html",
        {
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "server_count": "eval_count",
            "server_time": "eval_time",
            "server_text": "Evaluation",
            "server_url": "http://127.0.0.1:8000/evaluation/",
        },
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
    data = Metrics_long.objects.all().values()
    data_list = list(data)

    metrics = Metrics_long.objects.all().order_by("timestamp")
    unique_timestamps = sorted(metrics.values_list("timestamp", flat=True).distinct())
    unique_timestamps = [
        timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in unique_timestamps
    ]

    node_data = defaultdict(lambda: [None] * len(unique_timestamps))
    for metric in metrics:
        timestamp_index = unique_timestamps.index(
            metric.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        )
        node_data[metric.address][timestamp_index] = metric.accuracy

    node_data = dict(node_data)

    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "metrics.html",
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
    data = Metrics_long.objects.all().values()
    data_list = list(data)

    metrics = Metrics_long.objects.all().order_by("timestamp")
    unique_timestamps = sorted(metrics.values_list("timestamp", flat=True).distinct())
    unique_timestamps = [
        timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in unique_timestamps
    ]

    node_data = defaultdict(lambda: [None] * len(unique_timestamps))

    for metric in metrics:
        timestamp_index = unique_timestamps.index(
            metric.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        )
        node_data[metric.address][timestamp_index] = metric.f1

    node_data = dict(node_data)

    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "metrics.html",
        {
            "data": data_list,
            "unique_timestamps": unique_timestamps,
            "node_data": json.dumps(node_data),
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "metric_text": "F1-Score",
            "metric_name": "f1",
        },
    )


def recall(request):
    data = Metrics_long.objects.all().values()

    data_list = list(data)

    metrics = Metrics_long.objects.all().order_by("timestamp")
    unique_timestamps = sorted(metrics.values_list("timestamp", flat=True).distinct())
    unique_timestamps = [
        timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in unique_timestamps
    ]  # Format timestamps as strings

    node_data = defaultdict(
        lambda: [None] * len(unique_timestamps)
    )  # Initializes lists with 'None'
    for metric in metrics:
        timestamp_index = unique_timestamps.index(
            metric.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        )
        node_data[metric.address][timestamp_index] = metric.recall

    node_data = dict(node_data)

    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "metrics.html",
        {
            "data": data_list,
            "unique_timestamps": unique_timestamps,
            "node_data": json.dumps(node_data),
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "metric_text": "Recall",
            "metric_name": "recall",
        },
    )


def precision(request):
    data = Metrics_long.objects.all().values()

    data_list = list(data)

    metrics = Metrics_long.objects.all().order_by("timestamp")
    unique_timestamps = sorted(metrics.values_list("timestamp", flat=True).distinct())
    unique_timestamps = [
        timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in unique_timestamps
    ]

    node_data = defaultdict(lambda: [None] * len(unique_timestamps))
    for metric in metrics:
        timestamp_index = unique_timestamps.index(
            metric.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        )
        node_data[metric.address][timestamp_index] = metric.precision

    node_data = dict(node_data)

    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "metrics.html",
        {
            "data": data_list,
            "unique_timestamps": unique_timestamps,
            "node_data": json.dumps(node_data),
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "metric_text": "Precision",
            "metric_name": "precision",
        },
    )


def true_negative_rate(request):
    data = Metrics_long.objects.all().values()

    data_list = list(data)

    metrics = Metrics_long.objects.all().order_by("timestamp")
    unique_timestamps = sorted(metrics.values_list("timestamp", flat=True).distinct())
    unique_timestamps = [
        timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in unique_timestamps
    ]

    node_data = defaultdict(lambda: [None] * len(unique_timestamps))
    for metric in metrics:
        timestamp_index = unique_timestamps.index(
            metric.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        )
        node_data[metric.address][timestamp_index] = metric.true_negative_rate

    node_data = dict(node_data)

    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "metrics.html",
        {
            "data": data_list,
            "unique_timestamps": unique_timestamps,
            "node_data": json.dumps(node_data),
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "metric_text": "True Negative Rate",
            "metric_name": "true_negative_rate",
        },
    )


def false_negative_rate(request):
    data = Metrics_long.objects.all().values()

    data_list = list(data)

    metrics = Metrics_long.objects.all().order_by("timestamp")
    unique_timestamps = sorted(metrics.values_list("timestamp", flat=True).distinct())
    unique_timestamps = [
        timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in unique_timestamps
    ]

    node_data = defaultdict(lambda: [None] * len(unique_timestamps))
    for metric in metrics:
        timestamp_index = unique_timestamps.index(
            metric.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        )
        node_data[metric.address][timestamp_index] = metric.false_negative_rate

    node_data = dict(node_data)

    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "metrics.html",
        {
            "data": data_list,
            "unique_timestamps": unique_timestamps,
            "node_data": json.dumps(node_data),
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "metric_text": "False Negative Rate",
            "metric_name": "false_negative_rate",
        },
    )


def negative_predictive_value(request):
    data = Metrics_long.objects.all().values()

    data_list = list(data)

    metrics = Metrics_long.objects.all().order_by("timestamp")
    unique_timestamps = sorted(metrics.values_list("timestamp", flat=True).distinct())
    unique_timestamps = [
        timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in unique_timestamps
    ]

    node_data = defaultdict(lambda: [None] * len(unique_timestamps))
    for metric in metrics:
        timestamp_index = unique_timestamps.index(
            metric.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        )
        node_data[metric.address][timestamp_index] = metric.negative_predictive_value

    node_data = dict(node_data)

    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "metrics.html",
        {
            "data": data_list,
            "unique_timestamps": unique_timestamps,
            "node_data": json.dumps(node_data),
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "metric_text": "Negative Predictive Value",
            "metric_name": "negative_predictive_value",
        },
    )


def false_positive_rate(request):
    data = Metrics_long.objects.all().values()

    data_list = list(data)

    metrics = Metrics_long.objects.all().order_by("timestamp")
    unique_timestamps = sorted(metrics.values_list("timestamp", flat=True).distinct())
    unique_timestamps = [
        timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in unique_timestamps
    ]

    node_data = defaultdict(lambda: [None] * len(unique_timestamps))
    for metric in metrics:
        timestamp_index = unique_timestamps.index(
            metric.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        )
        node_data[metric.address][timestamp_index] = metric.false_positive_rate

    node_data = dict(node_data)

    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")
    return render(
        request,
        "metrics.html",
        {
            "data": data_list,
            "unique_timestamps": unique_timestamps,
            "node_data": json.dumps(node_data),
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "metric_text": "False Positive Rate",
            "metric_name": "false_positive_rate",
        },
    )


def data(request):
    metrics = Metrics_long.objects.all().order_by("address")
    nodes = sorted(metrics.values_list("address", flat=True).distinct())
    theme = request.session.get("is_dark_theme")
    smoothing = request.session.get("smoothing")
    interpolation = request.session.get("interpolation")

    trp = Metrics.objects.count() / num_shortterm_samples * 100
    trlp = Metrics_long.objects.count() / num_longterm_samples * 100
    trep = Metrics_export.objects.count() / num_export_samples * 100
    return render(
        request,
        "data.html",
        {
            "dark_theme": theme,
            "smoothing": smoothing,
            "interpolation": interpolation,
            "nodes": nodes,
            "total_records": Metrics.objects.count(),
            "total_records_long": Metrics_long.objects.count(),
            "total_records_export": Metrics_export.objects.count(),
            "export_percentage": trep,
            "long_percentage": trlp,
            "percentage": trp,
            "limit_long": num_longterm_samples,
            "limit_short": num_shortterm_samples,
            "limit_export": num_export_samples,
        },
    )


def freeStorage(request):
    Metrics_long.objects.all().delete()
    Metrics_export.objects.all().delete()
    Metrics.objects.all().delete()
    print("Freed Storage")
    return redirect("data")


def download_csv(request):
    if request.method == "POST":
        selected_metrics = request.POST.getlist("metrics")
        # selected_nodes = request.POST.getlist("nodes")
        metrics = Metrics_export.objects.order_by("timestamp")

        unique_timestamps = sorted(
            metrics.values_list("timestamp", flat=True).distinct()
        )
        unique_timestamps = [
            timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in unique_timestamps
        ]

        node_data = defaultdict(
            lambda: defaultdict(lambda: {metric: None for metric in selected_metrics})
        )

        for metric in metrics:
            timestamp_str = metric.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            for selected_metric in selected_metrics:
                node_data[metric.address][timestamp_str][selected_metric] = getattr(
                    metric, selected_metric
                )

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="data.csv"'
        writer = csv.writer(response)

        header = ["Timestamp"]
        for node in node_data.keys():
            for metric in selected_metrics:
                header.append(f"{metric.capitalize().replace('_', '-')}-{node}")
        writer.writerow(header)

        for timestamp in unique_timestamps:
            row = [timestamp]
            for node in node_data.keys():
                for metric in selected_metrics:
                    value = node_data[node][timestamp].get(metric)
                    row.append(
                        value if value is not None else ""
                    )  # Replace None with 'N/A'
            writer.writerow(row)

        return response

# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from django.urls import path

from daisy.federated_ids_components.dashboard.view import views

urlpatterns = [
    path("", views.index, name="index"),
    path("alerts/", views.alerts, name="alerts"),
    path("aggregate/", views.aggregate, name="aggregate"),
    path("evaluate/", views.evaluate, name="evaluate"),
    path("predict/", views.predict, name="predict"),
    path("nodes/", views.nodes, name="nodes"),
    path("terms/", views.terms, name="terms"),
    path("privacy/", views.privacy, name="privacy"),
    path("change_theme/", views.change_theme, name="change_theme"),
    path("change_smoothing/", views.change_smoothing, name="change_smoothing"),
    path(
        "change_interpolation/", views.change_interpolation, name="change_interpolation"
    ),
    path("accuracy/", views.accuracy, name="accuracy"),
    path("recall/", views.recall, name="recall"),
    path("precision/", views.precision, name="precision"),
    path("f1-score/", views.f1, name="f1-score"),
    path("true_negative_rate/", views.true_negative_rate, name="true_negative_rate"),
    path("false_negative_rate/", views.false_negative_rate, name="false_negative_rate"),
    path(
        "negative_predictive_value/",
        views.negative_predictive_value,
        name="negative_predictive_value",
    ),
    path("false_positive_rate/", views.false_positive_rate, name="false_positive_rate"),
    path("resolve/<uuid:alert_id>/", views.resolve, name="resolve"),
    path("restore/<uuid:alert_id>/", views.restore, name="restore"),
    path("delete/<uuid:alert_id>/", views.delete, name="delete"),
    path("deleteAll/", views.deleteAll, name="deleteAll"),
    path("resolveAll/", views.resolveAll, name="resolveAll"),
    path("download-csv/", views.download_csv, name="download_csv"),
    path("data/", views.data, name="data"),
    path("freestorage", views.freeStorage, name="freeStorage"),
]

# Copyright (C) 2024 DAI-Labor and others
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
    path("accuracy/", views.accuracy, name="accuracy"),
    path("recall/", views.recall, name="recall"),
    path("precision/", views.precision, name="precision"),
    path("f1-score/", views.f1, name="f1-score"),
    path("resolve/<uuid:alert_id>/", views.resolve, name="resolve"),
    path("restore/<uuid:alert_id>/", views.restore, name="restore"),
    path("delete/<uuid:alert_id>/", views.delete, name="delete"),
]

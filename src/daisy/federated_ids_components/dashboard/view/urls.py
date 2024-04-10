# Copyright (C) 2028 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("alerts/", views.alerts, name="alerts"),
    path("aggregate/", views.aggregate, name="aggregate"),
    path("evaluate/", views.evaluate, name="evaluate"),
    path("nodes/", views.nodes, name="nodes"),
    path("terms/", views.terms, name="terms"),
    path("privacy/", views.privacy, name="privacy"),
    path("change_theme/", views.change_theme, name="change_theme"),
    path("accuracy/", views.accuracy, name="accuracy"),
    path("recall/", views.recall, name="recall"),
    path("precision/", views.precision, name="precision"),
    path("f1-score/", views.f1, name="f1-score"),
]

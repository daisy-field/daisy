# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from django.urls import include, path
from rest_framework import routers

from . import views

router = routers.DefaultRouter()
router.register(r"metrics", views.MetricsSerializerView)
router.register(r"metricslong", views.MetricsLongSerializerView)
router.register(r"aggregation", views.AggregationSerializerView)
router.register(r"prediction", views.PredictionSerializerView)
router.register(r"alert", views.AlertsSerializerView)
router.register(r"node", views.NodeSerializerView)
router.register(r"evaluation", views.EvaluationSerializerView)


urlpatterns = [
    path("", include(router.urls)),
    path("api-auth/", include("rest_framework.urls", namespace="rest_framework")),
]

urlpatterns += router.urls

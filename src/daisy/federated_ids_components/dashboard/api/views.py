# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from rest_framework import permissions, viewsets

from api.serializers import (
    MetricsSerializer,
    AggregationSerializer,
    PredictionSerializer,
    AlertsSerializer,
    NodeSerializer,
    EvaluationSerializer,
    Evaluation,
    Metrics,
    Aggregation,
    Alerts,
    Prediction,
    Node,
    Metrics_long,
    MetricsLongSerializer,
)


class MetricsSerializerView(viewsets.ModelViewSet):
    queryset = Metrics.objects.all()
    serializer_class = MetricsSerializer
    permission_classes = [permissions.AllowAny]


class MetricsLongSerializerView(viewsets.ModelViewSet):
    queryset = Metrics_long.objects.all()
    serializer_class = MetricsLongSerializer
    permission_classes = [permissions.AllowAny]


class AggregationSerializerView(viewsets.ModelViewSet):
    queryset = Aggregation.objects.all()
    serializer_class = AggregationSerializer
    permission_classes = [permissions.AllowAny]


class EvaluationSerializerView(viewsets.ModelViewSet):
    queryset = Evaluation.objects.all()
    serializer_class = EvaluationSerializer
    permission_classes = [permissions.AllowAny]


class PredictionSerializerView(viewsets.ModelViewSet):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    permission_classes = [permissions.AllowAny]


class AlertsSerializerView(viewsets.ModelViewSet):
    queryset = Alerts.objects.all()
    serializer_class = AlertsSerializer
    permission_classes = [permissions.AllowAny]


class NodeSerializerView(viewsets.ModelViewSet):
    queryset = Node.objects.all()
    serializer_class = NodeSerializer
    permission_classes = [permissions.AllowAny]

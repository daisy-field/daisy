# Copyright (C) 2024 DAI-Labor and others
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
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = Metrics.objects.all()
    serializer_class = MetricsSerializer
    permission_classes = [permissions.AllowAny]


class MetricsLongSerializerView(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = Metrics_long.objects.all()
    serializer_class = MetricsLongSerializer
    permission_classes = [permissions.AllowAny]


class AggregationSerializerView(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = Aggregation.objects.all()
    serializer_class = AggregationSerializer
    permission_classes = [permissions.AllowAny]


class EvaluationSerializerView(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = Evaluation.objects.all()
    serializer_class = EvaluationSerializer
    permission_classes = [permissions.AllowAny]


class PredictionSerializerView(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    permission_classes = [permissions.AllowAny]


class AlertsSerializerView(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = Alerts.objects.all()
    serializer_class = AlertsSerializer
    permission_classes = [permissions.AllowAny]


class NodeSerializerView(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = Node.objects.all()
    serializer_class = NodeSerializer
    permission_classes = [permissions.AllowAny]

# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from rest_framework import permissions, viewsets

from .serializers import MetricsSerializer, AggregationSerializer,EvaluationSerializer, AlertsSerializer, NodeSerializer, Metrics, Aggregation, Alerts, Evaluation, Node


class MetricsSerializerView(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = Metrics.objects.all()
    serializer_class = MetricsSerializer
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

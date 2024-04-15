# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from rest_framework import serializers

from .models import *


class MetricsSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Metrics
        fields = ["address", "accuracy", "recall", "precision", "f1", "timestamp"]


class AggregationSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Aggregation
        fields = ["agg_status", "agg_count", "agg_time"]


class EvaluationSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Evaluation
        fields = ["eval_status", "eval_count", "eval_time"]


class AlertsSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Alerts
        fields = ["category", "active", "timestamp", "message"]


class NodeSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Node
        fields = ["address"]

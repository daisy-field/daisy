# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from rest_framework import serializers

from api.models import (
    Metrics,
    Aggregation,
    Alerts,
    Prediction,
    Node,
    Evaluation,
    Metrics_long,
)


class MetricsSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Metrics
        fields = [
            "address",
            "accuracy",
            "recall",
            "precision",
            "f1",
            "timestamp",
            "true_negative_rate",
            "false_negative_rate",
            "negative_predictive_value",
            "false_positive_rate",
        ]


class MetricsLongSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Metrics_long
        fields = [
            "address",
            "accuracy",
            "recall",
            "precision",
            "f1",
            "timestamp",
            "true_negative_rate",
            "false_negative_rate",
            "negative_predictive_value",
            "false_positive_rate",
        ]


class AggregationSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Aggregation
        fields = ["agg_status", "agg_count", "agg_time", "agg_nodes"]


class PredictionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Prediction
        fields = ["pred_status", "pred_count", "pred_time", "pred_nodes"]


class EvaluationSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Evaluation
        fields = ["eval_status", "eval_count", "eval_time", "eval_nodes"]


class AlertsSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Alerts
        fields = ["id", "address", "category", "active", "timestamp", "message"]


class NodeSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Node
        fields = ["address"]

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from rest_framework import serializers
from .models import *


class AccuracySerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Accuracy
        fields = ['accuracy']


class RecallSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Accuracy
        fields = ['recall']

class PrecisionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Accuracy
        fields = ['precision']

class F1Serializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Accuracy
        fields = ['f1']

class AggregationSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Aggregation
        fields = ['agg_status', 'agg_count', 'agg_time']


class EvaluationSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Evaluation
        fields = ['eval_status', 'eval_count', 'eval_time']

class AlertsSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Alerts
        fields = ['category', 'active', 'timestamp', 'message']


class NodeSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Node
        fields = ['address']
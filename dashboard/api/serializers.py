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
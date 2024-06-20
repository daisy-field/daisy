# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from django.db import models

import uuid

# Create your models here.


class Node(models.Model):
    address = models.CharField(max_length=255, unique=True)


class Alerts(models.Model):
    category = models.CharField(
        max_length=10,
        choices=(
            ("info", "Info"),
            ("warning", "Warning"),
            ("alert", "Alert"),
        ),
    )
    id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False, unique=True
    )
    active = models.BooleanField(default=True)
    message = models.CharField(max_length=255)
    address = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)


class Metrics(models.Model):
    address = models.CharField(max_length=255)
    accuracy = models.FloatField()
    f1 = models.FloatField()
    recall = models.FloatField()
    precision = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        total_records = Metrics.objects.count()
        while total_records >= 500:
            pks = Metrics.objects.values_list("pk")[:1]
            Metrics.objects.filter(pk__in=pks).delete()
            print("delete")
            total_records = Metrics.objects.count()

        else:
            super().save(*args, **kwargs)


class Aggregation(models.Model):
    agg_status = models.CharField(max_length=255)
    agg_count = models.IntegerField()
    agg_time = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        total_records = Aggregation.objects.count()
        while total_records >= 1000:
            pks = Aggregation.objects.values_list("pk")[:1]
            Aggregation.objects.filter(pk__in=pks).delete()
            total_records = Aggregation.objects.count()
        else:
            super().save(*args, **kwargs)


class Prediction(models.Model):
    pred_status = models.CharField(max_length=255)
    pred_count = models.IntegerField()
    pred_time = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        total_records = Aggregation.objects.count()
        while total_records >= 1000:
            pks = Aggregation.objects.values_list("pk")[:1]
            Aggregation.objects.filter(pk__in=pks).delete()
            total_records = Aggregation.objects.count()
        else:
            super().save(*args, **kwargs)


class Evaluation(models.Model):
    eval_status = models.CharField(max_length=255)
    eval_count = models.IntegerField()
    eval_time = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        total_records = Aggregation.objects.count()
        while total_records >= 1000:
            pks = Evaluation.objects.values_list("pk")[:1]
            Evaluation.objects.filter(pk__in=pks).delete()
            total_records = Evaluation.objects.count()
        else:
            super().save(*args, **kwargs)


# /alert
# - NodeID
# - Message
# - Timestamp

# /metrics
# - NodeID
# - Timestamp
# - Accuracy
# - F1
# - Precision
# - Recall

# /aggregation
# -

# /evaluation
# -

# /nodes
# - NodeID
# - Timestamp

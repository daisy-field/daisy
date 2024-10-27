# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from django.db import models

import uuid


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
    true_negative_rate = models.FloatField()
    false_negative_rate = models.FloatField()
    negative_predictive_value = models.FloatField()
    false_positive_rate = models.FloatField()

    def save(self, *args, **kwargs):
        total_records = Metrics.objects.count()
        while total_records >= 200:
            pks = Metrics.objects.values_list("pk")[:1]
            oldest_record = Metrics.objects.filter(pk__in=pks)[0]
            Metrics_long.objects.create(
                address=oldest_record.address,
                accuracy=oldest_record.accuracy,
                f1=oldest_record.f1,
                recall=oldest_record.recall,
                precision=oldest_record.precision,
                timestamp=oldest_record.timestamp,
                true_negative_rate=oldest_record.true_negative_rate,
                false_negative_rate=oldest_record.false_negative_rate,
                negative_predictive_value=oldest_record.negative_predictive_value,
                false_positive_rate=oldest_record.false_positive_rate,
            )
            oldest_record.delete()
            total_records = Metrics.objects.count()
        else:
            super().save(*args, **kwargs)


class Metrics_long(models.Model):
    address = models.CharField(max_length=255)
    accuracy = models.FloatField()
    f1 = models.FloatField()
    recall = models.FloatField()
    precision = models.FloatField()
    timestamp = models.DateTimeField()
    true_negative_rate = models.FloatField()
    false_negative_rate = models.FloatField()
    negative_predictive_value = models.FloatField()
    false_positive_rate = models.FloatField()

    def save(self, *args, **kwargs):
        total_records = Metrics_long.objects.count()
        while total_records >= 2000:
            pks = Metrics_long.objects.values_list("pk")[:1]
            Metrics_long.objects.filter(pk__in=pks).delete()
            print("Delete longterm object", total_records)
            total_records = Metrics_long.objects.count()

        else:
            super().save(*args, **kwargs)


class Aggregation(models.Model):
    agg_status = models.CharField(max_length=255)
    agg_count = models.IntegerField()
    agg_time = models.DateTimeField(auto_now_add=True)
    agg_nodes = models.CharField(max_length=500)

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
    pred_nodes = models.CharField(max_length=500)

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
    eval_nodes = models.CharField(max_length=500)

    def save(self, *args, **kwargs):
        total_records = Aggregation.objects.count()
        while total_records >= 1000:
            pks = Evaluation.objects.values_list("pk")[:1]
            Evaluation.objects.filter(pk__in=pks).delete()
            total_records = Evaluation.objects.count()
        else:
            super().save(*args, **kwargs)

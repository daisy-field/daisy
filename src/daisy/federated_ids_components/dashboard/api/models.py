# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from django.db import models

# Create your models here.

class Node(models.Model):
    adress = models.CharField(max_length=255, unique=True)
    last_connection =  models.DateTimeField()

class Accuracy(models.Model):
    #address = models.ForeignKey(Node, on_delete=models.CASCADE)
    accuracy = models.FloatField()
    #timestamp = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        total_records = Accuracy.objects.count()
        while total_records >= 50:
            pks = (Accuracy.objects
                   .values_list('pk')[:1])
            Accuracy.objects.filter(pk__in=pks).delete()
            print("delete")
            total_records = Accuracy.objects.count()

        else:
            super().save(*args, **kwargs)


class Aggregation(models.Model):
    agg_status = models.CharField(max_length=255)
    agg_count = models.IntegerField()
    agg_time = models.DateTimeField(auto_now_add=True)
    def save(self, *args, **kwargs):
        total_records = Aggregation.objects.count()
        while total_records >= 1000:
            pks = (Aggregation.objects
                   .values_list('pk')[:1])
            Aggregation.objects.filter(pk__in=pks).delete()
            total_records = Aggregation.objects.count()
        else:
            super().save(*args, **kwargs)


class Recall(models.Model):
    #address = models.ForeignKey(Node, on_delete=models.CASCADE)
    recall = models.FloatField()
    #timestamp = models.DateTimeField(auto_now_add=True)

class Precision(models.Model):
    #address = models.ForeignKey(Node, on_delete=models.CASCADE)
    precision = models.FloatField()
    #timestamp = models.DateTimeField(auto_now_add=True)


class F1(models.Model):
    #address = models.ForeignKey(Node, on_delete=models.CASCADE)
    f1 = models.FloatField()
    #timestamp = models.DateTimeField(auto_now_add=True)



#/alert
# - NodeID
# - Message
# - Timestamp

#/metrics
# - NodeID
# - Timestamp
# - Accuracy
# - F1
# - Precision
# - Recall

#/aggregation
# -

#/evaluation
# -

#/nodes
# - NodeID
# - Timestamp
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from django.db import models

# Create your models here.

class Node(models.Model):
    adress = models.CharField(max_length=255, unique=True)

class Accuracy(models.Model):
    #address = models.ForeignKey(Node, on_delete=models.CASCADE)
    accuracy = models.FloatField()
    #timestamp = models.DateTimeField(auto_now_add=True)


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

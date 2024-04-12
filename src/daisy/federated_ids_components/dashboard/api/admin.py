# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from django.contrib import admin

from .models import Metrics, Aggregation, Evaluation, Alerts, Node

admin.site.register(Metrics)
admin.site.register(Aggregation)
admin.site.register(Evaluation)
admin.site.register(Alerts)
admin.site.register(Node)
# Register your models here.

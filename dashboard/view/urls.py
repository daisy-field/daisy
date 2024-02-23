# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from django.urls import path, include
from . import views

urlpatterns = [
    path('',views.index, name = "index"),
    path('alerts/', views.alerts, name = "alerts"),
    path('nodes/', views.nodes, name="nodes"),
    path('change_theme/', views.change_theme, name="change_theme"),

]
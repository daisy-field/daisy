# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from rest_framework import permissions, viewsets
from .models import *
from .serializers import *


class AccuracySerializerView(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = Accuracy.objects.all()
    serializer_class = AccuracySerializer
    permission_classes = [permissions.AllowAny]

class F1SerializerView(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = F1.objects.all()
    serializer_class = AccuracySerializer
    permission_classes = [permissions.AllowAny]

class PrecisionSerializerView(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = Precision.objects.all()
    serializer_class = AccuracySerializer
    permission_classes = [permissions.AllowAny]

class RecallSerializerView(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = Recall.objects.all()
    serializer_class = AccuracySerializer
    permission_classes = [permissions.AllowAny]

class AggregationSerializerView(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = Aggregation.objects.all()
    serializer_class = AggregationSerializer
    permission_classes = [permissions.AllowAny]

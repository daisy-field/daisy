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

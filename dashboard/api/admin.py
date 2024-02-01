from django.contrib import admin
from .models import *


admin.site.register(Accuracy)
admin.site.register(F1)
admin.site.register(Precision)
admin.site.register(Recall)
# Register your models here.

from django.contrib.auth.mixins import PermissionRequiredMixin
from django.http import HttpResponse
from django.views import View
from django.shortcuts import render
from django.template import loader


def index(request):
    return render(request, 'calcul_default_risk/home.html')

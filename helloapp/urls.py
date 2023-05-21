from django.urls import path

from . import views

urlpatterns = [
    path('hello/', views.hello, name='hello'),
    path('process_video/', views.process_video, name='process_video'),
    path('det_video/', views.det_video, name='det_video'),
]

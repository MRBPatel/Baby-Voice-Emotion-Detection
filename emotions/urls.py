from django.urls import path
from emotions import views

app_name = 'emotions'
urlpatterns = [
    path('', views.predict_emotion, name='predict_emotion'),
    path('', views.home, name ="home"),

]

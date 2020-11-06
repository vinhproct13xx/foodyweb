from django.urls import path

from . import views

urlpatterns = [
    # ex: /polls/
    path('', views.index, name='index'),
    # ex: /polls/voice/
    path('voice/', views.voice, name='voice'),
    # ex: /polls/runalgorithm/
    path('runalgorithm/', views.runalgorithm, name='runalgorithm'),
    path('runnbh/<int:owner>/', views.runnbh, name='runnbh'),
    path('run_predict_score/<int:owner>/<int:resid>/', views.run_predict_score, name='run_predict_score')
]
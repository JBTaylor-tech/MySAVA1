from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('success', views.success, name='success'),
    path('signup.html', views.signup, name='signup'),
    path('activate/<uidb64>/<token>', views.activate, name='activate'),
    path('signin.html', views.signin, name='signin'),
    path('signout', views.signout, name='signout'),
]

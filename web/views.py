from django.shortcuts import render
from .Emporium_analyse_module import analyse

# Create your views here.
def index(request):
    dictionary = request.POST
    if dictionary:    
        analyse(dictionary['Date'])
    return render(request, "index.html")
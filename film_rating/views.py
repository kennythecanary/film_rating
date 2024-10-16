from django.shortcuts import render
from .models.model import predict


# home page view
def home(request):    
    return render(request, 'index.html')


# result page view
def result(request):
    text = str(request.GET['review'])
    result = predict(text)
    return render(request, 'result.html', {'result':result})

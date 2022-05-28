from django.http import HttpResponse
from django.template import loader
from predictions.static.predictions.weatherprediction import predict_weather

def index(request):
    template = loader.get_template('predictions/site.html')
    context = {}
    predict_weather()
    return HttpResponse(template.render(context, request))
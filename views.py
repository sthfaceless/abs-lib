from abslib.kp import KnowledgePatternManager, ConjunctKnowledgePatternItem, DisjunctKnowledgePatternItem, QuantKnowledgePatternItem
from django.http import JsonResponse, HttpResponseServerError
from django.views.decorators.csrf import csrf_exempt
import json


@csrf_exempt
def home(request):
   if request.method == 'POST':
      json_data = json.loads(request.body)
      try:
         dict_data = json_data['data']
         str_type = json_data['type']
      except KeyError:
         raise HttpResponseServerError("Malformed data.")

      arr_data = []
      key = 0
      while str(key) in dict_data:
         arr_data.append([float(dict_data[str(key)]['0']), float(dict_data[str(key)]['1'])])
         key+=1

      KnowledgePatternManager_obj = KnowledgePatternManager()
      if (str_type == 'conjuncts'):
         pattern = ConjunctKnowledgePatternItem(arr_data)
      elif (str_type == 'disjuncts'):
         pattern = DisjunctKnowledgePatternItem(arr_data)
      elif (str_type == 'quants'):
         pattern = QuantKnowledgePatternItem(arr_data)
      else:
         raise HttpResponseServerError("Wrong type of data.")
    
      return JsonResponse({'0': str(KnowledgePatternManager_obj.checkInconsistency(pattern).inconsistent)})       

   else: return HttpResponseServerError("Wrong request method: POST required.")
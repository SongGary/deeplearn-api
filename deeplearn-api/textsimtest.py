# -*- coding: utf-8 -*-
"""
Created on Mon Feb  27 16:03:58 2017

"""

import requests
import json

headers = {'content-type': 'application/json'}

#upload
upload = {"userName":"test",
         "password":"test",
         "datasetName":"dataset3",
         "target":"train",
         "dataPath":"/root/saas/deeplearn-api/semanticsim/SICK/train.txt"}
    
r = requests.post("http://localhost:5001/uploadfile",
                  headers=headers,
                  data=json.dumps(upload))       
    
print(r.text)
  
#train
model = {"userName":"test",
         "password":"test",
         "modelName":"mylrmodel2",
         "datasetName":"dataset3",
         "target":"train",        
         "algoName":"semanticsim"
         }
#model = json.dumps(info)
r = requests.post("http://localhost:5001/modeltrain",
                  headers=headers,
                  data=json.dumps(model))
                      
print(r.text)
 
#query
query = {"userName":"test",
         "password":"test",
         "modelName":"mylrmodel2"}
  
r = requests.post("http://localhost:5001/model/query",
                  headers=headers,
                  data=json.dumps(query))
                    
print(r.text)
 
#upload
upload = {"userName":"test",
         "password":"test",
         "datasetName":"dataset4",
         "target":"predict",
         "dataPath":"/root/saas/deeplearn-api/semanticsim/SICK/test.txt"}
   
r = requests.post("http://localhost:5001/uploadfile",
                  headers=headers,
                  data=json.dumps(upload))       
   
print(r.text)

#predict
predict = {"userName":"test",
         "password":"test",
         "modelName":"mylrmodel2",
         "algoName":"semanticsim",
         "datasetName":"dataset4",
         "target":"predict",   
         "outputName":"semanticsim.csv"}
 
r = requests.post("http://localhost:5001/model/batchpredict",
                  headers=headers,
                  data=json.dumps(predict))
 
print(r.text)
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
         "datasetName":"dataset5",
         "target":"train",
         "dataPath":"/root/saas/deeplearn-api/textsum/data/contents1.txt"}
     
r = requests.post("http://localhost:5000/dataset/upload",
                  headers=headers,
                  data=json.dumps(upload))       
     
print(r.text)
 
#upload
upload = {"userName":"test",
         "password":"test",
         "datasetName":"dataset6",
         "target":"train",
         "dataPath":"/root/saas/deeplearn-api/textsum/data/titles1.txt"}
     
r = requests.post("http://localhost:5000/dataset/upload",
                  headers=headers,
                  data=json.dumps(upload))       
     
print(r.text)
  
#train
model = {"userName":"test",
         "password":"test",
         "modelName":"mylrmodel3",
         "datasetName":"dataset5",
         "datasetName02":"dataset6",
         "target":"train",        
         "algoName":"textsum"
         }
#model = json.dumps(info)
r = requests.post("http://localhost:5000/model/train",
                  headers=headers,
                  data=json.dumps(model))
                       
print(r.text)
  
#query
query = {"userName":"test",
         "password":"test",
         "modelName":"mylrmodel3"}
   
r = requests.post("http://localhost:5000/model/query",
                  headers=headers,
                  data=json.dumps(query))
                     
print(r.text)
 
#upload
upload = {"userName":"test",
         "password":"test",
         "datasetName":"dataset7",
         "target":"predict",
         "dataPath":"/root/saas/deeplearn-api/textsum/data/contents1.txt"}
    
r = requests.post("http://localhost:5000/dataset/upload",
                  headers=headers,
                  data=json.dumps(upload))       
    
print(r.text)

#predict
predict = {"userName":"test",
         "password":"test",
         "modelName":"mylrmodel3",
         "algoName":"textsum",
         "datasetName":"dataset7",
         "target":"predict",   
         "outputName":"textsum.csv"}
 
r = requests.post("http://localhost:5000/model/prediction",
                  headers=headers,
                  data=json.dumps(predict))
 
print(r.text)
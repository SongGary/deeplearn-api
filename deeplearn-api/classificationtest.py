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
         "datasetName":"dataset1",
         "target":"train",
         "dataPath":"/root/saas/01textcalssification/data/AI100/training.txt"}
   
r = requests.post("http://localhost:5000/dataset/upload",
                  headers=headers,
                  data=json.dumps(upload))       
   
print(r.text)
 
#train
model = {"userName":"test",
         "password":"test",
         "modelName":"mylrmodel1",
         "datasetName":"dataset1",
         "target":"train",        
         "algoName":"textclassification"
         }
#model = json.dumps(info)
r = requests.post("http://localhost:5000/model/train",
                  headers=headers,
                  data=json.dumps(model))
                     
print(r.text)
 
#query
query = {"userName":"test",
         "password":"test",
         "modelName":"mylrmodel1"}
  
r = requests.post("http://localhost:5000/model/query",
                  headers=headers,
                  data=json.dumps(query))
                    
print(r.text)

#upload
upload = {"userName":"test",
         "password":"test",
         "datasetName":"dataset2",
         "target":"predict",
         "dataPath":"/root/saas/01textcalssification/data/AI100/testing.txt"}
  
r = requests.post("http://localhost:5000/dataset/upload",
                  headers=headers,
                  data=json.dumps(upload))       
  
print(r.text)

#predict
predict = {"userName":"test",
         "password":"test",
         "modelName":"mylrmodel1",
         "algoName":"textclassification",
         "datasetName":"dataset2",
         "target":"predict",   
         "outputName":"OUTPUT.csv"}
 
r = requests.post("http://localhost:5000/model/prediction",
                  headers=headers,
                  data=json.dumps(predict))
 
print(r.text)

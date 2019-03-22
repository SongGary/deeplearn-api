#! /usr/bin/env python


#upload_path='D:\AI\TensorFlow'
upload_path='/root/saas/deep/data'
file_path='/Users/glsong/Downloads/'

def getpath():
    return upload_path
def getfpath():
    return file_path

class DBPredict(object):
    datasetName=''
    userName="test"
    password="test"
    target="train"
    
    def __init__(self):
        self.dealt=0      
    def setcons(self, dataset, usage):
        self.datasetName=dataset
       
        if usage=='0':
            self.target="train"
        else:
            self.target="predict"
        
    
    def getUserName(self):
        print("start",self.userName)
        return self.userName
    
    def getPassword(self):
        return self.password
    
    def getDatasetName(self):
        return self.datasetName
       
    def getTarget(self):
        return self.target

class ModelSet(object):
    modelName='' 
    datasetName=''
    datasetName02='' 
    userName="test"
    password="test"
    target="train"
    algoName=""
    def __init__(self):
        self.dealt=0      
    def setcons(self,modelname,dataset,algo,dataset02):
        self.datasetName=dataset
        self.modelName=modelname
        self.algoName=algo
        self.datasetName02=dataset02
        
    def getModelName(self):
        print("start",self.modelName)
        return self.modelName
    
    def getUserName(self):
        print("start",self.userName)
        return self.userName
    
    def getPassword(self):
        return self.password
    
    def getDatasetName(self):
        return self.datasetName
    
    def getDatasetName02(self):
        return self.datasetName02
    
    def getTarget(self):
        return self.target
    
    def getAlgoName(self):
        return self.algoName
       
class SingleSet(object):
    userName="test"
    password="test"
    modelName='' 
    dataSet=''   
    dataType=''
    def __init__(self):
        self.dealt=0      
    def setcons(self,modelname,dataset,algo,datatype):
        self.dataSet=dataset
        self.modelName=modelname
        if datatype=='0':
            self.dataType="vectors"
        elif algo=='0' and datatype=='1':
            self.dataType="text"
        elif algo=='1' and datatype=='1':
            self.dataType="textlabel"
        
    def getModelName(self):
        print("start",self.modelName)
        return self.modelName
    
    def getUserName(self):
        print("start",self.userName)
        return self.userName
    
    def getPassword(self):
        return self.password
    
    def getDataSet(self):
        return self.dataSet
    
    def getDatatype(self):
        return self.dataType
    
class BatchSet(object):
    userName="test"
    password="test"
    target="predict"
    modelName='' 
    algoName=''
    datasetName=''   
    outputName=''
    outputPath=''
    def __init__(self):
        self.dealt=0      
    def setcons(self,modelname,dataset,algo,outputname):
        self.datasetName=dataset
        self.modelName=modelname
        self.outputName=outputname
        self.algoName=algo
    def setres(self,outputPath):
        self.outputPath=outputPath
    
    def getOutputPath(self):
        return self.outputPath    
        
    def getModelName(self):
        print("start",self.modelName)
        return self.modelName
    
    def getUserName(self):
        print("start",self.userName)
        return self.userName
    
    def getPassword(self):
        return self.password
    
    def getDataSetName(self):
        return self.datasetName
    
    def getOutputName(self):
        return self.outputName
    
    def getAlgoName(self):
        return self.algoName
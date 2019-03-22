# -*- coding: utf-8 -*-
"""
Created on 08 28 13:50:03 2017

@author: glsong
"""

from flask import Flask,jsonify,request,send_from_directory
import flask
import json
import database
import time
import os 
import subprocess
from time import strftime
import pymysql
from common.tool_util import CommonTool
from common.constans import getpath 
from common.constans import DBPredict,ModelSet,SingleSet,BatchSet
from werkzeug.utils import secure_filename
from collections import OrderedDict

upload_path=getpath()  #定义上传文件的保存路径
uploadfile=''

app = Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('login.tpl')
    #return flask.render_template('pageindex.tpl')
#     return "Welcome"

@app.route('/verified', methods=['POST'])
def verified():   
    username = request.values.get('username')  
    passwd = request.values.get('passwd')
    result=authentication(username,passwd)
    if result==-1:
        return jsonify({"status":-1,"info":"authentication failed"})
    else:
        return flask.render_template('pageindex.tpl')

@app.route('/mainframe')
def mainframe():
    return flask.render_template('pageindex.tpl')
#     return "Welcome"

@app.route('/db_setting',methods=['get'])
def do_db_setting():
    return flask.render_template('db_query_cons.tpl')

@app.route('/model_setting',methods=['get'])
def do_model_setting():
    return flask.render_template('model_query_cons.tpl')

@app.route('/batch_setting',methods=['get'])
def do_batch_setting():
    return flask.render_template('predict_batch_cons.tpl')

@app.route('/setcons', methods=['POST'])
def set_data():
    dataset = request.form.get('dataset')  
    usage = request.form.get('usage')
    DBPredict.setcons(DBPredict, dataset, usage)
    return flask.render_template('fileindex.tpl')

@app.route("/command",methods=['post'])
def command():

    print(request.form)
    print(request.form["name"])
    return "hello"
# curl -d "name=test" "http://10.10.0.144:5000/command"
@app.route("/register",methods=["post"])
def register():
    name = request.form["name"]
    password = request.form["password"]
# curl -d "name=guest&password=123456" "http://10.10.0.144:5000/register"
# curl -d "name=guest&password=123456&modelName=lr" "http://10.10.0.144:5000/model/query"    
@app.route("/model/query",methods=["post"])
def queryModel():
    userName = request.json["userName"]
    password = request.json["password"]
    modelName = request.json["modelName"]
    
    #Authentication
    userId = authentication(userName,password)
    if userId == -1:
        return jsonify({"status":-1,"info":"authentication failed"})
    userId = str(userId)
    
    sql = "select * from model where userId="+userId+" and modelName="+"\""+modelName+"\""
    result = database.search(mysql_host,mysql_user,mysql_password,"deeplearn",sql)
    if len(result) == 0:
        return jsonify({"status":-1,"info":"model: "+modelName+" does not exist"})
    
    info = {}
    info["modelName"] = result[0][1]
    info["status"] = result[0][2]
    info["time"] = result[0][4]
    info["createTime"] = result[0][6]
    info["datasetName"] = result[0][7]
    
    return jsonify(info)

@app.route('/setmodels', methods=['POST'])
def set_model():
    modelname = request.form.get('modelname')
    dataset = request.form.get('dataset')   
    algo = request.values.get('prov')
    dataset02 = request.form.get('dataset02')   
    ModelSet.setcons(ModelSet,modelname,dataset,algo,dataset02)
    return flask.render_template('buildmodel.tpl')
   
@app.route("/modeltrain",methods=["post"])
def trainModel():
#     datasetName02=""
#     userName = request.json["userName"]
#     password = request.json["password"]
#     datasetName = request.json["datasetName"]
#     target = request.json["target"]
#     modelName = request.json["modelName"]
#     algoName = request.json["algoName"]
#     if algoName == "textsum":
#         datasetName02 = request.json["datasetName02"]
#     algoPara = request.json["algoPara"]     
    ms = ModelSet() 
    userName = ms.userName
    password = ms.password
    datasetName = ms.datasetName
    datasetName02=ms.datasetName02
    target = ms.target
    modelName = ms.modelName
    algoName = ms.algoName
    #Authentication
    userId = authentication(userName,password)
    if userId == -1:
        return jsonify({"status":-1,"info":"authentication failed"})
    userId = str(userId)
    
    #search dataPath on hdfs
    dataPath = searchDataset(userId,datasetName,target)
    print("dataPath: ",dataPath)
    if dataPath == "":
        return jsonify({"status":-1,"info":"dataset: "+datasetName+" does not exist"})
     
    if datasetName02 != "":
        dataPath02 = searchDataset(userId,datasetName02,target)
        print("dataPath02: ",dataPath02)
        if dataPath02 == "":
            return jsonify({"status":-1,"info":"dataset: "+datasetName+" does not exist"})
    #model path
    modelPath = searchModel(userId,modelName)
    if modelPath != "":
        return jsonify({"status":-1,"info":"model: "+modelName+" already exists"})
    modelPath = "/opt/deeplearn/model/" + userName + "/" + modelName
    
    
    #update table model,set status to 1(running)
    #if model does not exist:insert,else:update
    sql = "select * from model where modelName="+"\""+modelName+"\""+" and userId="+ userId
                           
    if len(database.search(mysql_host,mysql_user,mysql_password,"deeplearn",sql)) == 0:
        sql = "insert into model(userId,modelName,status,modelPath,algoName,datasetName) values("+ \
                userId+",\""+modelName+"\","+"1"+",\""+modelPath+"\",\""+algoName+"\",\""+datasetName+"\""+")"
        print(sql)
        database.insert(mysql_host,mysql_user,mysql_password,"deeplearn",sql)
    else:
        sql = "update model set status=1,datasetName=\""+datasetName+"\" where userId="+"\""+ \
                userId+"\" and modelName="+"\""+modelName+"\""        
        database.update(mysql_host,mysql_user,mysql_password,"deeplearn",sql)
    
    resultfile=os.path.join(modelPath,"tmp")
    #run spark app
    t0 = time.time()
    #subCmd = cmdgenerator(algoName,algoPara)
    if algoName == "textsum":
        cmd = "python3 " + algoName + "/train.py" + \
            " --modelName=" + modelName + \
            " --dataPath=" + dataPath + \
            " --dataPath02=" + dataPath02 + \
            " --modelPath=" + modelPath + \
            " --resultfile=" + resultfile + \
            " --algoName=" + algoName
    else:
        cmd = "python3 " + algoName + "/train.py" + \
            " --modelName=" + modelName + \
            " --dataPath=" + dataPath + \
            " --modelPath=" + modelPath + \
            " --resultfile=" + resultfile + \
            " --algoName=" + algoName
    #cmd += subCmd
    
    status,output = subprocess.getstatusoutput(cmd)
    print(output)
    t1 = time.time()
    t = t1 - t0
    
    info = {"status":-1}
    if status == 0:
        #model is trained successfully,update database,set status to 0(finish)
        sql = "update model set status=0,time=" +str(t)+",createTime="+ str(t0)+\
                   " where userId="+"\""+userId+"\" and modelName="+"\""+modelName+"\""
        database.update(mysql_host,mysql_user,mysql_password,"deeplearn",sql)
        info = {"status":0,"modelName":modelName,"time":t}
    else:
        #failed,set status=-1
        sql = "update model set status=-1,time=" +str(t)+",createTime="+ str(t0)+\
                   " where userId="+"\""+userId+"\" and modelName="+"\""+modelName+"\""
        database.update(mysql_host,mysql_user,mysql_password,"deeplearn",sql)
        info["info"] = "train failed"
    
    
    return jsonify(info)

@app.route('/setpredict', methods=['POST'])
def set_predict():
    modelname = request.form.get('modelname')
    dataset = request.form.get('dataset') 
    algo = request.values.get('prov')  
    outputname = request.form.get('outputname') 
    BatchSet.setcons(BatchSet,modelname,dataset,algo,outputname)
    return flask.render_template('runpredict.tpl')

@app.route("/batchpredict",methods=["post"])
def batchPredction():
#     userName = request.json["userName"]
#     password = request.json["password"]
#     datasetName = request.json["datasetName"]
#     target = request.json["target"]
#     algoName = request.json["algoName"]
#     modelName = request.json["modelName"]
#     outputName = request.json["outputName"]
    
    bs = BatchSet() 
    userName = bs.userName
    password = bs.password
    datasetName = bs.datasetName
    target = bs.target
    algoName = bs.algoName
    modelName = bs.modelName
    outputName = bs.outputName
    #Authentication
    userId = authentication(userName,password)
    if userId == -1:
        return jsonify({"status":-1,"info":"authentication failed"})
    userId = str(userId)
    
    #outputName
    outputPath = searchDataset(userId,outputName,target)
    if outputPath != "":
        return jsonify({"status":-1,"info":"dataset: "+outputName+" already exists"})
    outputPath = "/opt/deeplearn/data/" + userName + "/dataset/" + outputName
    #search dataPath on hdfs
    dataPath = searchDataset(userId,datasetName,target)
    if dataPath == "":
        return jsonify({"status":-1,"info":"dataset: "+datasetName+" does not exist"})
    
    #search modelPath
    modelPath = searchModel01(userId,modelName)
    if modelPath == "":
        return jsonify({"status":-1,"info":"model: "+modelName+" does not exist"})
    
    resultfile=os.path.join(modelPath,"tmp")
    #submit spark job
    cmd = "python3 " + algoName + "/eval.py" + \
            " --dataPath=" + dataPath + \
            " --outputPath=" + outputPath + \
            " --modelPath=" + modelPath + \
            " --resultfile=" + resultfile
    
    t0 = time.time()
    status,output = subprocess.getstatusoutput(cmd)   
    t1 = time.time()
    print(output)
    d={}
    if algoName!='textsum':
        f=open(outputPath)
        for line in f:
            u1,u2=line.split(',')
            d[u1]=u2 
    d = OrderedDict(sorted(d.items(),key = lambda t:int(t[0])))  
    if status == 0:
        #success
        BatchSet.setres(BatchSet,outputPath)
        sql = "insert into dataset(userId,datasetName,dataPath,target,createTime) values(" +\
                userId+",\""+outputName+"\",\""+outputPath+"\",\""+target+"\","+str(t0)+ ")"
        database.insert(mysql_host,mysql_user,mysql_password,"deeplearn",sql)  
        #return jsonify({"status":0,"result":all_the_text,"prediction time":t1-t0}) 
        if algoName!='textsum':
            return flask.render_template('result_class_display.tpl',data=d)
        else:
            f=open(outputPath)
            reusltcont=f.read()
            return jsonify({"result":reusltcont}) 
    
    return jsonify({"status":-1,"info":"prediction failed"})   

def searchDataset(userId,dataset,target):
    
    sql = "select dataPath from dataset where userId="+ \
                    userId+" and datasetName="+"\""+dataset+"\" and target="+"\""+target+"\""
    
    result = database.search(mysql_host,mysql_user,mysql_password,"deeplearn",sql)
    
    if len(result) == 0:
        return ""
        
    dataPath = result[0][0]
    return dataPath

def searchModel(userId,modelName):
    sql = "select modelPath from model where userId="+ \
                    userId+" and modelName="+"\""+modelName+"\""+ \
                    " and status="+"0"
    
    result = database.search(mysql_host,mysql_user,mysql_password,"deeplearn",sql)
    if len(result) == 0:
        return ""
    
    modelPath = result[0][0]
    return modelPath

def searchModel01(userId,modelName):
    sql = "select modelPath from model where userId="+ \
                    userId+" and modelName="+"\""+modelName+"\""
    
    result = database.search(mysql_host,mysql_user,mysql_password,"deeplearn",sql)
    if len(result) == 0:
        return ""
    
    modelPath = result[0][0]
    return modelPath
    
def cmdgenerator(algoName,algoPara):
    """
    generate spark-submit command for each algorithm
    """
    if algoPara == None:
        return None
    cmd = ""    
    
        
    for key in algoPara:
        cmd += " --" + key + "=" + str(algoPara[key])
    
    return cmd
    

def authentication(user,passwd):
    sql = "select userId,password from user where userName="+"\""+user+"\""
    result = database.search(mysql_host,mysql_user,mysql_password,"deeplearn",sql)
 
    if len(result) == 0:
        return -1
        
    password = result[0][1]
    userId = result[0][0]
    
    if passwd == password:
        return userId
    else:
        return -1

#curl -l -H "Content-type: application/json" -X POST -d '{"userName":"test","password":"test","dataType":"libsvm","datasetName":"dataset1","dataPath":"sample_binary_classification_data.txt"}' "http://10.10.0.144:5000/dataset/upload" 
#upload dataset
@app.route('/uploadfile', methods=['POST'])
def file_upload():
    uploadfile=request.files['data'] #获取上传的文件
    filename = secure_filename(uploadfile.filename)
    uploadfile.save(os.path.join(upload_path,filename))#overwrite参数是指覆盖同名文件
    #return u"上传成功,文件名为：%s，文件类型为：%s"% (uploadfile.filename,uploadfile.content_type)
    #return flask.render_template('pageindex.tpl')
    #return template('view/review.tpl')
    inst = DBPredict()
    userName = inst.userName
    password = inst.password
    target = inst.target
    datasetName = inst.datasetName
    dataPath = os.path.join(upload_path,filename)
# @app.route('/dataset/upload', methods=['POST'])
# def upload_dataset():
    start_time = time.time()
#     #请求参数验证
#     if (not request.json or not 'datasetName' in request.json 
#                 or not 'dataPath' in request.json 
#                 or not 'userName' in request.json 
#                 or not 'password' in request.json ):
#         return jsonify({'status':-1,"info":"Wrong request para.",'time':-1}),200
#     
#     userName = request.json['userName']
#     password = request.json['password']
#     target = request.json['target']
#     datasetName = request.json['datasetName']
#     dataPath = request.json['dataPath']
    print("dataPath: ",dataPath)
    filename=os.path.basename(dataPath)
    save_dataPath= "/opt/deeplearn/data/"+userName+"/dataset/"+target+"/"+datasetName+"/"
    local_dataPath = dataPath
    
    #Authentication
    userId = authentication(userName,password)
    if userId == -1:
        return jsonify({"status":-1,"info":"authentication failed"})
    userId = str(userId)    
    
#     #localfile exist?
#     local_file_exists = 'test -e ' + local_dataPath
#     local_flag_not_exist = subprocess.call(local_file_exists, shell=True)
#     if local_flag_not_exist==1:
#         return jsonify({'status':-1,"info":dataPath+" doesn't exist.","time":-1}),200

    #dataset exist?
    dataPath = searchDataset(userId,datasetName,target)
    if dataPath != "":
        return jsonify({"status":-1,"info":"dataset: "+datasetName+" already exists"})
    
    #create hdfs dir
    cmd = "mkdir -p " + save_dataPath
    subprocess.call(cmd, shell=True)
    
    #upload to hdfs
    shell_to_hdfs = "mv " + local_dataPath + " " + save_dataPath
    subprocess.call(shell_to_hdfs, shell=True)
    createTime = strftime("%Y%m%d%H%M%S")
    
    savedPath=os.path.join(save_dataPath,filename)
    #update db
    conn = pymysql.connect(host=mysql_host, user=mysql_user,
                           passwd=mysql_password, db='deeplearn')
    cur = conn.cursor()
    sql_insert = "insert into dataset values('" + userId + "','" + datasetName + "','" + savedPath + "','" + target + "','" +createTime +"')"
    cur.execute(sql_insert)
    cur.close()
    conn.commit()
    conn.close()
    upload_time = time.time() - start_time
    return jsonify({'status':0, "datasetName":datasetName, "time":upload_time}),200

def loadConfig(path):
    f = open(path)
    config = json.load(f)
    f.close()
    return config["mysql_host"],config["mysql_user"],config["mysql_password"]      

@app.route('/web/js/<path:path>')
def send_js(path):
    static_root=CommonTool().current_path(__name__)
    return send_from_directory(static_root+'/web/js/', path)

@app.route('/web/css/<path:path>')
def send_css(path):
    static_root=CommonTool().current_path(__name__)
    return send_from_directory(static_root+'/web/css/', path)

@app.route('/web/img/<path:path>')
def send_img(path):
    static_root=CommonTool().current_path(__name__)
    return send_from_directory(static_root+'/web/img/', path)

@app.route('/web/icon/<path:path>')
def send_icon(path):
    static_root=CommonTool().current_path(__name__)
    return send_from_directory(static_root+'/web/icon/', path)

if  __name__ == "__main__":
    mysql_host,mysql_user,mysql_password = loadConfig("config.json")
    cwd = os.getcwd()
    app.config['JSON_AS_ASCII'] = False
    app.run(host="0.0.0.0",port=5001)

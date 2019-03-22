from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import math
import jieba
import subprocess

import argparse
#set argparser
parser = argparse.ArgumentParser()
parser.add_argument("--modelName")
parser.add_argument("--dataPath")
parser.add_argument("--modelPath")
parser.add_argument("--resultfile")
parser.add_argument("--algoName")
arguments = parser.parse_args()
modelName = arguments.modelName
print("modelName: ",modelName) 
dataPath = arguments.dataPath
print("dataPath: ",dataPath)  
modelPath = arguments.modelPath
print("modelPath: ",modelPath)  
resultfile = arguments.resultfile
print("resultfile: ",resultfile)
algoName = arguments.algoName
print("algoName: ",algoName)

if not os.path.exists(modelPath):
    os.makedirs(modelPath)
    
flags=tf.flags

lolfile = os.path.join(algoName,'lol.txt')
flags.DEFINE_string('word2vec_norm',lolfile,'Word2vec file with pre-trained embeddings')
#flags.DEFINE_string('word2vec_norm','lol.txt','Word2vec file with pre-trained embeddings')
#flags.DEFINE_string('data_path','SICK','SICK data set path')
flags.DEFINE_string('save_path',modelPath+'/','STS model output directory')
flags.DEFINE_integer('embedding_dim',128,'Dimensionality of word embedding')
flags.DEFINE_integer('max_length',40,'one sentence max length words which is in dictionary')
flags.DEFINE_bool('use_fp64',False,'Train using 64-bit floats instead of 32bit floats')



FLAGS=flags.FLAGS
FLAGS._parse_flags()
print('Parameters:')
for attr,value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr,value))
def data_type():
    return tf.float64 if FLAGS.use_fp64 else tf.float32
def build_vocab(word2vec_path=True):
    if word2vec_path:
        print('Load word2vec_norm file {}'.format(word2vec_path))
        with open(word2vec_path,encoding = 'utf-8') as f:
            header=f.readline()
            print(header.split())
            vocab_size,layer2_size=map(int,header.split())
            # initial matrix with random uniform
            init_W=np.random.uniform(-0.25,0.25,(vocab_size,FLAGS.embedding_dim))

            print('vocab_size={}'.format(vocab_size))
            dictionary=dict()
            while True:
                line=f.readline()
                if not line:break
                word=line.split()[0]
                dictionary[word]=len(dictionary)
                init_W[dictionary[word]]=np.array(line.split()[1:], dtype=np.float32)

    return dictionary,init_W
    
def file_to_word2vec_word_ids(filename,word_to_id,is_test=False):
    with open(filename,encoding = 'utf-8') as f:
        f.readline() # remove header
        sentences_A=[]
        sentencesA_length=[]
        sentences_B=[]
        sentencesB_length=[]
        relatedness_scores=[]
        pairIDs=[]
        while True:
            line=f.readline()
            if not line: break
            ID=line.split(' ')[0] # for test
            print(ID)
            pairIDs.append(ID)
            sentence_A=line.split(' ')[1]
            print(sentence_A)
            sentence_B=line.split(' ')[2]
            print(sentence_B)
            relatedness_score=line.split(' ')[3]    
            print(relatedness_score)
            _=[word_to_id[word] for word in jieba.cut(sentence_A, cut_all=False) if word in word_to_id]
            print(_)
            sentencesA_length.append(len(_)) # must be before [0]*(FLAGS.max_length-len(_))
            _+=[0]*(FLAGS.max_length-len(_))
            sentences_A.append(_)
            
            _=[word_to_id[word] for word in jieba.cut(sentence_B, cut_all=False) if word in word_to_id]
            print(_)
            sentencesB_length.append(len(_))
            _+=[0]*(FLAGS.max_length-len(_))
            sentences_B.append(_)
            
            relatedness_scores.append((float(relatedness_score)-1)/4)
    assert len(sentences_A)==len(sentencesA_length)==len(sentences_B)==len(sentencesB_length)==len(relatedness_scores)
    if not is_test: return STSInput(sentences_A,sentencesA_length,sentences_B,sentencesB_length,relatedness_scores)
    else:
        stsinput=STSInput(sentences_A,sentencesA_length,sentences_B,sentencesB_length,relatedness_scores)
        stsinput.pairIDs=pairIDs
        return stsinput

class STSInput(object):
    def __init__(self,sentences_A,sentencesA_length,sentences_B,sentencesB_length,relatedness_scores):
        self.sentences_A=sentences_A
        self.sentencesA_length=sentencesA_length
        self.sentences_B=sentences_B
        self.sentencesB_length=sentencesB_length
        self.relatedness_scores=relatedness_scores
    
    def sentences_A(self):
        return self.sentences_A
    
    def sentencesA_length(self):
        return self.sentencesA_length
    
    def sentences_B(self):
        return self.sentences_B
    
    def sentencesB_length(self):
        return self.sentencesB_length
    
    def relatedness_scores(self):
        return self.relatedness_scores


# train_path=os.path.join(FLAGS.data_path,'train.txt')
# valid_path=os.path.join(FLAGS.data_path,'trial.txt')
# test_path=os.path.join(FLAGS.data_path,'test.txt')
cmd = "python3 " + algoName + "/chinese_word2vec.py" + \
            " --dataPath=" + dataPath
print("cmd: ",cmd)
status,output = subprocess.getstatusoutput(cmd)
if status == 0:
    print("finished to build dict!")
else:
    print("dict can not be builded")
dictionary,init_W=build_vocab(FLAGS.word2vec_norm)
train_data=file_to_word2vec_word_ids(dataPath,dictionary)
# valid_data=file_to_word2vec_word_ids(valid_path,dictionary,is_test=True)
# test_data=file_to_word2vec_word_ids(test_path,dictionary,is_test=True)

def next_batch(start,end,input):
    inputs_A=input.sentences_A[start:end]
    inputsA_length=input.sentencesA_length[start:end]
    inputs_B=input.sentences_B[start:end]
    inputsB_length=input.sentencesB_length[start:end]
    labels=np.reshape(input.relatedness_scores[start:end],(-1))
    return STSInput(inputs_A,inputsA_length,inputs_B,inputsB_length,labels)

class Config(object):
    init_scale=0.2
    learning_rate=.01
    max_grad_norm=1.
    keep_prob=1.
    lr_decay=0.98
    batch_size=30
    max_epoch=8
    max_max_epoch=20
    num_layer=1
    
config=Config()
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True
def build_model(input_,input_length,dropout_):
    rnn_cell=tf.contrib.rnn.LSTMCell(num_units=50)
    rnn_cell=tf.contrib.rnn.DropoutWrapper(rnn_cell,output_keep_prob=dropout_)
    rnn_cell=tf.contrib.rnn.MultiRNNCell([rnn_cell]*config.num_layer)
        
    outputs,last_states=tf.nn.dynamic_rnn(
        cell=rnn_cell,
        dtype=data_type(),
        sequence_length=input_length,
        inputs=input_
    )
    return outputs,last_states
with tf.Graph().as_default():
    initializer=tf.contrib.layers.xavier_initializer()
    
    with tf.variable_scope('Model',initializer=initializer):
        sentences_A=tf.placeholder(tf.int32,shape=([None,FLAGS.max_length]),name='sentences_A')
        sentencesA_length=tf.placeholder(tf.int32,shape=([None]),name='sentencesA_length')
        sentences_B=tf.placeholder(tf.int32,shape=([None,FLAGS.max_length]),name='sentences_B')
        sentencesB_length=tf.placeholder(tf.int32,shape=([None]),name='sentencesB_length')
        labels=tf.placeholder(tf.float32,shape=([None]),name='relatedness_score_label')
        dropout_f=tf.placeholder(tf.float32)
        W=tf.Variable(tf.constant(0.0,shape=[len(dictionary),FLAGS.embedding_dim]),trainable=False,name='W')
        embedding_placeholder=tf.placeholder(data_type(),[len(dictionary),FLAGS.embedding_dim])
        embedding_init=W.assign(embedding_placeholder)

        sentences_A_emb=tf.nn.embedding_lookup(params=embedding_init,ids=sentences_A)
        sentences_B_emb=tf.nn.embedding_lookup(params=embedding_init,ids=sentences_B)

        # model
        with tf.variable_scope('siamese') as scope:
            outputs_A,last_states_A=build_model(sentences_A_emb,sentencesA_length,dropout_f)
            scope.reuse_variables()
            outputs_B,last_states_B=build_model(sentences_B_emb,sentencesB_length,dropout_f)

        # last_states[last_layer][0] cell states, last_states[last_layer][1] hidden states
        prediction=tf.exp(tf.multiply(-1.0,tf.reduce_mean(tf.abs(tf.subtract(last_states_A[config.num_layer-1][1],last_states_B[config.num_layer-1][1])),1)))
        
        # cost
        cost=tf.reduce_mean(tf.square(tf.subtract(prediction, labels)))

        lr=tf.Variable(0.0,trainable=False)
        tvars=tf.trainable_variables()
        grads,_=tf.clip_by_global_norm(tf.gradients(cost,tvars),config.max_grad_norm)
        optimizer=tf.train.AdamOptimizer(learning_rate=lr)
        train_op=optimizer.apply_gradients(zip(grads,tvars),global_step=tf.contrib.framework.get_or_create_global_step())
        new_lr=tf.placeholder(tf.float32,shape=[],name='new_learning_rate')
        lr_update=tf.assign(lr,new_lr)
        
        for v in tf.trainable_variables():
            print(v.name)
        saver = tf.train.Saver()
        
        with tf.Session(config=config_gpu) as sess:
            sess.run(tf.global_variables_initializer())

            total_batch=int(len(train_data.sentences_A)/config.batch_size)
            print('Total batch size: {}, data size: {}, batch size: {}'.format(total_batch,len(train_data.sentences_A),config.batch_size))
            print(config.max_grad_norm,config.keep_prob,config.lr_decay,config.max_epoch,config.max_max_epoch,config.num_layer)
            # py
            prev_train_cost=1
            prev_valid_cost=1
            for epoch in range(config.max_max_epoch):
                lr_decay=config.lr_decay**max(epoch+1-config.max_epoch,0.0)
                sess.run([lr,lr_update],feed_dict={new_lr:config.learning_rate*lr_decay})
                print('Epoch {} Learning rate: {}'.format(epoch,sess.run(lr)))
                
                avg_cost=0.
                for i in range(total_batch):
                    start=i*config.batch_size
                    end=(i+1)*config.batch_size

                    next_batch_input=next_batch(start,end,train_data)
                    _,train_cost,train_predict=sess.run([train_op,cost,prediction],feed_dict={
                            sentences_A:np.array(next_batch_input.sentences_A),
                            sentencesA_length:np.array(next_batch_input.sentencesA_length),
                            sentences_B:np.array(next_batch_input.sentences_B),
                            sentencesB_length:np.array(next_batch_input.sentencesB_length),
                            labels:next_batch_input.relatedness_scores,
                            dropout_f:config.keep_prob,
                            embedding_placeholder:init_W
                        })
                    avg_cost+=train_cost
                    
                start=total_batch*config.batch_size
                end=len(train_data.sentences_A)
                if not start==end:
                    next_batch_input=next_batch(start,end,train_data)
#                     print(next_batch_input.sentences_A)
#                     print(next_batch_input.sentences_B)
#                     print(np.array(next_batch_input.sentences_A))
#                     print(np.array(next_batch_input.sentencesA_length))
#                     print(np.array(next_batch_input.sentences_B))
#                     print(np.array(next_batch_input.sentencesB_length))
#                     print(next_batch_input.relatedness_scores)
#                     print(config.keep_prob)
#                     print(init_W)
#                     print(type(sentences_A),type(np.array(next_batch_input.sentences_A)))
#                     print(type(sentencesA_length),type(np.array(next_batch_input.sentencesA_length)))
#                     print(type(sentences_B),type(np.array(next_batch_input.sentences_B)))
#                     print(type(sentencesB_length),type(np.array(next_batch_input.sentencesB_length)))
#                     print(type(labels),type(next_batch_input.relatedness_scores))
#                     print(type(dropout_f),type(np.array(config.keep_prob)))
#                     print(type(embedding_placeholder),type(init_W))
                    _,train_cost,train_predict=sess.run([train_op,cost,prediction],feed_dict={
                            sentences_A:np.array(next_batch_input.sentences_A,),
                            sentencesA_length:np.array(next_batch_input.sentencesA_length),
                            sentences_B:np.array(next_batch_input.sentences_B),
                            sentencesB_length:np.array(next_batch_input.sentencesB_length),
                            labels:next_batch_input.relatedness_scores,
                            dropout_f:config.keep_prob,
                            embedding_placeholder:init_W
                        })
                    avg_cost+=train_cost
                
                if prev_train_cost>avg_cost/total_batch: print('Average cost:\t{} ↓'.format(avg_cost/total_batch))
                else: print('Average cost:\t{} ↑'.format(avg_cost/total_batch))
                prev_train_cost=avg_cost/total_batch
                
                # validation
#                 valid_cost,valid_predict=sess.run([cost,prediction],feed_dict={
#                     sentences_A:valid_data.sentences_A,
#                     sentencesA_length:valid_data.sentencesA_length,
#                     sentences_B:valid_data.sentences_B,
#                     sentencesB_length:valid_data.sentencesB_length,
#                     labels:np.reshape(valid_data.relatedness_scores,(-1)),
#                     embedding_placeholder:init_W,
#                     dropout_f:1.0
#                 })
#                 if prev_valid_cost>valid_cost: print('Valid cost:\t{} ↓'.format(valid_cost))
#                 else: print('Valid cost:\t{} ↑'.format(valid_cost))
#                 prev_valid_cost=valid_cost   
            #checkpoint_prefix = os.path.join(FLAGS.save_path, "model")
            saver.save(sess, FLAGS.save_path+'stslstm-model',global_step=config.max_max_epoch)
            #saver.save(sess, checkpoint_prefix,global_step=config.max_max_epoch)

            # test
#             test_cost,test_predict=sess.run([cost,prediction],feed_dict={
#                 sentences_A:test_data.sentences_A,
#                 sentencesA_length:test_data.sentencesA_length,
#                 sentences_B:test_data.sentences_B,
#                 sentencesB_length:test_data.sentencesB_length,
#                 labels:np.reshape(test_data.relatedness_scores,(-1)),
#                 embedding_placeholder:init_W,
#                 dropout_f:1.0
#             })
#             print("test_cost: ",test_cost)
# with open('SICK/stslstm_trial_result00.txt','w') as fw:
#     fw.write('pair_ID    relatedness_score    entailment_judgment\n')
#     for _ in range(len(valid_predict)):
#         fw.write(valid_data.pairIDs[_]+'\t'+str(round(valid_predict[_]*4+1,2))+'\tNA\n')
with tf.Graph().as_default():
    initializer=tf.contrib.layers.xavier_initializer()
    
    with tf.variable_scope('Model',initializer=initializer):
        sentences_A=tf.placeholder(tf.int32,shape=([None,FLAGS.max_length]),name='sentences_A')
        sentencesA_length=tf.placeholder(tf.int32,shape=([None]),name='sentencesA_length')
        sentences_B=tf.placeholder(tf.int32,shape=([None,FLAGS.max_length]),name='sentences_B')
        sentencesB_length=tf.placeholder(tf.int32,shape=([None]),name='sentencesB_length')
        labels=tf.placeholder(tf.float32,shape=([None]),name='relatedness_score_label')
        dropout_f=tf.placeholder(tf.float32)
        W=tf.Variable(tf.constant(0.0,shape=[len(dictionary),FLAGS.embedding_dim]),trainable=False,name='W')
        embedding_placeholder=tf.placeholder(data_type(),[len(dictionary),FLAGS.embedding_dim])
        embedding_init=W.assign(embedding_placeholder)

        sentences_A_emb=tf.nn.embedding_lookup(params=embedding_init,ids=sentences_A)
        sentences_B_emb=tf.nn.embedding_lookup(params=embedding_init,ids=sentences_B)

        with tf.variable_scope('siamese') as scope:
            outputs_A,last_states_A=build_model(sentences_A_emb,sentencesA_length,dropout_f)
            scope.reuse_variables()
            outputs_B,last_states_B=build_model(sentences_B_emb,sentencesB_length,dropout_f)
        
        with tf.Session(config=config_gpu) as sess:
            sess.run(tf.global_variables_initializer())

            
            lsA,outA=sess.run([last_states_A,outputs_A],feed_dict={
                    sentences_A:train_data.sentences_A,
                            sentencesA_length:train_data.sentencesA_length,
                            sentences_B:train_data.sentences_B,
                            sentencesB_length:train_data.sentencesB_length,
                            labels:np.reshape(train_data.relatedness_scores,(-1)),
                            dropout_f:config.keep_prob,
                            embedding_placeholder:init_W
                })
print(outA[0][6])
print(lsA[0][1][0])

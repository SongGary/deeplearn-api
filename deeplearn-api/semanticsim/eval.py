#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import sys
import math
import jieba

import argparse
#set argparser
parser = argparse.ArgumentParser()
parser.add_argument("--dataPath")
parser.add_argument("--modelPath")
parser.add_argument("--outputPath")
parser.add_argument("--resultfile")
arguments = parser.parse_args()
dataPath = arguments.dataPath
print("dataPath: ",dataPath)  
modelPath = arguments.modelPath
print("modelPath: ",modelPath)  
outputPath = arguments.outputPath
print("outputPath: ",outputPath) 
resultfile = arguments.resultfile
print("resultfile: ",resultfile)

# fo = open(resultfile, "r+")
# result = fo.read()

# Model Hyperparameters
flags=tf.flags

lolfile = os.path.join('semanticsim','lol.txt')
flags.DEFINE_string('word2vec_norm',lolfile,'Word2vec file with pre-trained embeddings')
#flags.DEFINE_string('data_path','SICK','SICK data set path')
flags.DEFINE_string('save_path',modelPath+'/','STS model output directory')
flags.DEFINE_integer('embedding_dim',128,'Dimensionality of word embedding')
flags.DEFINE_integer('max_length',40,'one sentence max length words which is in dictionary')
flags.DEFINE_bool('use_fp64',False,'Train using 64-bit floats instead of 32bit floats')
flags.DEFINE_string("checkpoint_dir", modelPath, "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS=flags.FLAGS
FLAGS._parse_flags()
print('Parameters:')
for attr,value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr,value))

def data_type():
    return tf.float64 if FLAGS.use_fp64 else tf.float32

def build_vocab(word2vec_path=None):
    if word2vec_path:
        print('Load word2vec_norm file {}'.format(word2vec_path))
        with open(word2vec_path,encoding = 'utf-8') as f:
            header=f.readline()
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
            pairIDs.append(ID)
            sentence_A=line.split(' ')[1]
            sentence_B=line.split(' ')[2]
            relatedness_score=line.split(' ')[3]    
            
            _=[word_to_id[word] for word in jieba.cut(sentence_A, cut_all=False) if word in word_to_id]
            sentencesA_length.append(len(_)) # must be before [0]*(FLAGS.max_length-len(_))
            _+=[0]*(FLAGS.max_length-len(_))
            sentences_A.append(_)
            
            _=[word_to_id[word] for word in jieba.cut(sentence_B, cut_all=False) if word in word_to_id]
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

class Config(object):
    init_scale=0.2
    learning_rate=.01
    max_grad_norm=1.
    keep_prob=1.
    lr_decay=0.98
    batch_size=30
    max_epoch=22
    max_max_epoch=300
    num_layer=1
    
config=Config()
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

# test_path=os.path.join(FLAGS.data_path,'test.txt')
dictionary,init_W=build_vocab(FLAGS.word2vec_norm)
#train_data=file_to_word2vec_word_ids(train_path,dictionary)
test_data=file_to_word2vec_word_ids(dataPath,dictionary,is_test=True)

print("\nEvaluating...\n")
# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
with tf.Graph().as_default():
    initializer=tf.contrib.layers.xavier_initializer()
    with tf.variable_scope('Model',initializer=initializer):
        # Load the saved meta graph and restore variables
        sentences_A=tf.placeholder(tf.int32,shape=([None,FLAGS.max_length]),name='sentences_A')
        sentencesA_length=tf.placeholder(tf.int32,shape=([None]),name='sentencesA_length')
        sentences_B=tf.placeholder(tf.int32,shape=([None,FLAGS.max_length]),name='sentences_B')
        sentencesB_length=tf.placeholder(tf.int32,shape=([None]),name='sentencesB_length')
        labels=tf.placeholder(tf.float32,shape=([None,1]),name='relatedness_score_label')
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
            
        prediction=tf.exp(tf.multiply(-1.0,tf.reduce_mean(tf.abs(tf.subtract(last_states_A[config.num_layer-1][1],last_states_B[config.num_layer-1][1])),1)))
        # cost
        cost=tf.reduce_mean(tf.square(tf.subtract(prediction, labels)))
            
        saver = tf.train.Saver()
        with tf.Session(config=config_gpu) as sess:
            #sess.run(tf.global_variables_initializer())
            saver.restore(sess,FLAGS.save_path+'stslstm-model-20')
            test_cost,test_predict=sess.run([cost,prediction],feed_dict={
                sentences_A:test_data.sentences_A,
                sentencesA_length:test_data.sentencesA_length,
                sentences_B:test_data.sentences_B,
                sentencesB_length:test_data.sentencesB_length,
                labels:np.reshape(test_data.relatedness_scores,(len(test_data.relatedness_scores),1)),
                embedding_placeholder:init_W,
                dropout_f:1.0
            })
        print(test_cost)
        for _ in range(len(test_predict)):
            print((test_data.pairIDs[_]+','+str(round(test_predict[_]*4+1,2))+'\n'))
            
            
with open(outputPath,'w') as fw:
    #fw.write('pair_ID    relatedness_score\n')
    for _ in range(len(test_predict)):
        fw.write(test_data.pairIDs[_]+','+str(round(test_predict[_]*4+1,2))+'\n')

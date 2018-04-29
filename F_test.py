# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 09:52:05 2018

@author: ashut
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt


data = pd.read_csv("data_last.csv")
data = data.drop("Unnamed: 0",1)
data = data.drop("CODE",1)

addi = data.sample(n = 48)

ffdata = data.append(addi)
predictors = list(ffdata.drop("CARatio",1))


SSE_final = []
SSTO_final = []

loss_compilation = []


for i in predictors:
    fdata = ffdata.drop(i,1)
    train, rest = train_test_split(fdata, test_size = 2000)
#valid,test =  train_test_split(fdata, test_size = 1000)


## Data split, scaling and recombining

    Xtr = train.drop("CARatio",1)
    Xre = rest.drop("CARatio",1)  
    min_max_scaler = preprocessing.MinMaxScaler()
    tr_scaled = min_max_scaler.fit_transform(Xtr)
    re_scaled = min_max_scaler.fit_transform(Xre)
    train = np.hstack((np.reshape(train["CARatio"].values,(16000,1)),pd.DataFrame(tr_scaled).values))   
    rest = np.hstack((np.reshape(rest["CARatio"].values,(2000,1)),pd.DataFrame(re_scaled).values))                   
        
    
    
    
    X_test = rest[0:1000,1:]
    Y_test = np.reshape(rest[0:1000,0] , (1000,1))
    X_valid = rest[1000:2000,1:]
    Y_valid = np.reshape(rest[1000:2000,0],(1000,1))
    
    
    
    
    y_mean =    np.mean(fdata["CARatio"].values)
    Y_mm = np.ones((16000,1), "float32")
    Y_m = y_mean*Y_mm
    
    
    
    def feed2(data,batch_size):
        size = len(train)
        #assert size%batch_size == 0
        _, batch = train_test_split(data, test_size = batch_size)
        X = batch[:,1:]
        Y = np.reshape(batch[:,0], (len(X),1))
        return X,Y
    batch_size = 15999
    size = batch_size
    

    
    
    
   
    
    
    
    
    
    
    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, 13))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,1 ))
        
        tf_test_dataset = tf.placeholder(tf.float32,
                                         shape=(1000, 13))
        tf_test_labels = tf.placeholder(tf.float32, shape=(1000,1 ))
        
        tf_valid_dataset = tf.placeholder(tf.float32,
                                          shape=(1000, 13))
        tf_valid_labels = tf.placeholder(tf.float32, shape=(1000,1 ))
        
        tf_dataset = tf.placeholder(tf.float32,
                                        shape=(16000, 13))
        tf_labels = tf.placeholder(tf.float32, shape=(16000 ,1 ))
        
        tf_pred = tf.placeholder(tf.float32,
                                        shape=(16000, 1))
        
        
        weights1 = tf.Variable(
                tf.truncated_normal([13, 32 ]))
        biases1 = tf.Variable(tf.zeros([32]))
        weights2 = tf.Variable(
                tf.truncated_normal([32,1 ]))
        biases2 = tf.Variable(tf.zeros([1]))
        # Training computation.
        logits1 = tf.matmul(tf_train_dataset, weights1) + biases1
        relu1 =  tf.nn.relu(logits1)
        logits2 = tf.matmul(relu1, weights2) + biases2
        
        loss = tf.losses.mean_squared_error(labels=tf_train_labels, predictions=logits2,weights=1.0) 
        optimizer = tf.train.AdagradOptimizer(.5).minimize(loss)
        
        tlogits1 = tf.matmul(tf_test_dataset, weights1) + biases1
        trelu1 =  tf.nn.relu(tlogits1)
        tlogits2 = tf.matmul(trelu1, weights2) + biases2
        
        tloss = tf.losses.mean_squared_error(labels=tf_test_labels, predictions=tlogits2,weights=1.0) 
        
        vlogits1 = tf.matmul(tf_valid_dataset, weights1) + biases1
        vrelu1 =  tf.nn.relu(vlogits1)
        vlogits2 = tf.matmul(vrelu1, weights2) + biases2
        
        vloss = tf.losses.mean_squared_error(labels=tf_test_labels, predictions=vlogits2,weights=1.0)
        
        flogits1 = tf.matmul(tf_dataset, weights1) + biases1
        frelu1 =  tf.nn.relu(flogits1)
        flogits2 = tf.matmul(frelu1, weights2) + biases2
        
        floss = tf.losses.mean_squared_error(labels=tf_labels, predictions=flogits2,weights=1.0)
        ttloss = tf.losses.mean_squared_error(labels=tf_labels, predictions= tf_pred ,weights=1.0)
        
        
        
    num_steps = 20000
    loss_test = []
    loss_valid =[]
    training_loss = []
    w_l = []
    b_l = []
    SSE = 0
    SSTO = 0
    with tf.Session(graph=graph) as session:
        
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            
            batch_data, batch_labels = feed2(train, batch_size)
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, tf_test_dataset : X_test, tf_test_labels : Y_test,tf_valid_dataset : X_valid, tf_valid_labels : Y_valid}
            _, l,j,k  = session.run([optimizer, loss,tloss,vloss], feed_dict=feed_dict)
            loss_test.append(j)
            loss_valid.append(k)
            training_loss.append(l)
            
            if (step %2000 == 0):
                print("Minibatch loss(MSE) at step %d: %f" % (step, l))
                print("test loss(MSE) at step %d: %f" % (step, j))
                print("valid loss(MSE) at step %d: %f" % (step, k))
            if (step == num_steps-1):
                w_l.append(session.run(weights1))
                w_l.append(session.run(weights2))
                b_l.append(session.run(biases1))
                b_l.append(session.run(biases2))
                dic = {tf_dataset : train[:,1:], tf_labels :np.reshape(train[:,0],(16000,1)) }
                SSE = session.run([floss], feed_dict= dic)
                tdic = { tf_labels : np.reshape(train[:,0],(16000,1)),tf_pred : Y_m}
                SSTO = session.run([ttloss], feed_dict= tdic)
                print("SSE: ", SSE)
                print("SSTO: ", SSTO)
                SSE_final.append(SSE)
                SSTO_final.append(SSTO)
                loss_compilation.append(training_loss)
                

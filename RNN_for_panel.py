# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 14:38:34 2018

@author: ashut
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random

data = pd.read_csv("data_last.csv")
data = data.drop("Unnamed: 0",1)
#data = data.drop("lagProdHRisk",1)
#data = data.drop("lagCAratio",1)
#data = data.drop("lagpRegARisk",1)


addi = data.sample(n = 48)

fdata = data.append(addi)

CODE_array_1 = fdata["CODE"].unique().tolist()
CODE_array_2 = random.sample(CODE_array_1, len(CODE_array_1))
CODE_array = CODE_array_2[0:1700]
CODE_array_test = CODE_array_2[1700:]



drop_col = ['ProdHRisk',
 'Atotal',
 'Ntype',
 'Ngroup',
 'RetOnCap',
 'LogATotal',
 'RBCratio',
 'year95',
 'year96',
 'year97',
 'lagpRegARisk',
 'lagProdHRisk',
 'CARatio',
 'lagCAratio']


batch_size = 150
test_batch_size = 189
vocabulary_size = 13
num_nodes = 32

loss_ledger = []
    
for a in drop_col:
    
    def feed_rnn(fdata,CODE_array,batch_size):
        code = np.random.choice(CODE_array,batch_size)
        data_list = []
        label_list = []

        for i in code:
            data1 = fdata.loc[fdata['CODE'] == i]
            data2 = data1.drop("CODE",1)
            Xm = data2.drop("lagpRegARisk",1) 
            Xn = Xm.drop('pRegARisk',1)  
            min_max_scaler = preprocessing.MinMaxScaler()
            np_scaled = min_max_scaler.fit_transform(Xn)
            X = pd.DataFrame(np_scaled).values                    
            Y = np.reshape(data2['pRegARisk'].values,(len(data2),1))
            z = 18 - len(X)
            np.asarray(data_list.append(np.vstack((X,np.zeros((z,13))))))
            np.asarray(label_list.append(np.vstack((Y,np.zeros((z,1))))))
        return np.swapaxes(data_list,0,1),np.swapaxes(label_list,0,1)



    graph = tf.Graph()
    with graph.as_default():
  
  # Parameters:
  # Input gate: input, previous output, and bias.
        ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -1, 1))
        im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -1, 1))
        ib = tf.Variable(tf.zeros([1, num_nodes]))
  # Forget gate: input, previous output, and bias.
        fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -1, 1))
        fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -1, 1))
        fb = tf.Variable(tf.zeros([1, num_nodes]))
  # Memory cell: input, state and bias.                             
        cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -1, 1))
        cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -1, 1))
        cb = tf.Variable(tf.zeros([1, num_nodes]))
  # Output gate: input, previous output, and bias.
        ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -1, 1))
        om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -1, 1))
        ob = tf.Variable(tf.zeros([1, num_nodes]))
  # Variables saving state across unrollings.
        saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
        w = tf.Variable(tf.truncated_normal([num_nodes, 1], -1, 1))
        b = tf.Variable(tf.zeros([1]))
    
    
    
  # Definition of the cell computation.
        def lstm_cell(i, o, state):
            """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
            Note that in this formulation, we omit the various connections between the
            previous state and the gates."""
            input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
            forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
            update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
            state = forget_gate * state + input_gate * tf.tanh(update)
            output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
            return output_gate * tf.tanh(state), state
    
    
    

        train_data = []
        train_label = []
        for i in range(18):
            train_data.append(
          tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
        for i in range(18):
            train_label.append(
          tf.placeholder(tf.float32, shape=[batch_size,1]))

        

    
      # Unrolled LSTM loop.
        outputs = []
        output = saved_output
        state = saved_state
        for i in train_data:
            output, state = lstm_cell(i, output, state)
            outputs.append(output)    
       # State saving across unrollings.
        with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
           # Classifier.
           logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
           loss = tf.reduce_mean(
                   tf.losses.mean_squared_error(labels= tf.concat(train_label, 0), predictions=logits,weights=1.0))##concat
   
        optimizer = tf.train.AdagradOptimizer(.05).minimize(loss)
    
    num_steps = 10000

    loss_list = []

    with tf.Session(graph=graph) as session:
    
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            x,y = feed_rnn(fdata,CODE_array,batch_size)

            feed_dict = dict()
            for i in range(len(y)):
                feed_dict[train_data[i]] = x[i]
                feed_dict[train_label[i]] = y[i]
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            loss_list.append(l)

        
            if (step %1000 == 0):
                print("Minibatch loss(MSE) at step %d: %f" % (step, l))
    loss_ledger.append(loss_list)
                
            



        




    
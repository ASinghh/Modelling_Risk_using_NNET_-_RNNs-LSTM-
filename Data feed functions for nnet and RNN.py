import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

############# Feed Function for Feed Forward Neural Net####################

def feed2(data,batch_size):
    size = len(train)
    #assert size%batch_size == 0
    _, batch = train_test_split(data, test_size = batch_size)
    X = batch[:,1:]
    Y = np.reshape(batch[:,0], (len(X),1))
    return X,Y


############# Feed function for RNN ########################################
Response = "lagpRegARisk"

def feed_rnn(fdata,CODE_array,batch_size):
    code = np.random.choice(CODE_array,batch_size)
    data_list = []
    label_list = []

    for i in code:
        data1 = fdata.loc[fdata['CODE'] == i] ## remove company code
        data2 = data1.drop("CODE",1)
        Xm = data2.drop(Response,1) 
        Xn = Xm.drop(Response,1)  
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(Xn)
        X = pd.DataFrame(np_scaled).values                    
        Y = np.reshape(data2[Response].values,(len(data2),1))
        z = 18 - len(X)
        np.asarray(data_list.append(np.vstack((X,np.zeros((z,13))))))
        np.asarray(label_list.append(np.vstack((Y,np.zeros((z,1))))))
        return np.swapaxes(data_list,0,1),np.swapaxes(label_list,0,1)
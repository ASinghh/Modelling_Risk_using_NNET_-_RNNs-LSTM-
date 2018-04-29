import numpy as np


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
predictors = list(ffdata)

def partial_f_t(array, sse):
    F_list = []
    t_list = []
    for i in array:
        F_list.append((i-sse)*100/(sse))
        t_list.append(np.sqrt((i-sse)/(sse/10000)))
    return F_list,t_list


see_h_risk = [0.0099421768,0.0081452522,0.010150982,0.0078331204,0.010685591,0.0094456365,0.012480779,0.011791273,0.0085500153,0.0089582065,0.0094800927,0.10784466,0.0098443013,0.0081553198]
predictors.remove("ProdHRisk")
predictors_h_risk =  predictors
sse_h_risk_full = 0.0077924528


F_h,t_h= partial_f_t(see_h_risk, sse_h_risk_full)

########################################################################333

predictors = list(ffdata)

predictors.remove("pRegARisk")

predictors_a_risk =  predictors

see_a_risk = [0.00035741398,
 0.001365024,
 0.00038194947,
 0.00057769706,
 0.0014379964,
 0.00043018081,
 0.00086278154,
 0.0005268649,
 0.00076944113,
 0.00052908564,
 0.0024601337,
 0.00059934141,
 0.00048612634,
 0.00042155289]
sse_a_risk_full = 0.00035161469

F_a,t_a= partial_f_t(see_a_risk, sse_a_risk_full)




#########################################################################

predictors = list(ffdata)

predictors.remove("CARatio")

predictors_c_risk =  predictors

see_c_risk = [0.062054645,
 0.061848912,
 0.061586052,
 0.062517002,
 0.062030938,
 0.062105391,
 0.062434126,
 0.062116772,
 0.061017387,
 0.062627532,
 0.061628871,
 0.062851518,
 0.061460014,
 0.08860933]


sse_c_risk_full = 0.061007324

F_c,t_c= partial_f_t(see_c_risk, sse_c_risk_full)

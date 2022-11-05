from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.metrics import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

def generate_data(N = 10000, noise = 0.05):
    
    lam1 = 20 #EC50 1
    lam2 = 5  #EC50 2
    h1 = 2
    h2 = 3
    dose1 = []
    dose2 = []
    uneff = []
    for _ in range(N):
        x1 = np.random.uniform(0,3*lam1)
        x2 = np.random.uniform(0,2*lam2)
        target = 1/((x1/lam1)**h1 + 1)/((x2/lam2)**h2 + 1) + np.random.normal(0,noise)
        target = np.clip(target,0,1)
        dose1.append(x1)
        dose2.append(x2)
        uneff.append(target)
    
    df = pd.DataFrame({'dose1':dose1,'dose2':dose2,'uneff':uneff})
    return df



data = generate_data(10000)

pred_targert = 'uneff'
featmod = data.drop(pred_targert,axis = 1)
labeln = data[pred_targert]
dtmod = pd.concat((featmod,labeln),axis = 1) 
ytall = []
ppall = []
bestpip = RandomForestRegressor(bootstrap=False, max_features=0.1, max_depth=10, min_samples_leaf=3, min_samples_split=3, n_estimators=200)
bestpip.fit(dtmod.drop(pred_targert,axis = 1),dtmod[pred_targert])

st = time.time()
for _ in range(50):
    cv = KFold(n_splits=5,shuffle = True)
    for i, (train, test) in enumerate(cv.split(featmod, labeln)):  #train and validation
        training_data = dtmod.iloc[train,:]        
        ptest = bestpip.predict(featmod.iloc[test,:])
        ytall.extend(list(labeln[test].values))
        ppall.extend(list(ptest))

print(mean_absolute_error(ytall, ppall))
print(time.time()-st)
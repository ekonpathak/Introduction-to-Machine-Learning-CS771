import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit(Z_trn):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# The first 64 columns contain the config bits
	# The next 4 columns contain the select bits for the first mux
	# The next 4 columns contain the select bits for the second mux
	# The first 64 + 4 + 4 = 72 columns constitute the challenge
	# The last column contains the response    
    
    idx=[] # for storing indexes whose output needed to be flipped
    for j in  range(len(Z_trn)):
        m1 = Z_trn[j][64]*8+Z_trn[j][65]*4+Z_trn[j][66]*2+Z_trn[j][67]
        m2 = Z_trn[j][68]*8+Z_trn[j][69]*4+Z_trn[j][70]*2+Z_trn[j][71]
        if m1>m2:
            idx.append(j) 
    Z_trn[idx,-1]=1-Z_trn[idx,-1]#flipping output
    model= fit(Z_trn) #feeding data to actual fit model
    Z_trn[idx,-1]=1-Z_trn[idx,-1]#reflipping output as we have modified the original data
    return model #returning model
def fit(Z_trn):
    data = {} #dictionary for data
    for i in range(16):
        for j in range(i+1,16):
            data[(i,j)]=[]
         
    for j in Z_trn:
        m1 = j[64]*8+j[65]*4+j[66]*2+j[67]
        m2 = j[68]*8+j[69]*4+j[70]*2+j[71]
        if m1>m2:
            m1,m2=m2,m1
        data[(m1,m2)].append(j) #storing data for corresponding dictionary value
    models={} #dictionary for models
    k=0
    ##hyper parameter value.....
    c=[   
         40,
         10,
         30,
         5,
         20,
         45,
         25,
         10,
         40,
         5,
         5,
         5,
         5,
         20,
         10,
         15,
         10,
         40,
         45,
         10,
         30,
         45,
         25,
         40,
         35,
         10,
         10,
         40,
         25,
         40,
         25,
         10,
         20,
         50,
         10,
         15,
         25,
         40,
         10,
         35,
         10,
         5,
         45,
         25,
         10,
         25,
         5,
         30,
         35,
         5,
         15,
         10,
         10,
         20,
         30,
         5,
         45,
         10,
         25,
         40,
         20,
         5,
         40,
         5,
         15,
         10,
         30,
         30,
         50,
         45,
         15,
         25,
         5,
         10,
         20,
         5,
         15,
         25,
         25,
         30,
         25,
         5,
         15,
         20,
         10,
         50,
         15,
         35,
         1,
         15,
         5,
         5,
         50,
         15,
         15,
         15,
         1,
         45,
         20,
         40,
         15,
         15,
         45,
         50,
         10,
         15,
         5,
         25,
         15,
         35,
         50,
         25,
         45,
         5,
         1,
         25,
         50,
         30,
         10,
         15
    ]
    for i in range(16):
        for j in range(i+1,16):
            #model = LinearSVC(C=3.59381366e-02,max_iter=10000,penalty='l2',dual=True,loss='hinge')
            model = LogisticRegression(solver='liblinear',max_iter=1000,C=c[k],dual=False)
            k+=1
            x=np.array(data[(i,j)])
            model.fit(x[:,:64],x[:,-1]) #fitting the model
            models[(i,j)]=model
    return models

################################
# Non Editable Region Starting #
################################
def my_predict( X_tst,models ):
################################
#  Non Editable Region Ending  #
################################

    pred = np.zeros(len(X_tst))
    data = {} #for storing indexes for data corresponding to give keys..
    idx=[] #storing indexes that needed to be flipped..
    for i in range(16):
        for j in range(i+1,16):
            data[(i,j)]=[]
    for j in  range(len(X_tst)):
        m1 = X_tst[j][64]*8+X_tst[j][65]*4+X_tst[j][66]*2+X_tst[j][67]
        m2 = X_tst[j][68]*8+X_tst[j][69]*4+X_tst[j][70]*2+X_tst[j][71]
        if m1>m2:
            m1,m2=m2,m1
            idx.append(j)
        data[(m1,m2)].append(j)
    for i in range(16):
        for j in range(i+1,16):
            x=X_tst[data[(i,j)],:64]     
            pred[data[(i,j)]]=models[(i,j)].predict(x)
    pred[idx]=1-pred[idx] #flipping outputs
    #pred=(pred+1)/2
    
    return pred

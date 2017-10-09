import numpy as np
from pyDOE import *
from sklearn import preprocessing

def strToFloat(datastr,start, end):
    strArr = datastr.strip().split(",")[start: end]
    floatArr = np.array([float(s) for s in strArr])
    return floatArr

def read_matrices(n_feats, filepath):
    train_file = open(filepath)
    train_data = train_file.readlines() 
    n_samples = len(train_data)  
    X_origin = np.array([strToFloat(s, 0, n_feats-1) for s in train_data])
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_origin)	
    Y_train = np.array([strToFloat(s, n_feats-1, n_feats) for s in train_data])
    
    length_scale = np.random.rand()
    magnitude = np.random.rand()
    ridge = np.ones(n_samples)*np.random.rand()
    	
    return X_train, Y_train, length_scale, magnitude, ridge

def LHS_samples(n_feats, test_size):
    X_test = lhs(n_feats, samples=test_size, criterion='center')
    return X_test

def compare(arr1, arr2, n_feats):
    count = 0
    for i in range(0, n_feats):
	dif = arr1[i]-arr2[i]
	if abs(dif) < 0.001:
	    count = count + 1
    if count==n_feats:
	print count
	return True
    else:
        return False

def isInArray(item, arr, n_feats):
    for i in arr:
	if not compare(item, i, n_feats):
	    return False
    return True

def explore_x_test(n_feats, test_size, X_train):
    X_test_draft = LHS_samples(n_feats, test_size)
    X_test = filter(lambda x: not isInArray(x, X_train, n_feats), X_test_draft)
    return X_test

def main():
    X_train, Y_train, length_scale, magnitude, ridge=read_matrices(3, filepath="/home/andy_shen/Desktop/experimenmt_result/SIGMOD/ep1/5/temp.csv")
    x_explore = explore_x_test(2, 100, X_train)
    for x in x_explore:
	print x
    print len(x_explore)
    

if __name__ == '__main__':
    main()


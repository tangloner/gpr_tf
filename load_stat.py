import numpy as np
from pyDOE import *
from sklearn import preprocessing

def rangeOfFeat(n_node, n_threads):
    '''
    the range of each features
    Args:
	n_node: number of nodes
	n_threads: numbers of max threads
    Returns:
	matrix: [n_ps, ps_strategy, sync_protocal, optimizer, batch size, learning_rate, individual op parallelism degree]
		n_ps: [[isInt, min , max]...]
		.....
		.....
		individual op parallelism degree: [[isInt, min , max]...]

    '''
    matrix = np.array([[1, 1, n_node],
			[1, 1, 3],
			[1, 1, 3],
			[1, 1, 9],
			[1, 1, 1000],
			[0, 0.00001, 1],
			[1, 0, n_threads]])
    return matrix


def strToFloat(datastr, start, end):
    strArr = datastr.strip().split(",")[start: end]
    floatArr = np.array([float(s) for s in strArr])
    return floatArr

def readMatrices(n_feats, filepath):
    train_file = open(filepath)
    train_data = train_file.readlines() 
    sum_features = len(train_data[0].split(","))
    n_samples = len(train_data)  
    X_origin = np.array([strToFloat(s, 0, n_feats) for s in train_data])
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_origin)	
    Y_train = np.array([strToFloat(s, n_feats, sum_features) for s in train_data])    
    length_scale = np.random.rand()
    magnitude = np.random.rand()
    ridge = np.ones(n_samples)*np.random.rand()    	
    return X_train, X_origin, Y_train, length_scale, magnitude, ridge, min_max_scaler.data_min_, min_max_scaler.data_range_ 

def LHS_samples(n_feats, test_size):
    X_test = lhs(n_feats, samples=test_size, criterion='center')
    return X_test

def compare(arr1, arr2, n_feats, range_matrix):
    for i in range(0, n_feats):
	dif=0
	if range_matrix[i][0]==1:
	    
	    dif = int(arr1[i]*(range_matrix[i][2]-range_matrix[i][1])+range_matrix[i][1])-arr2[i]
	else:
	    #print arr1[i]*(range_matrix[i][2]-range_matrix[i][1])+range_matrix[i][1]
	    dif = arr1[i]*(range_matrix[i][2]-range_matrix[i][1])+range_matrix[i][1]-arr2[i]
    	if dif!=0.0:
	    return False
    return True

def isInArray(item, arr, n_feats, range_matrix):
    for i in arr:
	if compare(item, i, n_feats, range_matrix):
	    return True
    return False

def explore_x_test(n_feats, test_size, X_origin, range_matrix):
    '''
    Args:
	n_feats: number of features
	test_size: numbers of generated test data
	X_origin: unnormalized train data
	range_matrix: the range of each features
    Returns:
	X_test: normalized test data which not included in train data

    '''
    X_test_draft = LHS_samples(n_feats, test_size)
    X_test = filter(lambda x: not isInArray(x, X_origin, n_feats,range_matrix), X_test_draft)
    return X_test

def recover(arr, range_matrix):
    '''
    Args:
	arr: an item of normalized test data
	range_matrix: the range of each features
    Returns:
	print the unnormalized value of this item

    '''
    i = 0
    for item in arr:
	dif = 0.0
	if range_matrix[i][0]==1:
	    
	    dif = int(item*(range_matrix[i][2]-range_matrix[i][1])+range_matrix[i][1])
	else:
	    dif = item*(range_matrix[i][2]-range_matrix[i][1])+range_matrix[i][1]
	print dif
	i = i+1

def main():
    range_matrix = rangeOfFeat(27, 16)
    X_train, X_origin, Y_train, length_scale, magnitude, ridge, data_min, data_range=readMatrices(7, filepath="/home/andy_shen/code/GD_tune/process.csv")
    x_explore = explore_x_test(7, 1, X_origin, range_matrix)
    for i in range (0, len(x_explore)):
        recover(x_explore[i], range_matrix)

if __name__ == '__main__':
    main()


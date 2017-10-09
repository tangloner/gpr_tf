import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def func(x,a):
    return a*np.power(x, float(-2))

def fit(x_data, y_data):
    popt, pcov = curve_fit(func, x_data, y_data)
    return popt

def predict(popt, x_data):
    return func(x_data, *popt)
 
def main():
    data = open("test3.csv").readlines()
    x_data = np.array([float(line.split(",")[0]) for line in data])
    y_data = np.array([float(line.split(",")[1]) for line in data])
    a = fit(x_data, y_data)
    print predict(a, 50)
if __name__ == '__main__':
    main()

import sklearn
from sklearn.datasets import *
from scipy.io import loadmat
import numpy 

def load_smiles():
	smiles = loadmat('smile_dataset.mat')
	return_val = sklearn.datasets.base.Bunch()
	return_val.data = smiles['X']
	return_val.target = smiles['expressions']
	# print "smiles %s" %smiles
	# print "hi"
	return return_val
    
if __name__ == '__main__':
    load_smiles()


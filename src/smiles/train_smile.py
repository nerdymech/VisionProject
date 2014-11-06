from load_smiles import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from pickle import dump
import numpy
# from numpy import *

def train_smiles ():
  data = load_smiles()
  model = LogisticRegression()
  #could also do SVC (Support Vector Machines) instead of LogisticRegression
  model.fit(data.data, data.target)
  #print model.score(data.data, data.target)
  
  # plt.matshow(numpy.reshape(model.raw_coef_[0][1:],(24,24)).transpose(),cmap='gray')
  
  # plt.show()
  
  return model
  
if __name__ == '__main__':
	model = train_smiles()

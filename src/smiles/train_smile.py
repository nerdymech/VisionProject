from load_smiles import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy
from numpy import *

data = load_smiles()
model = LogisticRegression()
print model
model.fit(data.data, data.target)

#import pdb
#pdb.set_trace()

#fig = plt.figure()

plt.matshow(numpy.reshape(model.raw_coef_[0][1:],(24,24)).transpose(),cmap='gray')

plt.show()
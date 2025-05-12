#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import random
import numpy as np
import matplotlib.pyplot as plt


def load_data(path='mnist.npz'): 
    with np.load(path, allow_pickle=True) as f:  # pylint: disable=unexpected-keyword-arg
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)
def findar(testlist,i,j):
    imax = len(testlist) - 1 
    jmax = len(testlist[0]) - 1 
    if (testlist[i][j] == 0):
         testlist[i][j] = 1
         if ((i == 0)&(j == 0)): 
             findar(testlist,i,j+1)
             findar(testlist,i+1,j)
         elif((i == 0)&(j != 0)&(j != jmax)):
             findar(testlist,i+1,j)
             findar(testlist,i,j-1)
             findar(testlist,i,j+1)
         elif ((i != 0)&(j == 0)&(i != imax)):
             findar(testlist,i+1,j)
             findar(testlist,i,j+1)
             findar(testlist,i-1,j)            
         elif ((i == imax)&(j == jmax)):
             findar(testlist,i,j-1)
             findar(testlist,i-1,j)
         elif((i == imax)&(j != 0)&(j != jmax)):
             findar(testlist,i-1,j)
             findar(testlist,i,j-1)
             findar(testlist,i,j+1)
         elif ((i != 0)&(i != imax) &(j == jmax)):
             findar(testlist,i-1,j)
             findar(testlist,i,j-1)
             findar(testlist,i+1,j)
         elif ((i == imax)&(j == 0)):
             findar(testlist,i,j+1)
             findar(testlist,i-1,j)
         elif ((i == 0)&(j == jmax)):
             findar(testlist,i,j-1)
             findar(testlist,i+1,j)
         else:
             findar(testlist,i,j+1)
             findar(testlist,i+1,j)
             findar(testlist,i-1,j)
             findar(testlist,i,j-1)
             
def kolar(testlist):
    flag = 0
    for i in range(len(testlist)):
        for j in range(len(testlist[0])):  
           if (testlist[i][j] == 0): 
              flag = flag + 1
              findar(testlist,i,j)
    return(flag)


# In[2]:


# файл может быть скачан по ссылке https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
(x_train, y_train), (x_test, y_test) = load_data(path='mnist.npz')
print(x_train.shape, y_train.shape)


# In[3]:


index = 123

print('label is %d' % y_train[index])
plt.imshow(x_train[index])
plt.show()


# In[4]:


class MyFirstClassifier(object):
    def __init__(self):
        pass
    def fit(self, x_train, y_train):
        pass
    def predict(self, x_test):
        predictions = []
        for x in x_test:
             n_pixels = np.sum(x>0)
             new_arr = copy.deepcopy(x)
             if n_pixels > 170:
                    predictions.append(8)
             elif n_pixels < 120:
                    predictions.append(1)
             elif n_pixels > 155:
                    predictions.append(0)
             else:
                     k = kolar(new_arr)
                     if (k == 3):
                        predictions.append(8)
                     elif (k == 2):
                        predictions.append(random.choice([2,6,9,0]))
                     elif (k == 1):
                        predictions.append(random.choice([1,3,4,5,7])) 
                     else:
                        predictions.append(random.randint(0, 9))
                     
        return(predictions)
    
def accuracy_score(pred, gt):
    return np.mean(pred==gt)


# In[5]:


cls = MyFirstClassifier()
cls.fit(x_train, y_train)
pred = cls.predict(x_test)

print('accuracy is %.4f' % accuracy_score(pred, y_test))







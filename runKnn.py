#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from os import listdir
from matplotlib import pyplot as plt
from PIL import Image
from collections import Counter
from math import sqrt
import time


# ## 定义路径和文件

# In[ ]:


path = 'D:/b/'
fle = 'D:/b/7.62.jpg'


# ## 图片文件转01向量

# In[ ]:


def jpg2arr(file):
    ima = Image.open(file)
    ima_arr = np.array(ima)
    ima_arr_fla = ima_arr.flatten()
    ima_01=np.zeros(784,int)
    ima_01=np.where(ima_arr_fla>100,1,0)
    #print (ima_01)
    return ima_01


# ## 函数：目录转成数据集和标签集

# In[ ]:


def getTrain(path):
    file_list = listdir(path) 
    train_data = np.zeros([len(file_list),784],int)
    train_label = np.zeros([len(file_list)],int)
    for i, item in enumerate(file_list):
        c=jpg2arr(path+item)
        d=item.split('.')[0]
        train_data[i] = c
        train_label[i] = d
    return train_data, train_label
#start = time.time()
all_data,all_label = getTrain(path)
#end = time.time()
#print("KNN run %d mins %0.2f sec"%(int((end-start)/60),(end-start)%60))
#np.savetxt('all_data.txt',all_data)
#np.savetxt('all_label.txt',all_label)


# ## 函数：数据标签集分成训练和测试集

# In[ ]:


def train_test_split(all_data,all_label,seed = None):
    if seed:
        np.random.seed(seed)
    rad = np.random.permutation(len(all_data))
    #print(rad)
    test_radio = 0.1
    test_size = int(len(all_data)*test_radio)
    #print(test_size)
    test_indexs = rad[:test_size]
    train_indexs = rad[test_size:]
    test_data = all_data[test_indexs]
    test_label = all_label[test_indexs]
    train_data = all_data[train_indexs]
    train_label = all_label[train_indexs]
    return train_data,train_label,test_data,test_label


# ##  调用KNN类实例化

# In[ ]:


all_data,all_label = getTrain(path)
get_ipython().run_line_magic('run', 'KNN.ipynb')
kfl = KNNclassifier(k=4)
train_data,train_label,test_data,test_label = train_test_split(all_data,all_label,seed=None)
kfl.fit(train_data,train_label)


# ## 输出单文件预测结果

# In[ ]:


#%run KNN.ipynb
imga = jpg2arr(fle)
rt = kfl._predict(imga)
print(rt)


# # 输出多文件预测结果，对比结果，输出运行时间及准确率

# In[ ]:


#%run KNN.ipynb
all_data,all_label = getTrain(path)
train_data,train_label,test_data,test_label = train_test_split(all_data,all_label,seed=None)
kfl = KNNclassifier(k=4)
start = time.time()
kfl.fit(train_data,train_label)
result = kfl.predict(test_data)
end = time.time()
print(result)
print(test_label)                                                                            
print("KNN run %d mins %0.2f sec"%(int((end-start)/60),(end-start)%60))
print("the Accuracy rate is %0.2f"%sum((result == test_label)/len(test_label)))


# ## 函数：单次计算

# In[ ]:


def go(k):
    all_data,all_label = getTrain(path)
    train_data,train_label,test_data,test_label = train_test_split(all_data,all_label,seed=None)
    kfl = KNNclassifier(k=k)
    start = time.time()
    kfl.fit(train_data,train_label)
    result = kfl.predict(test_data)
    end = time.time()
    #print(result)
    #print(test_label)                                                                            
    #print("KNN run %d mins %0.2f sec"%(int((end-start)/60),(end-start)%60))
    #print("the Accuracy rate is %0.2f"%sum((result == test_label)/len(test_label)))
    ru = sum((result == test_label)/len(test_label))
    return ru 


# ## 函数：根据运行次数和K值统计

# In[ ]:


def stas(tms,k):
    T = []
    for i in range(tms):
        ru = go(k)
        T.append(ru)
    print ("the Accuracy rate is %0.2f"%(sum(T)/len(T)))


# ## 重复运行，统计平均准确率

# In[ ]:


stas(tms=10,k=4)


# In[ ]:





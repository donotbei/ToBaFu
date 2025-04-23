#Classify samples using machine learning methods based on extracted topological features
#Number of combined topological features: 80 per channel
#Number of statistic features: 50 per channel
#Number of PI features: 30 per channel
#Method 1: KNN   Method 2: SVM   Method 3: RF


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf

file_path=r'...data/topo_features/CRC5000_F.csv'
#file_path=r"...data/topo_features/CRC5000_S.csv"
#file_path=r"...data/topo_features/CRC5000_PI.csv"

#file_path=r'...data/topo_features/BUS250_F.csv'
#file_path=r"...data/topo_features/BUS250_S.csv"
#file_path=r"...data/topo_features/BUS250_PI.csv"

#file_path=r'...data/topo_features/LC25000_F.csv'
#file_path=r"...data/topo_features/LC25000_S.csv"
#file_path=r"...data/topo_features/LC25000_PI.csv"

df = pd.read_csv(file_path,encoding='utf-8')  #Automatically skip the first row of header
df = df.iloc[:,1:]
data = df.to_numpy()
data = np.round(data, 2)

config = OmegaConf.load('.../config/config.yaml')
zero_count_threshold = config['data']['zero_count_threshold']
columns_to_keep = np.sum(data == 0, axis=0) <= zero_count_threshold
filtered_data = data[:, columns_to_keep]  
y=filtered_data[:,filtered_data.shape[1]-1]    
x=filtered_data[:,0:filtered_data.shape[1]-1]  
##############################
mean = np.mean(x, axis=0)  
std = np.std(x, axis=0)    
x_normalized = (x - mean)/std
X_train,X_test,y_train,y_test = train_test_split(x_normalized,y,test_size= 0.2,random_state=1)

#####################################
#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
KNN_classifier = KNeighborsClassifier(n_neighbors=4) 
KNN_classifier.fit(X_train,y_train)
y_pred = KNN_classifier.predict(X_test)
P=sum(y_pred==y_test)/len(y_test)
print ('\nKNN Accuracy:\n', P)

#RandomForest
from sklearn.ensemble import RandomForestClassifier as RFC
estimator=RFC(random_state=42,n_estimators=1000)
estimator.fit(X_train,y_train)
y_pred=estimator.predict(X_test)
score=estimator.score(X_test,y_test)
print ('\nRF Accuracy:\n', score)

#SVM
from sklearn import svm
clf = svm.SVC(kernel='rbf')  #precomputed、rbf
# linear kernel computation
gram_train = np.dot(X_train, X_train.T)
clf.fit(gram_train, y_train)
#SVC(kernel='precomputed')
# predict on training examples
gram_test = np.dot(X_test, X_train.T)
y_pred=clf.predict(gram_test)
P=sum(y_pred==y_test)/len(y_test)
print ('\nSVM Accuracy:\n', P)

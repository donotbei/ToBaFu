# Load features and labels for all training and test set samples for all classes 
# Remove columns with more than 125 zero elements(50%) from array data 
#1. Classification by KNN 
#2. Classification by SVM 
#3. Classification by RF

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

# Load data from CSV file
file_path1 = r"D:\AAdaily\7\13\BUS250_S.csv" 
df = pd.read_csv(file_path1, encoding='utf-8')

# Drop the first column (image path)
df = df.iloc[:, 1:]
data = df.to_numpy()
data = np.round(data, 2)

# Filter out columns with more than 125 zero elements
zero_count_threshold = 125
columns_to_keep = np.sum(data == 0, axis=0) <= zero_count_threshold
filtered_data = data[:, columns_to_keep]
y = filtered_data[:, -1]    # Labels are in the last column
x = filtered_data[:, :-1]   # Features are in all but the last column

# Normalize features
mean = np.mean(x, axis=0)  
std = np.std(x, axis=0)    
x_normalized = (x - mean)/std

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, stratify=y)

#####################################
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier as RFC
estimator = RFC(random_state=1, n_estimators=1000)  # Configure Random Forest
estimator.fit(X_train, y_train)
y_pred_rf = estimator.predict(X_test)
rf_accuracy = estimator.score(X_test, y_test)
print('\nRF Classification Accuracy:\n', rf_accuracy)

# SVM Classifier
from sklearn import svm
clf = svm.SVC(kernel='rbf')  # Radial Basis Function kernel
clf.fit(X_train, y_train)
y_pred_svm = clf.predict(X_test)
svm_accuracy = sum(y_pred_svm == y_test) / len(y_test)
print('\nSVM Classification Accuracy:\n', svm_accuracy)

# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=4)  # Configure KNN with 4 neighbors
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
knn_accuracy = sum(y_pred_knn == y_test) / len(y_test)
print('\nKNN Classification Accuracy:\n', knn_accuracy)


# Sequentially output Precision, Recall, F1-score for all classes, each as an 8*1 array.
print('------Performance------')
print('Precision', precision_score(y_test, y_pred, average=None))
print('Recall', recall_score(y_test, y_pred, average=None))
print('F1-score', f1_score(y_test, y_pred, average=None))

# Output weighted Precision, Recall, F1-score for all classes, each as a single value.
print('------Weighted------')
print('Weighted precision', precision_score(y_test, y_pred, average='weighted'))
print('Weighted recall', recall_score(y_test, y_pred, average='weighted'))
print('Weighted f1-score', f1_score(y_test, y_pred, average='weighted'))
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
# Plot the confusion matrix heatmap, with rows representing actual values and columns representing predicted values.
cm = confusion_matrix(y_test, y_pred)

# Create a plot
plt.figure(figsize=(8, 6))
# Draw the heatmap, adjusting the color mapping to softer colors
plt.imshow(cm, cmap='GnBu', interpolation='nearest')
# Display the integer values in each grid cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, int(cm[i, j]), ha='center', va='center', color='black', fontsize=18)

cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=18)
plt.xticks(np.arange(cm.shape[1]), ['1', '2'], fontsize=18)
plt.yticks(np.arange(cm.shape[0]), ['1', '2'], fontsize=18)
plt.xlabel('Predicted label', fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.savefig(r'D:\AAdaily\7\13\confusion_F.png', dpi=400)
#plt.savefig(r'D:\AAdaily\7\13\confusion_PI.png', dpi=400)
#plt.savefig(r'D:\AAdaily\7\13\confusion_PI.png', dpi=400)
plt.show()

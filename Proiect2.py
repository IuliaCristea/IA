from sklearn.neural_network import MLPClassifier # importul clasei
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(50,50),
activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
max_iter=200, shuffle=True, random_state=None, tol=0.00007,
momentum=0.9, early_stopping=False, validation_fraction=0.1,
n_iter_no_change=10)

train_images = pd.read_csv("ml-unibuc-2019-24/train_samples.csv", dtype = 'double',header = None)
train_labels =pd.read_csv("ml-unibuc-2019-24/train_labels.csv", dtype = 'double',header = None)
test_images = pd.read_csv("ml-unibuc-2019-24/test_samples.csv", dtype='double',header = None)
#
# print(train_images.shape)
# print(train_labels.shape)
# train_labels = train_labels.values.ravel()
# X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.25)
#
# print("done1")
#
trained_model = mlp_classifier_model.fit(train_images,train_labels)
# print(trained_model.score(X_test,y_test))



predicted_matrix = mlp_classifier_model.predict(test_images)


id=np.arange(1,5001)
d={'Id': id, 'Prediction': predicted_matrix}
data_frame=pd.DataFrame(data=d)
data_frame.to_csv('test_labels_none3.csv')

print("done")
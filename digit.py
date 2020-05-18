import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import seaborn as sns

test_dataset = "test.csv"
digit_dataset = "train.csv"

dataset = pd.read_csv(digit_dataset)
test = pd.read_csv(test_dataset)

#Total examples m= 42000
m = len(dataset)

X = np.array(dataset.iloc[:, 1:])
Y = np.array(dataset.iloc[:,0]).reshape(m,1)

X_test = np.array(test)

X_test_flat = X_test/255
print(X_test.shape)

#m_train = 33600, m_val = 8400
m_train = np.int(len(dataset) * 0.8)
m_val = np.int(len(dataset) * 0.2)

#print(Y.shape, X.shape)

#Dividing into Training and Validation Set
X_train_orig = X[:m_train, :]
Y_train_orig = Y[:m_train, :]

X_val_orig = X[m_train:, :]
Y_val_orig = Y[m_train:, :]

#Viewing Example Using Index
'''
plt.imshow(X_train_orig[1].reshape(28,28))
plt.show()
'''

#Normalizing
X_train = X_train_orig/255
X_val = X_val_orig/255


#Reshaping
Y_train = Y_train_orig
Y_val = Y_val_orig

#Neural Network Model
model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Dense(128, activation = tf.nn.leaky_relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
#model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer = "RMSprop", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])



history = model.fit(x = X_train, y = Y_train, epochs = 50)
#model.load_weights("model_digit.h5")

preds = model.evaluate(X_val,Y_val)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

print(np.array(X_test[1].shape))
Y_test = model.predict(X_test)

out = [0,1,2,3,4,5,6,7,8,9]


print((Y_test * out).shape)
Y_out = np.int32((np.amax((Y_test * out),axis = 1)))
print(Y_out)


submission = pd.DataFrame({ 'ImageId': range(1, 28001), 'Label': Y_out })
submission.to_csv("submission.csv", index=False)
'''
plt.imshow(X_test[0].reshape(28,28))
plt.show()
'''

model.summary()

plt.plot(history.history['acc'])
plt.show()
#Save the model
# serialize model to JSON
'''
model_digit_json = model.to_json()
with open("model_digit.json", "w") as json_file:
    json_file.write(model_digit_json)
# serialize weights to HDF5
model.save_weights("model_digit.h5")
print("Saved model to disk")
'''



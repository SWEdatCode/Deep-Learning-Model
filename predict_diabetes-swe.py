from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

dataset = loadtxt('pima-indians-diabetes.csv' ,delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
print(X.shape)
for i in range(5):
    print(dataset[i])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
mean_train = X_train.mean(axis=0)
std_train = X_train.std(axis=0)
X_train = (X_train-mean_train)/std_train
X_test = (X_test-mean_train)/std_train

model = Sequential()
model.add(Dense(12, input_dim=8, activation = 'relu'))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=120, batch_size=32, validation_split=0.25)
history.history.keys()

import matplotlib.pyplot as plt
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.grid(True)
plt.xlim([0,120])
plt.ylim([0,1.0])
plt.xlabel('epoch')
fig.add_subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.grid(True)
plt.xlim([0,120])
plt.ylim([0,1.0])
plt.xlabel('epoch')

_,accuracy = model.evaluate(X_test,y_test)
print('Accuracy on the test set: %.2f' % (accuracy*100))
y_predict = (model.predict(X_test) > 0.5).astype("int32")
for i in range(10):
    print('%s => %d (expected %d)' % ((X_test[i]).tolist(), y_predict[i], y_test[i]))

import numpy as np
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def randomize(a,b):
    permutation = np.random.permutation(a.shape[0])
    s_a = a[permutation]
    s_b = b[permutation]
    return s_a,s_b

f = open('glass.csv','r')
data = f.read()
f.close()

num_class = 7

data = data.replace('\n',',')
data = data.split(',')
material = data[0:10]

data = data[10:]
data = np.array(data)
data = np.delete(data, len(data) - 1)
data = data.reshape([214,10])
data = data.astype('float32')

classes = []
num_classes = np.zeros([num_class + 1])
label = np.zeros([len(data)])

for i in data:
    num_classes[int(i[9])] +=1
    
num_classes = np.delete(num_classes, 4)

for i in range(1,len(num_classes)):
    num_classes[i] +=num_classes[i-1]
    classes.append(data[int(num_classes[i - 1]):int(num_classes[i])])
    for j in range(int(num_classes[i-1]),int(num_classes[i])):
        label[j] = i

classes = np.array(classes)
data = data[:,:9]
label = to_categorical(label)

randomize(data, label)

train_data, train_label = data[:int(len(data) * 0.9)], label[:int(len(label) * 0.9)]
test_data, test_label = data[int(len(data) * 0.9):], label[int(len(label) * 0.9):]

model = models.Sequential()
model.add(layers.Dense(256, activation = 'relu', input_shape = (9,)))
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dense(num_class, activation = 'softmax'))

model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy',metrics = ['accuracy'])

history = model.fit(train_data,train_label,epochs = 500)
result = model.evaluate(test_data,test_label)

acc = history.history['acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label = 'Training acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

print(result)
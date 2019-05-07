from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation = 'relu', input_shape = (28,28,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(26,activation = 'softmax'))

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('alphabet_small/train', target_size = (28,28))
validation_generator = train_datagen.flow_from_directory('alphabet_small/validation', target_size = (28,28))
test_generator = train_datagen.flow_from_directory('alphabet_small/test', target_size = (28,28))

model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.RMSprop(lr = 1e-4),metrics = ['acc'])

history = model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 20, validation_data = validation_generator, validation_steps = 50)

model.save('alphabet_model.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo',label = 'Training acc')
plt.plot(epochs, val_acc, 'b',label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo',label = 'Training loss')
plt.plot(epochs, val_loss, 'b',label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
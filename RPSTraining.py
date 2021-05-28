from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "RPS-Game/rps"
data_gen = ImageDataGenerator(1/255, rotation_range=20, 
                            width_shift_range=0.2, height_shift_range=0.2, 
                            brightness_range=(0.5,1.1), zoom_range=0.8, 
                            horizontal_flip=True
                            )
training_data_generator = data_gen.flow_from_directory(train_dir, target_size=(150,150), 
                                                    class_mode="categorical",
                                                    batch_size=45 # tested and tuned and set to value 40
                                                    )
validation_generator = data_gen.flow_from_directory(
	train_dir,
	target_size=(150,150),
	class_mode='categorical'
)

l = training_data_generator.class_indices

# creating CNN Model
model=Sequential()
model.add(Conv2D(64,(3,3),activation="relu", input_shape=(150,150,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512,activation="relu"))
model.add(Dense(3,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adamax",metrics=["accuracy"])
model.summary()

history = model.fit(training_data_generator, validation_data=validation_generator, epochs=26, verbose=1)
print(history)
model.save("RPSmax-model.h5")


# plotting accurecy and loss curves
import matplotlib.pyplot as plt
print(history.history.keys())
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.show()

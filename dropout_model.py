from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout


num_classes = 1
# Define a sequential model here
model = models.Sequential()
# add more layers to the model...
# convolutional layer with 'relu' activation and input shape
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# a single flatten layer
model.add(Flatten())

# dropout layer (with 0.5 will approach the 80% val_acc) try the value less than 0.5
model.add(Dropout(0.5))

# "hidden" layer
model.add(Dense(64, activation='relu'))

# a final densely connected layer that outputs a single number # output 0 - 1
model.add(Dense(num_classes, activation='sigmoid'))

# Then, call model.compile()
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

# Train this compiled model by modifying basic_train 
# to import this model, then run:
#   python train.py

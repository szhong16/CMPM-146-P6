from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
# added by ourselves
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

num_classes = 1
# Define a sequential model here
model = models.Sequential()
# add more layers to the model...

# This idea came from Prof in Office hour
# 1_convolutional layer with 'relu' activation and input shape
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 2_
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 3_
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 4_
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 5_a single flatten layer
model.add(Flatten())

# 6_"hidden" layer
model.add(Dense(64, activation='relu'))

# 7_a final densely connected layer that outputs a single number # output 0 - 1
model.add(Dense(num_classes, activation='sigmoid'))

# model.summary()

# Then, call model.compile()
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

# Finally, train this compiled model by running:
# python train.py

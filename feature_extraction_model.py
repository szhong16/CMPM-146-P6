from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# Define a sequential model here
model = models.Sequential()
# add more layers to the model...

# Then, call model.compile()
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

# Finally, train this compiled model by running:
# python train_dropout.py
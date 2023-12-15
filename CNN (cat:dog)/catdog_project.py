from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
from keras import callbacks
import matplotlib.pyplot as plt


# Data Augmentation
train_datagen = ImageDataGenerator( # Initialize and specify how to augment the training data
  	rescale=1./255, rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True) 

val_datagen = ImageDataGenerator(rescale=1./255) # Setting RGB-values to range [0,1]. Only rescaling is needed on validation.

train_generator = train_datagen.flow_from_directory( # Loading and augmenting the training data
    'catdog_data/train', target_size=(180, 180), batch_size=32, class_mode='binary') 

validation_generator = val_datagen.flow_from_directory( # Loading the validation data
  	'catdog_data/validation', target_size=(180, 180), batch_size=32, class_mode='binary')


# Initializing the final model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3))) # Input layer of size 180x180x3
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5)) # Adding dropout to slow down learning and prevent overfitting
model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.002))) # Regularizing using Ridge, agian to slow down learning and prevent overfitting
model.add(layers.Dense(1, activation='sigmoid')) # Output layer using a sigmoid curve for binary classification


# Compiling the model
opt = optimizers.RMSprop(lr=0.0005, momentum=0.1) # Specifying metrics for the optimizer
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


# Saving the best model based on validation loss after fitting the model
callbacks = [
 callbacks.ModelCheckpoint(
 filepath="convnet.keras",
 save_best_only=True,
 monitor="val_loss"
 )
]

#Fitting the model 
history = model.fit(train_generator, epochs=100,  validation_data=validation_generator, callbacks=callbacks)


# Visualizing the loss and accuracy on both training and validation sets
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()


# Preparing test data
test_dataset = val_datagen.flow_from_directory(  
  	'catdog_data/test', target_size=(180, 180), batch_size=32, class_mode='binary') # Loading the test set reusing the same method as with the validation set. No change of parameters are needed.

# Loading the model and testing with the unseen test data
test_model = models.load_model("convnet.keras")
test_loss, test_acc = test_model.evaluate(test_dataset) 
print(f"Test accuracy: {test_acc:.3f}")
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from HodaDatasetReader import read_hoda_cdb


img_rows = 32
img_cols = 32
num_classes = 10


X_train, Y_train = read_hoda_dataset(dataset_path='Train 60000.cdb',
                                images_height=img_rows,
                                images_width=img_cols,
                                one_hot=True,
                                reshape=True)
X_test, Y_test = read_hoda_dataset(dataset_path='Test 20000.cdb',
                              images_height=img_rows,
                              images_width=img_cols,
                              one_hot=True,
                              reshape=True)
X_remaining, Y_remaining = read_hoda_dataset('RemainingSamples.cdb',
                                             images_height=img_rows,
                                             images_width=img_cols,
                                             one_hot=True,
                                             reshape=True)



X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
X_remaining = X_remaining.reshape(X_remaining.shape[0],img_rows,img_cols,1)

X_train = X_train/255
X_test = X_test/255

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(img_rows, img_cols, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model=define_model()

print(model.summary())

batch_size = 256
epochs = 10

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("test_model.h5")


model = load_model('test_model.h5')
opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

X_remaining2 = X_remaining/255

count = 0
result = model.predict(X_remaining2)
for i in range(len(result)):
    if np.argmax(Y_remaining[i]) == np.argmax(result[i]):
        count = count+1
print("Accuracy on Predictions: ", (count/len(result)*100), "%")


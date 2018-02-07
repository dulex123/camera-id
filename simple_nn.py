import os

import keras
from keras.optimizers import Adam
from keras.models import Sequential
from dataset import SinglePatchDataset, DerivOutDataset
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense


class SimpleNN():
    def __init__(self, dataset, val_dataset, model_type):
        # self.data_shape = (512, 512, 3)
        self.data_shape = (2, 2, 2048)
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.model_name = ""

        if model_type == "cnn":
            self.model = self.get_cnn()
        elif model_type == "fcn":
            self.model = self.get_fcn()
        else:
            print("Model not supported")
            exit()

    def train(self):
        os.makedirs("weights", exist_ok=True)
        weights_filepath = "weights/"+self.model_name+".hdf5"

        if os.path.isfile(weights_filepath):
            self.model.load_weights(weights_filepath)

        checkpointer = ModelCheckpoint(weights_filepath)
        rlrop = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=2,
                                  verbose=1,
                                  epsilon=1e-5,
                                  mode='max')

        tensorboard = keras.callbacks.TensorBoard(
            log_dir='logs/' + self.model_name + '/',
            histogram_freq=0,
            write_graph=True,
            write_images=False)

        self.model.fit_generator(self.dataset, len(self.dataset), epochs=100,
                                 validation_data=self.val_dataset,
                                 validation_steps=len(self.val_dataset),
                                 callbacks=[checkpointer, tensorboard])#,
        # rlrop])

    def get_cnn(self):
        self.model_name = "SimpleCNN"
        model = Sequential()
        model.add(Convolution2D(32, (5, 5),
                                input_shape=self.data_shape,
                                activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(16, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(8, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        optimizer = Adam(lr=0.0001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    def get_fcn(self):
        self.model_name = "SimpleFCNDerived"
        model = Sequential()
        model.add(Flatten(input_shape=self.data_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(10, activation='softmax'))
        optimizer=Adam(lr=0.00001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    def predict_single(self):
        pass


if __name__ == "__main__":
    # train_gen = SinglePatchDataset("data/single_patch/train.hdf5", 16)
    # val_gen = SinglePatchDataset("data/single_patch/val.hdf5", 16)
    train_gen = DerivOutDataset("data/deriv_outs/train.hdf5", 16)
    val_gen = DerivOutDataset("data/deriv_outs/val.hdf5", 16)
    model = SimpleNN(train_gen, val_gen, "fcn")
    model.train()

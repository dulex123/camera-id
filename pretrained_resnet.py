import os

import keras
from keras.applications import InceptionV3
from keras.engine import Model
from keras.optimizers import Adam
from keras.models import Sequential
from dataset import SinglePatchDataset
from dataset import AugPatchDataset
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


class PretrainedNNs:
    def __init__(self, dataset, val_dataset, model_type):
        self.data_shape = (512, 512, 3)
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.model_name = ""

        if model_type == "inception":
            self.model = self.get_inceptionNet()
        elif model_type == "resnet":
            self.model = self.get_resnet()
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

        self.model.fit_generator(self.dataset, len(self.dataset), epochs=25,
                                 validation_data=self.val_dataset,
                                 validation_steps=len(self.val_dataset),
                                 callbacks=[checkpointer, tensorboard, rlrop])

    def get_resnet(self):
        self.model_name = "resnet"
        # model = ResNet50(weights='imagenet')
        # preds = model.predict(x)
        # decode the results into a list of tuples (class, description, probability)
        # (one such list for each sample in the batch)
        # print('Predicted:', decode_predictions(preds, top=3)[0])
        # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
        return model

    def get_inceptionNet(self):
        self.model_name = "inceptionNet"
        base_model = InceptionV3(weights='imagenet', include_top=False,
                                 input_shape=self.data_shape)
        outputs = base_model.get_layer("activation_91").output
        print("INPUT", base_model.input_shape)
        x = Flatten()(outputs)
        x = Dense(1024, activation='relu')(x)
        x = Dense(10, activation='softmax')(x)
        model = Model(inputs=base_model.inputs, outputs=x)
        optimizer = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    def predict_single(self):
        pass


if __name__ == "__main__":
    train_gen = AugPatchDataset("data/aug_patch/train.hdf5", 2)
    val_gen = AugPatchDataset("data/aug_patch/val.hdf5", 2)
    model = PretrainedNNs(train_gen, val_gen, "inception")
    model.train()
    # train_gen = SinglePatchDataset("data/single_patch/train.hdf5", 2)
    # val_gen = SinglePatchDataset("data/single_patch/val.hdf5", 2)
    # model = SimpleNN(train_gen, val_gen, "fcn")
    # model.train()

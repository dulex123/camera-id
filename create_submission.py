import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from keras import Model, Sequential
from keras.applications import ResNet50, InceptionV3, DenseNet121
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

test_path = "data/vanilla/test"

# ResNet
base_res = ResNet50(weights='imagenet', include_top=False,
                    input_shape=[512, 512, 3])
res_outputs = base_res.output
res_model = Model(inputs=base_res.inputs, outputs=res_outputs)

# Inception net
base_incept = InceptionV3(weights='imagenet', include_top=False,
                          input_shape=[512, 512, 3])
incept_outputs = base_incept.output
incept_model = Model(inputs=base_incept.inputs, outputs=incept_outputs)

# DenseNet
base_dense = DenseNet121(include_top=False,
                         input_shape=[512, 512, 3])
dense_outputs = base_dense.output
dense_model = Model(inputs=base_dense.inputs, outputs=dense_outputs)
#
lab_to_text = {0: 'Samsung-Galaxy-Note3',
               1: 'Motorola-Droid-Maxx',
               2: 'LG-Nexus-5x',
               3: 'Motorola-X',
               4: 'Samsung-Galaxy-S4',
               5: 'Motorola-Nexus-6',
               6: 'HTC-1-M7',
               7: 'iPhone-4s',
               8: 'Sony-NEX-7',
               9: 'iPhone-6'}


def get_fcn():
    model_name = "SimpleFCNGuillotine"
    model = Sequential()
    # model.add(Flatten(input_shape=self.data_shape))
    model.add(Dense(128, activation='relu', input_shape=(671744,)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    optimizer = Adam(lr=0.00001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

model = get_fcn()
model.load_weights("weights/SimpleFCNGuillotine50val.hdf5")

filenames = []
labels = []
for entry in os.listdir(test_path):
    entry_path = os.path.join(test_path, entry)
    if os.path.isfile(entry_path):
        print("READING IMAGE")
        img = np.array(Image.open(entry_path)) / 255
        img = img[None, ...]


        res_out = res_model.predict(img)
        incept_out = incept_model.predict(img)
        dense_out = dense_model.predict(img)
        print("PREDICTED GUILL Stage1")

        res_batch = res_out.reshape(1, -1)
        incept_batch = incept_out.reshape(1, -1)
        dense_batch = dense_out.reshape(1, -1)

        batch_x = np.concatenate((res_batch, incept_batch, dense_batch), axis=1)

        label = model.predict(batch_x)
        print("PREDICTED GUILL Stage2")
        print(label.shape)

        filenames.append(entry)
        labels.append(lab_to_text[np.argmax(label)])


dframe = pd.DataFrame(columns=['fname', 'camera'])
dframe['fname'] = filenames
dframe['camera'] = labels
dframe.to_csv("submission.csv", index=False)






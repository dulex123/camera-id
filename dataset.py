import os
import cv2
import h5py
import numpy as np
from random import shuffle
from keras.utils import Sequence
# class CIFAR10Sequence(Sequenclass CIFAR10Sequence(Sequence):
#
#     def __init__(self, x_set, y_set, batch_size):
#         self.x, self.y = x_set, y_set
#         self.batch_size = batch_size
#
#     def __len__(self):
#         return math.ceil(len(self.x) / self.batch_size)
#
#     def __getitem__(self, idx):
#         batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
#
#         return np.array([
#             resize(imread(file_name), (200, 200))
#                for file_name in batch_x]), np.array(batch_y)


class SinglePatchDataset(Sequence):
    def __init__(self, hdf5_filepath, batch_size):
        self.h5file = h5py.File(hdf5_filepath, "r")
        self.num_class = 10
        self.batch_size = batch_size
        self.num = self.h5file["train_patches"].shape[0]
        print(self.num)

    def __len__(self):
        return self.num // self.batch_size

    def one_hot(self, batch_y):
        one_hot_y = np.zeros((self.batch_size, self.num_class))
        one_hot_y[np.arange(self.batch_size), batch_y] = 1
        return one_hot_y

    def __getitem__(self, idx):
        start = idx*self.batch_size
        end = (idx+1)*self.batch_size
        batch_x = self.h5file["train_patches"][start:end, ...]
        batch_y = self.one_hot(self.h5file["labels"][start:end])
        return batch_x, batch_y


class CIDDataset():
    def __init__(self, train_path):
        self.x_filenames = []
        self.y = []
        self.categories = {}

        # Add list of all filenames
        label = 0
        for root, dirs, files in os.walk(train_path, topdown=False):
            if root == train_path:
                continue
            for file in files:
                self.x_filenames.append(os.path.join(root, file))
                self.y.append(label)
            self.categories[label] = os.path.basename(root)
            label += 1

        print(self.categories)
        print(self.x_filenames)
        self.i = 0

    def patches_from_img(self, filepath, num_patches, patch_sz=512):
        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)/255
        img_h, img_w, channels = img.shape
        patches = np.zeros([num_patches, patch_sz, patch_sz, channels])
        patch_x_ind = np.random.randint(0, img_w-patch_sz, num_patches,
                                        dtype=np.int32)
        patch_y_ind = np.random.randint(0, img_h-patch_sz, num_patches,
                                        dtype=np.int32)
        for num in range(num_patches):
            x, y = patch_x_ind[num], patch_y_ind[num]
            patches[num, :, :, :] = img[y:y+patch_sz, x:x+patch_sz :]

        return patches

    def center_patch(self, filepath, patch_sz=512):
        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
        img_h, img_w, channels = img.shape
        x = img_h//2-patch_sz//2
        y = img_w//2-patch_sz//2
        crop = img[x:x+patch_sz, y:y+patch_sz, :]
        return crop

    def shuffle_data(self, val_percentage=0.2):
        shuf = list(zip(self.x_filenames, self.y))
        shuffle(shuf)
        shuf_filenames, shuf_labels = zip(*shuf)
        shuf_filenames = list(shuf_filenames)
        shuf_labels = list(shuf_labels)

        tstart, tend = 0, int(1.0-val_percentage * len(shuf_filenames))

        train_x = shuf_filenames[tstart:tend]
        train_y = shuf_labels[tstart:tend]
        val_x = shuf_filenames[tend:]
        val_y = shuf_labels[tend:]

        print(len(train_x))
        print(len(train_y))
        print(len(val_x))
        print(len(val_y))

        return train_x, train_y, val_x, val_y

    def single_patch_dataset(self, output_folder):

        patch_sz = 512

        train_x, train_y, val_x, val_y = self.shuffle_data()

        train_shape = (len(train_x), patch_sz, patch_sz, 3)
        val_shape = (len(val_x), patch_sz, patch_sz, 3)

        for a, b in zip(train_x, train_y):
            print(a, " ", b)
        exit()

        # Create validation set
        h5val_file = h5py.File(os.path.join(output_folder, "val.hdf5"))
        h5val_file.create_dataset("patches", val_shape, np.int8)
        h5val_file.create_dataset("labels", data=np.array(val_y))
        for i, filepath in enumerate(val_x):
            h5val_file["patches"][i, ...] = self.center_patch(filepath)
        h5val_file.close()

        # Create train set
        h5train_file = h5py.File(os.path.join(output_folder, "train.hdf5"))
        h5train_file.create_dataset("patches", train_shape, np.int8)
        h5train_file.create_dataset("labels", data=np.array(train_y))
        for i, filepath in enumerate(train_x):
            h5train_file["patches"][i, ...] = self.center_patch(filepath)
            print(filepath)
        h5train_file.close()

        # Parallel(n_jobs=4)(delayed(self.patches_from_img)(filepath, 3) for
        #     filepath in self.x_filenames)
            # bla.append(patches)
            # print(i)
            # i+=1


if __name__ == "__main__":
    dataset = CIDDataset("data/vanilla/train")
    dataset.single_patch_dataset("data/single_patch")

    # a = SinglePatchDataset("data/single_patch/train.hdf5", 16)
    # x, y = a[4]
    # for c in range(len(a)):
    #     x, y = a[c]
    #     print(x.shape)

    print("ende")
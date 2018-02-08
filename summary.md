# Camera model identification

This project was managed with github kanban, which can be [accessed here](https://github.com/dulex123/camera-id/projects/1). There are also [mini-milestones](https://github.com/dulex123/camera-id/milestones). 

## Folder structure:

- root
  - files - raw dataset .zip files
  - `data` - unpacked and preprocessed folders
    - `vanilla` - raw unzipped .jpg images
    - `single_patch` - single center 512x512 crop per image, stored as hdf5
    - `aug_patch` - multiple random 512x512 augmented crops
  - `logs` - tensorboard logs
  - **dataset.py** - dataset creation tool
    - _class AugPatchDataset_ - Keras sequence class, that reads aug_patch/*.hdf5 files
    - _class SinglePatchDataset_ - Keras sequence class, that reads single_patch/*.hdf5 files
    - _class CIDDataset_ - Creates AugPatch and SinglePatch .hdf5 training files
  - **pretrained_resnet.py**
  - **pretrained_deriv.py** 
  - **simple_nn.py** - simple neural network models
  - **summary.md** - source of this document
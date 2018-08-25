


from __future__ import print_function
from __future__ import division
import os
import cv2
import PIL.Image as Image
# import json
import torch
import random
import numpy as np
random.seed(74)
import matplotlib.pyplot as pl
from torch.utils.data import Dataset, DataLoader
import imgaug as ia
from imgaug import augmenters as iaa
import torchvision.transforms as transforms


# will implement all functionality (data augmentation) of doing
# 1. random crops,
# 2. random flips,
# 3. random rotations,


all_labels = {
            'ADVE'        : 0,
            'Email'       : 1,
            'Form'        : 2,
            'Letter'      : 3,
            'Memo'        : 4,
            'News'        : 5,
            'Note'        : 6,
            'Report'      : 7,
            'Resume'      : 8,
            'Scientific'  : 9
            }

def toTensor(image):
    "converts a single input image to tensor"
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image).float()

######################################################################################################
# Define our sequence of augmentation steps that will be applied to every image.
# random example images
# images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 3)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(1, 3)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                ]),
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                # # either change the brightness of the whole image (sometimes
                # # per channel) or change the brightness of subareas
                iaa.OneOf([
                    # iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.8, 1.2), per_channel=True),
                        second=iaa.ContrastNormalization((0.8, 1.2))
                    )
                ]),
                iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5), # improve or worsen the contrast
            ],
            random_order=True
        )
    ],
    random_order=True
)

######################################################################################################

def get_dataloaders(base_folder, batch_size):
    print('inside dataloading code...')

    class dataset(Dataset):
        def __init__(self, data_dictionary):
            super(dataset, self).__init__()
            self.example_dictionary = data_dictionary
            self.transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            pass

        def __getitem__(self, k):
            example_path, label_name = self.example_dictionary[k]
            example_array = Image.open(example_path).resize((227,227))
            example_array = np.asarray(example_array).astype(np.uint8)*1
            example_array = np.dstack((example_array, example_array, example_array))
            example_array = seq.augment_image(example_array) # augmentation using imgaug
            ###########################################
            # print(np.unique(example_array))
            # pl.imshow(example_array)
            # pl.show()
            ##########################################3
            # example_array = example_array.resize((227, 227), Image.BILINEAR)
            ### stack them depth-wise (from paper)
            # print(example_array.dtype)
            # example_array = np.resize(example_array, new_shape=(227,227,3))
            # example_array = cv2.resize(example_array, 227, 227)
            this_label = all_labels[label_name]
            example_array = toTensor(image=example_array)
            example_array = self.transform(example_array)
            return {'input': example_array, 'label': this_label}

        def __len__(self):
            return len(self.example_dictionary)

    # create training set examples dictionary
    all_examples = {}
    for folder in sorted([t for t in os.listdir(base_folder) if t in all_labels.keys()]): #
        # each folder name is a label itself
        # new folder, new dictionary!
        # print(folder)
        class_examples = []
        inner_path = os.path.join(base_folder, folder)
        for image in [x for x in os.listdir(inner_path) if x.endswith('.tif')]:
            image_path = os.path.join(inner_path, image)
            # for each index as key, we want to have its path and label as its items
            class_examples.append(image_path)
        all_examples[folder] = class_examples

    # split them into train and test
    train_dictionary, val_dictionary, test_dictionary = {}, {}, {}
    for class_name in all_examples.keys():
        class_examples = all_examples[class_name]
        # print(class_examples)
        random.shuffle(class_examples)
        total = len(class_examples)
        train_count = int(total * 0.8); train_ = class_examples[:train_count]
        test = class_examples[train_count:]
        total = len(train_)
        train_count = int(total * 0.9); train = train_[:train_count]
        validation = train_[train_count:]

        for example in train:
            train_dictionary[len(train_dictionary)] = (example, class_name)
        for example in test:
            test_dictionary[len(test_dictionary)] = (example, class_name)
        for example in validation:
            val_dictionary[len(val_dictionary)] = (example, class_name)

    # create dataset class instances
    # print(train_dictionary)
    train_data = dataset(data_dictionary=train_dictionary)
    val_data = dataset(data_dictionary=val_dictionary)
    test_data = dataset(data_dictionary=test_dictionary)
    print('train examples =', len(train_dictionary), 'val examples =', len(val_dictionary),
          'test examples =', len(test_dictionary))

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size,
                                shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size,
                                 shuffle=True, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader


# We shall use this at inference time on our custom downloaded images...
def get_inference_loader(image_path, batch_size):
    print('inside dataloading code...')

    class dataset(Dataset):
        def __init__(self, image_arr, index_dictionary):
            super(dataset, self).__init__()
            self.image_arr = image_arr
            self.index_dictionary = index_dictionary
            pass

        def __getitem__(self, k):
            x, x_, y, y_ = self.index_dictionary[k]
            example_array = self.image_arr[x:x_, y:y_, :]
            # this division is non-sense, but let's do it anyway...
            example_array = (example_array.astype(np.float)/4096)
            example_array = np.dstack((example_array[:,:,2],example_array[:,:,1],example_array[:,:,0]))
            example_array = toTensor(image=example_array)
            return {'input': example_array, 'indices': torch.Tensor([x, x_, y, y_]).long()}

        def __len__(self):
            return len(self.index_dictionary)

    # create training set examples dictionary
    patch = 64 # this is fixed and default
    image_file = np.load(image_path, mmap_mode='r') # we don't want to load it into memory because it's huge
    image_read = image_file['pixels']
    print(image_read.max())
    H, W = image_read.shape[0], image_read.shape[1]
    x_num = W // patch
    y_num = H //patch
    # get a dictionary of all possible indices to crop out of the actual tile image
    index_dict = {}
    for i in range(x_num):
        for j in range(y_num):
            index_dict[len(index_dict)] = (patch*i, patch*i+patch, j*patch, j*patch+patch)

    data = dataset(image_arr=image_read, index_dictionary=index_dict)
    print('number of test examples =', len(index_dict))

    train_dataloader = DataLoader(dataset=data, batch_size=batch_size,
                                  shuffle=False, num_workers=4)

    return train_dataloader, image_read.shape


def histogram_equalization(in_image):
    for i in range(in_image.shape[2]): # each channel
        image = in_image[:,:,i]
        prev_shape = image.shape
        # Flatten the image into 1 dimension: pixels
        pixels = image.flatten()

        # Generate a cumulative histogram
        cdf, bins, patches = pl.hist(pixels, bins=256, range=(0,256), normed=True, cumulative=True)
        new_pixels = np.interp(pixels, bins[:-1], cdf*255)
        in_image[:,:,i] = new_pixels.reshape(prev_shape)
    return in_image


def main():
    # train_dataloader, val_dataloader, test_dataloader = get_dataloaders(base_folder='/home/annus/Desktop/'
    #                                                                                 'forest_cover_change/'
    #                                                                                 'eurosat/images/tif',
    #                                                                     batch_size=1)
    # #
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(base_folder='../dataset',
                                                                        batch_size=16)

    count = 0
    reversed_labels = {v:k for k, v in all_labels.iteritems()}
    while True:
        count += 1
        for idx, data in enumerate(train_dataloader):
            examples, labels = data['input'], data['label']
            print('{} -> on batch {}/{}, {}'.format(count, idx+1, len(train_dataloader), examples.size()))
            if True:
                this = (examples[0].numpy()).transpose(1,2,0).astype(np.uint8)
                print(this.max(), np.unique(this))
                # print(this)
                pl.imshow(this)
                pl.title('{}'.format(reversed_labels[int(labels[0].numpy())]))
                pl.show()



def check_inference_loader():
    this_path = '/home/annus/Desktop/forest_images/test_images/muzaffarabad_pickle.pkl'
    inference_loader, _ = get_inference_loader(image_path=this_path, batch_size=4)
    count = 0
    while True:
        count += 1
        for idx, data in enumerate(inference_loader):
            examples, indices = data['input'], data['indices']
            print('{} -> on batch {}/{}, {}'.format(count, idx + 1, len(inference_loader), examples.size()))
            if True:
                this = np.max(examples[0].numpy())
                indices = indices.numpy()
                print(indices[:,0], indices[:,1], indices[:,2], indices[:,3])
                this = (examples[0].numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
                # this = histogram_equalization(this)
                pl.imshow(this)
                pl.show()
    pass


if __name__ == '__main__':
    main()















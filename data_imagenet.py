
from torchvision import datasets, transforms
from skimage.color import rgb2lab, rgb2gray
import torch
import os
import time
import numpy as np

scale_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip()
    #transforms.ToTensor()
])
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_locked(filepath):
    locked = None
    file_object = None
    if os.path.exists(filepath):
        try:
            buffer_size = 8
            # Opening file in append mode and read the first 8 characters.
            file_object = open(filepath, 'a', buffer_size)
            if file_object:
                locked = False
        except IOError as message:
            locked = True
        finally:
            if file_object:
                file_object.close()
    return locked

def wait_for_file(filepath):
    wait_time = 1
    while is_locked(filepath):
        time.sleep(wait_time)

class TrainImageFolder(datasets.ImageFolder):
    #def __init__(self, data_dir, transform):
    #   self.file_list=os.listdir(data_dir)
    #   self.transform=transform
    #   self.data_dir=data_dir
    def __getitem__(self, index):
 #       try:

         avoiding = True
         while avoiding:
             try:
                 path,_=self.imgs[index]
                 img = self.loader(path)
                 avoiding = False
             except Exception as e:
                 #print(e)
                 #print('lost ',path)
                 index = index + 1

         if self.transform is not None:
                img_original = self.transform(img)
                img_resize=transforms.Resize(56)(img_original)
                img_original = np.array(img_original)
                img_lab = rgb2lab(img_resize)
                img_ab = img_lab[:, :, 1:3]
                img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
                img_ab = img_ab.type(torch.FloatTensor)

                img_original = rgb2lab(img_original)[:,:,0]-50.
                img_original = torch.from_numpy(img_original)
                img_original = img_original.type(torch.FloatTensor)

                return img_original, img_ab
         else:
                print('no transformation')

    def __len__(self):
        return len(self.imgs)

class ValImageFolder(datasets.ImageFolder):

    def __getitem__(self, index):

        path,_=self.imgs[index]
        img=self.loader(path)
        img_scale = scale_transform(img)
        img_rescale = transforms.Resize((56,56))(img_scale)
        img_rescale = np.asarray(img_rescale)
        img_rescale = rgb2lab(img_rescale)[:,:,0]-50
        img_rescale = torch.from_numpy(img_rescale)
        img_scale = np.asarray(img_scale)
        img_scale = torch.from_numpy(img_scale)

        return img_scale,img_rescale

    def __len__(self):
        return len(self.imgs)

class ValTrainImageFolder(datasets.ImageFolder):
    def __getitem__(self,index):
        try:
            path,_=self.imgs[index] 
            img=self.loader(path)
            if self.transform is not None: img_original = self.transform(img)
            else: img_original = img

            img_resize=transforms.Resize(56)(img_original)
            #train
            img_lab = rgb2lab(img_resize)
            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
            img_ab = img_ab.type(torch.FloatTensor)
            
            img_grey = np.array(img_original)
            img_grey = rgb2lab(img_grey)[:,:,0]-50.
            img_grey = torch.from_numpy(img_grey)
            img_grey = img_grey.type(torch.FloatTensor)
            #val
            img_rescale = np.asarray(img_resize)
            img_rescale = rgb2lab(img_rescale)[:,:,0]-50
            img_rescale = torch.from_numpy(img_rescale)
            
            img_original = np.asarray(img_original)
            img_original = torch.from_numpy(img_original)

            return img_grey, img_ab, img_original, img_rescale

        except:
            print('exception')
            print(index)
            print(self.imgs[index])
            pass 

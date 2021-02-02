from Utils.utils import _decode
from torch import from_numpy, device
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model.U_CapsNet import CapsNet_MR
from Utils.data_imagenet import  ValImageFolder
import torch.optim as optim
import os
from PIL import Image
from Utils.utils import resume_model, psnr
import numpy as np
import torch.nn as nn
from skimage.color import rgb2lab
import shutil
from tqdm import tqdm


class RGB2LAB(object):

    def __init__(self):
        super(RGB2LAB,self).__init__()

    def __call__(self,img):

        # Original LAB image
        array_img = np.asarray(img).copy()
        lab_img_L = rgb2lab(array_img)
        output_lab = from_numpy(lab_img_L).float()

        # Resized LAB
        img_resize = transforms.Resize((56, 56), Image.LANCZOS)(img)
        img_resize = np.asarray(img_resize)
        lab_img_resize_lab = rgb2lab(img_resize).copy()
        lab_img_resize_l = lab_img_resize_lab[:, :, 0] - 50
        lab_img_resize_ab = lab_img_resize_lab[:, :, 1:3]
        output_resize_l = from_numpy(lab_img_resize_l).float()
        output_resize_ab = from_numpy(lab_img_resize_ab).float()

        return output_lab, output_resize_l, output_resize_ab, array_img #img_resize

transform = transforms.Compose([
    transforms.Resize((224, 224), Image.LANCZOS),
    RGB2LAB()
])
dataset = 'COCOstuff'
val_dir_name = "data/"+dataset
nome_modello = "U_CapsNet"
CUDA = "cuda"
batch_size = 32
# #Init and load dataset
ImageDataset = ValImageFolder(val_dir_name, transform=transform) #datasets.ImageNet(val_dir_name,split='train',transform=original_transform) #ValImageFolder(val_dir_name) #
dataloaders = DataLoader(ImageDataset, batch_size=batch_size, shuffle=False)  #, num_workers = num_workers)
#
# # SET DEVICE
device = device(CUDA)  # if cuda.is_available() else "cpu")
# # INIT MODEL AND OPTIM
model = CapsNet_MR(128)
optimizer = optim.Adam(model.parameters(), lr=0.0002)
criterion = nn.CrossEntropyLoss().cuda()

model_path = os.path.join('model/checkpoint__U_CapsNet.pth.tar')
model, optimizer, epoch_start = resume_model(model_path, model, optimizer, map_location=CUDA)
model = model.to(device)


mother_folder = "Results/RP_test_"+dataset
mother_folder_orig_clone = mother_folder + '_orig_clone'
if os.path.exists(mother_folder):
    shutil.rmtree(mother_folder, ignore_errors=True)
if os.path.exists(mother_folder_orig_clone):
    shutil.rmtree(mother_folder_orig_clone, ignore_errors=True)
os.mkdir(mother_folder)
os.mkdir(mother_folder_orig_clone)



psnr_sum = []
psnr_sum_self = []
for batch_id, (img_lab, target, name_path) in enumerate(tqdm(dataloaders)):
        img_or_lab = img_lab[0] #<-original image CIELAB [Batch_size,224,224,3]
        img_or_lab_resize = img_lab[1] #<-resized Lum image [Batch_size,56,56,1]
        img_or_lab_resize_ab = img_lab[2] #<-resized AB image [Batch_size,56,56,2]
        orig_img = img_lab[3].numpy() # Original RGB image

        val_img_L = img_or_lab[:, :, :, 0].unsqueeze(1) - 50 #[Batch_size,1,224,224]
        val_img_L = val_img_L.to(device)

        _, reconstructionQ = model(val_img_L)
        image_lab = _decode(img_or_lab[:,:,:,0], reconstructionQ)

        for j in range(len(image_lab)):
            im_rgb, im_lab = image_lab[j]
            im_rgb = (im_rgb*255).astype(np.uint8)
            psnr_sum.append(psnr(orig_img[j], im_rgb))

            pil_image = Image.fromarray(im_rgb, 'RGB')
            dstfolder = os.path.join(mother_folder, target[j])
            if not os.path.exists(dstfolder):
                os.mkdir(dstfolder)
                os.mkdir(os.path.join(mother_folder_orig_clone, target[j]))

            pil_image.save(os.path.join(dstfolder, name_path[j].replace('JPEG','png')))
            Image.fromarray(orig_img[j], 'RGB').save(os.path.join(mother_folder_orig_clone,
                                                                  target[j],
                                                                  name_path[j].replace('JPEG','png')))

print(np.mean(psnr_sum))

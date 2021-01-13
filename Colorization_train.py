from torch import from_numpy, Tensor, stack, device
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from training_layers import PriorBoostLayer, NNEncLayer, NonGrayMaskLayer
from data_imagenet import TrainImageFolder, ValImageFolder, ValTrainImageFolder
from utils import rfolder, save_checkpoint, isnan, _decodeAB, _decode
from U_CapsNet import CapsNet_MR as UCapsNet        # <---------------------MODEL
############################################################################################################
# Training procedure for U_CapsNet*.
# ###########SET VAR########################################################################################
AnnealedMean = False
file_model_name = "U_CapsNet_Niki_Q_batch_overfit"
CUDA, db_used = "cuda", 'ImageNet'
ADAM_LR = 1e-3
n_epochs = 100
epoch_val = 5
batch_size = 32
logits_num = 256
bool_lossQ, bool_lossAB = True, False
#######################
folder_results = rfolder("Results_/"+file_model_name+"_"+db_used, n_epochs)
#############################################################################################################

class RGB2LAB(object):
    def __init__(self):
        super(RGB2LAB,self).__init__()
    def __call__(self,img):
        arr_img = np.asarray(img)
        output = from_numpy(arr_img)
        lab_img = rgb2lab(arr_img)
        output_l = from_numpy(lab_img[:, :, 0] - 50)

        lab_img_resize = transforms.Resize((56, 56))(img)
        lab_img_resize = np.asarray(lab_img_resize)
        lab_img_resize = rgb2lab(lab_img_resize)
        output_r_l = from_numpy(lab_img_resize[:, :, 0] - 50)
        output_r_ab = from_numpy(lab_img_resize[:, :, 1:3])

        return output, output_l, output_r_l, output_r_ab

original_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    RGB2LAB()
])
dataset_path = '/media/TBData/Datasets/ImageNetOriginale'
ImageDataset = {'train': datasets.ImageNet(dataset_path,split='train',transform=original_transform),
                'val': datasets.ImageNet(dataset_path,split='val',transform=original_transform)}
dataloaders = {'train': DataLoader(ImageDataset['train'], batch_size=batch_size, shuffle=False),
               'val': DataLoader(ImageDataset['val'], batch_size=batch_size, shuffle=False)}

#SET DEVICE
device = device(CUDA)
#INIT MODEL AND OPTIM
model = UCapsNet(logits_num)
optimizer = optim.Adam(model.parameters(), lr=ADAM_LR)
criterion = nn.CrossEntropyLoss().to(device)
model = model.to(device)
print('%%%%%%%%%%%%%%%%%   MODEL   %%%%%%%%%%%%%%%%%%')
print(file_model_name)
with open(folder_results+"/_MODEL_STRUCTURE.txt", "a") as text_file:
    text_file.write(str(model))
    text_file.write(' ----- lossQ {} '.format(str(bool_lossQ)))
    text_file.write(' ----- lossAB {} '.format(str(bool_lossAB)))
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

#SET VAR FOR BOOST
encode_layer = NNEncLayer()
boost_layer = PriorBoostLayer()
nongray_mask = NonGrayMaskLayer()

#TRINING AND VALIDATION
loss_train = []
model.train()

#random batch for validation
(image_batch_val,target) = next(iter(dataloaders['val']))

pbar = tqdm(range(n_epochs))
for epoch in pbar: 
    train_loss, losted = 0, 0
    n_batch = 0
    #try:
    for batch_id, (image_batch,target)in enumerate(dataloaders['train']):
        img_or = image_batch[0]
        img_l = image_batch[1]
        img_r_l = image_batch[2]
        img_r_ab = image_batch[3]

        img_L = img_l.to(device)
        img_r_L = img_r_l.to(device)

        img_ab = img_r_ab.float()
        encode, max_encode = encode_layer.forward(img_ab.permute(0,3,1,2))
        targets = Tensor(max_encode).long().to(device)
        boost = Tensor(boost_layer.forward(encode)).float().to(device)
        mask = Tensor(nongray_mask.forward(img_ab)).float().to(device)
        boost_nongray = boost*mask

        img_ab = img_ab.to(device)
        output, reconstructionAB, reconstructionQ = model(img_L.unsqueeze(1).float())
        lossAB = model.reconstruction_loss(img_ab.permute(0,3,1,2), reconstructionAB)
        lossQ = (criterion(reconstructionQ, targets)*boost_nongray.squeeze(1)).mean()

        if bool_lossQ and bool_lossAB: loss = lossQ + lossAB
        else:
           if bool_lossQ: loss = lossQ
           else:
               if bool_lossAB: loss = lossAB

        if isnan(loss):
            print("loss lost batch_id {} lossQ {}/lossAB {}".format(0,lossQ,lossAB))
            losted += 1
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.data)
            n_batch += 1

        del encode, max_encode,  targets, boost, mask, boost_nongray, img_L, img_ab, output, lossAB, lossQ, loss, reconstructionAB,  reconstructionQ
    if n_batch > 0:
        loss_train.append(train_loss/(n_batch))
        pbar.set_postfix(loss='{:.4f}'.format(loss_train[-1]))
############################################################################################################
#--------------------------------------VALIDATION-----------------------------------------------------------
############################################################################################################
    if epoch % epoch_val == 0 and epoch > 0:
        print("Validation epoch {}".format(epoch))
        img_or = image_batch_val[0]
        img_l = image_batch_val[1]
        img_r_l = image_batch_val[2]
        img_r_ab = image_batch_val[3]

        val_img_L = img_l.to(device)
        output, reconstructionAB, reconstructionQ = model(val_img_L.unsqueeze(1).float())

        image_L = val_img_L.squeeze(1).cpu().detach().numpy()
        image_lab = _decode(img_r_l, reconstructionQ)

        if batch_size > 20:
            row = int((batch_size*2)/6)+int((batch_size*2)%6)
            col = 6
        else:
            row = int((batch_size*2)/4)
            col = 4
        fig = plt.figure()
        fig.set_figheight(30)
        fig.set_figwidth(30)
        Tot = row*col
        plt.axis('off')
        i = 1
        for j in range(1, (batch_size )):
            im_rgb, im_lab = image_lab[j-1]
            image_Or = img_or.cpu().detach().numpy()
            ax1 = fig.add_subplot(row, col, i)
            ax1.set_title('RGB original')
            ax1.imshow(image_Or[j-1].astype(np.uint8))
            ax1.grid(False)
            ax1.axis('off')
            i += 1
            ax2 = fig.add_subplot(row, col, i)
            ax2.set_title('RGB recon')
            ax2.imshow((im_rgb*255).astype(np.uint8))
            ax2.grid(False)
            ax2.axis('off')
            i += 1
        plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        plt.savefig(folder_results+"/_reconstructedQ_"+str(epoch)+".png")
        plt.close(fig)
        plt.clf()
        del im_rgb, im_lab, image_lab, image_L, reconstructionAB, reconstructionQ, output
        if epoch > (n_epochs - (n_epochs / 10)): save_checkpoint({
                'epoch': epoch + 1,
                'loss_type': file_model_name,
                'arch': 'CapsNet',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
             }, folder_results+"/BACKUP_model_log/checkpoint_"+"_"+file_model_name+"_"+str(epoch)+".pth.tar")

print("Loss value for training phase: {}".format(train_loss / len(dataloaders['train'])))
save_checkpoint({
        'epoch': epoch + 1,
        'loss_type': file_model_name,
        'arch': 'CapsNet',
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    },folder_results+"/model_log/checkpoint_"+"_"+file_model_name+"_"+str(epoch)+".pth.tar")

epochs = np.arange(1, len(loss_train)+1)
plt.plot(epochs, loss_train, color='g', label='loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training phase')
plt.savefig(folder_results+"/"+file_model_name+".png")
plt.clf()

print("Last Validation ...")
img_or = image_batch_val[0]
img_l = image_batch_val[1]
img_r_l = image_batch_val[2]
img_r_ab = image_batch_val[3]

val_img_L = img_l.to(device)
_, reconstructionAB, reconstructionQ = model(val_img_L)

image_L = val_img_L.squeeze(1).cpu().detach().numpy()
image_lab = _decode(img_r_l, reconstructionQ)
fig = plt.figure()
fig.set_figheight(30)
fig.set_figwidth(30)
Tot = row*col
plt.axis('off')
i = 1
for j in range(1, (batch_size )):
    im_rgb, im_lab = image_lab[j-1]
    image_Or = img_or.cpu().detach().numpy()
    ax1 = fig.add_subplot(row, col, i)
    ax1.set_title('RGB original')
    ax1.imshow(image_Or[j-1].astype(np.uint8))
    ax1.grid(False)
    ax1.axis('off')
    i += 1
    ax2 = fig.add_subplot(row, col, i)
    ax2.set_title('RGB recon')
    ax2.imshow((im_rgb*255).astype(np.uint8))
    ax2.grid(False)
    ax2.axis('off')
    i += 1
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
plt.savefig(folder_results+"/_reconstructed_final_"+str(epoch)+".png")
plt.close(fig)
plt.clf()
print('End')

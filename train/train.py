# This file tunes the feature extractor for tinyface.

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.simplefilter("ignore")

import numpy as np

from myUtil import resnet50, freeze_all, freeze_till_layer
from models.models.classifier import Classifier
from models.models.MLP import MLP
# from arcface_pytorch.models.metrics import ArcMarginProduct
from dataloader import Vggface2
from forward import forward


if __name__ == '__main__':

    batchsz = 96
    epochs = 100
    LR = 1e-5
    val_freq = 1
    
    ckp_path = r'C:\Users\JACK\Desktop\recognize_tinyface_continuous_v2\train\ckps\batchsz144_LR5e-05_16_32_64_128\e1_iter0_train0.00_3e-05.pth'
    
    save_root = f'ckps/batchsz{batchsz}_LR{LR}_16_32_64_128_tunebackbone'
    
    if not os.path.exists(save_root):
        os.makedirs(save_root)
        

    # -------------------------- DATA LOADER ----------------------------
    # train
    train_root = r'C:\Users\JACK\Vggface2\splitted\train'
    train_vggface = Vggface2(root=train_root, train=True, visualize=False)
    train_loader = DataLoader(dataset=train_vggface,
                                 batch_size=batchsz,
                                 num_workers=6,
                                 shuffle=True,
                                 pin_memory=True,
                                 drop_last=True,)


    train_len = len(train_loader)

    # ----------------------------- MODELS  ------------------------

    # 1. SRnet
    SRnet = nn.Upsample(size=(224, 224), mode='bicubic').cuda()

    # # 2. common feature extractor
    Resnet = resnet50('../models/resnet50_ft_weight.pkl', num_classes=8631).cuda()

    # # 3. specific feature extractor
    MLP = MLP().cuda()

    # # 4. Arcface classifier
    Classifier = Classifier().cuda()

    if (torch.cuda.device_count() > 1):
        print(f'Using {torch.cuda.device_count()} GPUs')
        
        SRnet = nn.DataParallel(SRnet)
        Resnet = nn.DataParallel(Resnet)
        MLP = nn.DataParallel(MLP)
        Classifier = nn.DataParallel(Classifier)

        
    if ckp_path is not None:
        ckp = torch.load(ckp_path)
        
        SRnet.load_state_dict(ckp['SRnet'])
        Resnet.load_state_dict(ckp['Resnet'])
        MLP.load_state_dict(ckp['MLP'])
        Classifier.load_state_dict(ckp['Classifier'])
        
        print(f'Successufully loaded ckp: {ckp_path}')

    # ---------------------- LOSS OPTIMIZER  ------------------------
    CEloss = torch.nn.CrossEntropyLoss()
    MSEloss = torch.nn.MSELoss()
    
    freeze_till_layer(Resnet, fix_till_layer=129)
    resnet_params_to_update = []
    for name, param in Resnet.named_parameters():
        if param.requires_grad == True:
            resnet_params_to_update.append(param)
            print("\t",name)
    
    optimizer = optim.Adam([{'params': resnet_params_to_update},
                            {'params': MLP.parameters()},
                            {'params': Classifier.parameters()}],
                            lr=LR,
                            weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.6)

    # --------------------- SummaryWriter -------------------------------
    writer = SummaryWriter(log_dir=os.path.join(save_root, 'runs'))

    # --------------------- TRAINING  ---------------------------------
    
    #freeze_all(Resnet)
    
    # freeze_all(MLP)
    
    for e in range(epochs):
        """""""""""""""""""""""""""""""""""""""
        1. Training
        """""""""""""""""""""""""""""""""""""""
        SRnet.train()
        Resnet.eval()
        
        MLP.train()
        Classifier.train()

        epoch_train_loss = 0
       
        for i, (img16, img32, img64, img128, label) in enumerate(train_loader):

            mseloss16, celoss16, loss16, \
            mseloss32, celoss32, loss32, \
            mseloss64, celoss64, loss64, \
            mseloss128, celoss128, loss128, \
            acc16_16, acc16_32, acc16_64, acc16_128, \
            acc32_16, acc32_32, acc32_64, acc32_128, \
            acc64_16, acc64_32, acc64_64, acc64_128, \
            acc128_16, acc128_32, acc128_64, acc128_128 = forward(img16, img32, img64, img128, label,
                                                     SRnet,
                                                     Resnet,
                                                     MLP,
                                                     Classifier, 
                                                     CEloss,
                                                     MSEloss,
                                                     optimizer)
            
            if i % 100 == 0:
                print(f'epoch: {e} | iter: {i}/{train_len} | loss16: {loss16:.2f}, loss32: {loss32:.2f}, loss64: {loss64:.2f}, loss128: {loss128:.2f}')
    
                writer.add_scalars('mse_loss', {'mseloss16':mseloss16, 'mseloss32':mseloss32, 'mseloss64':mseloss64, 'mseloss128':mseloss128}, train_len*e+i)
                writer.add_scalars('ce_loss', {'celoss16':celoss16, 'celoss32':celoss32, 'celoss64':celoss64, 'celoss128':celoss128}, train_len*e+i)
                writer.add_scalars('mse_loss+ce_loss', {'loss16':loss16, 'loss32':loss32, 'loss64':loss64, 'loss128':loss128}, train_len*e+i)
                
                writer.add_scalars('acc16', {'acc16_16':acc16_16, 'acc16_32':acc16_32, 'acc16_64':acc16_64, 'acc16_128':acc16_128}, train_len*e+i)
                writer.add_scalars('acc32', {'acc32_16':acc32_16, 'acc32_32':acc32_32, 'acc32_64':acc32_64, 'acc32_128':acc32_128}, train_len*e+i)
                writer.add_scalars('acc64', {'acc64_16':acc64_16, 'acc64_32':acc64_32, 'acc64_64':acc64_64, 'acc64_128':acc64_128}, train_len*e+i)
                writer.add_scalars('acc128', {'acc128_16':acc128_16, 'acc128_32':acc128_32, 'acc128_64':acc128_64, 'acc128_128':acc128_128}, train_len*e+i)
                
                writer.add_scalars('LR', {'LR': scheduler.optimizer.param_groups[0]['lr']}, train_len*e+i)
            
            total_loss = (loss16 + loss32 + loss64 + loss128) / 4
            epoch_train_loss += total_loss
            
            if (i % int(len(train_loader) / 4) == 0):
                """ saving skeckpoint"""

                epoch_train_loss /=  (int(len(train_loader) / 4))
                
                LR = scheduler.optimizer.param_groups[0]['lr']
                
                save_name = f'e{e}_iter{i}_train{epoch_train_loss:.2f}_{LR}.pth'
        
                state_dict = {'SRnet': SRnet.state_dict(),
                              'Resnet': Resnet.state_dict(),
                              'MLP':MLP.state_dict(),
                              'Classifier': Classifier.state_dict(),
                              'epoch': e,
                              'optimizer': optimizer.state_dict()}
                torch.save(state_dict, os.path.join(save_root, save_name))
                
                epoch_train_loss = 0
                
        scheduler.step()

        """""""""""""""""""""""""""""""""""""""
        2. Testing / saving checkpoint
        """""""""""""""""""""""""""""""""""""""
        
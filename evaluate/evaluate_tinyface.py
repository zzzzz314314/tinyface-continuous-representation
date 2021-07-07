# This file extract the features of test images.
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from scipy.io import loadmat
import torch
from torch import nn
# torch.backends.cudnn.benchmark = True
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.transforms
import PIL.Image
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import OrderedDict
import warnings
warnings.simplefilter("ignore")

from tqdm import tqdm
from myUtil import resnet50
from evaluate_forward import forward
from test_face_identification_parallel2 import calculate_acc

class Tinyface(Dataset):
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt

    def __init__(self, img_path, mode, mat_id_path=None):
        self._img_path = img_path
        self._mode = mode
        self._dataset = self._find_dataset(img_path)
        self._img_list = self._create_imglist(mat_id_path)


    def _find_dataset(self, img_path):
        last = os.path.basename(img_path)
        if last == 'Gallery_Match':
            return 'gallery'
        elif last == 'Probe':
            return 'probe'
        else:
            return 'distractor'

    def _create_imglist(self, mat_id_path):
        if mat_id_path is not None:
            # for gallery and probe
            annots = loadmat(mat_id_path)
            ids, _set = annots[self._dataset + '_ids'], annots[self._dataset + '_set']
            self._id_list = ids.reshape(-1)

            flatten_gallery_set = [_set[i, 0][0] for i in range(_set.shape[0])]
        else:
            # for distractor
            flatten_gallery_set = os.listdir(self._img_path)

        return flatten_gallery_set

    def __getitem__(self, index):
        img_name = self._img_list[index]
        img = PIL.Image.open(os.path.join(self._img_path, img_name))

        w, h = img.size
        img_size = (w + h) / 2
        img = self._transform(img, 160)
        
        return img, img_size
    

    def _transform(self, img, r):
        def transform2(img):
            img = img[:, :, ::-1]  # H, W, C(RGB) -> H, W, C(BGR)
            img = img.astype(np.float32)
            img -= self.mean_bgr
            img = img.transpose(2, 0, 1)  # H, W, C(BGR) -> C(BGR), H, W
            img = torch.from_numpy(img).float()
            return img

        #img = torchvision.transforms.GaussianBlur(kernel_size=7)(img)
        img = torchvision.transforms.Resize((r, r))(img)
        img = np.array(img, dtype=np.uint8)

        assert len(img.shape) == 3  # assumes color images and no alpha channel

        img = transform2(img)

        return img

    def __len__(self):
        return len(self._img_list)

def filter_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


def evaluate(ckp_path=None):

    print('startiiiiiiiiiiiiiiiing')
    
    file = open('ckp_accuracy.txt', 'a')
    
    # -------------------------- DATA LOADER START -------------------------------------
    dataset_root = r'C:\Users\JACK\tinyface\Testing_Set'

    gallery = Tinyface(os.path.join(dataset_root, 'Gallery_Match'),
                       'test',
                       os.path.join(dataset_root, 'gallery_match_img_ID_pairs.mat'))
    
    gallery_db = DataLoader(dataset=gallery,
                            batch_size=180,
                            num_workers=6,
                            pin_memory=True,
                            shuffle=False)
    
    probe = Tinyface(os.path.join(dataset_root, 'Probe'),
                        'test',
                       os.path.join(dataset_root, 'probe_img_ID_pairs.mat'))

    probe_db = DataLoader(dataset=probe,
                            batch_size=180,
                            num_workers=6,
                            pin_memory=True,
                            shuffle=False)
    
    gallery_distractor = Tinyface(os.path.join(dataset_root, 'Gallery_Distractor'),
                                    'test')

    distractor_db = DataLoader(dataset=gallery_distractor,
                               batch_size=180,
                               num_workers=6,
                               pin_memory=True,
                               shuffle=False)
    
    # --------------------- LOAD TRAINED MODEL  ---------------------------------

    # 1. SRnet
    # Later on use a 5 layer CNN SRnet to replace bicubic
    SRnet = nn.Upsample(size=(224, 224), mode='bilinear').cuda()

    # 2. common feature extractor
    Resnet = resnet50('../models/resnet50_ft_weight.pkl', num_classes=8631).cuda()

    # 3. specific feature extractor
    from models.models.MLP import MLP
    MLP = MLP().cuda()

    if ckp_path is not None:
        ckp = torch.load(ckp_path)
        SRnet.load_state_dict((ckp['SRnet']))
        Resnet.load_state_dict(ckp['Resnet'])
        MLP.load_state_dict((ckp['MLP']))
        print(f'Successufully loaded ckp: {ckp_path}')
  
    if (torch.cuda.device_count() > 1):
        print(f'Using {torch.cuda.device_count()} GPUs')

        SRnet = nn.DataParallel(SRnet)
        Resnet = nn.DataParallel(Resnet)
        MLP = nn.DataParallel(MLP)

    SRnet.eval()
    Resnet.eval()
    MLP.eval()

    # ---------------------EXTRACT FEATURES -----------------------------

    # gallery
    for (img, img_size) in tqdm(gallery_db):
        with torch.no_grad():
            
            
            
            features = forward(img, img_size,
                                SRnet, 
                                Resnet,
                                MLP)
            
            re_features = features.reshape(-1, features.shape[1]).cpu()

            try:
                output_matrix
            except NameError:
                output_matrix = re_features
            else:
                output_matrix = torch.cat((output_matrix, re_features), axis=0)
    gallery_matrix = output_matrix.numpy()
    del output_matrix

    # probe
    for (img, img_size) in tqdm(probe_db):
        with torch.no_grad():

            features = forward(img, img_size,
                                SRnet, 
                                Resnet,
                                MLP)
            re_features = features.reshape(-1, features.shape[1]).cpu()

            try:
                output_matrix
            except NameError:
                output_matrix = re_features
            else:
                output_matrix = torch.cat((output_matrix, re_features), axis=0)
    probe_matrix = output_matrix.numpy()
    del output_matrix
    
    # tsne(np.concatenate((normalize(gallery_matrix, axis=1, norm='l2'), normalize(probe_matrix, axis=1, norm='l2')), axis=0),
    #       np.concatenate((gallery._id_list, probe._id_list), axis=0))

    # distractor
    for (img, img_size) in tqdm(distractor_db):
        with torch.no_grad():
            features = forward(img, img_size,
                                SRnet, 
                                Resnet,
                                MLP)
            re_features = features.reshape(-1, features.shape[1]).cpu()

            try:
                output_matrix
            except NameError:
                output_matrix = re_features
            else:
                output_matrix = torch.cat((output_matrix, re_features), axis=0)
    distractor_matrix = output_matrix.numpy()
    del output_matrix
    # ---------------------------------------------------------------------------
    
    # This is as required as Tinyface protocal
    gallery_matrix = normalize(gallery_matrix)
    probe_matrix = normalize(probe_matrix)
    distractor_matrix = normalize(distractor_matrix)
    # --------------------------------------------------------------------------
    gallery_match_img_ID_pairs_path = r'C:\Users\JACK\tinyface\Testing_Set\gallery_match_img_ID_pairs.mat'
    probe_img_ID_pairs_path = r'C:\Users\JACK\tinyface\Testing_Set\probe_img_ID_pairs.mat'
    
    ap, CMC = calculate_acc(gallery_match_img_ID_pairs_path, probe_img_ID_pairs_path, gallery_matrix, probe_matrix, distractor_matrix)
    
    print(f'mAP = {ap}, r1 precision = {CMC[0]}, r5 precision = {CMC[4]}, r10 precision = {CMC[9]}, r20 precision = {CMC[19]}')
    file.write(f'{ckp_path}\n')
    file.write(f'mAP = {ap}, r1 precision = {CMC[0]}, r5 precision = {CMC[4]}, r10 precision = {CMC[9]}, r20 precision = {CMC[19]}\n')
    file.close()
    
    
if __name__ == '__main__':
    ckps = [
        r'C:\Users\JACK\Desktop\recognize_tinyface_continuous_v2\train\ckps\batchsz144_LR5e-05_16_32_64_128\e1_iter0_train0.00_3e-05.pth'
        ]

    for ckp in ckps:
        evaluate(ckp)
    
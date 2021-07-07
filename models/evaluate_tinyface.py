# This file extract the features of test images.
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from scipy.io import loadmat
import torch
from torch import nn
torch.backends.cudnn.benchmark = True
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms
import PIL.Image
from sklearn.preprocessing import normalize
import numpy as np
from tqdm import tqdm

from evaluate_tinyface_utils import calculate_acc
from models.resnet import resnet50

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

        if self._mode == 'train':
            img = torchvision.transforms.Resize((256, 256))(img)
            img = torchvision.transforms.RandomCrop(224)(img)
            img = torchvision.transforms.RandomGrayscale(p=0.2)(img)
        else:
            img = torchvision.transforms.Resize((96, 96))(img)
        img = np.array(img, dtype=np.uint8)
        assert len(img.shape) == 3  # assumes color images and no alpha channel

        return self.transform(img)

    def transform(self, img):
        img = img[:, :, ::-1]  # H, W, C(RGB) -> H, W, C(BGR)
        img = img.astype(np.float32) # to numpy
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # H, W, C(BGR) -> C(BGR), H, W 
        img = torch.from_numpy(img).float() # to tensor
        return img

    def __len__(self):
        return len(self._img_list)


if __name__ == '__main__':
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
    model = resnet50(r'C:\Users\JACK\Desktop\recognize_tinyface_continuous\models\resnet50_ft_weight.pkl', num_classes=8631).cuda()
    # if test with trained ckp:
    # ckp = torch.load(r'C:\Users\JACK\Desktop\Resnet_SPP\train\ckps\batchsz144_LR0.005_140_160_180\e0_iter13074_train6.42_0.005.pth')
    # resnet_weight = ckp['Resnet']
    # model = resnet50(num_classes=8631).cuda()
    # model.load_state_dict(resnet_weight)
    

    # ---------------------EXTRACT FEATURES -----------------------------
    model.eval()

    # gallery
    for imgs in tqdm(gallery_db):
        with torch.no_grad():
            _, features = model(imgs.cuda())
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
    for imgs in tqdm(probe_db):
        with torch.no_grad():

            _, features = model(imgs.cuda())
            re_features = features.reshape(-1, features.shape[1]).cpu()

            try:
                output_matrix
            except NameError:
                output_matrix = re_features
            else:
                output_matrix = torch.cat((output_matrix, re_features), axis=0)
    probe_matrix = output_matrix.numpy()
    del output_matrix

    # distractor
    for imgs in tqdm(distractor_db):
        with torch.no_grad():
            _, features = model(imgs.cuda())
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

    gallery_matrix = normalize(gallery_matrix)
    probe_matrix = normalize(probe_matrix)
    distractor_matrix = normalize(distractor_matrix)
    
    # --------------------------------------------------------------------------
    gallery_match_img_ID_pairs_path = r'C:\Users\JACK\tinyface\Testing_Set\gallery_match_img_ID_pairs.mat'
    probe_img_ID_pairs_path = r'C:\Users\JACK\tinyface\Testing_Set\probe_img_ID_pairs.mat'
    
    ap, CMC = calculate_acc(gallery_match_img_ID_pairs_path, 
                            probe_img_ID_pairs_path, 
                            gallery_matrix,
                            probe_matrix,
                            distractor_matrix)
    
    print(f'mAP = {ap}, r1 precision = {CMC[0]}, r5 precision = {CMC[4]}, r10 precision = {CMC[9]}, r20 precision = {CMC[19]}')
    
    # print('Visualizing TSNE. Distractors are not included.')
    # tsne(np.concatenate((normalize(gallery_matrix, axis=1, norm='l2'), normalize(probe_matrix, axis=1, norm='l2')), axis=0),
    #       np.concatenate((gallery._id_list, probe._id_list), axis=0))
    
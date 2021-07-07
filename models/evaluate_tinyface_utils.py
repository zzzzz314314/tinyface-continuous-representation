from scipy.io import loadmat
import numpy as np
import torch
from joblib import Parallel, delayed
import multiprocessing
from scipy.spatial import distance

import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

def compute_AP(good_image, index):
    cmc = np.zeros((len(index), 1))
    ngood = len(good_image)
    
    old_recall = 0.
    old_precision = 1.
    ap = 0.
    intersect_size = 0.
    j = 0.
    good_now = 0.
    
    for n in range(len(index)):
        flag = 0
        if index[n] in good_image:
            cmc[n:] = 1
            flag = 1
            good_now = good_now + 1
        if flag == 1:
            intersect_size += 1
        recall = intersect_size / ngood
        precision = intersect_size / (j + 1)
        ap = ap + (recall - old_recall) * ((old_precision + precision) / 2)
        old_recall = recall
        old_precision = precision
        j += 1
        
        if good_now == ngood:
            break
    
    return ap, cmc.T


def calculate_acc(gallery_match_img_ID_pairs_path, probe_img_ID_pairs_path, gallery_feature_map, probe_feature_map, distractor_feature_map):
    
    # read in the ids
    gallery_ids = loadmat(gallery_match_img_ID_pairs_path)\
                            ['gallery_ids'] # (4443,1)
    probe_ids = loadmat(probe_img_ID_pairs_path)\
                            ['probe_ids'] # (3728,1)

    
    gallery_feature_map = np.concatenate((gallery_feature_map, distractor_feature_map), axis=0)
       
    # concat the gallery id with the distractor id
    distractor_ids = -100 * np.ones((distractor_feature_map.shape[0],1))
    gallery_ids = np.concatenate((gallery_ids, distractor_ids), axis=0)
    
    dist = torch.cdist(torch.from_numpy(gallery_feature_map), \
                        torch.from_numpy(probe_feature_map)).numpy() # (157871,3728) = (#gallery, #probe)
        
    CMC = np.zeros((probe_feature_map.shape[0], gallery_feature_map.shape[0])) # (3728,157871)
    ap = np.zeros((probe_feature_map.shape[0]))
    
    num_cores = multiprocessing.cpu_count()
        
    x = Parallel(n_jobs=num_cores)(delayed(compute_AP)(np.where(gallery_ids == probe_ids[p,0])[0], np.argsort(dist[:, p])) for p in range(probe_feature_map.shape[0]))
    for i, (_ap, _cmc) in enumerate(x):
        ap[i] = _ap
        CMC[i, :] = _cmc
        
    CMC = np.mean(CMC, axis=0)  
    
    return np.mean(ap), CMC

if __name__ == '__main__':
    
    gallery_match_img_ID_pairs_path = '../tinyface/Face_Identification_Evaluation/gallery_match_img_ID_pairs.mat'
    probe_img_ID_pairs_path = '../tinyface/Face_Identification_Evaluation/probe_img_ID_pairs.mat'
    features_npz_path = 'D:\\ACMlab\\LRFR\\ResnetSE_FT\\NoTune.npz'
    
    ap, CMC = calculate_acc(gallery_match_img_ID_pairs_path, probe_img_ID_pairs_path, features_npz_path)
    
    print(f'mAP = {ap}, r1 precision = {CMC[0]}, r5 precision = {CMC[4]}, r10 precision = {CMC[9]}, r20 precision = {CMC[19]}')
    
    
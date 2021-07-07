from scipy.io import loadmat
import numpy as np
import torch
from joblib import Parallel, delayed
import multiprocessing
from scipy.spatial import distance

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

def eval_cmc(probe_features_norm, probe_subj_ids, gallery_features_norm, gallery_subj_ids, mAP=False):
    # This code is from 
    # https://github.com/fvmassoli/cross-resolution-face-recognition/blob/master/evaluation_protocols/classifier_metrics/metrics.py
    """
    Eval the Cumulative Match Characteristics for the 1:N Face Identification protocol (close set scenario)
    :param probe_features_norm: normalized probe features
    :param probe_subj_ids: subject ids of probe features
    :param gallery_features_norm: normalized gallery features
    :param gallery_subj_ids: subject ids of gallery features
    :param mAP: boolean. If True skip CMC final evaluation
    :return: points: ranks at which the CMC has been evaluated, retrieval_rates: CMC for each rank value
    """

    if len(gallery_subj_ids.shape) == 1:
        gallery_subj_ids = gallery_subj_ids[:, np.newaxis]
    if len(probe_subj_ids.shape) == 1:
        probe_subj_ids = probe_subj_ids[:, np.newaxis]

    print('\t\tEvaluating distance matrix...')

    distance_matrix = np.dot(probe_features_norm, gallery_features_norm.T)
    ranking = distance_matrix.argsort(axis=1)[:, ::-1]
    ranked_scores = distance_matrix[np.arange(probe_features_norm.shape[0])[:, np.newaxis], ranking]
    print('\t\tDistance matrix evaluated!!!')

    gallery_ids_expanded = np.tile(gallery_subj_ids, probe_features_norm.shape[0]).T
    gallery_ids_ranked = gallery_ids_expanded[np.arange(probe_features_norm.shape[0])[:, np.newaxis], ranking]

    ranked_gt = (gallery_ids_ranked == probe_subj_ids).astype(np.int8)

    nb_points = 50
    points = np.arange(1, nb_points+1)
    retrieval_rates = np.empty(shape=(nb_points, 1))

    if not mAP:
        for k in points:
            retrieval_rates_ = ranked_gt[:, :k].sum(axis=1)
            retrieval_rates_[retrieval_rates_ > 1] = 1
            retrieval_rates[k - 1] = np.average(retrieval_rates_)

    return points, retrieval_rates, ranked_scores, ranked_gt


def calculate_acc(gallery_match_img_ID_pairs_path, probe_img_ID_pairs_path, gallery_feature_map,probe_feature_map,distractor_feature_map):
    
    # read in the ids
    gallery_ids = loadmat(gallery_match_img_ID_pairs_path)\
                            ['gallery_ids'] # (4443,1)
    probe_ids = loadmat(probe_img_ID_pairs_path)\
                            ['probe_ids'] # (3728,1)
                            
    # read in the features
    # features = np.load(features_npz_path)
    # gallery_feature_map = features['gallery_feature_map']
    # probe_feature_map = features['probe_feature_map']
    # distractor_feature_map = features['distractor_feature_map']
    
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
    
    
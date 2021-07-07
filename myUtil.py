import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torch.nn as nn
# from models.models.resnet import ResNet, Bottleneck
# from backbone.models.senet import SENet, Bottleneck
from torchsummary import summary
from collections import OrderedDict

def resnet50(weights_path=None, **kwargs):
    model = ResNet(Bottleneck, [3,4,6,3], **kwargs)
    if weights_path:
        import pickle
        with open(weights_path, 'rb') as f:
            obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        model.load_state_dict(weights)
    return model

def senet50(weights_path=None, **kwargs):
    model = SENet(Bottleneck, [3,4,6,3], **kwargs)
    if weights_path:
        import pickle
        with open(weights_path, 'rb') as f:
            obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        model.load_state_dict(weights)
    return model
    
def freeze_till_layer(net, fix_till_layer=None):
    cnt = 0
    for name, param in net.named_parameters():
        # print (f'{cnt}: {name}', end='')
        if fix_till_layer is not None:

            if cnt < fix_till_layer:
                 param.requires_grad = False
            # print(f'{cnt}: {name}', end='')
            # double check freezing weight and bias
        if param.requires_grad:
            print(f'{cnt}: {name}', end='')
            print(' ---- not freezing ----')
        elif not param.requires_grad:
            print(f'{cnt}: {name}', end='')
            print(' ---- freezing ----')
        cnt += 1
    return net


def freeze_bn(net):

    for name ,child in (net.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            for param in child.parameters():
                param.requires_grad = False
                # print(f'freeze {name}')
        else:
            for param in child.parameters():
                param.requires_grad = True

def freeze_all(net):
    for name, param in net.named_parameters():
        param.requires_grad = False
    for name, param in net.named_parameters():
        if not param.requires_grad:
            print(f'Freezing: {name}')

def unfreeze_but_BN(net):
    for name ,child in (net.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            for param in child.parameters():
                param.requires_grad = False
                # print(f'freeze {name}')
                print(f'But freezing: {name}')
        else:
            for param in child.parameters():
                param.requires_grad = True
                print(f'Opening: {name}')

def unfreeze_all(net):
    for name, param in net.module.named_parameters():
        param.requires_grad = True
    for name, param in net.module.named_parameters():
        if param.requires_grad:
            print(f'Opening: {name}')
            

def eval_for_frozen_BN(net, till_module_cnt):
    cnt = 0
    for name, module in net.named_modules():
        if isinstance(module, nn.BatchNorm2d) and cnt <= till_module_cnt:
            module.eval()
            print(cnt, name, '--- eval ---')
        elif isinstance(module, nn.BatchNorm2d) and cnt > till_module_cnt:
            module.train()
            print(cnt, name, '--- train ---')
        cnt += 1
        
def check_status_grad(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(f'{name}', end='')
            print(' ---- not freezing ----')
        elif not param.requires_grad:
            print(f'{name}', end='')
            print(' ---- freezing ----')
def check_status_mode(net):
    for name, module in net.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if module.training:
                print(name, '--- train ---')
            else:
                print(name, '--- eval ---')
                
def filter_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict

def tsne (matrix, label):
    print(matrix.shape, label.shape)
    feature_matrix_embed = TSNE(n_components=2).fit_transform(matrix)
    fig, ax = plt.subplots()
    colors = cm.rainbow(np.linspace(0, 1, 50))#len(np.unique(label))))

    for c, color in zip(np.unique(label), colors):

        index = np.where(label == c)
        print(f'identity: {c}, {index[0]}')
        ax.scatter(feature_matrix_embed[index, 0], feature_matrix_embed[index, 1], color=color)
        for i in index[0]:
            # print(feature_matrix_embed[i, 0])
            ax.annotate(str(c), (feature_matrix_embed[i, 0], feature_matrix_embed[i, 1]))
        if c == 49:
            break
    plt.show()

if __name__ == '__main__':

    # model = resnet50('./resnet50_ft_weight.pkl', num_classes=8631)
    model = resnet50('models/resnet50_ft_weight.pkl', num_classes=8631).cuda()
    # summary(model, (3,224,224))
    # model = nn.DataParallel(model)
    freeze_till_layer(model, fix_till_layer=129)
    # eval_for_frozen_BN(model, 68)
    

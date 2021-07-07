import torch
import torch.nn.functional as F
from torch.autograd import Variable

def forward(img, img_size,
            SRnet, 
            Resnet,
            MLP):

    img = img.cuda() # img size: (224, 224)


    """ SR """
    # SR done in dataloader.
    
    size = img_size.unsqueeze(1).float().cuda()
    target_size = Variable((torch.ones(len(img), 1)*16).cuda(), requires_grad=False)
    
    # print(size.shape, size64.shape)

    _, f = Resnet(img) # discard original classifier
    fp = MLP(torch.cat([f, target_size/size], 1))

    return fp
    # return f

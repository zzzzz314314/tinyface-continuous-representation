import torch
from torch.autograd import Variable

def forward(img16, img32, img64, img128, label,
            SRnet,
            Resnet,
            MLP,
            Classifier,
            CEloss,
            MSEloss,
            optimizer):

    img16, img32, img64, img128, label = img16.cuda(), img32.cuda(), img64.cuda(), img128.cuda(), label.cuda()
    
    optimizer.zero_grad()

    """ SR """
    img16 = SRnet(img16)
    img32 = SRnet(img32)
    img64 = SRnet(img64)
    img128 = SRnet(img128)
    
    size16 = Variable((torch.ones(len(label), 1)*16).cuda(), requires_grad=False)
    size32 = Variable((torch.ones(len(label), 1)*32).cuda(), requires_grad=False)
    size64 = Variable((torch.ones(len(label), 1)*64).cuda(), requires_grad=False)
    size128 = Variable((torch.ones(len(label), 1)*128).cuda(), requires_grad=False)

    """ feature extraction """
    _, f16 = Resnet(img16) # discard original classifier
    _, f32 = Resnet(img32) # discard original classifier
    _, f64 = Resnet(img64) # discard original classifier
    _, f128 = Resnet(img128) # discard original classifier
    
    """ feature synthesize from 16 to 16, 32, 64 """

    fp16 = MLP(torch.cat([f16, size16 / size16], 1))
    fp32 = MLP(torch.cat([f16, size32 / size16], 1))
    fp64 = MLP(torch.cat([f16, size64 / size16], 1))
    fp128 = MLP(torch.cat([f16, size128 / size16], 1))
    
    pred16 = Classifier(fp16)
    pred32 = Classifier(fp32)
    pred64 = Classifier(fp64)
    pred128 = Classifier(fp128)
    
    acc16_16 = torch.mean((torch.argmax(pred16, axis=1) == label).float())
    acc16_32 = torch.mean((torch.argmax(pred32, axis=1) == label).float())
    acc16_64 = torch.mean((torch.argmax(pred64, axis=1) == label).float())
    acc16_128 = torch.mean((torch.argmax(pred128, axis=1) == label).float())
    
    mseloss16 = MSEloss(fp16, f16.detach()) + MSEloss(fp32, f32.detach()) + MSEloss(fp64, f64.detach()) + MSEloss(fp128, f128.detach())
    celoss16 = CEloss(pred16, label) + CEloss(pred32, label) + CEloss(pred64, label) + CEloss(pred128, label)
    loss16 = mseloss16 + celoss16
    loss16.backward()
    
    """ feature synthesize from 32 to 16, 32, 64 """
    fp16 = MLP(torch.cat([f32, size16 / size32], 1))
    fp32 = MLP(torch.cat([f32, size32 / size32], 1))
    fp64 = MLP(torch.cat([f32, size64 / size32], 1))
    fp128 = MLP(torch.cat([f32, size128 / size32], 1))
    
    pred16 = Classifier(fp16)
    pred32 = Classifier(fp32)
    pred64 = Classifier(fp64)
    pred128 = Classifier(fp128)
    
    acc32_16 = torch.mean((torch.argmax(pred16, axis=1) == label).float())
    acc32_32 = torch.mean((torch.argmax(pred32, axis=1) == label).float())
    acc32_64 = torch.mean((torch.argmax(pred64, axis=1) == label).float())
    acc32_128 = torch.mean((torch.argmax(pred128, axis=1) == label).float())
    
    mseloss32 = MSEloss(fp16, f16.detach()) + MSEloss(fp32, f32.detach()) + MSEloss(fp64, f64.detach()) + MSEloss(fp128, f128.detach())
    celoss32 = CEloss(pred16, label) + CEloss(pred32, label) + CEloss(pred64, label) + CEloss(pred128, label)
    loss32 = mseloss32 + celoss32
    loss32.backward()
    
    """ feature synthesize from 64 to 16, 32, 64 """
    fp16 = MLP(torch.cat([f64, size16 / size64], 1))
    fp32 = MLP(torch.cat([f64, size32 / size64], 1))
    fp64 = MLP(torch.cat([f64, size64 / size64], 1))
    fp128 = MLP(torch.cat([f64, size128 / size64], 1))
    
    pred16 = Classifier(fp16)
    pred32 = Classifier(fp32)
    pred64 = Classifier(fp64)
    pred128 = Classifier(fp128)
    
    acc64_16 = torch.mean((torch.argmax(pred16, axis=1) == label).float())
    acc64_32 = torch.mean((torch.argmax(pred32, axis=1) == label).float())
    acc64_64 = torch.mean((torch.argmax(pred64, axis=1) == label).float())
    acc64_128 = torch.mean((torch.argmax(pred128, axis=1) == label).float())
    
    mseloss64 = MSEloss(fp16, f16.detach()) + MSEloss(fp32, f32.detach()) + MSEloss(fp64, f64.detach()) + MSEloss(fp128, f128.detach())
    celoss64 = CEloss(pred16, label) + CEloss(pred32, label) + CEloss(pred64, label) + CEloss(pred128, label)
    loss64 = mseloss64 + celoss64
    loss64.backward()
    
    """ feature synthesize from 128 to 16, 32, 64, 128 """
    fp16 = MLP(torch.cat([f128, size16 / size128], 1))
    fp32 = MLP(torch.cat([f128, size32 / size128], 1))
    fp64 = MLP(torch.cat([f128, size64 / size128], 1))
    fp128 = MLP(torch.cat([f128, size128 / size128], 1))
    
    pred16 = Classifier(fp16)
    pred32 = Classifier(fp32)
    pred64 = Classifier(fp64)
    pred128 = Classifier(fp128)
    
    acc128_16 = torch.mean((torch.argmax(pred16, axis=1) == label).float())
    acc128_32 = torch.mean((torch.argmax(pred32, axis=1) == label).float())
    acc128_64 = torch.mean((torch.argmax(pred64, axis=1) == label).float())
    acc128_128 = torch.mean((torch.argmax(pred128, axis=1) == label).float())
    
    mseloss128 = MSEloss(fp16, f16.detach()) + MSEloss(fp32, f32.detach()) + MSEloss(fp64, f64.detach()) + MSEloss(fp128, f128.detach())
    celoss128 = CEloss(pred16, label) + CEloss(pred32, label) + CEloss(pred64, label) + CEloss(pred128, label)
    loss128 = mseloss128 + celoss128
    loss128.backward()
    
    optimizer.step()

    return mseloss16.item(), celoss16.item(), loss16.item(), \
           mseloss32.item(), celoss32.item(), loss32.item(), \
           mseloss64.item(), celoss64.item(), loss64.item(), \
           mseloss128.item(), celoss128.item(), loss128.item(), \
           acc16_16.item(), acc16_32.item(), acc16_64.item(), acc16_128.item(),\
           acc32_16.item(), acc32_32.item(), acc32_64.item(), acc32_128.item(),\
           acc64_16.item(), acc64_32.item(), acc64_64.item(), acc64_128.item(),\
           acc128_16.item(), acc128_32.item(), acc128_64.item(), acc128_128.item()

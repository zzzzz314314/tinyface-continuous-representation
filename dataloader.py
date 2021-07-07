import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
torch.backends.cudnn.benchmark = True
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
import torchvision.transforms
import PIL.Image
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')


class Vggface2(Dataset):
    # The Vggface2 dataset is the loose cropped version
    # The original Vggface2 is trained on a tightly cropped version
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt

    def __init__(self, root, train, visualize=False):
        self._root = root
        self._train = train
        self._visualize = visualize
        self._classes, self._class_to_idx = self._find_classes()
        self._samples = self._make_dataset()
        print(f'total num of classes: {len(self._classes)}')
        # print(self._samples[:300])


    def _find_classes(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self._root) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self._root) if os.path.isdir(os.path.join(self._root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self):
        images = []
        dir = os.path.expanduser(self._root)
        progress_bar = tqdm(
                        sorted(self._class_to_idx.keys()),
                        desc='Making data training set' if self._train else 'Making data validation set',
                        total=len(self._class_to_idx.keys()),
                        leave=False
                    )
        for target in progress_bar:
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self._class_to_idx[target])
                    images.append(item)
            progress_bar.update(n=1)
        progress_bar.close()
        return images

    def _transform(self, img, r1, r2, r3, r4):
        def transform2(img):
            img = img[:, :, ::-1]  # H, W, C(RGB) -> H, W, C(BGR)
            img = img.astype(np.float32)
            if not self._visualize:
                img -= self.mean_bgr
            img = img.transpose(2, 0, 1)  # H, W, C(BGR) -> C(BGR), H, W
            img = torch.from_numpy(img).float()
            return img


        # img = torchvision.transforms.GaussianBlur(kernel_size=7)(img)
        if self._train:
            if not self._visualize:
                img = torchvision.transforms.RandomGrayscale(p=0.2)(img)
                img = torchvision.transforms.RandomHorizontalFlip(p=0.5)(img)
                # img = torchvision.transforms.RandomVerticalFlip(p=0.2)(img)
                img = torchvision.transforms.RandomRotation(degrees=15)(img)

        else:
            pass

        img16 = torchvision.transforms.Resize((r1, r1))(img)
        img32 = torchvision.transforms.Resize((r2, r2))(img)
        img64 = torchvision.transforms.Resize((r3, r3))(img)
        img128 = torchvision.transforms.Resize((r4, r4))(img)

        img16 = np.array(img16, dtype=np.uint8)
        img32 = np.array(img32, dtype=np.uint8)
        img64 = np.array(img64, dtype=np.uint8)
        img128 = np.array(img128, dtype=np.uint8)
        assert len(img16.shape) == 3  # assumes color images and no alpha channel

        img16 = transform2(img16)
        img32 = transform2(img32)
        img64 = transform2(img64)
        img128 = transform2(img128)

        return img16, img32, img64, img128



    def __getitem__(self, index):
        img_path, label = self._samples[index]
        img = PIL.Image.open(img_path)

        # img = self.gaussian_blur(img, kernel_size)
        img16, img32, img64, img128 = self._transform(img, 16, 32, 64, 128)

        return img16, img32, img64, img128, label

    def __len__(self):
        return len(self._samples)

def tensor_CHWbgr_2_np_HWCrgb(img):
    CHWbgr = img.int().numpy()
    HWCbgr = CHWbgr.transpose(1, 2, 0)
    HWCrgb = HWCbgr[:, :, ::-1]
    return HWCrgb

if __name__ == '__main__':
    img_root = r'C:\Users\JACK\Vggface2\splitted\train'
    vggface = Vggface2(root=img_root, train=True, visualize=True)
    vggface_loader = DataLoader(dataset=vggface,
                                 batch_size=1,
                                 num_workers=0,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)

    for b, (img16, img32, img64, img128, label) in enumerate(vggface_loader):

        print(f'b:{b} | img16:{img16.shape} | img32:{img32.shape} | img64:{img64.shape} | img128:{img128.shape} |label: {label}')

        test = [tensor_CHWbgr_2_np_HWCrgb(img16[0]),
                tensor_CHWbgr_2_np_HWCrgb(img32[0]),
                tensor_CHWbgr_2_np_HWCrgb(img64[0]),
                tensor_CHWbgr_2_np_HWCrgb(img128[0])]
        plt.subplot(4,1,1);plt.imshow(test[0]);plt.axis('off')
        plt.subplot(4,1,2);plt.imshow(test[1]);plt.axis('off')
        plt.subplot(4,1,3);plt.imshow(test[2]);plt.axis('off')
        plt.subplot(4,1,4);plt.imshow(test[3]);plt.axis('off')
        # test = [tensor_CHWbgr_2_np_HWCrgb(img16[1]),
        #         tensor_CHWbgr_2_np_HWCrgb(img35[1]),
        #         tensor_CHWbgr_2_np_HWCrgb(img50[1])]
        # plt.subplot(3,3,4);plt.imshow(test[0])
        # plt.subplot(3,3,5);plt.imshow(test[1])
        # plt.subplot(3,3,6);plt.imshow(test[2])
        # test = [tensor_CHWbgr_2_np_HWCrgb(img16[2]),
        #         tensor_CHWbgr_2_np_HWCrgb(img35[2]),
        #         tensor_CHWbgr_2_np_HWCrgb(img50[2])]
        # plt.subplot(3,3,7);plt.imshow(test[0])
        # plt.subplot(3,3,8);plt.imshow(test[1])
        # plt.subplot(3,3,9);plt.imshow(test[2])

        break
    plt.show()

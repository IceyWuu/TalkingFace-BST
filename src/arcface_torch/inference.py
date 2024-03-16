import argparse

import cv2
import numpy as np
import torch

from backbones import get_model


@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img).numpy()
    
    return feat
    #print(feat)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network') # r50
    parser.add_argument('--weight', type=str, default='./checkpoints/glint360k_r100.pth')
    parser.add_argument('--img', type=str, default='0.png')# default=None)
    args = parser.parse_args()
    #inference(args.weight, args.network, args.img)

    
    x1 = inference('./checkpoints/glint360k_r100.pth','r100','image_for_test/1.png')
    x2 = inference('./checkpoints/glint360k_r100.pth','r100','image_for_test/2.png').T
    
    re = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    print(re)


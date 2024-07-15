import torch
import os
from datetime import datetime
from torch.nn import functional as F
from net.Network import build_PANet
from torch.utils.data import DataLoader
from data_loader.FLOW import DUTLF_V2
import cv2

#load data and model
load_root = 'pretrain/pvt_v2_b2.pth'
data_root = ''
# type = [test_DUTV2, test_HFUT, test_Lytro]
test_dataloader = DataLoader(DUTLF_V2(data_root, type=''), batch_size=1, shuffle=False)


# build model
my_model = build_PANet()

def test(model, my_model, dataset, save_sal):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    my_model = my_model.to(device)
    my_model.load_state_dict(torch.load(model))

    my_model.eval()

    if not os.path.exists(save_sal):
        os.makedirs(save_sal)

    with torch.no_grad(): 
        prev_time = datetime.now()
        print(prev_time)
       
        for i, (up_, left_, cvi_, names) in enumerate(dataset):       
            up, left, cvi = up_.to(device), left_.to(device), cvi_.to(device) 
            pre_rgb, pre_u, pre_v = my_model(up, left, cvi) 
            name = names[0]
            print(name)

            res = pre_rgb.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = res * 255
            cv2.imwrite(save_sal + name.split('.')[0]+'.png', res)
        cur_time = datetime.now()
        print(cur_time)

if __name__ == '__main__':
    save_sal = ''
    test(load_root, my_model, test_dataloader, save_sal)
        
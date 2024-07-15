import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
import os
import os.path as osp
import cv2
import random

def randomFlip(image, up, left, gt):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        up = up.transpose(Image.FLIP_LEFT_RIGHT)
        left = left.transpose(Image.FLIP_LEFT_RIGHT)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)       
    return image, up, left, gt 

def randomCrop(image, up, left, gt):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    
    return image.crop(random_region), up.crop(random_region), left.crop(random_region), gt.crop(random_region)

def randomRotation(image, up, left, gt):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        up = up.rotate(random_angle, mode)
        left = left.rotate(random_angle, mode)
        gt = gt.rotate(random_angle, mode)
    return image, up, left, gt 

def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomPeper(img):
    #img=np.asarray(img)
    img=np.array(img)  #python 3.8要求
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):
        randX=random.randint(0,img.shape[0]-1)
        randY=random.randint(0,img.shape[1]-1)
        if random.randint(0,1)==0:
            img[randX,randY]=0
        else:
            img[randX,randY]=255 
    return Image.fromarray(img) 

class DUTLF_V2(Dataset):
    def __init__(self, root, type='train'):
        self.type = type
        self.jpgnames = None
        self.pngnames = None
        
        if type == 'train':
            self.updir = osp.join(root, '')
            self.leftdir = osp.join(root, '')
            self.cvidir = osp.join(root, '')
            self.maskdir = osp.join(root, '')
    
        elif type == 'test':
            self.updir = osp.join(root, '')
            self.leftdir = osp.join(root, '')
            self.cvidir = osp.join(root, '')

        elif type == 'test_Lytro':
            self.updir = ''
            self.leftdir = ''
            self.cvidir = ''

        elif type == 'test_HFUT':
            self.updir = ''
            self.leftdir = ''
            self.cvidir = ''   

        else:
            print('Wrong Dataset Type! Please Check!!')

        self.jpgnames = sorted(os.listdir(self.cvidir)) 

    def __len__(self):
        return len(self.jpgnames)

    def __getitem__(self, item):
        jpg_name = self.jpgnames[item]
        png_name = self.jpgnames[item].split('.')[0] + '.png'
        
        if self.type == 'train':
            up = Image.open(self.updir  + jpg_name.split('.')[0] + '_up.jpg')
            up = up.convert('RGB')
            up = up.resize((256, 256))
            left = Image.open(self.leftdir  + jpg_name.split('.')[0] + '_left.jpg')
            left = left.convert('RGB')
            left = left.resize((256, 256))
            cvi = Image.open(self.cvidir + jpg_name)
            cvi = cvi.convert('RGB')
            cvi = cvi.resize((256, 256))
            mask = Image.open(self.maskdir + png_name)
            mask = mask.convert('L')
            mask = mask.resize((256, 256))

            cvi, up, left, mask = randomFlip(cvi, up, left, mask)
            cvi, up, left, mask = randomCrop(cvi, up, left, mask)
            cvi, up, left, mask = randomRotation(cvi, up, left, mask)
            cvi = colorEnhance(cvi)
            mask = randomPeper(mask)
        
            self.up_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.left_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.cvi_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.mask_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()])
            sample = self.transform_train(up, left, cvi, mask, png_name)

        elif self.type == 'test':
            up = Image.open(self.updir  + jpg_name.split('.')[0] + '_up.jpg')
            up = up.convert('RGB')
            left = Image.open(self.leftdir  + jpg_name.split('.')[0] + '_left.jpg')
            left = left.convert('RGB')
            cvi = Image.open(self.cvidir + jpg_name)
            cvi = cvi.convert('RGB')

            self.up_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.left_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.cvi_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            sample = self.transform_test(up, left, cvi, png_name)

        elif self.type == 'test_Lytro':
            up = Image.open(self.updir  + jpg_name.split('.')[0] + '_up.jpg')
            up = up.convert('RGB')
            left = Image.open(self.leftdir  + jpg_name.split('.')[0] + '_left.jpg')
            left = left.convert('RGB')
            cvi = Image.open(self.cvidir + jpg_name)
            cvi = cvi.convert('RGB')

            self.up_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.left_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.cvi_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            sample = self.transform_test(up, left, cvi, png_name)

        elif self.type == 'test_HFUT':
            up = Image.open(self.updir  + jpg_name.split('.')[0] + '_up.jpg')
            up = up.convert('RGB')
            left = Image.open(self.leftdir  + jpg_name.split('.')[0] + '_left.jpg')
            left = left.convert('RGB')
            cvi = Image.open(self.cvidir + jpg_name)
            cvi = cvi.convert('RGB')

            self.up_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.left_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.cvi_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            sample = self.transform_test(up, left, cvi, png_name)
        
        return sample

    def transform_train(self, up, left, cvi, mask, name):
        up = self.up_transform(up)
        left = self.left_transform(left)
        cvi = self.cvi_transform(cvi)
        mask = self.mask_transform(mask)
       
        return up, left, cvi, mask, name

    def transform_test(self, up, left, cvi, name):
        up = self.up_transform(up)
        left = self.left_transform(left)
        cvi = self.cvi_transform(cvi)  
        
        return up, left, cvi, name




#Algumas classes que eu usei na minha IC

import random
import torch
import skimage
from PIL import Image
from torchvision import transforms
import numpy as np

#Contains classes for transform

class Resize(object):
    #Resize a PIL image to the desired size
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = transforms.functional.resize(image, (self.h, self.w))

        return {"image": image, "label": label}


class ToTensor(object):
    #Convert a PIL image to a tensor
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = transforms.functional.to_tensor(image)
        return {"image": image, "label": label}
        

class HorizontalFlip(object):
    def __call__(self, sample):
        transform = transforms.RandomHorizontalFlip()
        sample["image"] = transform(sample["image"])
        return sample
        

class VerticalFlip(object):
    def __call__(self, sample):
        transform = transforms.RandomVerticalFlip()
        sample["image"] = transform(sample["image"])
        return sample


class RandomCrop(object):
    
    def __init__(self, h, w):
        self.h = h
        self.w = w
    
    def __call__(self, sample):
        transform = transforms.RandomCrop(self.h, self.w)
        sample["image"] = transform(sample["image"])
        return sample
        

class RandomRotation(object):
    
    def __init__(self, angle):
        self.angle = angle
        
    def __call__(self, sample):
        transform = transforms.RandomRotation(self.angle)
        sample["image"] = transform(sample["image"])
        return sample
        

class GaussianNoise(object):
    def __init__(self, var):
        self.var = var
    
    def __call__(self, sample):
        var = random.uniform(0, self.var)
        
        arrImg = transforms.functional.to_tensor(sample)
        arrImg = arrImg.permute(1, 2, 0).numpy()
        arrImg = skimage.util.random_noise(arrImg, mode='gaussian', var=var)
        arrImg = (arrImg * 255).astype('uint8')
        arrImg = transforms.functional.to_pil_image(arrImg)
        return arrImg
        
        
        
class RandomPerspective(object):
    
    def __call__(self, sample):
        transform = transforms.RandomPerspective(distortion_scale=0.2)
        sample["image"] = transform(sample["image"])
        return sample
        
        
class CenterCrop(object):
    
    def __init__(self, H, W):
        self.H = H
        self.W = W
        
    def __call__(self, sample):
        transform = transforms.CenterCrop((self.H, self.W))
        sample["image"] = transform(sample["image"])
        return sample
        
        
class ColorJitter(object):
    
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
    def __call__(self, sample):
        transform = transforms.ColorJitter(
            brightness=self.brightness, 
            contrast=self.contrast, 
            saturation=self.saturation, 
            hue=self.hue
        )
        
        sample["image"] = transform(sample["image"])
        return sample
        
        
        
class Normalize(object):
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, sample):
        transform = transforms.Normalize(mean=self.mean, std=self.std)
                                         
        sample["image"] = transform(sample["image"])
        return sample

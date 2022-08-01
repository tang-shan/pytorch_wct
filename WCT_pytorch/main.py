import torch
from torch import nn
from PIL import Image
import cfg
from torchvision import transforms
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

#utils

#load pil image
def load_pil(path):
    return Image.open(path).convert('RGB')

#resize image size, Keep the width and height the same
def resize_pil(img,size):
    w,h=img.size
    if h>w:
        img=img.resize((size,size*h//w))
    else:
        img=img.resize((size*w//h,size))
    img=img.crop((0,0,size,size))
    return img

#convert pil to tensor
def pil_to_tensor(img):
    transformer=transforms.ToTensor()
    img=transformer(img)
    return img.unsqueeze(0)

#convert tensor to numpy (0-255)
def tensor_to_imgnp(tensor):
    tensor=tensor.squeeze()
    tensor=tensor.permute(1,2,0)
    tensor=tensor-tensor.min()
    tensor=tensor/tensor.max()
    tensor=tensor*255.
    np_array=tensor.detach().numpy()
    np_array=np_array.clip(0,255)
    np_array=np_array.astype(np.uint8)
    return np_array

#load image as tensor
def load_tensor_img(path,size):
    pil=load_pil(path)
    pil=resize_pil(pil,size)
    img=pil_to_tensor(pil)
    return img

#model
#encoder part of network
class Encoder(nn.Module):
    def __init__(self,pretrained_path='vgg_normalised_conv5_1.pth'):
        super(Encoder,self).__init__()
        self.net=nn.Sequential(
            #encoder1
            nn.Conv2d(3, 3, 1),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            #encoder2
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2,ceil_mode=True),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            #encoder3
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2,ceil_mode=True),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            #encoder4
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2,ceil_mode=True),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            #encoder5
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.MaxPool2d(2,ceil_mode=True),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(512, 512, 3),
            nn.ReLU()
            )
        self.net.load_state_dict(torch.load(os.path.join(cfg.ckpt_dir, pretrained_path), map_location=lambda storage, loc: storage))
    
    def forward(self,x,target):
        if target==1:
            return self.net[:4](x)
        elif target==2:
            return self.net[:11](x)
        elif target==3:
            return self.net[:18](x)
        elif target==4:
            return self.net[:31](x)
        elif target==5:
            return self.net(x)
      
#decoder part of network
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.net=nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, 3))
        self.decoder1=nn.Sequential(*copy.deepcopy(list(self.net.children())[-2:]))
        self.decoder1.load_state_dict(torch.load(os.path.join(cfg.ckpt_dir, "decoder_relu1_1.pth")))
        self.decoder2=nn.Sequential(*copy.deepcopy(list(self.net.children())[-9:]))
        self.decoder2.load_state_dict(torch.load(os.path.join(cfg.ckpt_dir, "decoder_relu2_1.pth")))
        self.decoder3=nn.Sequential(*copy.deepcopy(list(self.net.children())[-16:]))
        self.decoder3.load_state_dict(torch.load(os.path.join(cfg.ckpt_dir, "decoder_relu3_1.pth")))
        self.decoder4=nn.Sequential(*copy.deepcopy(list(self.net.children())[-29:]))
        self.decoder4.load_state_dict(torch.load(os.path.join(cfg.ckpt_dir, "decoder_relu4_1.pth")))
        self.decoder5=nn.Sequential(*copy.deepcopy(list(self.net.children())))
        self.decoder5.load_state_dict(torch.load(os.path.join(cfg.ckpt_dir, "decoder_relu5_1.pth")))
    def forward(self,x,target):
        if target==1:
            return self.decoder1(x)
        elif target==2:
            return self.decoder2(x)
        elif target==3:
            return self.decoder3(x)
        elif target==4:
            return self.decoder4(x)
        elif target==5:
            return self.decoder5(x)

#wct 
#feature transform module
def whiten_and_color(content_feature,style_feature,alpha=1):
    cf = content_feature.squeeze(0)
    c, ch, cw = cf.shape
    cf = cf.reshape(c, -1)
    c_mean = torch.mean(cf, 1, keepdim=True)
    cf = cf - c_mean
    c_cov = torch.mm(cf, cf.t()).div(ch*cw - 1)
    c_u, c_e, c_v = torch.svd(c_cov)
    
    c_d=c_e.pow(-0.5)
    
    whiten=torch.mm(c_u,torch.diag(c_d))
    whiten=torch.mm(whiten,c_v.T)
    whiten=torch.mm(whiten,cf)
    
    sf = style_feature.squeeze(0)
    c, sh, sw = sf.shape
    sf = sf.reshape(c, -1)
    s_mean = torch.mean(sf, 1, keepdim=True)
    sf = sf - s_mean
    s_cov = torch.mm(sf, sf.t()).div(sh*sw - 1)
    s_u, s_e, s_v = torch.svd(s_cov)
    
    s_d=s_e.pow(0.5)
    
    colored=torch.mm(s_u,torch.diag(s_d))
    colored=torch.mm(colored,s_v.T)
    colored=torch.mm(colored,whiten)+s_mean
    colored=colored.reshape(c, ch, cw)
    return alpha*colored+(1-alpha)*content_feature

#main network
class WCT(nn.Module):
    def __init__(self):
        super(WCT,self).__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()
        self.decoder5=self.decoder.decoder5
        self.decoder4=self.decoder.decoder4
        self.decoder3=self.decoder.decoder3
        self.decoder2=self.decoder.decoder2
        self.decoder1=self.decoder.decoder1
    def transform(self,content_img,style_img,alpha,level):
        content_feature=self.encoder(content_img,level)
        style_feature=self.encoder(style_img,level)
        transformed_feature=whiten_and_color(content_feature, style_feature,alpha)
        return getattr(self, f'decoder{level}')(transformed_feature)
    
    def forward(self,content_img,style_img,alpha=1):
        r5=self.transform(content_img, style_img, alpha, 5)
        r4=self.transform(r5, style_img, alpha, 4)
        r3 = self.transform(r4, style_img, alpha, 3)
        r2 = self.transform(r3, style_img, alpha, 2)
        r1 = self.transform(r2, style_img, alpha, 1)
        return r1

if __name__ == '__main__':
    content_img=load_tensor_img(cfg.content_img_path, cfg.img_size)
    style_img=load_tensor_img(cfg.style_img_path, cfg.img_size)
    wct=WCT()
    construct_img=wct(content_img,style_img)
    construct_img=tensor_to_imgnp(construct_img)
    plt.imshow(construct_img)
    construct_img=Image.fromarray(construct_img)
    construct_img.save(cfg.out_path)
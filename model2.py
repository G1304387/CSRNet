import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url
import re
from collections import OrderedDict
from utils import save_net,load_net,make_layers
from itertools import islice 
model_url = "https://download.pytorch.org/models/vgg16-397923af.pth"

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            self._initialize_weights()
            mod =  torch.hub.load_state_dict_from_url(model_url)
        
            print ("lengt front end",len(self.frontend.state_dict().items()))
            print("length model vgg",len(mod.items())) 
                        
            i = 1
            for k, v in mod.items():
                print ("type k",type(k))
                print ("type v",type(v))
                print ("k=", k)
                #k = re.sub('group\d+\.', '', k)
                k = re.sub('(features.|classfier.)', '', k)
                print ("k.sub =",k)
                v = v.data
                print ("type v.data",type(v))
               
                if i in range(len(self.frontend.state_dict().items())):

                    print ("tensor v",v)
                    print ("shape tensor v",v.shape)
                    print ("shape front end",self.frontend.state_dict()[k].shape)
                    self.frontend.state_dict()[k].copy_(v) 
                    print ("tensor %d",k,self.frontend.state_dict()[k])
                    
                i+=1
                    
            print ("lengt front end",len(self.frontend.state_dict().items()))
            print("model items",self.frontend.state_dict().items())
           
    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    #def make_layers(self, cfg, in_channels = 3,batch_norm=False,dilation = False):
    #    if dilation:
    #        d_rate = 2
    #    else:
    #        d_rate = 1
    #    layers = []
    #    for v in cfg:
    #        print('v cfg', v)
    #        if v == 'M':
    #            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    #        else:
    #            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
    #            if batch_norm:
    #                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
     #           else:
    #                layers += [conv2d, nn.ReLU(inplace=True)]
    #            in_channels = v
     #   return nn.Sequential(*layers)
        
import torch
import torch.nn as nn
from collections import OrderedDict
from .MaskFlownet import *
import yaml

class Reader:
    def __init__(self, obj, full_attr=""):
        self._object = obj
        self._full_attr = full_attr
    
    def __getattr__(self, name):
        if self._object is None:
            ret = None
        else:
            ret = self._object.get(name, None)
        return Reader(ret, self._full_attr + '.' + name)

    def get(self, default=None):
        if self._object is None:
            print('Default FLAGS.{} to {}'.format(self._full_attr, default))
            return default
        else:
            return self._object

    @property
    def value(self):
        return self._object

class FlowNetS(nn.Module):
    def __init__(self, args):
        super(FlowNetS,self).__init__()
       
        with open(args.pretrained_flownet_path + "/kitti.yaml") as f:
            config = Reader(yaml.safe_load(f))
            
        self.maskflownet = MaskFlownet(config)

    def forward(self, input):

        feature, flow = self.maskflownet(input)

        return feature, flow

if __name__ == '__main__':
    dummy_input = torch.randn(8, 6, 384, 640)
    encoder = FlowNetS()
    feature = encoder(dummy_input)
    print(feature.shape)
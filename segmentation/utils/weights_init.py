from torch.nn import init
from torch import nn

def init_model(net):
    if isinstance(net, nn.Conv3d) or isinstance(net, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(net.weight.data, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        if net.bias is not None:
            net.bias = nn.init.constant_(net.bias, 0)
        #nn.init.constant_(net.bias.data, 0)
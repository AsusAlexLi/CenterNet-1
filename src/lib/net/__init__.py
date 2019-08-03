from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from net.CenterNet import CenterNet
NETS = {'CenterNet':CenterNet}

def get_models(name, **kwargs):
    print(NETS)
    return NETS[name](**kwargs)
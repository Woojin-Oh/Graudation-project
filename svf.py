import copy
import inspect
from math import floor

#import paddle
#import paddle.nn as nn
#import paddle.vision.models as models

from torch import nn
import torch

import torchvision.models as models

from models import c3d, squeezenet, mobilenet, shufflenet, mobilenetv2, shufflenetv2, resnext, resnet, resnetl
from models.resnetl import get_fine_tuning_parameters


#Conv3d(4, 16, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)

'''
def d_nsvd(matrix, rank=1):
    U, S, V = paddle.linalg.svd(matrix)
    S = S[:rank]
    U = U[:, :rank]  # * S.view(1, -1)
    V = V[:, :rank]  # * S.view(1, -1)
    V = paddle.transpose(V, perm=[1, 0])
    return U, S, V
    '''
def d_nsvd(matrix, rank=1):
    #max_value = torch.max(matrix)
    #print('max_value of matrix: ', max_value)
    U,S,V = torch.linalg.svd(matrix)
    #max_value_S = torch.max(S)
    #print('max value of S: ', max_value_S)
    S=S[:rank]
    U=U[:,:rank] 
    V=V[:,:rank]
    return (U,S,V.t())


#class SVD_Conv2d(nn.Layer):
class SVD_Conv2d(nn.Module):
    """Kernel Number first SVD Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias,
                 padding_mode='zeros', device=None, dtype=None,
                 rank=1):
        super(SVD_Conv2d, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.conv_U = nn.Conv2d(rank, out_channels, (1, 1), (1, 1), 0, (1, 1), 1, bias)
        self.conv_V = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, dilation, groups, False)
        #self.vector_S = nn.ParameterList(paddle.empty((1, rank, 1, 1), **factory_kwargs))
        #self.vector_S = nn.ParameterList(torch.empty((1, rank, 1, 1), **factory_kwargs))
        self.vector_S = nn.Parameter(torch.empty((1, rank, 1, 1), **factory_kwargs))
        #print('rank: ', rank) #64
        

    def forward(self, x):
        x = self.conv_V(x)
        x = x.mul(self.vector_S)
        output = self.conv_U(x)
        return output
    
#nn.Conv3d( 3, 16, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
class SVD_Conv3d(nn.Module):
    """Kernel Number first SVD Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias,
                 padding_mode='zeros', device=None, dtype=None,
                 rank=1):
        super(SVD_Conv3d, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.conv_U = nn.Conv3d(rank, out_channels, (1, 1, 1), (1, 1, 1), 0, (1, 1, 1), 1, bias)
        #print('groups, rank: ', groups, rank)
        self.conv_V = nn.Conv3d(in_channels, rank, kernel_size, stride, padding, dilation, groups, False)
        #self.vector_S = nn.ParameterList(paddle.empty((1, rank, 1, 1), **factory_kwargs))
        #self.vector_S = nn.ParameterList(torch.empty((1, rank, 1, 1), **factory_kwargs))
        self.vector_S = nn.Parameter(torch.empty((1, rank, 1, 1, 1), **factory_kwargs))
        #print('rank: ', rank) #64
        

    def forward(self, x):
        x = self.conv_V(x)
        x = x.mul(self.vector_S)
        output = self.conv_U(x)
        return output

#class SVD_Linear(nn.Layer):
class SVD_Linear(nn.Module):

    def __init__(self, in_features, out_features, bias, device=None, dtype=None, rank=1):
        super(SVD_Linear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.fc_V = nn.Linear(in_features, rank, False)
        #self.vector_S = nn.ParameterList(paddle.empty((1, rank), **factory_kwargs))
        #self.vector_S = nn.ParameterList(torch.empty((1, rank), **factory_kwargs))
        self.vector_S = nn.Parameter(torch.empty((1, rank), **factory_kwargs))
        self.fc_U = nn.Linear(rank, out_features, bias)

    def forward(self, x):
        x = self.fc_V(x)
        x = x.mul(self.vector_S)
        output = self.fc_U(x)
        return output


full2low_mapping_n = {
    nn.Conv3d: SVD_Conv3d,
    nn.Conv2d: SVD_Conv2d,
    nn.Linear: SVD_Linear
}

'''
replace_fullrank_with_lowrank(
            self.low_rank_model_cpu,
            full2low_mapping=full2low_mapping_n,
            layer_rank=self.layer_rank,
            lowrank_param_dict=self.param_lowrank_decomp_dict,
            module_name=""
        )
'''

def replace_fullrank_with_lowrank(model, full2low_mapping={}, layer_rank={}, lowrank_param_dict={},
                                  module_name=""):
    """Recursively replace original full-rank ops with low-rank ops.
    """
    if len(full2low_mapping) == 0 or full2low_mapping is None:
        return model
    else:
        for sub_module_name in model._modules:
            current_module_name = sub_module_name if module_name == "" else \
                module_name + "." + sub_module_name
            # has children # Bottleneck - layer
            if len(model._modules[sub_module_name]._modules) > 0:
                #print('modules: ',model._modules[sub_module_name]._modules)
                replace_fullrank_with_lowrank(model._modules[sub_module_name],
                                              full2low_mapping,
                                              layer_rank,
                                              lowrank_param_dict,
                                              current_module_name)
            else:
                if type(getattr(model, sub_module_name)) in full2low_mapping and \
                        current_module_name in layer_rank.keys():
                    _attr_dict = getattr(model, sub_module_name).__dict__
                    # use inspect.signature to know args and kwargs of __init__
                    _sig = inspect.signature(
                        type(getattr(model, sub_module_name)))
                    #print('attr dict keys: ', _attr_dict.keys()) #dict_keys(['training', '_parameters', '_buffers', '_non_persistent_buffers_set', '_backward_hooks', '_is_full_backward_hook', '_forward_hooks', '_forward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks', '_load_state_dict_post_hooks', '_modules', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'transposed', 'output_padding', 'groups', 'padding_mode', '_reversed_padding_repeated_twice'])
                    #print('_sig: ', _sig) #(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]] = 1, padding: Union[str, int, Tuple[int, int]] = 0, dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None
                    _kwargs = {}
                    for param in _sig.parameters.values():
                        if param.name not in _attr_dict.keys():
                            if 'bias' in param.name:
                                if getattr(model, sub_module_name).bias is not None:
                                    value = True
                                else:
                                    value = False
                            elif 'stride' in param.name:
                                value = 1
                            elif 'padding' in param.name:
                                value = 0
                            elif 'dilation' in param.name:
                                value = 1
                            elif 'groups' in param.name:
                                value = 1
                            elif 'padding_mode' in param.name:
                                value = 'zeros'
                            else:
                                value = None
                            _kwargs[param.name] = value
                        else:
                            _kwargs[param.name] = _attr_dict[param.name]
                    _kwargs['rank'] = layer_rank[current_module_name]
                    _layer_new = full2low_mapping[type( #여기서 실질적으로 SVD conv layer 받아옴
                        getattr(model, sub_module_name))](**_kwargs)
                    old_module = getattr(model, sub_module_name)
                    old_type = type(old_module)
                    bias_tensor = None
                    if _kwargs['bias'] == True:
                        bias_tensor = old_module.bias.data
                    setattr(model, sub_module_name, _layer_new)
                    new_module = model._modules[sub_module_name]
                    if old_type == nn.Conv2d or old_type == nn.Conv3d:
                        conv1 = new_module._modules["conv_V"]
                        conv2 = new_module._modules["conv_U"]
                        param_list = lowrank_param_dict[current_module_name] #self.param_lowrank_decomp_dict[m_name] = [ U.reshape(tensor_shape[0], self.layer_rank[m_name], 1, 1), V.reshape( self.layer_rank[m_name], tensor_shape[1], tensor_shape[2], tensor_shape[3]), S.reshape(1, self.layer_rank[m_name], 1, 1) ]
                        conv1.weight.data.copy_(param_list[1]) #여기서 U,V,S의 weight 복사됨
                        conv2.weight.data.copy_(param_list[0])
                        new_module.vector_S.data.copy_(param_list[2])
                        #print('new_module S max: ', torch.max(param_list[2]))
                        if bias_tensor is not None:
                            #print('here') #없음
                            conv2.bias.data.copy_(bias_tensor)
    return model


class DatafreeSVD(object):

    def __init__(self, model, global_rank_ratio=1.0,
                 excluded_layers=[], customized_layer_rank_ratio={}, skip_1x1=True, skip_3x3=True):
        # class-independent initialization
        super(DatafreeSVD, self).__init__()
        self.model = model
        self.layer_rank = {}
        model_dict_key = list(model.state_dict().keys())[0]
        model_data_parallel = True if str(
            model_dict_key).startswith('module') else False # dataparallel 사용하면 'module.conv1.weight', 아니면  'conv1.weight'
        self.model_cpu = self.model.module.to(
            "cpu") if model_data_parallel else self.model.to("cpu")
        #self.model_named_modules = self.model_cpu.named_modules() #for name, module in model_cpu.named_modules(): 로 사용하면, 'conv1', 'bn1', 'relu', 'maxpool' 등의 문자열 반환받음
        self.model_named_modules = self.model_cpu.modules()
        #self.rank_base = 4
        self.rank_base = 32
        self.global_rank_ratio = global_rank_ratio
        self.excluded_layers = excluded_layers
        self.customized_layer_rank_ratio = customized_layer_rank_ratio
        self.skip_1x1 = skip_1x1
        self.skip_3x3 = skip_3x3

        

        self.param_lowrank_decomp_dict = {}
        registered_param_op = [nn.Conv3d, nn.Conv2d, nn.Linear]

        #for m_name, m in self.model_named_modules:
        for m_name, m in self.model.named_modules():
            if type(m) in registered_param_op and m_name not in self.excluded_layers: #conv, linear layer의 rank 구함
                weights_tensor = m.weight.data
                tensor_shape = weights_tensor.squeeze().shape # ex)[1, 3, 1, 2] -> [3,2]
                param_1x1 = False
                param_3x3 = False
                depthwise_conv = False
                if len(tensor_shape) == 2: #singular value matrix의 rank 정함(R)
                    full_rank = min(tensor_shape[0], tensor_shape[1])
                    param_1x1 = True
                elif len(tensor_shape) == 4:
                    full_rank = min(
                        tensor_shape[0], tensor_shape[1] * tensor_shape[2] * tensor_shape[3])
                    if tensor_shape[2] == 1 and tensor_shape[3] == 1:
                        param_1x1 = True #1x1 conv
                    else:
                        param_3x3 = True
                elif len(tensor_shape) == 5: #3d conv
                    full_rank = min(
                        tensor_shape[0], tensor_shape[1] * tensor_shape[2] * tensor_shape[3] * tensor_shape[4])
                    if tensor_shape[2] == 1 and tensor_shape[3] == 1 and tensor_shape[4] == 1:
                        param_1x1 = True
                    else:
                        param_3x3 = True
                else:
                    full_rank = 1
                    depthwise_conv = True 

                if self.skip_1x1 and param_1x1:
                    continue
                if self.skip_3x3 and param_3x3:
                    continue
                if depthwise_conv:
                    continue

                low_rank = round_to_nearest(full_rank, 
                                                ratio=self.global_rank_ratio,
                                                base_number=self.rank_base,
                                                allow_rank_eq1=True) #4의 배수로 반올림

                self.layer_rank[m_name] = low_rank

    def decompose_layers(self):
        self.model_named_modules = self.model_cpu.named_modules()
        for m_name, m in self.model_named_modules:
            if m_name in self.layer_rank.keys(): #nn.conv2d, linear
                weights_tensor = m.weight.data
                tensor_shape = weights_tensor.shape #(out_channels, in_channels, kernel_height, kernel_width)
                if len(tensor_shape) == 1:
                    self.layer_rank[m_name] = 1
                    continue
                elif len(tensor_shape) == 2:
                    weights_matrix = m.weight.data
                    U, S, V = d_nsvd(weights_matrix, self.layer_rank[m_name])
                    self.param_lowrank_decomp_dict[m_name] = [
                        U, V, S.reshape(1, self.layer_rank[m_name])]
                elif len(tensor_shape) == 4:
                    weights_matrix = m.weight.data.reshape(tensor_shape[0], -1)
                    U, S, V = d_nsvd(weights_matrix, self.layer_rank[m_name])
                    #print('U shape: ', U.shape) # ex)[256, 64]
                    self.param_lowrank_decomp_dict[m_name] = [
                        U.reshape(tensor_shape[0],
                                  self.layer_rank[m_name], 1, 1),
                        V.reshape(
                            self.layer_rank[m_name], tensor_shape[1], tensor_shape[2], tensor_shape[3]),
                        S.reshape(1, self.layer_rank[m_name], 1, 1)    
                    ]
                elif len(tensor_shape) == 5:
                    weights_matrix = m.weight.data.reshape(tensor_shape[0], -1)
                    U, S, V = d_nsvd(weights_matrix, self.layer_rank[m_name])
                    #print('U shape: ', U.shape) # ex)[256, 64]
                    #print('V shape: ', V.shape)
                    self.param_lowrank_decomp_dict[m_name] = [
                        U.reshape(tensor_shape[0],
                                  self.layer_rank[m_name], 1, 1, 1),
                        V.reshape(
                            self.layer_rank[m_name], tensor_shape[1], tensor_shape[2], tensor_shape[3], tensor_shape[4]),
                        S.reshape(1, self.layer_rank[m_name], 1, 1, 1)    
                    ]

    def reconstruct_lowrank_network(self):
        self.low_rank_model_cpu = copy.deepcopy(self.model_cpu)
        self.low_rank_model_cpu = replace_fullrank_with_lowrank(
            self.low_rank_model_cpu,
            full2low_mapping=full2low_mapping_n,
            layer_rank=self.layer_rank,
            lowrank_param_dict=self.param_lowrank_decomp_dict,
            module_name=""
        )
        return self.low_rank_model_cpu

def round_to_nearest(n, ratio=1.0, base_number=4, allow_rank_eq1=False):
    rank = floor(floor(n * ratio) / base_number) * base_number
    rank = min(max(rank, 1), n)
    if rank == 1:
        rank = rank if allow_rank_eq1 else n
    return rank

def resolver(
        model,
        global_low_rank_ratio=1.0,
        excluded_layers=[],
        customized_layers_low_rank_ratio={},
        skip_1x1=False, #skip하면 decompose 안함
        skip_3x3=False
):
    lowrank_resolver = DatafreeSVD(model,
                                   global_rank_ratio=global_low_rank_ratio,
                                   excluded_layers=excluded_layers,
                                   customized_layer_rank_ratio=customized_layers_low_rank_ratio,
                                   skip_1x1=skip_1x1,
                                   skip_3x3=skip_3x3)
    lowrank_resolver.decompose_layers()
    lowrank_cpu_model = lowrank_resolver.reconstruct_lowrank_network()
    return lowrank_cpu_model


if __name__ == "__main__":

    #model = models.resnet18(pretrained=True)
    # Pretrained ResNet-50
    #model = models.resnet50(pretrained=True)
    model = resnetl.resnetl10(
                num_classes=2,
                shortcut_type='A',
                sample_size= 112,
                sample_duration= 16)

    model = resolver(model,
                        global_low_rank_ratio=1.0,  # no need to change
                        skip_1x1=False,  # we will decompose 1x1 conv layers
                        skip_3x3=False  # we will decompose 3x3 conv layers
                                    )

    #for name, module in model.named_modules():
    #for name, parameter in model.named_parameters():
    #    print(f"Layer: {name} | Size: {parameter.size()}  \n")
    for name, module in model.named_modules():
        print(f"Module Name: {name}")
        print(f"Module Object: {module}")
    print('finish')
    #origin_model = FSS_model
    #final_model = resolver(origin_model)
    #self.layer1 = svf.resolver(self.layer1, global_low_rank_ratio=1.0, skip_1x1=False, skip_3x3=False)
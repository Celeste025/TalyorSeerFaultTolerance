import torch
from torch import nn
from functools import partial
import pdb
from torch.nn.modules.conv import Conv2d
import random
import numpy as np

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=4):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w, scales


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=4):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w, scales


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=4):
    t_shape = t.shape
    t.reshape(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t, scales


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=4):
    t_shape = t.shape
    t.contiguous().view(-1, t_shape[-1])  ##contiguous added
    scales = t.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t, scales


def identity(x):
    return x

class W8A8Conv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', act_quant='per_token', weight_quant='per_tensor', quantize_output=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        
        if not hasattr(self, 'weight'):
            print("randn_weight.")
            self.register_buffer('weight', torch.randn(self.out_channels, self.in_channels // self.groups, *self.kernel_size, dtype=torch.float16, requires_grad=False))
        
        if bias:
            if not hasattr(self, 'bias'):
                print("randn_bias")
                self.register_buffer('bias', torch.zeros(self.out_channels, dtype=torch.float16, requires_grad=False))
        else:
            if not hasattr(self, 'bias'):
                self.register_buffer('bias', None)
        
        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=4)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=4)
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')
        
        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x
    
    def to(self, *args, **kwargs):
        super(W8A8Conv2d, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def forward(self, x):
        q_x, _ = self.act_quant(x)
        y = self._conv_forward(q_x, self.weight, self.bias)
        if self.output_quant_name == "None":
            q_y = self.output_quant(y)
        else:
            q_y, _ = self.output_quant(y)
        return q_y
    
    @staticmethod
    def from_float(module, weight_quant='per_channel', act_quant='per_token', quantize_output=False):
        assert isinstance(module, nn.Conv2d)
        new_module = W8A8Conv2d(
            module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, 
            module.dilation, module.groups, module.bias is not None, module.padding_mode, act_quant=act_quant, 
            weight_quant=weight_quant, quantize_output=quantize_output)
        if weight_quant == 'per_channel':
            new_module.weight, _ = quantize_weight_per_channel_absmax(
                module.weight, n_bits=4)
        elif weight_quant == 'per_tensor':
            new_module.weight, _ = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=4)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module
    
class NoisyW8A8Conv2d(W8A8Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', act_quant='per_token', weight_quant='per_tensor', quantize_output=False, err_prob=0.0, accumulation_bitw=32):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, act_quant, weight_quant, quantize_output)
        assert isinstance(err_prob, list) or isinstance(err_prob, float)
        self.err_prob = err_prob
        self.accumulation_bitw = accumulation_bitw
        self.w_scales = None

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    @torch.no_grad()
    def inject_error(self, y, w_scales, a_scales, err_prob):
        y_not_quantized = y
        result = y.to(torch.float32) / (w_scales * a_scales)  ## integer of y
        result = result.round().to(torch.int32)
        result_injected = result

        flip_bit = 30
        err = torch.tensor([2**flip_bit], dtype=torch.int32).to(result.device)
        prob_tensor = torch.full(result.shape, err_prob).to(result.device)
        mask = torch.bernoulli(prob_tensor).bool().to(result.device)

        result_injected[mask] = torch.bitwise_xor(result[mask], err)
        
        result_injected = result_injected.to(torch.float32) * a_scales * w_scales
        result_injected = result_injected.to(y.dtype)
        y_not_quantized[mask] = result_injected[mask]

        return y_not_quantized

    @torch.no_grad()
    def forward(self, x):  
        #print("NoisyConv2d forwarding")
        q_x, x_scales = self.act_quant(x) ## q_x is multiplied by scale
        y = self._conv_forward(q_x, self.weight, bias=self.bias)
        y_for_quant = self._conv_forward(q_x, self.weight, bias=self.bias)
        y_injected = self.inject_error(y, self.w_scales, x_scales, self.err_prob)
        # if self.bias is not None:
        #     y_injected = y_injected + self.bias

        if self.output_quant_name == "None":
            q_y = self.output_quant(y_injected)  
        else:
            _, out_scale = self.output_quant(y_for_quant)  
            q_y = torch.clamp(torch.round(y_injected / (out_scale)), -7, 7) * out_scale ## quant according to out_scale

        return q_y

    @staticmethod
    def from_float(module, weight_quant='per_channel', act_quant='per_token', quantize_output=False, err_prob=0.0, accumulation_width=32):
        assert isinstance(module, Conv2d)
        new_module = NoisyW8A8Conv2d(
            module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None, module.padding_mode, act_quant=act_quant, weight_quant=weight_quant, quantize_output=quantize_output, err_prob=err_prob, accumulation_bitw=accumulation_width)
        if weight_quant == 'per_channel':
            new_module.weight, new_module.w_scales = quantize_weight_per_channel_absmax(
                module.weight, n_bits=4)  # use 4-bit integer for weight
        elif weight_quant == 'per_tensor':
            new_module.weight, new_module.w_scales = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=4)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f'NoisyW8A8Conv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None}, padding_mode={self.padding_mode}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name}, err_prob={self.err_prob})'
    
class NoisyW8A8Conv2dProtected(NoisyW8A8Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', act_quant='per_token', weight_quant='per_tensor', quantize_output=False, err_prob=0.0, accumulation_bitw=32, method=None, thre=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, act_quant, weight_quant, quantize_output, err_prob, accumulation_bitw)
        self.method = method
        self.thre = thre
        self.recompute_count = 0

    def inject_error(self, y, w_scales, a_scales, err_prob):
        y_not_quantized = y
        result = y.to(torch.float32) / (w_scales * a_scales)
        result = result.round().to(torch.int32)
        result_injected = result.clone()

        flip_bit = random.randint(0, 30)
        #print("flip_bit = ", flip_bit)
        err = torch.tensor([2**flip_bit], dtype=torch.int32).to(result.device)
        prob_tensor = torch.full(result.shape, err_prob).to(result.device)
        mask = torch.bernoulli(prob_tensor).bool().to(result.device)

        result_injected[mask] = torch.bitwise_xor(result[mask], err)

        # 计算注错前后的差异
        diff = torch.abs(result - result_injected).sum().item()
        #print(f"Difference between original and injected int matrices: {diff}")
        result_injected = result_injected.to(torch.float32) * a_scales * w_scales
        result_injected = result_injected.to(y.dtype)
        y_not_quantized[mask] = result_injected[mask]

        return y_not_quantized, diff
    
    @torch.no_grad()
    def forward(self, x):  
        thre_record = self.thre

        #print("NoisyConv2d forwarding")
        q_x, x_scales = self.act_quant(x) ## q_x is multiplied by scale
        y = self._conv_forward(q_x, self.weight, bias=self.bias)
        y_for_quant = self._conv_forward(q_x, self.weight, bias=self.bias)
        y_injected, diff = self.inject_error(y, self.w_scales, x_scales, self.err_prob)
      

        if self.output_quant_name == "None":  #quantize_out=False
            q_y = self.output_quant(y_injected)  
        else:  #quantize_output=True
            #import pdb; pdb.set_trace()
            _, out_scale = self.output_quant(y_for_quant)  
            #print(out_scale.shape)
            q_y_bit = torch.round(y_injected / out_scale) ## quant according to out_scale
            q_y = torch.clamp(torch.round(y_injected / out_scale), -7, 7) * out_scale ## quant according to out_scale

        _, p_out_scale = quantize_activation_per_tensor_absmax(y_for_quant, n_bits=4)
        # Apply protection based on the specified method
        if self.method == "AD":
            if self.output_quant_name == "None":  #quantize_out=False
                if self.thre is None:
                    self.thre = (-7*p_out_scale, 7*p_out_scale)
                y_injected = torch.where(  
                    (y_injected < self.thre[0]) | (y_injected > self.thre[1]),
                    torch.tensor(0.0, device=y_injected.device),
                    y_injected
                )   #将超出阈值的值置为0
                q_y = self.output_quant(y_injected)

                self.thre = thre_record
                return q_y
            
            else: #quantize_output=True
                if self.thre is None:
                    self.thre = (-7,7)
                tmp = q_y_bit   #q_y_bit 不一定在-127，127范围
                # condition = (tmp < self.thre[0]) | (tmp > self.thre[1])
                # modified_num =  torch.sum(condition)
                # if modified_num > 0:
                #     print(f"(conv quantize_output=True) AD modified {modified_num}.")
                #     tmp_size = tmp.numel()
                #     print(f"estimated error rate: {modified_num/(tmp_size)}")
                    #import pdb; pdb.set_trace()
                tmp = torch.where(
                    (tmp < self.thre[0]) | (tmp > self.thre[1]),
                    torch.tensor(0, device=tmp.device),
                    tmp
                )
                q_y = tmp*out_scale

                self.thre = thre_record 
                return q_y

        elif self.method == "ABFT":
            if self.thre is None:
                self.thre = 2**30 * 1  #默认1个bit翻转
            #print(diff,self.thre)
            if diff >= self.thre:
                if self.recompute_count < 1:
                    print("ABFT recompute.")
                    self.recompute_count += 1
                    q_y = self.forward(x)
                else:  #已经重算过了，直接return
                    self.recompute_count = 0
                    self.thre = thre_record
                    return q_y
            self.recompute_count = 0
            self.thre = thre_record
            return q_y
        
        elif self.method == "RC":
            if diff != 0 : #检查到不一样就重算三次，视为无错误
                print("RC applied.")
                if self.bias is not None:
                    y_right = y + self.bias
                if self.output_quant_name== "None":
                    q_y = self.output_quant(y_right)  #q_y float
                else:
                    q_y=torch.clamp(torch.round(y_right/out_scale),-7,7)*out_scale ## quant according to out_scale
                return q_y
            else: #直接出结果
                return q_y
            
        else: #无保护
            return q_y
       
    
    @staticmethod
    def from_float(module, weight_quant='per_tensor', act_quant='per_tensor', quantize_output=False, err_prob=0.0, accumulation_width=32, method=None, thre=None):
        assert isinstance(module, nn.Conv2d)
        new_module = NoisyW8A8Conv2dProtected(
            module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None, module.padding_mode, act_quant=act_quant, weight_quant=weight_quant, quantize_output=quantize_output, err_prob=err_prob, accumulation_bitw=accumulation_width, method=method, thre=thre)
        if weight_quant == 'per_channel':
            new_module.weight, new_module.w_scales = quantize_weight_per_channel_absmax(
                module.weight, n_bits=4)  # use 4-bit integer for weight
        elif weight_quant == 'per_tensor':
            new_module.weight, new_module.w_scales = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=4)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

class W8A8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, act_quant='per_token', weight_quant='per_tensor', quantize_output=False): ## weight_quant added
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_quant_name='per_tensor'

        self.register_buffer('weight', torch.randn(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False))
        
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=4)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=4)
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = identity

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x, _ = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)    
        if self.output_quant_name== "None":
            q_y = self.output_quant(y)
        else:
            q_y, _ = self.output_quant(y)    
        return q_y

    @staticmethod
    def from_float(module, weight_quant='per_channel', act_quant='per_token', quantize_output=False):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features, module.out_features, module.bias is not None, act_quant=act_quant, weight_quant=weight_quant, quantize_output=quantize_output)
        if weight_quant == 'per_channel':
            new_module.weight, _ = quantize_weight_per_channel_absmax(
                module.weight, n_bits=4)  # use 4-bit integer for weight
        elif weight_quant == 'per_tensor':
            new_module.weight, _ = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=4)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias    
        return new_module

    def __repr__(self):
        return f'W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})'



class NoisyW8A8Linear(W8A8Linear):
    def __init__(self, in_features, out_features, bias=True, act_quant='per_token', quantize_output=False, err_prob=0.0, accumulation_bitw=32):
        super().__init__(in_features,out_features,bias,act_quant,quantize_output)
        assert isinstance(err_prob, list) or isinstance(err_prob, float)
        self.err_prob=err_prob
        self.accumulation_bitw = accumulation_bitw
        self.w_scales=None

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = identity
    
    @torch.no_grad()
    def inject_error(self,y, w_scales, a_scales, err_prob):
        y_not_quantized=y
        y_div_a_scales=y/a_scales
        result = y_div_a_scales/w_scales.view(1,1,-1)
        #result=y.to(torch.float32)/(w_scales*a_scales)  ## integer of y
        result=result.round().to(torch.int32)
        result_injected=result

        flip_bit=30
        err=torch.tensor([2**flip_bit],dtype=torch.int32).to(result.device)
        prob_tensor=torch.full(result.shape, err_prob).to(result.device)
        mask=torch.bernoulli(prob_tensor).bool().to(result.device)

        result_injected[mask]=torch.bitwise_xor(result[mask],err)
        
        result_injected=result_injected.to(torch.float32)*a_scales*w_scales.view(1,1,-1)
        #result_injected=result_injected.to(torch.float32)*a_scales*w_scales
        result_injected=result_injected.to(y.dtype)
        y_not_quantized[mask]=result_injected[mask]

        # if not (y_not_quantized == y).all():
        #     import pdb; pdb.set_trace()
        return y_not_quantized

    @torch.no_grad()
    def forward(self, x):  
        q_x, x_scales = self.act_quant(x) ## q_x is multiplied by scale
        y = torch.functional.F.linear(q_x, self.weight, bias=None)
        y_for_quant= torch.functional.F.linear(q_x, self.weight, bias=self.bias)
        y_injected=self.inject_error(y, self.w_scales, x_scales, self.err_prob)
        if self.bias is not None:
            y_injected=y_injected + self.bias

        if self.output_quant_name== "None":
            q_y = self.output_quant(y_injected)  
            # clipping=y_for_quant.abs().max()

            # q_y=torch.where((q_y>clipping)|(q_y<-clipping), torch.zeros_like(q_y), q_y)
            # q_y = torch.clamp(q_y,-clipping,clipping) ## avoid overflowing of float16
        else:
            _, out_scale = self.output_quant(y_for_quant)  
            q_y=torch.clamp(torch.round(y_injected/out_scale),-7,7)*out_scale ## quant according to out_scale
        return q_y

    @staticmethod
    def from_float(module, weight_quant='per_channel', act_quant='per_token', quantize_output=False, err_prob=0.0,accumulation_width=32):
        assert isinstance(module, torch.nn.Linear)
        new_module = NoisyW8A8Linear(
            module.in_features, module.out_features, module.bias is not None, act_quant=act_quant, quantize_output=quantize_output,err_prob=err_prob,accumulation_bitw=accumulation_width)
        if weight_quant == 'per_channel':
            new_module.weight, new_module.w_scales = quantize_weight_per_channel_absmax(
                module.weight, n_bits=4)  # use 4-bit integer for weight
        elif weight_quant == 'per_tensor':
            new_module.weight, new_module.w_scales = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=4)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f'NoisyW8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name}, err_prob={self.err_prob})'
    

def hadamard(n):
    # Base case
    if n == 1:
        return np.array([[1]])
    
    # Recursive construction
    H_n_minus_1 = hadamard(n // 2)
    top = np.hstack((H_n_minus_1, H_n_minus_1))
    bottom = np.hstack((H_n_minus_1, -H_n_minus_1))
    return np.vstack((top, bottom))

# Compute Hadamard matrix H1024
H_1024 = hadamard(1024)
H_4096 = hadamard(4096)
Q = torch.from_numpy(H_4096 / 64).to(torch.device('cuda'))
Q = Q.to(torch.float32)

class NoisyW8A8LinearProtected(NoisyW8A8Linear):
    def __init__(self, in_features, out_features, bias=True, act_quant='per_token', quantize_output=False, err_prob=0.0, accumulation_bitw=32, method=None, thre=(0.0, 0.0)):
        super().__init__(in_features, out_features, bias, act_quant, quantize_output, err_prob, accumulation_bitw)
        self.method = method
        self.thre = thre
        self.recompute_count = 0

    def inject_error(self, y, w_scales, a_scales, err_prob):
        y_not_quantized = y.clone()
        y_div_a_scales=y/a_scales
        result = y_div_a_scales/w_scales.view(1,1,-1)
        #result=y.to(torch.float32)/(w_scales*a_scales)  ## integer of y
        result = result.round().to(torch.int32)
        result_injected = result.clone()

        flip_bit = torch.randint(0, 31, result.size(), dtype=torch.int, device=result.device)
        #print("flip_bit = ", flip_bit)
        err = torch.pow(2, flip_bit).to(torch.int32)
        prob_tensor = torch.full(result.shape, err_prob, device=result.device)
        mask = torch.bernoulli(prob_tensor).bool()

        result_injected[mask] = torch.bitwise_xor(result[mask], err[mask])

        # 计算注错前后的差异
        diff = torch.abs(result - result_injected).sum().item()
        result_injected = result_injected.to(torch.float32)*a_scales*w_scales.view(1,1,-1)
        result_injected = result_injected.to(y.dtype)
        y_not_quantized[mask] = result_injected[mask]
        return result_injected, diff

    @torch.no_grad()
    def forward(self, x):
        # Save the original threshold and compute common variables.
        original_thre = self.thre
        q_x, x_scales = self.act_quant(x)
        
        weight_matrix = self.weight
        weight_scales = self.w_scales
        if "QuaRot" in self.method:
            #print("QuaRot enabled.")
            global Q
            wm_dtype = weight_matrix.dtype
            weight_matrix = weight_matrix.to(torch.float32)
            weight_matrix = torch.matmul(weight_matrix, Q)
            weight_matrix, weight_scales = quantize_weight_per_channel_absmax(weight_matrix, n_bits=4)
            weight_matrix = weight_matrix.to(wm_dtype)

        y = torch.functional.F.linear(q_x, weight_matrix, bias=None)
        y_for_quant = torch.functional.F.linear(q_x, weight_matrix, bias=self.bias)
        y_injected, diff = self.inject_error(y, weight_scales, x_scales, self.err_prob)

        if self.bias is not None:
            y_injected = y_injected + self.bias

        # Quantize the result.
        if self.output_quant_name != "None":
            _, out_scale = self.output_quant(y_for_quant)
            q_y = torch.clamp(torch.round(y_injected / out_scale), -7, 7) * out_scale
        else:
            q_y = self.output_quant(y_injected)

        # Protection: shared post-processing.
        p_out_scale = x_scales * weight_scales.view(1,1,-1)
        #_, p_out_scale = quantize_activation_per_token_absmax(y_for_quant, n_bits=4)

        if "AD" in self.method:
            #print("AD enabled.")
            # Set default thresholds if not provided.
            if self.thre is None:
                if self.output_quant_name != "None":
                    self.thre = (-7, 7)
                else:
                    self.thre = (-7 * p_out_scale, 7 * p_out_scale)
            if self.output_quant_name != "None":
                q_y_bit = torch.round(y_injected / out_scale)
                q_y_bit = torch.where(
                    (q_y_bit < self.thre[0]) | (q_y_bit > self.thre[1]),
                    torch.tensor(0, device=q_y_bit.device),
                    q_y_bit
                )
                # Clamp the quantized values to the threshold range.
                q_y = q_y_bit * out_scale
            else:
                y_injected = torch.where(
                    (y_injected < self.thre[0]) | (y_injected > self.thre[1]),
                    torch.tensor(0.0, device=y_injected.device),
                    y_injected
                )
                q_y = self.output_quant(y_injected)

        elif self.method == "ABFT":
            if self.thre is None:
                self.thre = 2 ** 30  # Default threshold for ABFT.
            if diff >= self.thre and self.recompute_count < 1:
                print("ABFT recompute.")
                self.recompute_count += 1
                q_y = self.forward(x)
            else:
                self.recompute_count = 0

        elif self.method == "RC":
            if diff != 0:
                print("RC applied.")
                y_right = y + (self.bias if self.bias is not None else 0)
                if self.output_quant_name != "None":
                    q_y = torch.clamp(torch.round(y_right / out_scale), -7, 7) * out_scale
                else:
                    q_y = self.output_quant(y_right)
            # else: keep the original q_y

        # Restore original threshold and return final result.
        self.thre = original_thre
        return q_y


    @staticmethod
    def from_float(module, weight_quant='per_channel', act_quant='per_token', quantize_output=False, err_prob=0.0, accumulation_width=32, method=None, thre=None):
        assert isinstance(module, torch.nn.Linear)
        new_module = NoisyW8A8LinearProtected(
            module.in_features, module.out_features, module.bias is not None, act_quant=act_quant, quantize_output=quantize_output, err_prob=err_prob, accumulation_bitw=accumulation_width, method=method, thre=thre)
        if weight_quant == 'per_channel':
            new_module.weight, new_module.w_scales = quantize_weight_per_channel_absmax(
                module.weight, n_bits=4)  # use 4-bit integer for weight
        elif weight_quant == 'per_tensor':
            new_module.weight, new_module.w_scales = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=4)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f'NoisyW8A8LinearProtected({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name}, err_prob={self.err_prob}, method={self.method}, thre={self.thre})'


class W8A8BMM(nn.Module):
    def __init__(self, act_quant='per_token',quantize_output=False):
        super().__init__()
        
        if act_quant=='per_token':
            self.act_quant_name='per_token'
            self.act_quant=partial(quantize_activation_per_token_absmax,n_bits=4)
        elif act_quant=='per_tensor':
            self.act_quant_name='per_tensor'
            self.act_quant=partial(quantize_activation_per_tensor_absmax,n_bits=4)
        else:
            raise ValueError(f'Invalid act_qunant: {act_quant}')
        
        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = identity

    @torch.no_grad()
    def forward(self, input1, input2):
        # pdb.set_trace()
        q_input1, _ = self.act_quant(input1)
        q_input2, _ = self.act_quant(input2)
        y = torch.bmm(q_input1, q_input2)
        
        if self.output_quant_name== "None":
            q_y = self.output_quant(y)
        else:
            q_y, _ = self.output_quant(y)
        return q_y

    def __repr__(self):
        return f'W8A8BMM(act_quant={self.act_quant_name}, output_quant={self.output_quant_name})'




class NoisyW8A8BMM(W8A8BMM):
    def __init__(self, act_quant='per_token', quantize_output=False, err_prob=0.0, accumulation_bitw=32):
        super().__init__(act_quant,quantize_output)
        assert isinstance(err_prob, list) or isinstance(err_prob, float)
        self.err_prob=err_prob
        self.accumulation_bitw = accumulation_bitw
        
    @torch.no_grad()
    def inject_error(self,y, w_scales, a_scales, err_prob):
        y_not_quantized=y
        y_div_a_scales=y/a_scales
        result = y_div_a_scales/w_scales.view(1,1,-1)
        #result=y.to(torch.float32)/(w_scales*a_scales)  ## integer of y
        result=result.round().to(torch.int32)
        result_injected=result
        flip_bit=30
        err=torch.tensor([2**flip_bit],dtype=torch.int32).to(result.device)
        prob_tensor=torch.full(result.shape, err_prob).to(result.device)
        mask=torch.bernoulli(prob_tensor).bool().to(result.device)
        result_injected[mask]=torch.bitwise_xor(result[mask],err)
        #result_injected=result_injected.to(torch.float32)*a_scales*w_scales
        result_injected=result_injected.to(torch.float32)*a_scales*w_scales.view(1,1,-1)
        result_injected=result_injected.to(y.dtype)
        y_not_quantized[mask]=result_injected[mask]

        return y_not_quantized
    
    @torch.no_grad()
    def forward(self, input1, input2):
        q_input1, input1_scale = self.act_quant(input1)
        q_input2, input2_scale = self.act_quant(input2)
        y = torch.bmm(q_input1, q_input2)
        y_clone=y.clone()
        y_injected=self.inject_error(y_clone,input1_scale,input2_scale,self.err_prob)
        # y_injected = y
        
        if self.output_quant_name== "None":
            q_y = self.output_quant(y_injected)
            # q_y = torch.clamp(q_y,-32768,32768) ## avoid overfitting of float16
        else:
            _, out_scale = self.output_quant(y)
            q_y=torch.clamp(torch.round(y_injected/out_scale),-7,7)*out_scale ## quant according to out_scale    
            # q_y, _ = self.output_quant(y)
        return q_y
    
    def __repr__(self):
        return f'NoisyW8A8BMM(act_quant={self.act_quant_name}, output_quant={self.output_quant_name}, err_prob={self.err_prob})'

class layer_norm_without_outlier(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, percentage=0.1):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self.percentage = percentage

    @torch.no_grad()
    def forward(self,x):
        upper=x.quantile(self.percentage)
        lower=x.quantile(1-self.percentage)
        mask=(x<lower)|(x>upper)
        medium_elements=x.clone()
        medium_elements[mask]=0.
        mu=medium_elements.mean(dim=1,keepdim=True)
        sigma2=medium_elements.var(dim=1,keepdim=True)
        x_normalized=(x-mu)/torch.sqrt(sigma2+self.eps)*self.weight+self.bias
        return x_normalized
    
    @staticmethod
    def from_float(module,percentage):
        new_module=layer_norm_without_outlier(normalized_shape=module.normalized_shape, 
                                              eps=module.eps, 
                                              elementwise_affine=module.elementwise_affine,
                                              percentage=percentage).to('cuda')
        new_module.weight=module.weight
        new_module.bias=module.bias
        return new_module
    
    def __repr__(self):
        return f'layer_norm_without_outlier(percentage={self.percentage})'
    
class W8A8MatMul(nn.Module):
    def __init__(self, act_quant='per_token',quantize_output=False):
        super().__init__()
        
        if act_quant=='per_token':
            self.act_quant_name='per_token'
            self.act_quant=partial(quantize_activation_per_token_absmax,n_bits=4)
        elif act_quant=='per_tensor':
            self.act_quant_name='per_tensor'
            self.act_quant=partial(quantize_activation_per_tensor_absmax,n_bits=4)
        else:
            raise ValueError(f'Invalid act_qunant: {act_quant}')
        
        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = identity

    @torch.no_grad()
    def forward(self, input1, input2):
        # pdb.set_trace()
        q_input1, _ = self.act_quant(input1)
        q_input2, _ = self.act_quant(input2)
        y = torch.matmul(q_input1, q_input2)
        
        if self.output_quant_name== "None":
            q_y = self.output_quant(y)
        else:
            q_y, _ = self.output_quant(y)
        return q_y

    def __repr__(self):
        return f'W8A8MatMul(act_quant={self.act_quant_name}, output_quant={self.output_quant_name})'
    

class NoisyW8A8MatMul(W8A8MatMul):
    def __init__(self, act_quant='per_token', quantize_output=False, err_prob=0.0, accumulation_bitw=32):
        super().__init__(act_quant,quantize_output)
        assert isinstance(err_prob, list) or isinstance(err_prob, float)
        self.err_prob=err_prob
        self.accumulation_bitw = accumulation_bitw
        
    @torch.no_grad()
    def inject_error(self,y, w_scales, a_scales, err_prob):
        y_not_quantized=y
        scale=torch.matmul(w_scales,a_scales.permute(0,1,3,2))
        result=y/scale
       
        #result=y.to(torch.float32)/(w_scales*a_scales)  ## integer of y
        result=result.round().to(torch.int32)
        result_injected=result
        flip_bit=30
        err=torch.tensor([2**flip_bit],dtype=torch.int32).to(result.device)
        prob_tensor=torch.full(result.shape, err_prob).to(result.device)
        mask=torch.bernoulli(prob_tensor).bool().to(result.device)
        result_injected[mask]=torch.bitwise_xor(result[mask],err)
        result_injected=result_injected.to(torch.float32)*scale
        result_injected=result_injected.to(y.dtype)
        y_not_quantized[mask]=result_injected[mask]
        
        return y_not_quantized
    
    @torch.no_grad()
    def forward(self, input1, input2):
        q_input1, input1_scale = self.act_quant(input1)
        q_input2, input2_scale = self.act_quant(input2.permute(0,1,3,2))
        y = torch.matmul(q_input1, q_input2.permute(0,1,3,2))
        y_clone=y.clone()
        y_injected=self.inject_error(y_clone,input1_scale,input2_scale,self.err_prob)
        # y_injected = y
        
        if self.output_quant_name== "None":
            q_y = self.output_quant(y_injected)
            # q_y = torch.clamp(q_y,-32768,32768) ## avoid overfitting of float16
        else:
            _, out_scale = self.output_quant(y)
            q_y=torch.clamp(torch.round(y_injected/out_scale),-7,7)*out_scale ## quant according to out_scale    
            # q_y, _ = self.output_quant(y)
        return q_y
    
    def __repr__(self):
        return f'NoisyW8A8MatMul(act_quant={self.act_quant_name}, output_quant={self.output_quant_name}, err_prob={self.err_prob})'

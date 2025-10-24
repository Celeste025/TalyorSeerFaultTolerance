from fault_injection.fake_quant import NoisyW8A8Conv2d, W8A8Conv2d, W8A8Linear, W8A8BMM, NoisyW8A8Linear, NoisyW8A8BMM, NoisyW8A8LinearProtected, NoisyW8A8Conv2dProtected
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaMLP, LlamaForCausalLM
from transformers.models.clip.modeling_clip import CLIPMLP, CLIPSdpaAttention
from transformers.models.llava_next.modeling_llava_next import LlavaNextMultiModalProjector
from jarvis.mineclip.clip import VisionTransformer
from jarvis.mineclip.pooling import TemporalPooling
from jarvis.mineclip.head import CLIPScoreHead
from torch.nn.modules.linear import Linear
from torch.nn.modules.conv import Conv2d
from jarvis.steveI.steveI_lib.VPT.lib.impala_cnn import ImpalaCNN, CnnDownStack,CnnBasicBlock
from jarvis.steveI.steveI_lib.embed_conditioned_policy import ImgObsProcess
from jarvis.steveI.steveI_lib.VPT.lib.util import ResidualRecurrentBlocks
from jarvis.steveI.steveI_lib.embed_conditioned_policy import MinecraftAgentPolicy
import pdb

import re
import torch.multiprocessing as mp
from rich import print as rprint
import torch
import dill

def print_activations(module, input, output):
    with open('output.txt', 'a') as outfile:
        print(f"Layer: {module.__class__.__name__}", file=outfile)
        print(f"Output (activations): {output}\n", file=outfile)

def modify_llama_attention(module, weight_quant, act_quant, quantize_bmm_input, err_prob):
    module.q_proj = NoisyW8A8LinearProtected.from_float(module.q_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob, method="ABFT", thre=17*2**30)
    module.k_proj = NoisyW8A8LinearProtected.from_float(module.k_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob, method="ABFT", thre=17*2**30)
    module.v_proj = NoisyW8A8LinearProtected.from_float(module.v_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob, method="ABFT", thre=17*2**30)
    module.o_proj = NoisyW8A8LinearProtected.from_float(module.o_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=False, err_prob=err_prob, method="ABFT", thre=2**20)
    module.to('cpu')
    return module

def modify_llama_attention_k_proj(module, weight_quant, act_quant, quantize_bmm_input, err_prob):
    module.k_proj = NoisyW8A8Linear.from_float(module.k_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
    module.to('cpu')
    return module

def modify_clip_attention(module, weight_quant, act_quant, quantize_bmm_input, err_prob):
    module.q_proj = NoisyW8A8Linear.from_float(module.q_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
    module.k_proj = NoisyW8A8Linear.from_float(module.k_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
    module.v_proj = NoisyW8A8Linear.from_float(module.v_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
    module.out_proj = NoisyW8A8Linear.from_float(module.out_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=False, err_prob=err_prob)
    module.to('cpu')
    return module
    
def modify_llama_mlp(module, weight_quant, act_quant, err_prob):
    module.gate_proj = NoisyW8A8LinearProtected.from_float(module.gate_proj, weight_quant=weight_quant,
        act_quant=act_quant, quantize_output=False, err_prob=err_prob, method="ABFT", thre=17*2**30)
    module.up_proj = NoisyW8A8LinearProtected.from_float(module.up_proj, weight_quant=weight_quant,
        act_quant=act_quant, quantize_output=False, err_prob=err_prob, method="ABFT", thre=17*2**30)
    module.down_proj = NoisyW8A8LinearProtected.from_float(module.down_proj, weight_quant=weight_quant,
        act_quant=act_quant, quantize_output=False, err_prob=err_prob, method="ABFT", thre=2**20)
    module.to('cpu')
    return module
    
    
def modify_clip_mlp(module, weight_quant, act_quant, err_prob):
    module.fc1 = NoisyW8A8Linear.from_float(module.fc1, weight_quant=weight_quant,
        act_quant=act_quant, quantize_output=False, err_prob=err_prob)
    module.fc2 = NoisyW8A8Linear.from_float(module.fc2, weight_quant=weight_quant,
        act_quant=act_quant, quantize_output=False, err_prob=err_prob)
    module.to('cpu')
    return module


class FIassistant:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def is_target_layer(layer_name, target_layers):
        # 如果目标层列表为空，返回 True
        if not target_layers:
            return True
        match = re.search(r'\.(\d+)\.', layer_name)
        if match:
            first_layer_number = int(match.group(1))
            return (first_layer_number in target_layers)
        return False
    
    def inject_fault_to_module(self, target="", weight_quant='per_channel', act_quant='per_token', quantize_bmm_input=False, err_prob=0.0, target_layers=[]):
        #target_layers is used to mark which layers need to inject fault, an empty list means all layers
        ###planner part
        if target == "planner_all":
            self.inject_fault_to_module('planner_llama_all', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('planner_clip_all',
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('planner_multi_model_projector', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            
        elif target == "planner_llama_all":
            self.inject_fault_to_module('planner_llama_attention', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('planner_llama_mlp', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            """self.inject_fault_to_module('planner_llama_lm_head', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=[])"""
        
        elif target == 'planner_llama_attention':
            mp.set_start_method('spawn', force=True)
            mp.get_context().forking_pickler = dill
            args = [
                (m, weight_quant, act_quant, quantize_bmm_input, err_prob)
                for name, m in self.model.named_modules()
                if isinstance(m, LlamaSdpaAttention) and self.is_target_layer(name, target_layers)
            ]
            with mp.Pool(processes=20) as pool:
                modified_modules = pool.starmap(modify_llama_attention, args)
            
            
            module_to_modify = [
                (name, m) for name, m in self.model.named_modules()
                if isinstance(m, LlamaSdpaAttention) and self.is_target_layer(name, target_layers)
            ]
            
            for (name, m), modified_module in zip(module_to_modify, modified_modules):
                name_parts = name.split('.')
                module = self.model
                for part in name_parts[:-1]:
                    module = getattr(module, part)
                getattr(module, name_parts[-1]).to("cpu")
                setattr(module, name_parts[-1], modified_module.to('cuda'))
                
            rprint("injected: all llama attention layers")
            

        elif target == "planner_llama_mlp":
            mp.set_start_method('spawn', force=True)
            mp.get_context().forking_pickler = dill
            
            args = [
                (m, weight_quant, act_quant, err_prob)
                for name, m in self.model.named_modules()
                if isinstance(m, LlamaMLP) and self.is_target_layer(name, target_layers)
            ]
            with mp.Pool(processes=20) as pool:
                modified_modules = pool.starmap(modify_llama_mlp, args)
                
            module_to_modify = [
                (name, m) for name, m in self.model.named_modules()
                if isinstance(m, LlamaMLP) and self.is_target_layer(name, target_layers)
            ]
            
            for (name, m), modified_module in zip(module_to_modify, modified_modules):
                name_parts = name.split('.')
                module = self.model
                for part in name_parts[:-1]:
                    module = getattr(module, part)
                getattr(module, name_parts[-1]).to("cpu")
                setattr(module, name_parts[-1], modified_module.to('cuda'))
                
            
            rprint("injected: all llama mlp layers")

        elif target == 'planner_llama_attention_k_proj':
            mp.set_start_method('spawn', force=True)
            mp.get_context().forking_pickler = dill
            args = [
                (m, weight_quant, act_quant, quantize_bmm_input, err_prob)
                for name, m in self.model.named_modules()
                if isinstance(m, LlamaSdpaAttention) and self.is_target_layer(name, target_layers)
            ]
            with mp.Pool(processes=20) as pool:
                modified_modules = pool.starmap(modify_llama_attention_k_proj, args)
            
            
            module_to_modify = [
                (name, m) for name, m in self.model.named_modules()
                if isinstance(m, LlamaSdpaAttention) and self.is_target_layer(name, target_layers)
            ]
            
            for (name, m), modified_module in zip(module_to_modify, modified_modules):
                name_parts = name.split('.')
                module = self.model
                for part in name_parts[:-1]:
                    module = getattr(module, part)
                getattr(module, name_parts[-1]).to("cpu")
                setattr(module, name_parts[-1], modified_module.to('cuda'))
                
            rprint("injected: all llama attention k_proj layers")
        
        elif target == "planner_llama_lm_head":   
            for name, m in self.model.named_modules():   
                if isinstance(m, LlamaForCausalLM):   #only one layer
                    m.lm_head = NoisyW8A8Linear.from_float(m.lm_head, weight_quant=weight_quant,
                        act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                    print("injected:", name,"lm_head")
                    
        elif target == "planner_clip_all":
            self.inject_fault_to_module('planner_clip_attention', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('planner_clip_mlp', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
        
        elif target == "planner_clip_attention":
            mp.set_start_method('spawn', force=True)
            mp.get_context().forking_pickler = dill
            
            args = [
                (m, weight_quant, act_quant, quantize_bmm_input, err_prob)
                for name, m in self.model.named_modules()
                if isinstance(m, CLIPSdpaAttention) and self.is_target_layer(name, target_layers)
            ]
            with mp.Pool(processes=20) as pool:
                modified_modules = pool.starmap(modify_clip_attention, args)
            
            module_to_modify = [
                (name, m) for name, m in self.model.named_modules()
                if isinstance(m, CLIPSdpaAttention) and self.is_target_layer(name, target_layers)
            ]
            
            for (name, m), modified_module in zip(module_to_modify, modified_modules):
                name_parts = name.split('.')
                module = self.model
                for part in name_parts[:-1]:
                    module = getattr(module, part)
                getattr(module, name_parts[-1]).to("cpu")
                setattr(module, name_parts[-1], modified_module.to('cuda'))
            
            rprint("injected: all CLIP attention layers")
            

        elif target == "planner_clip_mlp":
            mp.set_start_method('spawn', force=True)
            mp.get_context().forking_pickler = dill
            
            args = [
                (m, weight_quant, act_quant, err_prob)
                for name, m in self.model.named_modules()
                if isinstance(m, CLIPMLP) and self.is_target_layer(name, target_layers)
            ]
            
            with mp.Pool(processes=20) as pool:
                modified_modules = pool.starmap(modify_clip_mlp, args)
            
            module_to_modify = [
                (name, m) for name, m in self.model.named_modules()
                if isinstance(m, CLIPMLP) and self.is_target_layer(name, target_layers)
            ]
            
            for (name, m), modified_module in zip(module_to_modify, modified_modules):
                name_parts = name.split('.')
                module = self.model
                for part in name_parts[:-1]:
                    module = getattr(module, part)
                getattr(module, name_parts[-1]).to("cpu")
                setattr(module, name_parts[-1], modified_module.to('cuda'))
            
            rprint("injected: all CLIP mlp layers")


        elif target == "planner_multi_model_projector":
            for name, m in self.model.named_modules():  #only one layer
                if isinstance(m, LlavaNextMultiModalProjector):
                    m.linear_1 = NoisyW8A8Linear.from_float(m.linear_1, weight_quant=weight_quant,
                        act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                    m.linear_2 = NoisyW8A8Linear.from_float(m.linear_2, weight_quant=weight_quant,
                        act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                    print("injected:", name)

            
        
        
        elif target == "controller_all":
            self.inject_fault_to_module('planner_llama_all', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('planner_clip_all', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('planner_multi_model_projector', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            '''self.inject_fault_to_module('controller_image_encoder', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('controller_temporal_encoder', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('controller_reward_head_mlp', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('controller_reward_head_video_adapter',
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)'''
        
        
        elif target == "MCPolicy_all":
            self.inject_fault_to_module('MCPolicy_CNN', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('MCPolicy_RNN', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('MCPolicy_Heads_and_Embeddings', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            
        elif target == "MCPolicy_CNN":
            self.inject_fault_to_module('MCPolicy_CNN_conv2d', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('MCPolicy_CNN_linear', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
        
        elif target == "MCPolicy_CNN_conv2d":
            for name,m in self.model.named_modules():
                if isinstance(m, ImpalaCNN):
                    m = m.stacks
                    if target_layers == []:
                        target_layers = [x for x in range(len(m))]
                    for i in target_layers: #0,1,2
                        m[i].firstconv.layer = NoisyW8A8Conv2d.from_float(
                            m[i].firstconv.layer, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                        m[i].blocks[0].conv0.layer = NoisyW8A8Conv2d.from_float(
                            m[i].blocks[0].conv0.layer, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                        m[i].blocks[0].conv1.layer = NoisyW8A8Conv2d.from_float(
                            m[i].blocks[0].conv1.layer, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                        m[i].blocks[1].conv0.layer = NoisyW8A8Conv2d.from_float(
                            m[i].blocks[1].conv0.layer, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                        m[i].blocks[1].conv1.layer = NoisyW8A8Conv2d.from_float(
                            m[i].blocks[1].conv1.layer, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                    '''for i in [0, 1, 2]:
                        m[i].firstconv.layer.register_forward_hook(print_activations)
                        m[i].blocks[0].conv0.layer.register_forward_hook(print_activations)
                        m[i].blocks[0].conv1.layer.register_forward_hook(print_activations)
                        m[i].blocks[1].conv0.layer.register_forward_hook(print_activations)
                        m[i].blocks[1].conv1.layer.register_forward_hook(print_activations)'''
                    
                        
        elif target == "MCPolicy_CNN_linear":
             for name,m in self.model.named_modules():
                if isinstance(m, ImgObsProcess): #不分层
                    m.cnn.dense.layer = NoisyW8A8Linear.from_float(
                        m.cnn.dense.layer, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)    
                    m.cnn.dense.layer.register_forward_hook(print_activations)
                    
                    m.linear.layer = NoisyW8A8Linear.from_float(
                        m.linear.layer, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                    m.linear.layer.register_forward_hook(print_activations)
        
        elif target == "MCPolicy_RNN":      
            for name, m in self.model.named_modules():
                if isinstance(m, ResidualRecurrentBlocks):
                    if target_layers == []:
                        target_layers = [x for x in range(len(m.blocks))]
                    for i in target_layers:
                        q = m.blocks[i]
                        q.mlp0.layer = NoisyW8A8Linear.from_float(q.mlp0.layer, weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                        q.mlp1.layer = NoisyW8A8Linear.from_float(q.mlp1.layer, weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                        q.r.orc_block.q_layer = NoisyW8A8Linear.from_float(q.r.orc_block.q_layer, weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                        q.r.orc_block.k_layer = NoisyW8A8Linear.from_float(q.r.orc_block.k_layer, weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                        q.r.orc_block.v_layer = NoisyW8A8Linear.from_float(q.r.orc_block.v_layer, weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                        q.r.orc_block.proj_layer = NoisyW8A8Linear.from_float(q.r.orc_block.proj_layer, weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                        q.r.orc_block.r_layer = NoisyW8A8Linear.from_float(q.r.orc_block.r_layer, weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                    '''for i in [0, 1, 2, 3]:
                        q = m.blocks[i]
                        q.mlp0.layer.register_forward_hook(print_activations)
                        q.mlp1.layer.register_forward_hook(print_activations)
                        q.r.orc_block.q_layer.register_forward_hook(print_activations)
                        q.r.orc_block.k_layer.register_forward_hook(print_activations)
                        q.r.orc_block.v_layer.register_forward_hook(print_activations)
                        q.r.orc_block.proj_layer.register_forward_hook(print_activations)
                        q.r.orc_block.r_layer.register_forward_hook(print_activations)'''
                        
        elif target == "MCPolicy_Heads_and_Embeddings":
            for name, m in self.model.named_modules():
                if isinstance(m, MinecraftAgentPolicy):     
                    m.pi_head.camera.linear_layer = NoisyW8A8Linear.from_float(m.pi_head.camera.linear_layer, weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                    m.pi_head.buttons.linear_layer = NoisyW8A8Linear.from_float(m.pi_head.buttons.linear_layer, weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                    m.value_head.linear = NoisyW8A8Linear.from_float(m.value_head.linear, weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
        
        elif target == "MCPolicy_Lastlayer_and_Mineclip":
            for name, m in self.model.named_modules():
                if isinstance(m, MinecraftAgentPolicy):     
                    m.net.lastlayer.layer = NoisyW8A8Linear.from_float(m.net.lastlayer.layer, weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                    m.net.mineclip_embed_linear = NoisyW8A8Linear.from_float(m.net.mineclip_embed_linear, weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                    
                    """m.net.lastlayer.layer.register_forward_hook(print_activations)
                    m.net.mineclip_embed_linear.register_forward_hook(print_activations)"""
        
        elif target == "VAE":
            self.model.encoder[0] = NoisyW8A8Linear.from_float(self.model.encoder[0], weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
            self.model.encoder[3] = NoisyW8A8Linear.from_float(self.model.encoder[3], weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
            self.model.encoder[6] = NoisyW8A8Linear.from_float(self.model.encoder[6], weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
            self.model.decoder[0] = NoisyW8A8Linear.from_float(self.model.decoder[0], weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
            self.model.decoder[3] = NoisyW8A8Linear.from_float(self.model.decoder[3], weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
            self.model.decoder[6] = NoisyW8A8Linear.from_float(self.model.decoder[6], weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob) 
        
        else:
            "Target not supported. Don't inject error."

        print(f'Fault injected to {target} finished.')
        #print(self.model)
        return None



        """
        elif target == "controller_image_encoder":
            for name, m in self.model.named_modules():  
                if isinstance(m, VisionTransformer):
                    if target_layers == []:
                        target_layers = [x for x in range(len(m.blocks))]
                    for i in target_layers:
                        m.blocks[i].mlp.c_fc = NoisyW8A8Linear.from_float(m.blocks[i].mlp.c_fc, weight_quant=weight_quant,
                            act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                        m.blocks[i].mlp.c_proj = NoisyW8A8Linear.from_float(m.blocks[i].mlp.c_proj, weight_quant=weight_quant,
                            act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                        print("injected:", name, "mlp")
    
        elif target == "controller_temporal_encoder":
            for name, m in self.model.named_modules():  
                if isinstance(m, TemporalPooling):
                    m = m.attn.model.attn_layers
                    if target_layers == []:
                        target_layers = [x for x in range(len(m.layers))]
                    for i in target_layers:
                        if i == 0 or i == 2:
                            m.layers[i][1].to_q = NoisyW8A8Linear.from_float(m.layers[i][1].to_q, weight_quant=weight_quant,
                                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                            m.layers[i][1].to_k = NoisyW8A8Linear.from_float(m.layers[i][1].to_k, weight_quant=weight_quant,
                                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                            m.layers[i][1].to_v = NoisyW8A8Linear.from_float(m.layers[i][1].to_v, weight_quant=weight_quant,
                                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                            m.layers[i][1].to_out = NoisyW8A8Linear.from_float(m.layers[i][1].to_out, weight_quant=weight_quant,
                                act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                            print("injected:", name, "attn")
                        elif i == 1 or i == 3:
                            m.layers[i][1].net[0].proj = NoisyW8A8Linear.from_float(m.layers[i][1].net[0].proj, weight_quant=weight_quant,
                                act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                            m.layers[i][1].net[3] = NoisyW8A8Linear.from_float(m.layers[i][1].net[3], weight_quant=weight_quant,
                                act_quant=act_quant, quantize_output=False, err_prob=err_prob)

        elif target == "controller_reward_head_mlp":
            for name, m in self.model.named_modules():  
                if isinstance(m, CLIPScoreHead):
                    m = m.clip_model.text_model
                    if target_layers == []:
                        target_layers = [x for x in range(len(m.blocks))]
                    for i in target_layers:
                        m.blocks[i].mlp.c_fc = NoisyW8A8Linear.from_float(m.blocks[i].mlp.c_fc, weight_quant=weight_quant,
                            act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                        m.blocks[i].mlp.c_proj = NoisyW8A8Linear.from_float(m.blocks[i].mlp.c_proj, weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                        print("injected:", name, "mlp")

        elif target == "controller_reward_head_video_adapter":
            for name,m in self.model.named_modules():
                if isinstance(m, CLIPScoreHead):
                    for i in range(len(m.video_adapter)):
                        if isinstance(m.video_adapter[i], Linear):
                            m.video_adapter[i] = NoisyW8A8Linear.from_float(m.video_adapter[i], weight_quant=weight_quant,
                                act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                            print("injected:", name, "video_adapter", i)
        
        """
from fault_injection.fake_quant import W16A16Linear, W16A16BMM, NoisyW16A16Linear, NoisyW16A16BMM
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaMLP, LlamaForCausalLM
from transformers.models.clip.modeling_clip import CLIPMLP, CLIPSdpaAttention
from transformers.models.llava_next.modeling_llava_next import LlavaNextMultiModalProjector
from jarvis.mineclip.clip import VisionTransformer
from jarvis.mineclip.pooling import TemporalPooling
from jarvis.mineclip.head import CLIPScoreHead
from torch.nn.modules.linear import Linear
import re
import torch.multiprocessing as mp
from rich import print as rprint
import torch
import dill

def modify_llama_attention(module, weight_quant, act_quant, quantize_bmm_input, err_prob):
    module.q_proj = NoisyW16A16Linear.from_float(module.q_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
    module.k_proj = NoisyW16A16Linear.from_float(module.k_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
    module.v_proj = NoisyW16A16Linear.from_float(module.v_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
    module.o_proj = NoisyW16A16Linear.from_float(module.o_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=False, err_prob=err_prob)
    module.to('cpu')
    return module
    
def modify_clip_attention(module, weight_quant, act_quant, quantize_bmm_input, err_prob):
    module.q_proj = NoisyW16A16Linear.from_float(module.q_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
    module.k_proj = NoisyW16A16Linear.from_float(module.k_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
    module.v_proj = NoisyW16A16Linear.from_float(module.v_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
    module.out_proj = NoisyW16A16Linear.from_float(module.out_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=False, err_prob=err_prob)
    module.to('cpu')
    return module
    
def modify_llama_mlp(module, weight_quant, act_quant, err_prob):
    module.gate_proj = NoisyW16A16Linear.from_float(module.gate_proj, weight_quant=weight_quant,
        act_quant=act_quant, quantize_output=False, err_prob=err_prob)
    module.up_proj = NoisyW16A16Linear.from_float(module.up_proj, weight_quant=weight_quant,
        act_quant=act_quant, quantize_output=False, err_prob=err_prob)
    module.down_proj = NoisyW16A16Linear.from_float(module.down_proj, weight_quant=weight_quant,
        act_quant=act_quant, quantize_output=False, err_prob=err_prob)
    module.to('cpu')
    return module
    
    
def modify_clip_mlp(module, weight_quant, act_quant, err_prob):
    module.fc1 = NoisyW16A16Linear.from_float(module.fc1, weight_quant=weight_quant,
        act_quant=act_quant, quantize_output=False, err_prob=err_prob)
    module.fc2 = NoisyW16A16Linear.from_float(module.fc2, weight_quant=weight_quant,
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
    
    def inject_fault_to_module(self, target, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=False, err_prob=0.0, target_layers=[]):
        #target_layers is used to mark which layers need to inject fault, an empty list means all layers
        ###planner part
        if target == 'planner_llama_attention':
            mp.set_start_method('spawn', force=True)
            mp.get_context().forking_pickler = dill

            args = [
                (m, weight_quant, act_quant, quantize_bmm_input, err_prob)
                for name, m in self.model.named_modules()
                if isinstance(m, LlamaSdpaAttention) and self.is_target_layer(name, target_layers)
            ]
            with mp.Pool(processes=10) as pool:
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
            with mp.Pool(processes=10) as pool:
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

        
        elif target == "planner_llama_lm_head":   
            for name, m in self.model.named_modules():   
                if isinstance(m, LlamaForCausalLM):   #only one layer
                    m.lm_head = NoisyW16A16Linear.from_float(m.lm_head, weight_quant=weight_quant,
                        act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                    print("injected:", name,"lm_head")
                    
        elif target == "planner_llama_all":
            self.inject_fault_to_module('planner_llama_attention', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('planner_llama_mlp', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('planner_llama_lm_head', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=[])
        
        elif target == "planner_clip_attention":
            mp.set_start_method('spawn', force=True)
            mp.get_context().forking_pickler = dill
            
            args = [
                (m, weight_quant, act_quant, quantize_bmm_input, err_prob)
                for name, m in self.model.named_modules()
                if isinstance(m, CLIPSdpaAttention) and self.is_target_layer(name, target_layers)
            ]
            with mp.Pool(processes=10) as pool:
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
            
            with mp.Pool(processes=10) as pool:
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


        elif target == "planner_clip_all":
            self.inject_fault_to_module('planner_clip_attention', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('planner_clip_mlp', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            
        elif target == "planner_multi_model_projector":
            for name, m in self.model.named_modules():  #only one layer
                if isinstance(m, LlavaNextMultiModalProjector):
                    m.linear_1 = NoisyW16A16Linear.from_float(m.linear_1, weight_quant=weight_quant,
                        act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                    m.linear_2 = NoisyW16A16Linear.from_float(m.linear_2, weight_quant=weight_quant,
                        act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                    print("injected:", name)

            
        elif target == "planner_all":
            self.inject_fault_to_module('planner_llama_all', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('planner_clip_all', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('planner_multi_model_projector', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
        
        ###controller part  not finished yet.
        elif target == "controller_all":
            self.inject_fault_to_module('controller_image_encoder', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('controller_temporal_encoder', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('controller_reward_head_mlp', 
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            self.inject_fault_to_module('controller_reward_head_video_adapter',
                weight_quant=weight_quant, act_quant=act_quant, quantize_bmm_input=quantize_bmm_input, err_prob=err_prob, target_layers=target_layers)
            print(self.model)
        
        elif target == "controller_image_encoder":
            for name, m in self.model.named_modules():  
                if isinstance(m, VisionTransformer):
                    if target_layers == []:
                        target_layers = [x for x in range(len(m.blocks))]
                    for i in target_layers:
                        m.blocks[i].mlp.c_fc = NoisyW16A16Linear.from_float(m.blocks[i].mlp.c_fc, weight_quant=weight_quant,
                            act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                        m.blocks[i].mlp.c_proj = NoisyW16A16Linear.from_float(m.blocks[i].mlp.c_proj, weight_quant=weight_quant,
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
                            m.layers[i][1].to_q = NoisyW16A16Linear.from_float(m.layers[i][1].to_q, weight_quant=weight_quant,
                                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                            m.layers[i][1].to_k = NoisyW16A16Linear.from_float(m.layers[i][1].to_k, weight_quant=weight_quant,
                                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                            m.layers[i][1].to_v = NoisyW16A16Linear.from_float(m.layers[i][1].to_v, weight_quant=weight_quant,
                                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                            m.layers[i][1].to_out = NoisyW16A16Linear.from_float(m.layers[i][1].to_out, weight_quant=weight_quant,
                                act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                            print("injected:", name, "attn")
                        elif i == 1 or i == 3:
                            m.layers[i][1].net[0].proj = NoisyW16A16Linear.from_float(m.layers[i][1].net[0].proj, weight_quant=weight_quant,
                                act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                            m.layers[i][1].net[3] = NoisyW16A16Linear.from_float(m.layers[i][1].net[3], weight_quant=weight_quant,
                                act_quant=act_quant, quantize_output=False, err_prob=err_prob)

        elif target == "controller_reward_head_mlp":
            for name, m in self.model.named_modules():  
                if isinstance(m, CLIPScoreHead):
                    m = m.clip_model.text_model
                    if target_layers == []:
                        target_layers = [x for x in range(len(m.blocks))]
                    for i in target_layers:
                        m.blocks[i].mlp.c_fc = NoisyW16A16Linear.from_float(m.blocks[i].mlp.c_fc, weight_quant=weight_quant,
                            act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                        m.blocks[i].mlp.c_proj = NoisyW16A16Linear.from_float(m.blocks[i].mlp.c_proj, weight_quant=weight_quant,  
                            act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                        print("injected:", name, "mlp")

        elif target == "controller_reward_head_video_adapter":
            for name,m in self.model.named_modules():
                if isinstance(m, CLIPScoreHead):
                    for i in range(len(m.video_adapter)):
                        if isinstance(m.video_adapter[i], Linear):
                            m.video_adapter[i] = NoisyW16A16Linear.from_float(m.video_adapter[i], weight_quant=weight_quant,
                                act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                            print("injected:", name, "video_adapter", i)
                              
        else:
            "Target not supported."

        print(f'Fault injected to {target} finished.')
        #print(self.model)
        return None

                   
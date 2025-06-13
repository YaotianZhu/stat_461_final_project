import collections
import pickle
from dis import dis
import os
import random
from pathlib import Path
import logging
import shutil
from packaging import version

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from param import parse_args
from pretrain_data import get_loader,load_pickle 
from utils import LossMeter
from dist_utils import reduce_dict, new_reduce_dict

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

_use_native_amp = False
_use_apex = False

_use_native_amp = True
from torch.cuda.amp import autocast

from trainer_base import TrainerBase
from pretrain_model import InstructGLM

def save_pickle(data, filename):
    # Get the directory part of the filename
    # For example, if filename is './some_dir/another_dir/file.pkl',
    # dir_name will be './some_dir/another_dir'
    dir_name = os.path.dirname(filename)
    
    # If dir_name is not empty (meaning the filename includes a directory path)
    # and the directory does not already exist, create it.
    if dir_name and not os.path.exists(dir_name):
        # os.makedirs will create all necessary parent directories in the path.
        # exist_ok=True means it won't raise an error if the directory already exists
        # (e.g., if another process/rank created it just moments before).
        os.makedirs(dir_name, exist_ok=True)
        print(f"DEBUG: Created directory: {dir_name}", flush=True) # Optional: for logging
        
    # Now, proceed to save the file as before
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

class FP(nn.Module):
    def __init__(self,llama_embed,real):
        super(FP,self).__init__()
        self.trans_1=nn.Linear(1433,512,bias=False)  
        self.trans_2=nn.Linear(512,4096,bias=False)
        self.rac=nn.ELU()
        self.sln=nn.LayerNorm(512)

        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal_(mean=0.0, std=1.0)
        self.embed_tokens=llama_embed
        self.real_feature=real
        

    def forward(self, input_ids):
        transfered=self.trans_2(self.rac(self.sln(self.trans_1(self.real_feature)))) 

        inputs_embeds = transfered[input_ids] + self.embed_tokens[input_ids] ### embedding step - add HERE ###

        return inputs_embeds


# The Trainer inherits TrainerBase in trainer_base.py
class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True, val_list=None):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)   


        model_class = InstructGLM
        self.m_class = InstructGLM   

        config = self.create_config()
        self.m_config=config     
        self.tokenizer = self.create_tokenizer()

        re_start=0
        if train:  
            self.model = self.create_model(model_class, config)
            self.model.tokenizer = self.tokenizer  

            self.model = prepare_model_for_int8_training(self.model)

            lora_r=16 
            lora_alpha=16
            lora_target_modules=['q_proj','k_proj','v_proj','o_proj','lm_head']
            lora_dropout=0.05

            LORA_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=lora_target_modules,lora_dropout=lora_dropout,bias="none",task_type="CAUSAL_LM")
            if re_start!=2:
                self.model = get_peft_model(self.model, LORA_config)

            if self.verbose and re_start!=2:
                print()
                self.model.print_trainable_parameters()
                print()
            dist.barrier()
        
            if re_start==1:
                print('Main model re-starting')
                doc_prefix='./your_folder_path/'
                for gg in range(32):
                    self.model.base_model.model.model.layers[gg].self_attn.q_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Cora_qa_{}_{}_{}.pkl".format(gg,self.args.lr,self.args.gradient_accumulation_steps)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.k_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Cora_ka_{}_{}_{}.pkl".format(gg,self.args.lr,self.args.gradient_accumulation_steps)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.v_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Cora_va_{}_{}_{}.pkl".format(gg,self.args.lr,self.args.gradient_accumulation_steps)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.o_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Cora_oa_{}_{}_{}.pkl".format(gg,self.args.lr,self.args.gradient_accumulation_steps)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.q_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Cora_qb_{}_{}_{}.pkl".format(gg,self.args.lr,self.args.gradient_accumulation_steps)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.k_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Cora_kb_{}_{}_{}.pkl".format(gg,self.args.lr,self.args.gradient_accumulation_steps)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.v_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Cora_vb_{}_{}_{}.pkl".format(gg,self.args.lr,self.args.gradient_accumulation_steps)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.o_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Cora_ob_{}_{}_{}.pkl".format(gg,self.args.lr,self.args.gradient_accumulation_steps)).to(args.gpu)
                self.model.base_model.model.lm_head.lora_A.default.weight.data=load_pickle(doc_prefix+"Cora_lm_a_{}_{}.pkl".format(self.args.lr,self.args.gradient_accumulation_steps)).data.to(args.gpu)
                self.model.base_model.model.lm_head.lora_B.default.weight.data=load_pickle(doc_prefix+"Cora_lm_b_{}_{}.pkl".format(self.args.lr,self.args.gradient_accumulation_steps)).data.to(args.gpu)
                print('Main model loaded.')
                if self.verbose:
                    self.model.save_pretrained("Cora_restart")

            if re_start==2:
                from peft import PeftModel, PeftConfig
                peft_model_id ='./Cora_restart'
                print('now we are loading peft model')

                self.model = PeftModel.from_pretrained(self.model, peft_model_id)
                for n,p in self.model.named_parameters():
                    if 'lora' in n:
                        p.requires_grad_()
                if self.verbose:
                    print()
                    self.model.print_trainable_parameters()
                    print()
                
            self.model = self.model.to(args.gpu)




        else:
            self.model = self.create_model(model_class, config)
            self.model.tokenizer = self.tokenizer  
            if True:
                for name, param in self.model.named_parameters():

                    param.requires_grad = False

                for param in self.model.parameters():
                    if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                        param.data = param.data.to(torch.float32)

            lora_r=16 
            lora_alpha=16
            lora_target_modules=['q_proj','k_proj','v_proj','o_proj','lm_head']
            lora_dropout=0.05

            LORA_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=lora_target_modules,lora_dropout=lora_dropout,bias="none",task_type="CAUSAL_LM")
            self.model = get_peft_model(self.model, LORA_config)

            if self.verbose:
                print()
                self.model.print_trainable_parameters()
                print()
            dist.barrier()
            self.model = self.model.to(args.gpu)

        if train:
            self.first_model=FP(llama_embed=self.train_loader.dataset.llama_embed.to(args.gpu),real=self.train_loader.dataset.real_feature.to(args.gpu))
        else:
            self.first_model=FP(llama_embed=self.val_loader.dataset.llama_embed.to(args.gpu),real=self.val_loader.dataset.real_feature.to(args.gpu))
        
        re_start=0
        if train and re_start==1:
            print('All processes re-starting first-model')
            ckpt_path="yours.pth"
        
            self.load_checkpoint(ckpt_path)

            print('first_model loaded.')

        if train and re_start==2:   
            self.first_model.sln.bias.data=load_pickle('./Cora_first_restart/sln_bias.pkl')
            self.first_model.sln.weight.data=load_pickle('./Cora_first_restart/sln_weight.pkl')
            self.first_model.trans_1.weight.data=load_pickle('./Cora_first_restart/trans_1_weight.pkl')
            self.first_model.trans_2.weight.data=load_pickle('./Cora_first_restart/trans_2_weight.pkl')

        self.first_model=self.first_model.to(args.gpu)

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()


        if args.multiGPU and not args.inference:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()    

            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu])
                self.first_model=DDP(self.first_model, device_ids=[args.gpu])
                
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

        self.val_list=val_list

    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME

        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            best_eval_loss = 100000.

            project_name = "Natural Language is All a Graph Needs"

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)

        if self.args.distributed:
            dist.barrier()

        for epoch in range(self.args.epoch):
            global_step=0

            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)    #keep in mind this

            # Train
            self.model.train()   
            self.first_model.train()

            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=275)

            epoch_results = {}   
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for step_i, batch in enumerate(self.train_loader):   
                torch.cuda.empty_cache()
                dist.barrier()

                if self.args.fp16 and _use_native_amp:
                    pass
                else:
                    if self.args.distributed:
                        
                        dddd = next(self.model.parameters()).device

                        input_ids = batch['input_ids'].to(dddd)
                        lm_labels = batch["target_ids"].to(dddd)
                        attention_mask=batch['attn_mask'].to(dddd)

                        loss_weights = batch["loss_weights"].to(dddd)
                        B, L = lm_labels.size()

                        embeds = self.first_model(  # forward
                            input_ids=input_ids
                        )
                        output=self.model(inputs_embeds=embeds,attention_mask=attention_mask,labels=lm_labels)

                        lm_mask = lm_labels[:,1:] != -100
                        lm_mask = lm_mask.float()

                        loss = output['loss']

                        loss = loss.view(B, L-1) * lm_mask   

                        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)   

                        results = {}    

                        results['loss'] = (loss * loss_weights).mean()    
                        results['total_loss'] = loss.detach().sum()
                        results['total_loss_count'] = len(loss)

                        task_counts = {task: 0 for task in self.model.module.losses}
                        task_loss = {task: 0 for task in self.model.module.losses}

                        for _loss, task in zip(loss.detach(), batch['task']):
                            task_loss[task] += _loss
                            task_counts[task] += 1

                        for task in self.model.module.losses:
                            if task_counts[task] > 0:
                                results[f'{task}_loss'] = task_loss[task]
                                results[f'{task}_loss_count'] = task_counts[task]

                loss = results['loss']/self.args.gradient_accumulation_steps
                dist.barrier()
                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss = loss.detach()

                # Update Parameters
                if step_i % self.args.gradient_accumulation_steps==0:
                    #
                    if self.args.clip_grad_norm > 0:
                        if self.args.fp16 and _use_native_amp:
                            self.scaler.unscale_(self.optim)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                        elif self.args.fp16 and _use_apex:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.args.clip_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                            torch.nn.utils.clip_grad_norm_(self.first_model.parameters(), self.args.clip_grad_norm)

                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()    # Update

                    if self.lr_scheduler:
                        self.lr_scheduler.step()
 
                    for param in self.model.parameters():    
                        param.grad = None
                    for param in self.first_model.parameters():    
                        param.grad = None

                global_step += 1
            
                if global_step==len(self.train_loader)//8:
                    torch.cuda.empty_cache()
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mmid1/Cora_qa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mmid1/Cora_ka_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mmid1/Cora_va_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mmid1/Cora_oa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mmid1/Cora_qb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mmid1/Cora_kb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mmid1/Cora_vb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mmid1/Cora_ob_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mmid1/Cora_lm_a_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mmid1/Cora_lm_b_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))

                        torch.save(self.first_model.state_dict(),"Cora_first_{}_{}_8_{}_{}_mmid1.pth".format(epoch+1,self.args.lr,self.args.train,self.args.gradient_accumulation_steps))
                if global_step==len(self.train_loader)//4:
                    torch.cuda.empty_cache()
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mid1/Cora_qa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mid1/Cora_ka_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mid1/Cora_va_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mid1/Cora_oa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mid1/Cora_qb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mid1/Cora_kb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mid1/Cora_vb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mid1/Cora_ob_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mid1/Cora_lm_a_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mid1/Cora_lm_b_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))

                        torch.save(self.first_model.state_dict(),"Cora_first_{}_{}_8_{}_{}_mid1.pth".format(epoch+1,self.args.lr,self.args.train,self.args.gradient_accumulation_steps))
                if global_step==len(self.train_loader)*3//8:
                    torch.cuda.empty_cache()
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mmid2/Cora_qa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mmid2/Cora_ka_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mmid2/Cora_va_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mmid2/Cora_oa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mmid2/Cora_qb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mmid2/Cora_kb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mmid2/Cora_vb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mmid2/Cora_ob_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mmid2/Cora_lm_a_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mmid2/Cora_lm_b_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))

                        torch.save(self.first_model.state_dict(),"Cora_first_{}_{}_8_{}_{}_mmid2.pth".format(epoch+1,self.args.lr,self.args.train,self.args.gradient_accumulation_steps))
                if global_step==len(self.train_loader)//2:
                    torch.cuda.empty_cache()
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mid2/Cora_qa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mid2/Cora_ka_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mid2/Cora_va_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mid2/Cora_oa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mid2/Cora_qb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mid2/Cora_kb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mid2/Cora_vb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mid2/Cora_ob_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mid2/Cora_lm_a_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mid2/Cora_lm_b_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))

                        torch.save(self.first_model.state_dict(),"Cora_first_{}_{}_8_{}_{}_mid2.pth".format(epoch+1,self.args.lr,self.args.train,self.args.gradient_accumulation_steps))
                if global_step==len(self.train_loader)*5//8:
                    torch.cuda.empty_cache()
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mmid3/Cora_qa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mmid3/Cora_ka_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mmid3/Cora_va_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mmid3/Cora_oa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mmid3/Cora_qb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mmid3/Cora_kb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mmid3/Cora_vb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mmid3/Cora_ob_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mmid3/Cora_lm_a_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mmid3/Cora_lm_b_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))

                        torch.save(self.first_model.state_dict(),"Cora_first_{}_{}_8_{}_{}_mmid3.pth".format(epoch+1,self.args.lr,self.args.train,self.args.gradient_accumulation_steps))
                if global_step==len(self.train_loader)*3//4:
                    torch.cuda.empty_cache()
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mid3/Cora_qa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mid3/Cora_ka_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mid3/Cora_va_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mid3/Cora_oa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mid3/Cora_qb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mid3/Cora_kb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mid3/Cora_vb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mid3/Cora_ob_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mid3/Cora_lm_a_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mid3/Cora_lm_b_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))

                        torch.save(self.first_model.state_dict(),"Cora_first_{}_{}_8_{}_{}_mid3.pth".format(epoch+1,self.args.lr,self.args.train,self.args.gradient_accumulation_steps))
                if global_step==len(self.train_loader)*7//8:
                    torch.cuda.empty_cache()
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mend/Cora_qa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mend/Cora_ka_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mend/Cora_va_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mend/Cora_oa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mend/Cora_qb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mend/Cora_kb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mend/Cora_vb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mend/Cora_ob_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mend/Cora_lm_a_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mend/Cora_lm_b_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))

                        torch.save(self.first_model.state_dict(),"Cora_first_{}_{}_8_{}_{}_mend.pth".format(epoch+1,self.args.lr,self.args.train,self.args.gradient_accumulation_steps))

                dist.barrier()

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    lr=self.optim.param_groups[-1]['lr']

                for k, v in results.items():    
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 1==0:     
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'    

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:  
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']    
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)
                dist.barrier()

            if self.verbose:
                pbar.close()

            dist.barrier()

            results = reduce_dict(epoch_results,average=False)    # For global info

            dist.barrier()


            if self.verbose:
                train_loss = results['total_loss']
                train_loss_count = results['total_loss_count']

                avg_train_loss = train_loss / train_loss_count
                losses_str = f"Train Loss: {avg_train_loss:.3f}\n"

                for name, loss in results.items():
                    if name[-4:] == 'loss':
                        loss_count = int(results[name+'_count'])
                        if loss_count > 0:
                            avg_loss = loss/loss_count
                            losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                losses_str += '\n'
                print(losses_str)      

            dist.barrier()
            
            torch.cuda.empty_cache()
            if self.verbose:
                for ig in range(32):
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_end/Cora_qa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_end/Cora_ka_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_end/Cora_va_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_end/Cora_oa_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_end/Cora_qb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_end/Cora_kb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_end/Cora_vb_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_end/Cora_ob_{}_{}_{}.pkl".format(epoch+1,ig,self.args.lr,self.args.gradient_accumulation_steps))
                save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_end/Cora_lm_a_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))
                save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_end/Cora_lm_b_{}_{}.pkl".format(epoch+1,self.args.lr,self.args.gradient_accumulation_steps))

                torch.save(self.first_model.state_dict(),"Cora_first_{}_{}_8_{}_{}_end.pth".format(epoch+1,self.args.lr,self.args.train,self.args.gradient_accumulation_steps))
            dist.barrier()
            

    def test(self):
        # 假设 self.args.distributed 和 self.verbose 已经正确设置
        # 并且 torch.distributed (dist) 在分布式模式下会被正确导入和初始化
        if self.args.distributed:
            import torch.distributed as dist

        # Define the base path to snap/cora-7b relative to current script location
        script_dir = Path(__file__).resolve().parent  # This is .../InstructGLM/llama_cora_src/
        cora_checkpoint_base = (script_dir / "../snap/cora-7b/").resolve()

        for epoch_idx_in_loop in range(8 * self.args.epoch): # 使用不同的变量名避免与类/实例的 epoch 混淆
            # 1. 确定当前的 subfolder_name 和 pth_filename (主模型文件名)
            current_outer_epoch = epoch_idx_in_loop // 8 + 1
            remainder = (epoch_idx_in_loop + 1) % 8

            if remainder == 1:
                subfolder_name = 'llama_{}_mmid1'.format(current_outer_epoch)
                pth_filename = "Cora_first_{}_{}_8_{}_{}_mmid1.pth".format(
                    current_outer_epoch, self.args.lr, self.args.train, self.args.gradient_accumulation_steps)
            elif remainder == 2:
                subfolder_name = 'llama_{}_mid1'.format(current_outer_epoch)
                pth_filename = "Cora_first_{}_{}_8_{}_{}_mid1.pth".format(
                    current_outer_epoch, self.args.lr, self.args.train, self.args.gradient_accumulation_steps)
            elif remainder == 3:
                subfolder_name = 'llama_{}_mmid2'.format(current_outer_epoch)
                pth_filename = "Cora_first_{}_{}_8_{}_{}_mmid2.pth".format(
                    current_outer_epoch, self.args.lr, self.args.train, self.args.gradient_accumulation_steps)
            elif remainder == 4:
                subfolder_name = 'llama_{}_mid2'.format(current_outer_epoch)
                pth_filename = "Cora_first_{}_{}_8_{}_{}_mid2.pth".format(
                    current_outer_epoch, self.args.lr, self.args.train, self.args.gradient_accumulation_steps)
            elif remainder == 5:
                subfolder_name = 'llama_{}_mmid3'.format(current_outer_epoch)
                pth_filename = "Cora_first_{}_{}_8_{}_{}_mmid3.pth".format(
                    current_outer_epoch, self.args.lr, self.args.train, self.args.gradient_accumulation_steps)
            elif remainder == 6:
                subfolder_name = 'llama_{}_mid3'.format(current_outer_epoch)
                pth_filename = "Cora_first_{}_{}_8_{}_{}_mid3.pth".format(
                    current_outer_epoch, self.args.lr, self.args.train, self.args.gradient_accumulation_steps)
            elif remainder == 7:
                subfolder_name = 'llama_{}_mend'.format(current_outer_epoch)
                pth_filename = "Cora_first_{}_{}_8_{}_{}_mend.pth".format(
                    current_outer_epoch, self.args.lr, self.args.train, self.args.gradient_accumulation_steps)
            else:  # remainder == 0
                subfolder_name = 'llama_{}_end'.format(current_outer_epoch)
                pth_filename = "Cora_first_{}_{}_8_{}_{}_end.pth".format(
                    current_outer_epoch, self.args.lr, self.args.train, self.args.gradient_accumulation_steps)

            # 2. 构建 .pth 主模型检查点的完整路径
            checkpoint_subfolder = cora_checkpoint_base / subfolder_name
            full_pth_checkpoint_path = checkpoint_subfolder / pth_filename

            if self.verbose:
                print(f"Loop iteration: {epoch_idx_in_loop + 1}, Outer epoch: {current_outer_epoch}, Subfolder: {subfolder_name}")
                print(f"Attempting to load .pth checkpoint from: {full_pth_checkpoint_path}")
            
            self.load_checkpoint(str(full_pth_checkpoint_path))
            # 假设 load_checkpoint 加载到 self.model 或 self.first_model
            # 原始代码: self.first_model=self.first_model.to(self.args.gpu)
            # 如果 self.load_checkpoint 加载的是 self.model 的基础权重, 那么 self.model 稍后会被移到GPU
            # 如果它加载的是 self.first_model, 并且 self.first_model 与 self.model 不同, 则需要小心处理
            # 为保持与原意接近，如果 self.first_model 被 load_checkpoint 修改:
            if hasattr(self, 'first_model') and self.first_model is not None: # 确保 first_model 存在
                self.first_model = self.first_model.to(self.args.gpu)


            # 3. 构建 .pkl LoRA 权重文件的基础路径
            pickle_base_path = checkpoint_subfolder

            # 加载 LoRA 权重到 self.model
            for gg in range(32): # 假设有32层
                lora_A_filenames = {
                    "q": "Cora_qa_{}_{}_{}.pkl".format(gg, self.args.lr, self.args.gradient_accumulation_steps),
                    "k": "Cora_ka_{}_{}_{}.pkl".format(gg, self.args.lr, self.args.gradient_accumulation_steps),
                    "v": "Cora_va_{}_{}_{}.pkl".format(gg, self.args.lr, self.args.gradient_accumulation_steps),
                    "o": "Cora_oa_{}_{}_{}.pkl".format(gg, self.args.lr, self.args.gradient_accumulation_steps),
                }
                lora_B_filenames = {
                    "q": "Cora_qb_{}_{}_{}.pkl".format(gg, self.args.lr, self.args.gradient_accumulation_steps),
                    "k": "Cora_kb_{}_{}_{}.pkl".format(gg, self.args.lr, self.args.gradient_accumulation_steps),
                    "v": "Cora_vb_{}_{}_{}.pkl".format(gg, self.args.lr, self.args.gradient_accumulation_steps),
                    "o": "Cora_ob_{}_{}_{}.pkl".format(gg, self.args.lr, self.args.gradient_accumulation_steps),
                }

                layer = self.model.base_model.model.model.layers[gg].self_attn
                for proj_type in ["q", "k", "v", "o"]:
                    proj_layer_A = getattr(layer, f"{proj_type}_proj").lora_A.default
                    proj_layer_B = getattr(layer, f"{proj_type}_proj").lora_B.default
                    
                    proj_layer_A.weight.data = load_pickle(str(pickle_base_path / lora_A_filenames[proj_type])).to(self.args.gpu)
                    proj_layer_B.weight.data = load_pickle(str(pickle_base_path / lora_B_filenames[proj_type])).to(self.args.gpu)

            # 加载 lm_head 的 LoRA 权重
            lm_a_filename = "Cora_lm_a_{}_{}.pkl".format(self.args.lr, self.args.gradient_accumulation_steps)
            lm_b_filename = "Cora_lm_b_{}_{}.pkl".format(self.args.lr, self.args.gradient_accumulation_steps)

            self.model.base_model.model.lm_head.lora_A.default.weight.data = \
                load_pickle(str(pickle_base_path / lm_a_filename)).data.to(self.args.gpu)
            self.model.base_model.model.lm_head.lora_B.default.weight.data = \
                load_pickle(str(pickle_base_path / lm_b_filename)).data.to(self.args.gpu)

            if self.verbose:
                print('Main model checkpoint and LoRA adapter weights loaded.')
            
            # 将整个（可能已修改的）模型移到 GPU
            self.model = self.model.to(self.args.gpu)
            
            # 5. 执行评估
            valid_results = self.evaluate_epoch()
            if self.args.distributed:
                dist.barrier()
                valid_results = new_reduce_dict(valid_results) # 确保 new_reduce_dict 已定义并可用
                dist.barrier()

            # 6. 打印和记录结果
            if self.verbose:
                print("\nValidation Results:")
                for kk in valid_results.keys():
                    if kk.endswith('transductive'):
                        if self.args.train == 'Cora':
                            if hasattr(self.val_loader.dataset, 'len_transductive') and self.val_loader.dataset.len_transductive > 0:
                                valid_results[kk] = valid_results[kk].item() / self.val_loader.dataset.len_transductive
                            else:
                                print(f"Warning: len_transductive not available or zero for {kk}")
                                valid_results[kk] = 0.0 # 或者其他合适的错误处理
                    # F1 scores are already in the correct format (float), no need to normalize
                print("Accuracy and Macro-F1 Results:")
                print(valid_results)
                print("\n")

            if self.args.distributed:
                dist.barrier() # 在写文件前同步，确保 rank 0 有最终结果

            # 只在主进程 (rank 0) 中写入文件，以避免多进程同时写入导致文件损坏
            if self.verbose: # self.verbose 通常等价于 self.args.gpu == 0 或 self.args.local_rank == 0
                accuracy_log_path = 'Cora_7b_eval.txt' # 更新日志文件名以反映从snap加载
                
                with open(accuracy_log_path, 'a') as acc_file:
                    suffix_map = {
                        1: '_mmid1', 2: '_mid1', 3: '_mmid2', 4: '_mid2',
                        5: '_mmid3', 6: '_mid3', 7: '_mend', 0: '_end' # remainder 0 for the 'else' case
                    }
                    log_suffix = suffix_map.get(remainder, f'_unknown_rem{remainder}') # 获取后缀
                    
                    acc_file.write(str(current_outer_epoch) + log_suffix + '\n')
                    acc_file.write(str(valid_results) + '\n\n')
            
            if self.args.distributed:
                dist.barrier()


    def evaluate_epoch(self):   
        ACC={}
        # Add data structures for F1 calculation
        predictions_by_template = {}  # Store predictions for each template
        true_labels_by_template = {}  # Store true labels for each template
        
        for k in list(self.val_list.keys()):
            if k=='classification':
                if self.args.train=='Cora':
                    templates=[]
                    for tems in self.val_list[k]:
                        templates=templates+tems
                    for thing in templates:
                        ACC[thing+'-'+'transductive']=0
                        # Initialize prediction and label storage for each template
                        predictions_by_template[thing+'-'+'transductive'] = []
                        true_labels_by_template[thing+'-'+'transductive'] = []
            elif k=='link':
                pass

        self.first_model.eval()
        self.model.eval()
        with torch.no_grad():
            for step_i, batch in tqdm(enumerate(self.val_loader)):   
                torch.cuda.empty_cache()

                if self.args.distributed:
                    
                    device = next(self.model.parameters()).device
                    input_ids = batch['input_ids'].to(device)
                    embeds = self.first_model(input_ids=input_ids)
                    attention_mask=batch['attn_mask'].to(device)
                    results = self.model.g_step(in_embeds=embeds, attention_mask=attention_mask)

                for iiid in range(len(results)):    
                    task=batch['task'][iiid]
                    temp_id=batch['temp_ids'][iiid]
                    if task=='classification':
                        
                        cate=batch['cate'][iiid] 
                        if temp_id.endswith('2') or temp_id.endswith('4') or temp_id.endswith('6') or temp_id.endswith('7'):  
                            template_key = temp_id+'-'+cate
                            predicted_label = results[iiid].lower()
                            true_label = batch['target_text'][iiid]
                            
                            # Store predictions and true labels for F1 calculation
                            if template_key in predictions_by_template:
                                predictions_by_template[template_key].append(predicted_label)
                                true_labels_by_template[template_key].append(true_label)
                            
                            if predicted_label == true_label:
                                ACC[template_key]+=1

                    elif task=='link':
                        pass

                dist.barrier()

            # Calculate Macro-F1 for each template
            F1_SCORES = {}
            for template_key in predictions_by_template.keys():
                if len(predictions_by_template[template_key]) > 0:
                    f1_score = self.calculate_macro_f1(
                        true_labels_by_template[template_key], 
                        predictions_by_template[template_key]
                    )
                    F1_SCORES[template_key.replace('transductive', 'macro_f1')] = f1_score
            
            # Merge ACC and F1_SCORES for return
            result_dict = ACC.copy()
            result_dict.update(F1_SCORES)
            
            return result_dict   

    def calculate_macro_f1(self, true_labels, predictions):
        """
        Calculate Macro-F1 score for multi-class classification.
        
        Args:
            true_labels: List of true labels
            predictions: List of predicted labels
            
        Returns:
            macro_f1: Macro-averaged F1 score
        """
        from collections import defaultdict, Counter
        
        # Get all unique labels
        all_labels = list(set(true_labels + predictions))
        
        if len(all_labels) == 0:
            return 0.0
        
        f1_scores = []
        
        for label in all_labels:
            # Calculate TP, FP, FN for this label
            tp = sum(1 for true, pred in zip(true_labels, predictions) if true == label and pred == label)
            fp = sum(1 for true, pred in zip(true_labels, predictions) if true != label and pred == label)
            fn = sum(1 for true, pred in zip(true_labels, predictions) if true == label and pred != label)
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Calculate F1 score for this label
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
                
            f1_scores.append(f1)
        
        # Return macro-averaged F1 score
        macro_f1 = sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0.0
        return macro_f1


def main_worker(gpu, args):     # the gpu is the local_rank in DDP 
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    # Training Console
    if not args.inference:
        print(f'Building train loader at GPU {gpu}')
        if args.train == 'Cora':
            train_task_list = {
            'link':[['1-1-1-1','1-3-1-1'],['1-1-2-1','1-1-2-3','1-3-2-1','1-3-2-3'],['1-1-3-1','1-1-3-3','1-3-3-1','1-3-3-3']],
            'classification':[['6-6-6-6','6-6-6-7'],['2-3-1-2','2-1-1-2'],['2-3-2-2','2-1-2-2','2-3-2-4','2-1-2-4'],['2-3-3-2','2-1-3-2','2-3-3-4','2-1-3-4']]
            }

        train_sample_numbers = {}   # Abandoned

        train_loader = get_loader(
            args,
            train_task_list,
            train_sample_numbers,
            split=args.train, 
            mode='train',
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=args.distributed
        )

        if args.gpu==0:
            print('Length of train dataset:', len(train_loader.dataset))
        trainer = Trainer(args,train_loader= train_loader,  train=True)   
        trainer.train()

    # Inference/valid Console
    if args.inference:
        print(f'Building val loader at GPU {gpu}')
        if args.valid == 'Cora':
            val_task_list = {
            'classification':[['6-6-6-6','6-6-6-7'],['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4'],['2-1-3-2','2-1-3-4','2-3-3-2','2-3-3-4'],['2-1-1-2','2-3-1-2']]
            }

        val_sample_numbers = {} # Abandoned
        val_loader = get_loader(
            args,
            val_task_list,
            val_sample_numbers,
            split=args.valid, 
            mode='val',
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=args.distributed
        )

        if args.gpu==0:
            print('Length of test dataset:', len(val_loader.dataset))

        trainer = Trainer(args, val_loader= val_loader, train=False, val_list=val_task_list)   
        trainer.test()


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    if args.local_rank in [0, -1]:
        print(args)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
    if args.local_rank in [0, -1]:
        print(LOSSES_NAME)
    LOSSES_NAME.append('total_loss')

    args.LOSSES_NAME = LOSSES_NAME

    comments = []
    dsets = []

    if 'Cora' in args.train:
        dsets.append('Cora')

    comments.append(''.join(dsets))
    comments.append(args.backbone)
    comments.append(''.join(args.losses.split(',')))
    if args.comment != '':
        comments.append(args.comment)
    comment = '_'.join(comments)

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M')

    project_dir = Path(__file__).resolve().parent.parent

    if args.local_rank in [0, -1]:
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'
        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)

import io
import logging
import math
import os
import pprint
import sys
import time
import json
import pdb 
from tqdm import tqdm
from datetime import datetime
import transformers
import torch
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM  
from trl import DPOTrainer, DPOConfig
import torch.multiprocessing
from dataset_lm.reindent import run as run_reindent
from datasets import Dataset

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3'
from typing import List, Dict

from typing import List, Dict
import torch
import torch.nn.functional as F

import torch
def dpo_with_skel_collate(
    samples,
    tokenizer,
    *,
    max_concat_len: int = 2048,   
    max_prompt_len: int = 1024,    
    max_code_len: int = 2048,     
    max_skel_len: int = 2048,     
    min_resp_tok: int = 128     
):
    def S(s, k, d=""):
        v = s.get(k, d)
        return v if isinstance(v, str) else ("" if v is None else str(v))
    #print(samples)
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "tokenizer.pad_token_id not None"

    code_prompts   = [S(s, "code_prompt", S(s, "prompt", "")) for s in samples]
    code_chosen    = [S(s, "code_chosen", "")   for s in samples]
    code_rejected  = [S(s, "code_rejected", "") for s in samples]
    prompts_tok   = [tokenizer.encode(t or "", add_special_tokens=False) for t in code_prompts]
    chosen_tok    = [tokenizer.encode(t or "", add_special_tokens=False) for t in code_chosen]
    rejected_tok  = [tokenizer.encode(t or "", add_special_tokens=False) for t in code_rejected]

    B = len(samples)
    ch_full_ids, ch_full_mask = [], []
    rj_full_ids, rj_full_mask = [], []
    p_ids, p_mask = [], []

    for i in range(B):
        p_raw = torch.tensor(prompts_tok[i],  dtype=torch.long)
        ch_r  = torch.tensor(chosen_tok[i],   dtype=torch.long)
        rj_r  = torch.tensor(rejected_tok[i], dtype=torch.long)

        p_len, ch_len, rj_len = int(p_raw.size(0)), int(ch_r.size(0)), int(rj_r.size(0))

        ch_floor = min(max(min_resp_tok, 1), ch_len, max_code_len - 1)
        rj_floor = min(max(min_resp_tok, 1), rj_len, max_code_len - 1)

        p_cap_ch = min(max_code_len - ch_floor, max_prompt_len)
        p_cap_rj = min(max_code_len - rj_floor, max_prompt_len)

        use_p_ch = min(p_len, p_cap_ch)
        use_p_rj = min(p_len, p_cap_rj)
        use_ch   = min(ch_len, max_code_len - use_p_ch)
        use_rj   = min(rj_len, max_code_len - use_p_rj)

        p_keep_ch = p_raw[-use_p_ch:] if use_p_ch > 0 else p_raw[:0]
        p_keep_rj = p_raw[-use_p_rj:] if use_p_rj > 0 else p_raw[:0]
        ch_keep   = ch_r[:use_ch]      if use_ch   > 0 else ch_r[:0]
        rj_keep   = rj_r[:use_rj]      if use_rj   > 0 else rj_r[:0]

        cur = torch.cat([p_keep_ch, ch_keep], dim=0)
        if cur.size(0) < max_code_len:
            cur = torch.cat([cur, cur.new_full((max_code_len - cur.size(0),), pad_id)], dim=0)
        m = torch.zeros_like(cur); m[:use_p_ch + use_ch] = 1
        ch_full_ids.append(cur); ch_full_mask.append(m)

        cur = torch.cat([p_keep_rj, rj_keep], dim=0)
        if cur.size(0) < max_code_len:
            cur = torch.cat([cur, cur.new_full((max_code_len - cur.size(0),), pad_id)], dim=0)
        m = torch.zeros_like(cur); m[:use_p_rj + use_rj] = 1
        rj_full_ids.append(cur); rj_full_mask.append(m)

        p_only = p_raw[-min(p_len, max_prompt_len):] if p_len > 0 else p_raw[:0]
        if p_only.size(0) < max_prompt_len:
            p_only = torch.cat([p_only, p_only.new_full((max_prompt_len - p_only.size(0),), pad_id)], dim=0)
        pm = torch.zeros_like(p_only); pm[:min(p_len, max_prompt_len)] = 1
        p_ids.append(p_only); p_mask.append(pm)

    ch_full_ids = torch.stack(ch_full_ids).to(torch.long)
    ch_full_mask = torch.stack(ch_full_mask).to(torch.long)
    rj_full_ids = torch.stack(rj_full_ids).to(torch.long)
    rj_full_mask = torch.stack(rj_full_mask).to(torch.long)
    p_ids = torch.stack(p_ids).to(torch.long)
    p_mask = torch.stack(p_mask).to(torch.long)

    skel_pos_full = [S(s, "skel_prompt", "") + S(s, "skel_chosen", "")   for s in samples]
    skel_neg_full = [S(s, "skel_prompt", "") + S(s, "skel_rejected", "") for s in samples]

    sk_pos_tok = tokenizer(skel_pos_full, return_tensors="pt", padding=True, truncation=True, max_length=max_skel_len)
    sk_neg_tok = tokenizer(skel_neg_full, return_tensors="pt", padding=True, truncation=True, max_length=max_skel_len)

    sk_pos_masks, sk_neg_masks = [], []
    for s, ids_pos, ids_neg in zip(samples, sk_pos_tok["input_ids"], sk_neg_tok["input_ids"]):
        sp  = S(s, "skel_prompt", "")
        sch = S(s, "skel_chosen", "")
        srj = S(s, "skel_rejected", "")

        sp_len  = len(tokenizer(sp,  add_special_tokens=False)["input_ids"])
        sch_len = len(tokenizer(sch, add_special_tokens=False)["input_ids"])
        srj_len = len(tokenizer(srj, add_special_tokens=False)["input_ids"])

        m_pos = torch.zeros_like(ids_pos, dtype=torch.float)
        end_p = min(sp_len + sch_len, ids_pos.size(0))
        if end_p > sp_len:
            m_pos[sp_len:end_p] = 1.0
        sk_pos_masks.append(m_pos)

        m_neg = torch.zeros_like(ids_neg, dtype=torch.float)
        end_n = min(sp_len + srj_len, ids_neg.size(0))
        if end_n > sp_len:
            m_neg[sp_len:end_n] = 1.0
        sk_neg_masks.append(m_neg)

    batch = {
        "prompt_input_ids":        p_ids,
        "prompt_attention_mask":   p_mask,
        "chosen_input_ids":        ch_full_ids,
        "chosen_attention_mask":   ch_full_mask,
        "rejected_input_ids":      rj_full_ids,
        "rejected_attention_mask": rj_full_mask,

        "skel_chosen_input_ids":        sk_pos_tok["input_ids"].to(torch.long),
        "skel_chosen_attention_mask":   sk_pos_tok["attention_mask"].to(torch.long),
        "skel_chosen_response_mask":    torch.stack(sk_pos_masks).to(torch.float),
        "skel_rejected_input_ids":      sk_neg_tok["input_ids"].to(torch.long),
        "skel_rejected_attention_mask": sk_neg_tok["attention_mask"].to(torch.long),
        "skel_rejected_response_mask":  torch.stack(sk_neg_masks).to(torch.float),
    }

    with torch.no_grad():
        eq_rows = (ch_full_ids == rj_full_ids).all(dim=1)
        ch_len  = ch_full_mask.sum(dim=1)
        rj_len  = rj_full_mask.sum(dim=1)
        batch["code/debug/eq_rows_frac"]        = float(eq_rows.float().mean().item())
        batch["code/debug/chosen_len_mean"]     = float(ch_len.float().mean().item())
        batch["code/debug/rejected_len_mean"]   = float(rj_len.float().mean().item())
        batch["code/debug/resp_len_mean"]       = float(((ch_len - p_mask.sum(dim=1)).clamp_min(0).float().mean()
                                                        + (rj_len - p_mask.sum(dim=1)).clamp_min(0).float().mean()) / 2)

    with torch.no_grad():
        p_len   = p_mask.sum(dim=1)
        ch_len  = ch_full_mask.sum(dim=1)
        rj_len  = rj_full_mask.sum(dim=1)
        resp_ch = (ch_len - p_len).clamp_min(0)
        resp_rj = (rj_len - p_len).clamp_min(0)
        eq_rows = (ch_full_ids == rj_full_ids).all(dim=1)
        batch["code/debug/prompt_len_mean"] = float(p_len.float().mean().item())
        batch["code/debug/resp_len_mean"]   = float(((resp_ch.float().mean() + resp_rj.float().mean()) / 2).item())
        batch["code/debug/eq_rows_frac"]    = float(eq_rows.float().mean().item())

    return batch


class DPOWithSkeletonTrainer(DPOTrainer):
    def __init__(self, *args, skel_weight: float = 0.7, debug_first_n_steps: int = 30, **kwargs):
        super().__init__(*args, **kwargs)
        self.skel_weight = float(skel_weight)
        self._debug_first_n_steps = int(debug_first_n_steps)
        self._global_step_seen = 0

    def _prepare_dataset(self, dataset, processing_class, args, stage):
        return dataset

    def _shift_and_gather_logps(self, logits, ids):
        logps = logits.log_softmax(dim=-1)[:, :-1, :]        
        tok_ids = ids[:, 1:].unsqueeze(-1)                   
        return torch.gather(logps, -1, tok_ids).squeeze(-1)  

    def _seq_avg_logprob(self, model, input_ids, attention_mask, response_mask):
        
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        tok_lp = self._shift_and_gather_logps(out.logits, input_ids)  
        mask   = response_mask[:, 1:]                                  
        den    = mask.sum(dim=1)                                       
        safe_den = den.clamp_min(1.0)
        avg   = (tok_lp * mask).sum(dim=1) / safe_den
        return avg, den

    def _batch_debug_code_inputs(self, batch):
        try:
            chi, rji = batch["chosen_input_ids"], batch["rejected_input_ids"]
            cam, ram = batch["chosen_attention_mask"], batch["rejected_attention_mask"]
        except KeyError:
            return {}
        ch_len = cam.sum(dim=1); rj_len = ram.sum(dim=1)
        eq_frac = (chi == rji).all(dim=1).float().mean().item()
        zero_frac = ((ch_len == 0) | (rj_len == 0)).float().mean().item()
        return {
            "code/debug/eq_rows_frac": float(eq_frac),
            "code/debug/zero_len_frac": float(zero_frac),
            "code/debug/chosen_len_mean": float(ch_len.float().mean().item()),
            "code/debug/rejected_len_mean": float(rj_len.float().mean().item()),
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        beta = getattr(self.args, "beta", 0.2)
        device = next(model.parameters()).device

        sk_keys = [
            "skel_chosen_input_ids","skel_chosen_attention_mask","skel_chosen_response_mask",
            "skel_rejected_input_ids","skel_rejected_attention_mask","skel_rejected_response_mask"
        ]
        has_skel = all(k in inputs for k in sk_keys)
        sk = {k: inputs[k] for k in sk_keys} if has_skel else None

        skel_loss = torch.tensor(0.0, device=device)
        if sk is not None and self.skel_weight > 0.0:
            pos_lp, den_pos = self._seq_avg_logprob(model, sk["skel_chosen_input_ids"],  sk["skel_chosen_attention_mask"],  sk["skel_chosen_response_mask"])
            neg_lp, den_neg = self._seq_avg_logprob(model, sk["skel_rejected_input_ids"], sk["skel_rejected_attention_mask"], sk["skel_rejected_response_mask"])
            valid = (den_pos > 0) & (den_neg > 0)

            if valid.any():
                with torch.no_grad():
                    pos_ref, _ = self._seq_avg_logprob(self.model, sk["skel_chosen_input_ids"],  sk["skel_chosen_attention_mask"],  sk["skel_chosen_response_mask"])
                    neg_ref, _ = self._seq_avg_logprob(self.model, sk["skel_rejected_input_ids"], sk["skel_rejected_attention_mask"], sk["skel_rejected_response_mask"])

                r_pos = beta * (pos_lp - pos_ref)
                r_neg = beta * (neg_lp - neg_ref)
                delta = (r_pos - r_neg)
                skel_loss = -F.logsigmoid(delta[valid]).mean()

                acc    = (r_pos[valid] > r_neg[valid]).float().mean()
                margin = (r_pos[valid] - r_neg[valid]).mean()
                self.store_metrics({
                    "loss_dpo_skel": float(skel_loss.detach().cpu()),
                    "lambda_skel":   float(self.skel_weight),
                    "skel/logps/chosen": float(pos_lp[valid].mean().detach().cpu()),
                    "skel/logps/rejected": float(neg_lp[valid].mean().detach().cpu()),
                    "skel/rewards/chosen": float(r_pos[valid].mean().detach().cpu()),
                    "skel/rewards/rejected": float(r_neg[valid].mean().detach().cpu()),
                    "skel/rewards/accuracies": float(acc.detach().cpu()),
                    "skel/rewards/margins": float(margin.detach().cpu()),
                    "skel/debug/den_pos_mean": float(den_pos[valid].float().mean().detach().cpu()),
                    "skel/debug/den_neg_mean": float(den_neg[valid].float().mean().detach().cpu()),
                    "skel/debug/den_zero_frac": float(((den_pos==0)|(den_neg==0)).float().mean().detach().cpu()),
                }, train_eval="train")
            else:
                self.store_metrics({
                    "loss_dpo_skel": 0.0, "lambda_skel": float(self.skel_weight),
                    "skel/logps/chosen": 0.0, "skel/logps/rejected": 0.0,
                    "skel/rewards/chosen": 0.0, "skel/rewards/rejected": 0.0,
                    "skel/rewards/accuracies": 0.0, "skel/rewards/margins": 0.0,
                    "skel/debug/den_pos_mean": 0.0, "skel/debug/den_neg_mean": 0.0, "skel/debug/den_zero_frac": 1.0,
                }, train_eval="train")

    
        chi = inputs["chosen_input_ids"]            
        cam = inputs["chosen_attention_mask"]       
        rji = inputs["rejected_input_ids"]          
        ram = inputs["rejected_attention_mask"]     
        pam = inputs["prompt_attention_mask"]       

        p_len = pam.sum(dim=1)                       
        ch_len = cam.sum(dim=1)                     
        rj_len = ram.sum(dim=1)                      

        T = chi.size(1)
        idx = torch.arange(T, device=device).unsqueeze(0)  

        resp_mask_ch = (idx >= p_len.unsqueeze(1)) & (idx < ch_len.unsqueeze(1))
        resp_mask_rj = (idx >= p_len.unsqueeze(1)) & (idx < rj_len.unsqueeze(1))
        resp_mask_ch = resp_mask_ch.to(chi.dtype)   
        resp_mask_rj = resp_mask_rj.to(rji.dtype)

        pos_lp, den_pos = self._seq_avg_logprob(model, chi, cam, resp_mask_ch)
        neg_lp, den_neg = self._seq_avg_logprob(model, rji, ram, resp_mask_rj)
        valid = (den_pos > 0) & (den_neg > 0)

        with torch.no_grad():
            pos_ref, _ = self._seq_avg_logprob(self.model, chi, cam, resp_mask_ch)
            neg_ref, _ = self._seq_avg_logprob(self.model, rji, ram, resp_mask_rj)

        r_pos = beta * (pos_lp - pos_ref)
        r_neg = beta * (neg_lp - neg_ref)
        delta = r_pos - r_neg

        code_loss = -F.logsigmoid(delta[valid]).mean() if valid.any() else torch.tensor(0.0, device=device)

        if self._global_step_seen < self._debug_first_n_steps:
            code_dbg = self._batch_debug_code_inputs(inputs)
            if code_dbg:
                self.store_metrics(code_dbg, train_eval="train")
        self._global_step_seen += 1

        with torch.no_grad():
            acc    = (r_pos[valid] > r_neg[valid]).float().mean() if valid.any() else torch.tensor(0.0, device=device)
            margin = (r_pos[valid] - r_neg[valid]).mean()          if valid.any() else torch.tensor(0.0, device=device)
            self.store_metrics({
                "loss_dpo_code": float(code_loss.detach().cpu()),
                "rewards/chosen": float(r_pos[valid].mean().detach().cpu()) if valid.any() else 0.0,
                "rewards/rejected": float(r_neg[valid].mean().detach().cpu()) if valid.any() else 0.0,
                "rewards/accuracies": float(acc.detach().cpu()),
                "rewards/margins": float(margin.detach().cpu()),
                "logps/chosen": float(pos_lp[valid].mean().detach().cpu()) if valid.any() else 0.0,
                "logps/rejected": float(neg_lp[valid].mean().detach().cpu()) if valid.any() else 0.0,
            }, train_eval="train")

        
        loss = code_loss + skel_loss
        if return_outputs:
            return loss, {}   
        return loss



dpo_peft_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules = []
)

def run_training(args, train_data):

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"  

    sft_adapter_path = ""

    model = AutoPeftModelForCausalLM.from_pretrained(
        sft_adapter_path, 
        device_map="auto"
    )

    print('Finished loading model {}'.format(args.model))
    print(f"Starting main loop")

    training_args = DPOConfig(
        output_dir=args.save_dir,
        overwrite_output_dir=True,

        do_train=True,
        do_eval=False,
        do_predict=True,

        max_steps=800,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        lr_scheduler_type='constant_with_warmup',
        warmup_ratio=0.05,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_steps=args.save_freq,
        save_total_limit=args.save_total_limit,

        dataloader_drop_last=True,
        dataloader_num_workers=0 if args.db else 8,

        local_rank=args.local_rank,

        deepspeed=args.deepspeed,
        fp16=args.fp16,

        remove_unused_columns=False,

        beta=0.1,                            
        max_length=2048,                
        max_prompt_length=1024,  
        padding_value=tokenizer.eos_token_id,

    )
    
    trainer = DPOWithSkeletonTrainer(
        
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=lambda s: dpo_with_skel_collate(
            s, tokenizer,
            max_concat_len=2048,
            max_prompt_len=1024,   
            max_code_len=2048,
            max_skel_len=2048
        ),
        skel_weight= 0,   
    )

    trainer.train()

    model.save_pretrained(args.save_dir)


def get_dataset(args):
    pairCode_dict_json = ""
    with open(pairCode_dict_json,'r',encoding='utf-8') as pci:
        pariCode_dict = json.load(pci)

    print(f"Loading {len(pariCode_dict)} problems.")
    dpo_solutions = []
    for key, value in tqdm(pariCode_dict.items()):
        efficientCode = value[0]
        efficientCodeSkeleton = value[1]
        unefficientCode = value[2]
        unefficientCodeSkeleton = value[3]
        question_fname = os.path.join(args.train_path, key, "question.txt")

        with open(question_fname, 'r') as q:
            question_str = q.read()

        efficientCode_str           = reindent_code(efficientCode)
        unefficientCode_str         = reindent_code(unefficientCode)
        efficientCodeSkeleton_str   = reindent_code(efficientCodeSkeleton)
        unefficientCodeSkeleton_str = reindent_code(unefficientCodeSkeleton)

        code_prompt = (
            "[GEN_CODE]\n"
            "Please generate an efficient and correct program that solves the given problem.\n"
            "QUESTION:\n" + question_str + "\n"
            "ANSWER:\n"
        )
        skel_prompt = (
            "[GEN_CODESKELETON]\n"
            "Please generate the efficient code skeleton that reflects high-performance patterns of implementation.\n"
            "QUESTION:\n" + question_str + "\n"
            "Let's think by codeskeleton:\n"
        )

        dpo_solutions.append(
            {
                "code_prompt":  code_prompt,
                "code_chosen":  efficientCode_str,
                "code_rejected": unefficientCode_str,

                "skel_prompt":   skel_prompt,
                "skel_chosen":   efficientCodeSkeleton_str,
                "skel_rejected": unefficientCodeSkeleton_str,
            }
        )

    return dpo_solutions

def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr, 
        ret, 
        config = {
            "dry-run": False,
            "help": False,
            "to": 4,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 4,
            "all-tabs": False
        }
    )

    return ret.getvalue()

def sanity_check_samples(samples, n_print=5):
    bad = []
    for i, s in enumerate(samples):
        has_code = all(k in s and isinstance(s[k], str) for k in ["code_chosen","code_rejected"])
        has_prompt = ("code_prompt" in s or "prompt" in s)
        if not has_code or not has_prompt:
            bad.append((i, list(s.keys())))
    if bad:
        print(f"[WARN] Found {len(bad)} bad samples missing keys (showing first {n_print}):")
        for i,(idx, keys) in enumerate(bad[:n_print]):
            print(f"  idx={idx}, keys={keys}")
    else:
        print("[OK] All samples contain required keys.")

def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    train_data = get_dataset(args)
    # Save args to file
    json.dump(argsdict, open(os.path.join(args.save_dir, "args.json"), 'w'))

    # Load and train model; save model checkpoints
    train_dataset = Dataset.from_list(train_data)
    print(len(train_dataset))
    sanity_check_samples(train_data)
    run_training(args, train_dataset)


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Training a model for code generation")
    parser.add_argument('--model', default="", type=str, help='type of transformers model as model backbone')
    parser.add_argument('--model_path', default="", type=str, help='path to model backbone pretrained weights')
    parser.add_argument('--save_dir', default='', type=str, help='path to save trained model checkpoints')

    # Dataloading
    parser.add_argument('--train_path', default='', type=str, help='path to training data')
    parser.add_argument('--sample_mode', default='uniform_sol', help='sampling output programs following a uniform distribution by program population')

    # Training
    parser.add_argument('--beta', default=0.1, type=float, help='the beta parameter for DPO loss')
    parser.add_argument('--lr', default=1e-5, type=float, help='training learning rate')
    parser.add_argument('--batch_size_per_replica', default=4, type=int, help='batch size per GPU')
    parser.add_argument('--grad_acc_steps', default=16, type=int, help='number of training steps before each gradient update')
    parser.add_argument('--deepspeed', default=None, type=str, help='path to deepspeed configuration file; set None if not using deepspeed')
    parser.add_argument('--fp16', default=True, action='store_true', help='set 16-bit training to reduce memory usage')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--db', default=False, action='store_true', help='set to turn on debug mode i.e. using dummy small data split and only 1 data worker')

    # Logging
    parser.add_argument('--log-freq', default=10, type=int, help='save training log after this number of training steps')
    parser.add_argument('--save-freq', default=10, type=int, help='save model checkpoints after this number of training steps')
    parser.add_argument('--save_total_limit', default=1, type=int, help='total of number checkpoints to keep; only keep the latest ones')

    args = parser.parse_args()
    main(args)
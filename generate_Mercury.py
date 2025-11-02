import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pprint
import torch
import pdb 
import glob 
from tqdm import tqdm
import pickle as pkl 
import numpy as np 
from peft import AutoPeftModelForCausalLM
from collections import Counter 
import transformers
from peft import PeftModel, PeftConfig
import csv
import pandas as pd
from datasets import load_dataset
from collections import defaultdict
import re
import textwrap

def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    problems = pd.read_parquet(args.test_path)
    dataset = load_dataset("parquet", data_files="../Mercury-main/data/eval-00000-of-00001.parquet", streaming=True)
    streaming_dataset = dataset["train"]
    problems = streaming_dataset.to_list()
    print(len(problems))
    
    # Set up model
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    print("Loading model from {}...".format(args.model_path))
    peft_model_id = " "
    model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, pad_token_id=tokenizer.eos_token_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    solutions = {}
    for index, problem in tqdm(enumerate(problems), ncols=0, total=len(problems)):
        try:
            pretty_content = problem["pretty_content"][0]
            prompt = problem["prompt"]
            slug_name = problem["slug_name"]
            input_text = f"[GEN_CODE]\nPlease generate an efficient and correct program that solves the given problem.\nQUESTION:\n{pretty_content}\n{prompt}"
            input_text += "\nANSWER:\n"
        except:
            with open("save_error_.txt", 'a') as f:
                f.write( + '\n')
            continue

        with torch.no_grad():

            try:
            
                input_ids = tokenizer.encode(input_text, verbose=False, max_length=1024)
                input_ids = torch.LongTensor(input_ids).unsqueeze(0).cuda()
            except:
                with open("information_tensor_new.txt", 'a') as fff:
                    fff.write('\n')
            num_loops = int(args.num_seqs / args.num_seqs_per_iter)
            output_programs = []
            
            for i in tqdm(range(num_loops), ncols=0, total=num_loops, leave=False):
                output_ids = model.generate(
                        input_ids=input_ids, 
                        do_sample=True,            
                        temperature=args.temperature, 
                        max_length=1024, 
                        num_return_sequences=args.num_seqs_per_iter,
                        top_p=0.95
                        )

                for output_id in output_ids: 
                    output_str = tokenizer.decode(output_id, skip_special_tokens=True)
                    if len(output_str):
                        if "ANSWER:\n " in output_str:
                            try:
                                output_program = output_str.split("ANSWER:\n ")[1].replace("<|endoftext|>", "")
                                output_programs.append(output_program)
                            except:
                                continue
                        else:
                            try:
                                output_program = output_str.split("ANSWER:\n")[1].replace("<|endoftext|>", "")
                                output_programs.append(output_program)
                            except:
                                continue
                    else:
                        output_programs.append("None code!")
            
            if len(output_programs) == 0:
                output_programs = ["No code!"]*10
            solutions[slug_name] = [{
                "task_id": slug_name,
                "completion": output_programs,
            }]
        with open(args.output_path, 'w') as f:
            json.dump(solutions, f, indent=2)


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Run model to generate Python programs.")
    parser.add_argument("-t","--test_path", default="", type=str, help='Path to test samples')
    parser.add_argument("--output_path", default="",type=str, help='Path to save output programs')
    parser.add_argument("--model_path", default="",type=str, help='Path of trained model')
    parser.add_argument("--tokenizer_path", type=str, help='Path to the tokenizer')  

    parser.add_argument("--num_seqs", default=1, type=int, help='Number of total generated programs per test sample')
    parser.add_argument('--num_seqs_per_iter', default=1, type=int, help='Number of possible minibatch to generate programs per iteration, depending on GPU memory')

    parser.add_argument("--max_len", default=1024, type=int, help='Maximum length of output sequence') 
    parser.add_argument("--temperature", default=0.8, type=float, help='temperature for sampling tokens')

    args = parser.parse_args()
    
    main(args)

        
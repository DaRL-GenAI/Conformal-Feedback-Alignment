import torch
import time
import math
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from collections import defaultdict
import json
import numpy as np
from gensim.test.utils import common_texts, datapath
from gensim.models import FastText
from tqdm import tqdm
import pickle

device_name = [f"cuda:0"]

np.random.seed(42)
torch.manual_seed(42)

model = LlamaForCausalLM.from_pretrained("/home/Model/Llama2-7b-RLUF", torch_dtype=torch.bfloat16)
model.to(device_name[0])

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


dataset = load_dataset("openai/summarize_from_feedback", "axis")
val_data = dataset["test"]

unique_data = {}
for example in val_data:
    info = example["info"]
    this_id = info["id"]

    if this_id not in unique_data:
        unique_data[this_id] = info["article"]

for key in unique_data.keys():
    full_question = "Please summarize this text:\n" + unique_data[key] + "\nSummary:"
    unique_data[key] = full_question

response_res = {}
for key in tqdm(unique_data.keys()):
    prompt = unique_data[key]
    inputs = tokenizer(prompt, return_tensors="pt").to(device_name[0])
    attention_mask = inputs['attention_mask']
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=200,
    )[0]
    generate_text = tokenizer.decode(outputs[input_length:],skip_special_tokens=True)
    response_res[key] = generate_text

with open('test_dict_question.pkl', 'wb') as f:
    pickle.dump(unique_data, f)

with open('test_dict_RLUF.pkl', 'wb') as f:
    pickle.dump(response_res, f)

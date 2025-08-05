import ast
from datasets import load_dataset
import numpy as np
import torch
import itertools
import json
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
import json
import openai

import random

adjust_format = []
response_dict = np.load("./Generation With CP/response_dict_llama2.pkl",allow_pickle=True)
for key in response_dict.keys():
    second_dict = response_dict[key]
    response_list = []
    for res_key in second_dict.keys():
        response_list.append(res_key)
    if len(response_list) < 2:
        continue
    all_pairs = list(itertools.combinations(response_list, 2))
    for pair in all_pairs:
        temp_dict = {}
        temp_dict['instruction'] = key
        temp_dict['input'] = ""
        temp_dict['output_1'] = pair[0]
        temp_dict['output_2'] = pair[1]
        adjust_format.append(temp_dict)


openai.api_key = "xxxxxx"

annotator = PairwiseAutoAnnotator()
annotated = annotator.annotate_pairs(adjust_format)

# file_path = "AI_feedback_llama2.json"
#
# with open(file_path, "w", encoding="utf-8") as f:
#     json.dump(annotated, f, ensure_ascii=False, indent=4)

new_data = []
for entry in annotated:
    # Construct the prompt (instruction + input if needed)
    prompt = entry["instruction"]
    # Determine chosen vs. rejected based on preference
    if entry["preference"] == 1:
        chosen = entry["output_1"]
        rejected = entry["output_2"]
    else:
        chosen = entry["output_2"]
        rejected = entry["output_1"]

    new_data.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    })

# Save the new data in DPO-friendly format
with open("dpo_data_llama2.json", "w", encoding="utf-8") as f:
    for item in new_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")










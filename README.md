# RLUF

## Usage

To use our code, please consider the following package:

- alpacafarm
- gensim
- transformers==4.51.0
- openai==0.28.1
- datasets==3.5.0
- torch==2.3.0
- trl==0.7.0

Please also replace the huggingface token and openai key in all codes.

## Introduction

This repo now provides the sample code for RLUF with DPO on Llama2 and Summarize Dataset. It mainly contains three stages:

1. Generation with CP

To do it, firstly run ```sft.py``` to get the sft model and then run ```run_generation.py``` to get the calibration set and test set of conformal prediction. Finally run ```run_cp.py``` to do calibration and inference of conformal prediction to get the prediction set. To get the default prediction set, ```run_cp.py``` should be exexcuted twice and *quantile_bar* should be set to 0.2 and 0.5 for two runs. 

2. AI Feedback

This part is to get the AI feedback for the generation. Run ```AI_feedback.py``` to get the preference informtion using package alpacafarm. Then using ```obtain_uncertainty_data.py``` to get the weight for each perference pair. 

3. Training and Testing

To train the model with RLUF, please use the code ```dpo_ours_train.py```. After training, please use ```inference-generation.py``` to get the answers for the test dataset. Finally, ```AI_response_evaluation.py``` is used to get the final evaluation scores.

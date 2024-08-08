import os
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
dtype = torch.float16 if is_half is True else torch.float32

CNHUBERT_BASE_PATH = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
BERT_PATH = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"


def get_device():
    return device


def get_is_half():
    return is_half


def get_dtype():
    return dtype

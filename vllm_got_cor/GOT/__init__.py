import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .model import GOTQwenForCausalLM
from .utils.utils import disable_torch_init
from .model.plug.blip_process import BlipImageEvalProcessor
from .model import Qwen2GotForCausalLM

from loguru import logger
from vllm import LLM, ModelRegistry

# from GOT.model import GOTQwenForCausalLM # 无vllm加速原版
ModelRegistry.register_model("GOTQwenForCausalLM", Qwen2GotForCausalLM)
# ModelRegistry.register_model("Qwen2GotForCausalLM", Qwen2GotForCausalLM)


def init_model_GOT(model_name):
    """ 初始化并加载模型，输入参数为模型权重路径 """
    # 禁用 torch 的一些初始化设置
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 非vllm形式加载
    # model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()
    # model.to(device='cuda', dtype=torch.bfloat16)
    # vllm形式加载
    model = LLM(model=model_name, max_model_len=8192, trust_remote_code=True, enforce_eager=True, gpu_memory_utilization=0.3)

    # 加载图像处理器
    image_processor = BlipImageEvalProcessor(image_size=1024)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)
    return model, tokenizer, image_processor, image_processor_high


# 调用初始化函数，项目启动时加载模型
model_path = 'GOT-OCR2_0/'
logger.info('GOT start loading')
model, tokenizer, image_processor, image_processor_high = init_model_GOT(model_name=model_path)
logger.info('GOT finish loading')

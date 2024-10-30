#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vllm_got_ocr.py
@Time    :   2024/10/29
@Author  :   Winter.Yu 
@Version :   1.0
@Contact :   winter741258@126.com
@Desc    :   None
'''

# here put the import lib
import os
import torch
import requests
from PIL import Image
from io import BytesIO


from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import KeywordsStoppingCriteria
from GOT import model, tokenizer, image_processor, image_processor_high

from vllm.sampling_params import SamplingParams



DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


def load_image(image_bytes):
     # 从bytes加载图像并转换为RGB格式
    if isinstance(image_bytes, bytes):
        try:       
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    # 如果是文件或者下载链接，用于测试
    image_file = image_bytes
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def vllm_got(image_list, type='format', is_save=False):
    # 使用修改后的load_image函数来加载图像
    images = []
    for image_bytes in image_list:
        image = load_image(image_bytes)
        if image is None:
            return "Image loading failed."
        images.append(image)
    
    image_tensors = []
    image_tensors_1 = []
    for image in images:
        image_tensor = image_processor(image)
        image_tensor_1 = image_processor_high(image.copy())
        image_tensors.append(image_tensor)
        image_tensors_1.append(image_tensor_1)


    # 构建提示符
    qs = f'OCR with format: ' if type == 'format' else 'OCR: '
    use_im_start_end = True
    image_token_len = 256

    if use_im_start_end:
        qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_PATCH_TOKEN * image_token_len}{DEFAULT_IM_END_TOKEN}\n{qs}"
    else:
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"

    # 配置对话模板
    conv_mode = "mpt"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer(prompt)
    
    input_ids = inputs.input_ids
    tokenizer.eos_token_id = tokenizer.pad_token_id
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
    new_keyword_ids = [keyword_id[0] for keyword_id in keyword_ids]
    sampling_param = SamplingParams(temperature=0, top_p=0.8, repetition_penalty=1.25, max_tokens=2048, stop_token_ids = new_keyword_ids)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            prompts = [
                {
                    'prompt_token_ids': input_ids,
                    'multi_modal_data':{
                        'image':image_tensor.unsqueeze(0).half().cuda()
                    }
                } for image_tensor in image_tensors_1
            ],
            sampling_params = sampling_param
        )
        
        generated_text = ""
        for o in output_ids:
            print('-'*100,'\n\n')
            generated_text += o.outputs[0].text
        
    if not generated_text:
        print("未识别成功")
        return None
    if is_save:
        with open('./result.txt', 'w', encoding='utf-8') as f:
            f.write(generated_text)
    return generated_text


# just for test
if __name__ == '__main__':
    import time
    start = time.time()
    image_list = [
        './img/test1.png'
    ]          
    res = vllm_got(image_list, type='ocr', is_save=True)
    print(res)
    end = time.time()
    print(f"Time cost: {end-start:.3f} seconds")

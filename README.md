# vllm_GOT2_OCR
Accelerating GOT-OCRv2 with VLLM
# Install
pip install vllm==0.5.3.post1 transformer  pytorch=2.3.1

## The output may sometimes miss characters, please adjust sample parameter in vllm, such as SamplingParams(temperature=0.0, top_p=0.9, repetition_penalty=1.0)

# Acknowledgement
 [GOT-OCR](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
 
 [基于MinerU和GOT-OCR2.0 实现pdf解析](https://github.com/liunian-Jay/MU-GOT)
 
 


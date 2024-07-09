# vllm-mcqa
Simple code to force mcqa outputs from VLLM.

## Usage
```python
from vllm import LLM, SamplingParams
from logits import ban_illegal_tokens, get_allowed_token_ids

llm = LLM(model="facebook/opt-1.3b")

allowed_token_ids = get_allowed_token_ids(llm, ['A','B','C','D'])

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1,
    logits_processors=[lambda token_ids, logits: ban_illegal_tokens(token_ids, logits, allowed_token_ids)]
)

prompts = ["""### Question: 한국채택국제회계기준(K-IFRS)하에서 금융자산으로 분류되지 않는 것은?
### Options:
    A. 대여금
    B. 재고자산
    C. 매출채권
    D. 만기보유금융자산
### Answer:"""]

outputs = llm.generate(prompts*10, sampling_params)

print(prompts[0])
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")
```

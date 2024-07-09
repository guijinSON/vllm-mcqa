import torch

def get_allowed_token_ids(llm, allowed_tokens=['A','B','C','D']):
    return llm.llm_engine.tokenizer.tokenizer.convert_tokens_to_ids(allowed_tokens)

def ban_illegal_tokens(token_ids, logits, allowed_tokens):
    mask = torch.zeros_like(logits, dtype=torch.bool) # Mask for allowed tokens
    mask[allowed_tokens] = True
    
    logits = torch.where(mask, logits, torch.tensor(-float('inf')))
    return logits

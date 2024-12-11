from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
torch_d_type = torch.float16 if torch.cuda.is_available() else torch.float32

base_model_name = "albert-base-v2"

finetuned_model_name = "./steam_review_model"

classifier = pipeline(
    task="text-classification",
    model=finetuned_model_name,
    tokenizer=base_model_name,
    device=device,
    top_k=None,
    truncation=True,
    max_length=512,
    torch_dtype=torch_d_type)

review = "This game is really fun and easy access to play i do not regret a single penny!"
result = classifier(review)
print(result)
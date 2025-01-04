import torch
from transformers import pipeline

REVIEW_TO_CHECK = "very good game, good graphics, good music and good character development"

BASE_MODEL_NAME = "albert-base-v2"
FINETUNED_MODEL_PATH = "./training/steam_review_model"

device = 0 if torch.cuda.is_available() else -1
torch_d_type = torch.float16 if torch.cuda.is_available() else torch.float32

classifier = pipeline(
    task="text-classification",
    model=FINETUNED_MODEL_PATH,
    tokenizer=BASE_MODEL_NAME,
    device=device,
    top_k=None,
    truncation=True,
    max_length=512,
    torch_dtype=torch_d_type)

result = classifier(REVIEW_TO_CHECK)[0]

if result[0]['label'] == 'LABEL_1':
    print("Review is helpful. Helpful Score: {} | Unhelpful Score: {}".format(result[0]['score'], result[1]['score']))
else:
    print("Review is NOT helpful. Helpful Score: {} | Unhelpful Score: {}".format(result[1]['score'], result[0]['score']))

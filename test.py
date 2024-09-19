import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Initialize Variables
model_name = "AI-Sweden-Models/gpt-sw3-126m"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
prompt = "Vad ska man göra om man har ont i huvudet?"

# Initialize Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
model.to(device)


input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

generated_token_ids = model.generate(
    inputs=input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.6,
    top_p=1,
)[0]

generated_text = tokenizer.decode(generated_token_ids)

print(generated_text)
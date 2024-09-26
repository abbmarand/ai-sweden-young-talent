from transformers import pipeline
summarizer = pipeline("summarization")
classifier = pipeline("text-classification")
ner_tagger = pipeline("ner", aggregation_strategy="simple")
translator = pipeline("translation_en_to_sv", model="Helsinki-NLP/opus-mt-en-sv") #This one you can change to other language then swedish if you like!
generator = pipeline("text-generation")
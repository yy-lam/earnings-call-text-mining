import sys
import os
import spacy
import re

def tokenize(text):
    paragraphs = text.split('\n')
    processed_text = ""
    for p in paragraphs:
        if len(p) < 15:
            continue
        doc = nlp(p)
        for token in doc:
            if not token.is_punct:
                replaced_tok = token.lemma_
                if token.lemma_ == '-PRON-':
                    replaced_tok = 'it'
                processed_text += (replaced_tok + " ")
            else:
                processed_text += (token.text + " ")
    processed_text = re.sub(r"(good morning( everyone)*)", "", processed_text)
    processed_text = re.sub(r"(thank you)", "", processed_text)
    processed_text = re.sub(r"(thank(s)*)", "", processed_text)
    return processed_text

def read_file(directory):
    all_files = os.listdir(directory + '/preprocess')
    for text_dir in all_files:
        if not text_dir.endswith(".txt") or text_dir.startswith("."):
            continue
        if os.path.isdir(text_dir):
            continue
        with open(directory + '/' + 'preprocess/' + text_dir, 'r', encoding='utf-8') as f:
            text = f.read()
        print("Processing", text_dir)
        with open(directory + '/' + text_dir, 'w', encoding='utf-8') as p:
            p.write(tokenize(text))

        
        
if __name__ == '__main__':
    directory = sys.argv[1]
    nlp = spacy.load('en_core_web_sm')
    read_file(directory)
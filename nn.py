import torch
import numpy as np
import pandas as pd
import os, sys

import torch
#import torchtext
import torch.optim as optim
import torch.nn as nn
from collections import defaultdict, Counter
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
torch.cuda.is_available()
device = torch.device("cuda")
torch.cuda.set_device(0)
torch.cuda.device(0)


def load_data(data, label):
    for file in os.listdir(label):
        if file.endswith('.txt') and not file.startswith('.'):
            file_path = label + '/' + file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                data[text] = label

class ClassifierRunner(object):
    def __init__(self, data, voca_size, in_dim, hid_dim, word_embedding):
        self.data = data
        self.clf = Classifier(voca_size, in_dim, hid_dim, word_embedding).cuda()
        self.optimizer = optim.Adam(self.clf.parameters())
        # criterion
        self.ce_loss = nn.CrossEntropyLoss()
        #self.ce_loss = nn.PoissonNLLLoss()

    def run_epoch(self, split):
        """Runs an epoch, during which the classifier is trained or applied
        on the data. Returns the predicted labels of the instances."""

        if split == "dev": self.clf.train()
        else: self.clf.eval()
        epoch_loss = 0
        labels_pred = []
        for i, (words, label) in enumerate(self.data[split]):
            #m = nn.Dropout(p=0.9, inplace = True)
            logit = self.clf(torch.LongTensor(words).cuda())
            #logit = m(logit)
            # Optimize
            if split == "dev":
                loss = self.ce_loss(logit, torch.LongTensor([label]).cuda())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #print(loss.item())
                epoch_loss += loss.item()

            idx = torch.argmax(logit)
            #print(idx)
            #print("------------")
            
            labels_pred.append(idx2label[idx])
        if split == "test":
            tp_default = 0
            tp_ndefault = 0
            total_default = 0
            total_ndefault = 0
            predicted_default = 0
            predicted_ndefault = 0
            total = len(self.data[split])

            for i in range(len(self.data[split])):
                _, test_label = self.data[split][i]
                
                print("Lable:", test_label)
                print("Pridicted:", labels_pred[i])
                
                if test_label == "default":
                    total_default += 1
                    if labels_pred[i] == "default":
                        tp_default += 1
                else:
                    total_ndefault += 1
                    if labels_pred[i] == "ndefault":
                        tp_ndefault += 1
                if labels_pred[i] == "default":
                    predicted_default += 1
                else:
                    predicted_ndefault += 1
            print("Overall accuracy:", (tp_default+tp_ndefault)/total)
            print("Precision for default:", tp_default/total_default)
            print("Precision for ndefault:", tp_ndefault/total_ndefault)
            if predicted_default != 0:
                print("Recall for default:", tp_default/predicted_default)
            else:
                print("Fail at recalling default")
            if predicted_ndefault != 0:
                print("Recall for ndefault:", tp_ndefault/predicted_ndefault)
            else:
                print("Fail at recalling ndefault")
            print("---------------------------------------------------------")
    
        return labels_pred


class Classifier(nn.Module):
    def __init__(self, voca_size, in_dim, hid_dim, word_embedding):
        super(Classifier, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim

        # Layers
        #self.word2wemb = nn.Embedding(voca_size, rnn_in_dim).cuda()
        #print(word_embedding.shape)
        self.word2wemb = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding).cuda())
        self.lstm = nn.LSTM(in_dim, hid_dim, bias = False, num_layers = 2, dropout = 0.9)
        #self.rnn1 = nn.RNN(rnn_in_dim, rnn_hid_dim)
        
        self.dropout = nn.Dropout(0.6)
        self.rnn = nn.RNN(hid_dim, 15)
        #self.relu = nn.ReLU()
        self.fc = nn.Linear(15, 2)
        #self.runlogit = nn.LogSoftmax(hid_dim, 2)

    def init_rnn_hid(self):
        """Initial hidden state."""
        return torch.zeros(2, 1, self.hid_dim).cuda()

    def forward(self, words):
        """Feeds the words into the neural network and returns the value
        of the output layer."""
        wembs = self.word2wemb(words) # (seq_len, rnn_in_dim)
        lstm_outs, _ = self.lstm(wembs.unsqueeze(1))
        #rnn_outs, _ = self.rnn1(wembs.unsqueeze(1), self.init_rnn_hid()) 
        #drop_outs = self.dropout(lstm_outs)
        rnn_outs, _ = self.rnn(lstm_outs) 
                                      # (seq_len, 1, rnn_hid_dim)
        #fc_outs = self.fc(drop_outs) # (1 x 3)
        logit = self.fc(rnn_outs[-1])
        return logit



if __name__ == '__main__':
	import random, time
	df = {}
	load_data(df, 'default')
	load_data(df, 'ndefault')
	train_df = {}
	test_df = {}
	print(len(df))
	random.seed(123)
	split = random.sample(list(df), k=int(len(df)*0.8))
	for s in split:
	    train_df[s] = df[s]
	for text in df:
	    if text not in train_df:
	        test_df[text] = df[text]
	print(len(train_df), len(test_df))
	wnl = WordNetLemmatizer()
	OUT_HELDOUT_PATH = "pred_nn.txt"
	idx2label = ["default", "ndefault"]
	label2idx = {label: idx for idx, label in enumerate(idx2label)}
	start_time = time.time()
	print("Reading data...")
	data_raw = defaultdict(list)
	voca_cnt = Counter()
	for text in train_df:
	    label = train_df[text]
	    words = word_tokenize(text.strip())
	    selected_words = []
	    for word in words:
	        if word in set(stopwords.words('english')):
	            continue
	        #word = wnl.lemmatize(word.lower())
	        selected_words.append(word)
	    #tagged = nltk.pos_tag(words)
	#     for (word, tag) in tagged:
	#         if tag in ['VBD', 'JJ']:
	#             target_words.append(word)
	#     if len(target_words) == 0:
	#         target_words.append("UNK")
	#     data_raw["dev"].append((words, label2idx[label.strip()]))
	#     voca_cnt.update(words)
	    # words -> target_words
	    data_raw["dev"].append((selected_words, label2idx[label.strip()]))
	    voca_cnt.update(words)
	    
	for text in test_df:
	    words = word_tokenize(text.strip())
	    for word in words:
	        if word in set(stopwords.words('english')):
	            continue
	        word = wnl.lemmatize(word.lower())
	        selected_words.append(word)
	    words = word_tokenize(text.strip())
	    label = test_df[text]
	    data_raw["test"].append((selected_words, label))
	    
	print(f'Number of training examples: {len(train_df)}')
	print(f'Number of testing examples: {len(test_df)}')

	print("Building voca...")
	word_idx = {"[UNK]": 0}
	for word in voca_cnt.keys():
	    word_idx[word] = len(word_idx)
	print("n_voca:", len(word_idx))

	print("Indexing words...")
	data = defaultdict(list)
	for split in ["dev", "test"]:
	    for words, label in data_raw[split]:
	        data[split].append(([word_idx.get(w, 0) for w in words], label))

	print("Importing GloVe...") 

	with open('glove-2/glove.6B.300d.txt', encoding="utf-8", mode="r") as f:
	    word_embedding = np.random.normal(0, 1, (len(word_idx), 300))
	    
	    for line in f:
	        # Separate the values from the word
	        line = line.split()
	        word = line[0]

	        # If word is in our vocab, then update the corresponding weights
	        idx = word_idx.get(word, None)
	        if word in word_idx:
	            word_embedding[idx] = np.array(line[1:], dtype=np.float32)

	print("Running classifier...")
	#M = ClassifierRunner(data, len(word_idx), args.rnn_in_dim, args.rnn_hid_dim)
	M = ClassifierRunner(data, len(word_idx), 300, 80, word_embedding)

	# -epochs = 10
	for epoch in range(5):
	    print("Epoch", epoch+1)

	    # Train
	    M.run_epoch("dev")

	    # Test
	    with torch.no_grad():
	        labels_pred = M.run_epoch("test")
	        
	    with open(OUT_HELDOUT_PATH, "w") as f:
	        f.write("\n".join(labels_pred))
	end_time = time.time()
	print("Time elapsed:", round((end_time - start_time)/60, 2), 'minutes')
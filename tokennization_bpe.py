import pandas as pd  
import numpy as np 
from collections import defaultdict


def load_data(file_path):
    with open(file_path, 'r', encoding= "utf-8") as f:
        text = f.read()
    print(f"loaded data: {text[:100]}....")
    return text

def prepare_data(text):
    #spilt the text into characters
    text = list(text)

    print(f"prepared data: {text[:100]}....")
    return text

def intialize_vocabulary(text):
    vocab = defaultdict(int)
    for char in text:
        vocab[char] += 1
    print(f"intialized vocabulary: {dict(list(vocab.items())[:10])}....")
    return vocab 

def get_pair_statistic(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    print(f"pair statistic: {dict(list(pairs.items())[:10])}....")
    return pairs

def merge_vocab(pair, in_vocab):
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair) 
    for word in in_vocab:
        new_word = word.replace(bigram, "".join(pair))
        new_vocab[new_word] = in_vocab[word]
    print(f"Merged '{bigram}' into '{replacement}'. New vocabulary: {dict(list(new_vocab.items())[:10])}")
    return new_vocab   

def byte_pair_encoding(text, num_merges):
    vocab = intialize_vocabulary(text)
    for i in range(num_merges):
        pairs = get_pair_statistic(vocab)
        if not pairs:
            print("No pair statistics to merge")
            break
        most_frequent_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(most_frequent_pair, vocab)
    return vocab

file_path = "drake_lyrics.txt"
text = load_data(file_path)
prepared_text = prepare_data(text)
vocab = byte_pair_encoding(prepared_text, num_merges=10)
print("\nFinal vocabulary:")
print
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
import json

import pandas as pd
import numpy as np
import random
import os
import re
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Custom Tokenizer
# class CustomTokenizer:
#     def __init__(self):
#         self.word_index = {}
#         self.index_word = {}
#         self.vocab_size = 0
#
#     def fit_on_texts(self, texts):
#         for text in texts:
#             self._update_vocab(text)
#
#     def _update_vocab(self, text):
#         words = text.split()
#         for word in words:
#             if word not in self.word_index:
#                 self.word_index[word] = self.vocab_size
#                 self.index_word[self.vocab_size] = word
#                 self.vocab_size += 1
#
#     def texts_to_sequences(self, texts):
#         sequences = []
#         for text in texts:
#             sequence = [self.word_index.get(word, 0) for word in text.split()]
#             sequences.append(sequence)
#         return sequences
#
#     def sequences_to_texts(self, sequences):
#         texts = []
#         for sequence in sequences:
#             text = ' '.join([self.index_word.get(idx, '') for idx in sequence])
#             texts.append(text)
#         return texts


# Step 1: Load and clean the dataset
def clean_text(text_to_clean):
    text_to_clean = text_to_clean.lower()

    # Remove special characters
    text_to_clean = re.sub(r'[^a-zA-Z0-9\s]', '', text_to_clean)

    # Remove numbers
    text_to_clean = re.sub(r'\d+', '', text_to_clean)

    # Remove punctuation
    text_to_clean = re.sub(r'[^\w\s]', '', text_to_clean)

    # Tokenize text
    tokens = word_tokenize(text_to_clean)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens
    cleaned_text = ' '.join(tokens)
    return text_to_clean


df = pd.read_csv('./datasets/Training_Essay_Data.csv')

# Clean the text column
df['clean_text'] = df['text'].apply(clean_text)
df.drop(columns=['generated'], inplace=True)

# Step 2: Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['clean_text'])
vocab_size = len(tokenizer.word_index) + 1

# Save the vocabulary, if does not exist
file_path = 'tokenizer_vocab_with_frequency.csv'
if not os.path.exists(file_path):
    word_index = tokenizer.word_index
    word_counts = tokenizer.word_counts
    word_index_df = pd.DataFrame(word_index.items(), columns=['Word', 'Index'])
    word_counts_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency'])
    tokenizer_vocab_df = pd.merge(word_index_df, word_counts_df, on='Word', how='left')
    tokenizer_vocab_df = tokenizer_vocab_df.sort_values(by='Frequency', ascending=False)
    tokenizer_vocab_df = tokenizer_vocab_df[['Word', 'Index', 'Frequency']]
    tokenizer_vocab_df.to_csv('tokenizer_vocab_with_frequency.csv', index=False)
    print("Vocabulary saved")
else:
    tokenizer_vocab_df = pd.read_csv(file_path)

# Compute word counts
# word_counts = {}
# total_words = 0
# for text in df['clean_text']:
#     for word in text.split():
#         word_counts[word] = word_counts.get(word, 0) + 1
#         total_words += 1
#
# # Print the vocabulary with word frequencies
# for word, index in tokenizer.word_index.items():
#     frequency = word_counts.get(word, 0)
#     print(f'Word: {word}, Index: {index}, Frequency: {frequency}')
#
# # Print the total number of words
# print(f'Total number of words: {total_words}')


# Handling Missing Words
position_index_pairs_all = []
def create_missing_word_examples(text, tokenizer, min_missing_words=1, max_missing_words=4):
    words = text.split()
    if len(words) == 0:
        return 0
    num_missing_words = random.randint(min_missing_words, min(max_missing_words, len(words)))
    missing_word_indices = random.sample(range(len(words)), num_missing_words)
    missing_word_examples = []
    words_with_marks = words.copy()  # Create a copy of the original words list to modify
    position_index_pairs = []
    for index in missing_word_indices:
        # Createing missing word example texts
        missing_word = words[index]
        words_with_marks[index] = '*' + missing_word + '*'
        # position missing word pairs
        position = index
        index_in_vocab = tokenizer.word_index.get(missing_word, 0)
        position_index_pairs.append((position, index_in_vocab))
    position_index_pairs_all.append(position_index_pairs)
    input_text = ' '.join(words_with_marks)  # Join the words with marks back into a string
    output_word_indices = [tokenizer.word_index.get(words[index], 0) for index in missing_word_indices]
    # print(missing_word_indices, output_word_indices)
    # missing_word_examples.append((input_text, output_word_indices))
    missing_word_examples.append(input_text)
    return missing_word_examples


# Creating Training Examples
training_examples = []
for text in df['clean_text']:
    examples = create_missing_word_examples(text, tokenizer)
    if examples == 0:
        continue
    training_examples.extend(examples)

# Convert position_index_pairs to a NumPy array
position_index_pairs_array = np.array(position_index_pairs_all, dtype=object)

file_path2 = 'position_index_pairs.npy'
if not os.path.exists(file_path2):
    # Save the NumPy array to a file
    np.save(file_path2, position_index_pairs_array)
    print("Position-index pairs saved to NumPy array:", file_path2)
else:
    print("Position-index pairs NumPy array file already exists:", file_path2)
    position_index_pairs = np.load(file_path2, allow_pickle=True)

# print(len(training_examples), len(position_index_pairs_all))
# position_index_pairs = np.load('position_index_pairs.npy', allow_pickle=True)
# for sublist in position_index_pairs:
#     print(sublist)

# print("asd")
#
# def text_to_tensor(text, tokenizer):
#     # Tokenize the text
#     tokens = tokenizer.texts_to_sequences([text])[0]
#     tensor_representation = []
#
#     # Iterate through the tokens
#     for i, token in enumerate(tokens):
#         # Check if the token represents the missing word
#         if token in missing_word_index:
#             tensor_representation.append(-1)  # Mark the missing word with -1
#         else:
#             tensor_representation.append(token)  # Append the token index
#
#     return tensor_representation
#
# essays_tensor = []
#
# # Iterate through each essay
# for essay in df['clean_text']:
#     # Convert the essay to tensor representation
#     tensor_representation = text_to_tensor(essay, tokenizer)
#     # Append the tensor representation to the list
#     essays_tensor.append(tensor_representation)
#
# # Convert the list to a numpy array
# essays_tensor = np.array(essays_tensor)
#
# # Print the tensor representation of the first essay
# print("Tensor representation of the first essay:")
# print(essays_tensor[0])
# Vectorization
# max_sequence_length = max(len(text.split()) for text, _ in training_examples)
# X = []
# Y = []
# for text, missing_word_index in training_examples:
#     encoded_text = tokenizer.texts_to_sequences([text])[0]
#     padded_text = pad_sequences([encoded_text], maxlen=max_sequence_length, padding='post')[0]
#     X.append(padded_text)
#     Y.append([missing_word_index])
#
# X = np.array(X)
# Y = np.array(Y)
#
# save_dir = 'dataset'
# os.makedirs(save_dir, exist_ok=True)
#
# X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
# X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
#
# print(X_train)
# Save the datasets to files
# np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
# np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
# np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
# np.save(os.path.join(save_dir, 'Y_train.npy'), Y_train)
# np.save(os.path.join(save_dir, 'Y_val.npy'), Y_val)
# np.save(os.path.join(save_dir, 'Y_test.npy'), Y_test)

import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Custom Tokenizer
class CustomTokenizer:
    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.vocab_size = 0

    def fit_on_texts(self, texts):
        for text in texts:
            self._update_vocab(text)

    def _update_vocab(self, text):
        words = text.split()
        for word in words:
            if word not in self.word_index:
                self.word_index[word] = self.vocab_size
                self.index_word[self.vocab_size] = word
                self.vocab_size += 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = [self.word_index.get(word, 0) for word in text.split()]
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = ' '.join([self.index_word.get(idx, '') for idx in sequence])
            texts.append(text)
        return texts


# Step 1: Load and clean the dataset
def clean_text(text):
    return text


# Load your dataset into a pandas DataFrame
# Assuming your dataset is stored in a CSV file named 'Training_Essay_Data.csv' with a column named 'text'
df = pd.read_csv('./datasets/Training_Essay_Data.csv')

# Clean the text column
df['clean_text'] = df['text'].apply(clean_text)
df.drop(columns=['generated'], inplace=True)

# Step 2: Tokenization
tokenizer = CustomTokenizer()
tokenizer.fit_on_texts(df['clean_text'])
vocab_size = tokenizer.vocab_size + 1

# Compute word counts
word_counts = {}
total_words = 0
for text in df['clean_text']:
    for word in text.split():
        word_counts[word] = word_counts.get(word, 0) + 1
        total_words += 1

# Print the vocabulary with word frequencies
for word, index in tokenizer.word_index.items():
    frequency = word_counts.get(word, 0)
    print(f'Word: {word}, Index: {index}, Frequency: {frequency}')

# Print the total number of words
print(f'Total number of words: {total_words}')


# Step 3: Handling Missing Words
def create_missing_word_examples(text, tokenizer, min_missing_words=2, max_missing_words=4):
    words = text.split()
    num_missing_words = random.randint(min_missing_words, min(max_missing_words, len(words)))
    missing_word_indices = random.sample(range(len(words)), num_missing_words)
    missing_word_examples = []
    words_with_marks = words.copy()  # Create a copy of the original words list to modify
    for index in missing_word_indices:
        missing_word = words[index]
        words_with_marks[index] = '*' + missing_word + '*'  # Mark the missing word with asterisks
    input_text = ' '.join(words_with_marks)  # Join the words with marks back into a string
    output_word_indices = [tokenizer.word_index.get(words[index], 0) for index in missing_word_indices]
    missing_word_examples.append((input_text, output_word_indices))
    return missing_word_examples


# Step 4: Creating Training Examples
training_examples = []
for text in df['clean_text']:
    examples = create_missing_word_examples(text, tokenizer)
    training_examples.extend(examples)

# Step 5: Vectorization
max_sequence_length = max(len(text.split()) for text, _ in training_examples)
X = []
Y = []
for text, missing_word_index in training_examples:
    encoded_text = tokenizer.texts_to_sequences([text])[0]
    padded_text = pad_sequences([encoded_text], maxlen=max_sequence_length, padding='post')[0]
    X.append(padded_text)
    Y.append(missing_word_index)

X = np.array(X)
Y = np.array(Y)

# Step 6: Splitting Dataset
save_dir = 'dataset'
os.makedirs(save_dir, exist_ok=True)

# Splitting Dataset
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

print(X_train)
# Save the datasets to files
# np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
# np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
# np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
# np.save(os.path.join(save_dir, 'Y_train.npy'), Y_train)
# np.save(os.path.join(save_dir, 'Y_val.npy'), Y_val)
# np.save(os.path.join(save_dir, 'Y_test.npy'), Y_test)

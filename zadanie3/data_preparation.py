import pandas as pd
import numpy as np
import random
import os
import re
import torch
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')


# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def text_to_tensor(text, tokenizer, position_index_pairs_all):
    # Tokenize the text
    tokens = tokenizer.texts_to_sequences([text])[0]
    tensor_representation = []

    # Iterate through the tokens
    for i, token in enumerate(tokens):
        tensor_representation.append(token)

    return tensor_representation


def create_missing_word_examples(text, tokenizer_vocab_df, min_missing_words=1, max_missing_words=1, max_essay_length=150):
    words = text.split()[:max_essay_length]
    if len(words) <= max_missing_words + 2:
        return 0

    missing_word_examples = []
    words_with_marks = words.copy()
    position_index_pairs = []

    while len(missing_word_examples) < 1:
        num_missing_words = random.randint(min_missing_words, max_missing_words)
        missing_word_indices = random.sample(range(1, len(words) - 1), num_missing_words)

        for index in missing_word_indices:
            # Choose a random word until it's in the vocabulary
            try_choice = 0
            while True:
                missing_word = words[index]
                index_in_vocab = tokenizer_vocab_df[tokenizer_vocab_df['Word'] == missing_word]['Index'].values
                if len(index_in_vocab) > 0:
                    break

                # Choose another random word
                missing_word = random.choice(list(tokenizer_vocab_df['Word']))
                index_in_vocab = tokenizer_vocab_df[tokenizer_vocab_df['Word'] == missing_word]['Index'].values
                words[index] = missing_word
                try_choice += 1

            words_with_marks[index] = missing_word  # '*' + missing_word + '*'
            # Position missing word pairs
            position = index
            position_index_pairs.append((position, index_in_vocab[0]))

        input_text = ' '.join(words_with_marks)  # Join the words with marks back into a string
        output_word_indices = [tokenizer_vocab_df[tokenizer_vocab_df['Word'] == words[index]]['Index'].values[0] for index in missing_word_indices]
        missing_word_examples.append(input_text)

    position_index_pairs_all.append(position_index_pairs)

    return missing_word_examples


def clean_text(text_to_clean):
    text_to_clean = text_to_clean.lower()

    # Remove special characters, numbers, and punctuation
    text_to_clean = re.sub(r'[^a-zA-Z\s]', '', text_to_clean)

    # Tokenize text
    tokens = word_tokenize(text_to_clean)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize tokens using WordNet
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]

    # Join tokens
    cleaned_text = ' '.join(tokens)
    return cleaned_text


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


os.chdir(r"C:\Users\matul\Desktop\NSIETE\zadanie3")
df = pd.read_csv('./datasets/Training_Essay_Data.csv')

# Check if the cleaned text CSV file exists
cleaned_text_file = 'cleaned_text.csv'
if not os.path.exists(cleaned_text_file):
    print("Cleaning dataset")
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.split().apply(len) > 0]
    df.to_csv(cleaned_text_file, index=False)
    print("Cleaned text saved to:", cleaned_text_file)
else:
    df = pd.read_csv(cleaned_text_file)
    print("Cleaned text loaded from:", cleaned_text_file)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['clean_text'])
vocab_size = len(tokenizer.word_index) + 1

# Save the vocabulary, if does not exist
file_path = 'tokenizer_vocab_with_frequency_embedding.csv'
if not os.path.exists(file_path):
    word_index = tokenizer.word_index
    word_counts = tokenizer.word_counts
    word_index_df = pd.DataFrame(word_index.items(), columns=['Word', 'Index'])
    word_counts_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency'])
    tokenizer_vocab_df = pd.merge(word_index_df, word_counts_df, on='Word', how='left')
    tokenizer_vocab_df = tokenizer_vocab_df.sort_values(by='Frequency', ascending=False)
    tokenizer_vocab_df = tokenizer_vocab_df[['Word', 'Index', 'Frequency']]
    tokenizer_vocab_df = tokenizer_vocab_df[tokenizer_vocab_df['Frequency'] >= 20]

    # Load pre-trained GloVe embeddings
    embeddings_index = {}
    embedding_dim = 50
    glove_file = 'glove.6B.50d.txt'

    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # "Unknown" token and initialization of embedding vector
    unknown_token = '<UNK>'
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    unknown_vector = np.random.normal(size=(embedding_dim,))

    # Update the vocabulary and embedding matrix based on GloVe embeddings
    for index, row in tokenizer_vocab_df.iterrows():
        word = row['Word']
        i = row['Index']
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]
        else:
            tokenizer_vocab_df.at[index, 'Word'] = unknown_token
            embedding_matrix[i] = unknown_vector

    # Remove duplicates and reset indices of the vocabulary DataFrame
    tokenizer_vocab_df = tokenizer_vocab_df.drop_duplicates(subset='Word').reset_index(drop=True)

    tokenizer_vocab_df.to_csv(file_path, index=False)
    print("Vocabulary saved")
else:
    tokenizer_vocab_df = pd.read_csv(file_path)
    print("Vocabulary loaded")

# Handling Missing Words
position_index_pairs_all = []

# Creating Training Examples
training_examples = []
k = 0
for text in df['clean_text']:
    examples = create_missing_word_examples(text, tokenizer_vocab_df)
    if examples == 0:
        continue
    training_examples.extend(examples)
    print(f"Example {k} created")
    k += 1

# Convert position_index_pairs to a NumPy array
file_path2 = 'position_index_pairs.npy'
position_index_pairs_array = np.array(position_index_pairs_all, dtype=object)
np.save(file_path2, position_index_pairs_array)
print("Position-index pairs saved to NumPy array:", file_path2)

# Calculate average and minimum length of essays
essay_lengths = [len(text.split()) for text in df['clean_text']]
average_length = sum(essay_lengths) / len(essay_lengths)
min_length = min(essay_lengths)

# Vectorization
max_sequence_length = 150

essays_tensor = []

# Iterate through each essay
k = 0
for essay, position_index_pair in zip(training_examples, position_index_pairs_all):
    tokens = essay.split()
    token_indices = []

    # Convert each token to its corresponding index in the vocabulary or -1 if missing
    for i, token in enumerate(tokens):
        if any(position == i for position, _ in position_index_pair):
            token_indices.append(-1)  # Missing word represented as -1
        else:
            index = tokenizer_vocab_df[tokenizer_vocab_df['Word'] == token]['Index'].values
            if len(index) > 0:
                token_indices.append(index[0])
            else:
                # If the token is not in the vocabulary -> unknown token
                token_indices.append(-2)

    # Pad the sequence to the maximum length
    if len(token_indices) > max_sequence_length:
        token_indices = token_indices[:max_sequence_length]
    else:
        token_indices += [0] * (max_sequence_length - len(token_indices))

    essays_tensor.append(token_indices)
    print(f"token indices {k} appended")
    k += 1


# Convert the list to a numpy array
essays_tensor = np.array(essays_tensor)

# Save essays as a number representation in a tensor
torch.save(essays_tensor, 'essays_tensor.pt')

# Convert the list to a numpy array
essays_tensor = np.array(essays_tensor)

# Print statistics
print("Average length of essays:", average_length)
print("Minimum length of essays:", min_length)
print("Maximum length of essays:", max_sequence_length)

# Print the tensor representation of the first essay
print("Tensor representation of the first essay:")
print(essays_tensor[0])


# Load essays tensor and position index pairs
essays_tensor = torch.load('essays_tensor.pt')
position_index_pairs_array = np.load('position_index_pairs.npy', allow_pickle=True)
position_index_pairs_array = np.array(position_index_pairs_array)  # Convert to NumPy array
essay_representation = [' '.join(map(str, essay)) for essay in essays_tensor]
position_index_pairs = position_index_pairs_all

# Save the essay representations and position-index pairs separately in the output folder
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)
essay_representation_file = os.path.join(output_folder, 'essays_tensor_representation_max150_1miss.npy')
np.save(essay_representation_file, np.array(essays_tensor))
print("Essay representations saved to:", essay_representation_file, f"Type: {essay_representation}")
position_index_pairs_file = os.path.join(output_folder, 'position_index_pairs_max150_1miss.npy')
np.save(position_index_pairs_file, position_index_pairs_array)
print("Position-index pairs saved to:", position_index_pairs_file, f"Type: {position_index_pairs_array.dtype}")


data = {
    'EssayRepresentation': essay_representation,
    'PositionIndexPairs': position_index_pairs
}

df = pd.DataFrame(data)
df.to_csv(os.path.join(output_folder, 'essays_with_positions_max150_1miss.csv'), index=False)

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

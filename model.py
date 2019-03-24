from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from os import listdir
import string
from keras.models import Model

from os import listdir
import string

"""
Data Preprocessing
"""


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a document into news story and highlights
def split_story(doc):
    # find first highlight
    index = doc.find('@highlight')
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights


# load all stories in a directory
def load_stories(directory):
    stories = list()
    for name in listdir(directory):
        filename = directory + '/' + name
        # load document
        doc = load_doc(filename)
        # split into story and highlights
        story, highlights = split_story(doc)
        # store
        stories.append({'story': story, 'highlights': highlights})
    return stories


# clean a list of lines
def clean_lines(lines):
    cleaned = list()
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # strip source cnn office if it exists
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index + len('(CNN)'):]
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [w.translate(table) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        # store as string
        cleaned.append(' '.join(line))
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned


# load stories
directory = 'cnn/story/'
stories = load_stories(directory)
print('Loaded Stories %d' % len(stories))

# clean stories
for example in stories:
    example['story'] = clean_lines(example['story'].split('\n'))
    example['highlights'] = clean_lines(example['highlights'])


"""
Building the model
"""


MAX_SEQUENCE_LENGTH = 1000

tokenizer = Tokenizer()
X, y = [' '.join(t['story']) for t in stories], [' '.join(t['highlights']) for t in stories]

total = X + y
tokenizer.fit_on_texts(total)
sequences_X = tokenizer.texts_to_sequences(X)
sequences_y = tokenizer.texts_to_sequences(y)

word_index = tokenizer.word_index

data = pad_sequences(sequences_X, maxlen=MAX_SEQUENCE_LENGTH)
labels = pad_sequences(sequences_y, maxlen=100) # test with maxlen=100

# train/test split
TEST_SIZE = 5
X_train, y_train, X_test, y_test = data[:-TEST_SIZE], labels[:-TEST_SIZE], data[-TEST_SIZE:], labels[-TEST_SIZE:]

# create model
# encoder
inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

encoder1 = Embedding(len(word_index) + 1, 128, input_length=MAX_SEQUENCE_LENGTH)(inputs)
encoder2 = LSTM(128)(encoder1)
encoder3 = RepeatVector(2)(encoder2)

# decoder
decoder1 = LSTM(128, return_sequences=True)(encoder3)
outputs = TimeDistributed(Dense(len(word_index) + 1, activation='softmax'))(decoder1)

model = Model(inputs=inputs, outputs=outputs)

# loss function
model.compile(loss='categorical_crossentropy', optimizer='adam')

batch_size = 3
epochs = 4

# Train
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)

# Predict

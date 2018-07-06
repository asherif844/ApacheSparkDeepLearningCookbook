
# coding: utf-8

# In[1]:


from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


# In[2]:


pwd


# In[3]:


cd '/Users/Chanti/Desktop/Cookbook/Chapter 8'


# In[4]:


pwd


# In[5]:


# load doc into memory
def load_document(name):
    file = open(name, 'r')
    text = file.read()
    file.close()
    return text


# In[6]:


# load document
input_filename = 'junglebook.txt'
doc = load_document(input_filename)
print(doc[:2000])


# In[7]:


import string
 
# turn a document into clean tokens
def clean_document(doc):
    doc = doc.replace('--', ' ')
    tokens = doc.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens


# In[8]:


# clean document
tokens = clean_document(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))


# In[9]:


# organize into sequences (of length 50) of tokens
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    sequences.append(line)
print('Total Sequences: %d' % len(sequences))


# In[10]:


# save tokens to file, one dialog per line
def save_document(lines, name):
    data = '\n'.join(lines)
    file = open(name, 'w')
    file.write(data)
    file.close()


# In[11]:


# save sequences to file
output_filename = 'junglebook_sequences.txt'
save_document(sequences, output_filename)


# In[12]:


# load document into memory
def load_document(name):
    file = open(name, 'r')
    text = file.read()
    file.close()
    return text
 
# load
input_filename = 'junglebook_sequences.txt'
doc = load_document(input_filename)
lines = doc.split('\n')


# In[13]:


# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)


# In[14]:


# vocabulary size
vocab_size = len(tokenizer.word_index) + 1 


# In[15]:


# separate into input and output
sequences = array(sequences)
Input, Output = sequences[:,:-1], sequences[:,-1]
Output = to_categorical(Output, num_classes=vocab_size)
sequence_length = Input.shape[1]


# In[16]:


# define model
from keras.layers import Dropout
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=sequence_length))
model.add(LSTM(200, return_sequences=True))
model.add(LSTM(200))
model.add(Dropout(0.3))
model.add(Dense(200, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())


# In[17]:


# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(Input, Output, batch_size=250, epochs=75)


# In[18]:


# save the model to file
model.save('junglebook_trained.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))


# In[19]:


# load doc into memory
def load_document(name):
    file = open(name, 'r')
    text = file.read()
    file.close()
    return text
 
# load cleaned text sequences
input_filename = 'junglebook_sequences.txt'
doc = load_document(input_filename)
lines = doc.split('\n')


# In[20]:


sequence_length = len(lines[0].split()) - 1


# In[21]:


# load the model
from keras.models import load_model
model = load_model('junglebook_trained.h5')


# In[22]:


# select a seed text
from random import randint
seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n')


# In[23]:


encoded = tokenizer.texts_to_sequences([seed_text])[0]


# In[24]:


from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
 
# load doc into memory
def load_document(name):
    file = open(name, 'r')
    text = file.read()
    file.close()
    return text
 
# generate a sequence from a language model
def generate_sequence(model, tokenizer, sequence_length, seed_text, n_words):
	result = list()
	input_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([input_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		prediction = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == prediction:
				out_word = word
				break
		# append to input
		input_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)
 
# load cleaned text sequences
input_filename = 'junglebook_sequences.txt'
doc = load_document(input_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1


# In[25]:


# load the model
model = load_model('junglebook_trained.h5')
 
# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
 
# select a seed text
seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n')
 
# generate new text
generated = generate_sequence(model, tokenizer, sequence_length, seed_text, 50)
print(generated)


# In[26]:


# load the model
model = load_model('junglebook_trained.h5')
 
# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
 
# select a seed text
seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n')
 
# generate new text
generated = generate_sequence(model, tokenizer, sequence_length, seed_text, 50)
print(generated)


# In[29]:


# load the model
model = load_model('junglebook_trained.h5')
 
# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
 
# select a seed text
seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n')
 
# generate new text
generated = generate_sequence(model, tokenizer, sequence_length, seed_text, 50)
print(generated)


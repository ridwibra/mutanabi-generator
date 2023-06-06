from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from keras.layers import Embedding, GRU, Dense
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import PyPDF2
import docx
import re
from keras.losses import sparse_categorical_crossentropy

app = FastAPI()

origins = [
    # "http://localhost.tiangolo.com",
    # "https://localhost.tiangolo.com",
    # "http://localhost",
    #"http://localhost:3000",
    "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


files = ['ديوان المتنبي.pdf', '/ديوان أبي الطيب المتنبي.doc', 'mutanabiTwo.docx', 'mutanabiOne.docx']
text = ''

for file in files:
    if file.endswith('.docx'):
        doc = docx.Document(file)
        text += '\n'.join([p.text for p in doc.paragraphs]) + '\n\n'
    elif file.endswith('.pdf'):
        with open(file, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            for page in range(len(pdf.pages)):
                text += pdf.pages[page].extract_text() + '\n\n'

# Split the text into separate lines
text = text.replace(' / ', '\n')

# Remove httpwwwshamelaws
text = text.replace('httpwwwshamelaws', '')

# Remove parentheses but keep their content
text = re.sub(r'\(([^)]+)\)', r'\1', text)

# Remove square brackets and their content
text = re.sub(r'\[[^\]]+\]', '', text)

# Remove fractions
text = re.sub(r'\d+/\d+', '', text)

# Remove horizontal lines
text = re.sub(r'_{3,}', '', text)

# Put each text in parentheses on a separate line
text = re.sub(r'\(([^)]+)\)', r'\n\1\n', text)

# Put each number at the beginning of a sentence
text = re.sub(r'(?<=\n)\d+\.', r'\n\g<0>', text)

# Remove special characters except for asterisks
text = re.sub(r'[^\w\s*]', '', text)

# Remove standalone occurrences of the Arabic letter ص
text = re.sub(r'(?<=\n)ص(?=\n)', '', text)

# Split the text into paragraphs
paragraphs = text.split('\n\n')

# Remove short lines below the paragraph before the last paragraph
if len(paragraphs) > 1:
    paragraphs[-2] = re.sub(r'\n-+\n$', '', paragraphs[-2])

# Join the paragraphs back together
text = '\n\n'.join(paragraphs)

# Split the text into sentences and remove duplicates
sentences = list(set(text.split('\n')))
sentences.sort(key=text.index)
text = '\n'.join(sentences)

# Remove all numbers from the text
text = re.sub(r'\d+', '', text)
# # Read, then decode for py2 compat.
# text = open('poems', 'rb').read().decode(encoding='utf-8')

# # remove some exteranous chars
# execluded = '!()*-.1:=[]«»;؛,،~?؟#\u200f\ufeff'
# out = ""
#
# for char in text:
#     if char not in execluded:
#         out += char
# text = out
# text = text.replace("\t\t\t", "\t")
# text = text.replace("\r\r\n", "\n")
# text = text.replace("\r\n", "\n")
# text = text.replace("\t\n", "\n")
vocab = sorted(set(text))

char_to_ind = {char: ind for ind, char in enumerate(vocab)}
ind_to_char = np.array(vocab)

def sparse_cat_loss(y_true, y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim,
                        batch_input_shape=[batch_size, None]))
    model.add(GRU(rnn_neurons, return_sequences=True, stateful=True,
                  recurrent_initializer='glorot_uniform', reset_after=True))
    model.add(Dense(vocab_size))
    model.compile('adam', loss=sparse_cat_loss)
    return model


model = create_model(vocab_size=len(vocab), embed_dim=64,
                     rnn_neurons=1024, batch_size=1)

model.load_weights('my_model_weights.h5')
model.build(tf.TensorShape([1, None]))



def generate_text(model, start_seed, gen_size=500, temp=1.0):
    # number to generate
    num_generate = gen_size
    # evaluate the input text and convert the text to index
    input_eval = [char_to_ind[s] for s in start_seed]
    # expand it to meet the batch format shape
    input_eval = tf.expand_dims(input_eval, 0)
    # holds the generated text
    text_generated = []
    # how surprising you want the results to be
    temperature = temp
    # reset the state of the model
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch shape dimension
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(ind_to_char[predicted_id])
    return (start_seed + "".join(text_generated))


@app.get('/')
async def root():
    return "Welcome"


@app.get('/generate')
# async def receive(seed: str = Form(...), length: int = Form(...)):
async def receive(seed: str, length: int):
    # print(seed)
    # print(length)
    result = generate_text(model, seed, gen_size=length)
    # print(result)
    return result

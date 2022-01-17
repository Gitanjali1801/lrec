
# %%
import warnings
import os

# %%
from tensorflow.python.training.tracking.util import Checkpoint
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import tensorflow as tf

# %%
from tensorflow.python.keras import backend as K

# %%
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
K.set_session(tf.compat.v1.Session(config=config))
warnings.filterwarnings('ignore', category=FutureWarning)
tf.compat.v1.enable_eager_execution()

# %%
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input
import tensorflow as tf
import tensorflow.keras as keras
import os,re, codecs
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import *
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm
from tensorflow.keras.models import Model
import random
# import bert
from tensorflow.keras.layers import Layer, Input, GRU,Bidirectional,Dense, Dropout, Embedding, Reshape, LSTM, Flatten, Conv1D,SpatialDropout1D, MaxPooling1D, GlobalMaxPooling1D,MaxPool2D, MaxPool1D, Concatenate,Conv2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, BatchNormalization, LayerNormalization

# %% [markdown]
# %%

# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras import initializers
import random
from tensorflow.keras.callbacks import ModelCheckpoint
from numpy import asarray, zeros
import os
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import string
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

print(tf.__version__)

import tensorflow_hub as hub

# %%
import tensorflow as tf
import numpy as np
from transformers import TFRobertaModel, RobertaTokenizer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# %%
from transformers import AutoTokenizer, AutoModel
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Input, Dropout, Flatten
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np

# %%
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from collections import defaultdict
import regex as re
from tensorflow.keras.preprocessing import image as keras_image
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# %%
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# %%
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# %%
def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r))

# %%
epochs=32
batch_size=64
metrics = False
plot_model_diagram=False
summary=False
epochs = epochs
metrics = metrics
batch_size = batch_size
plot_model_diagram = plot_model_diagram
summary = summary
seq_len = 42
bert_layer = TFBertModel.from_pretrained('bert-base-multilingual-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
vgg = VGG19(weights='imagenet', include_top=False)
bert_layer.trainable = False
vgg.trainable = False

# %% [markdown]
# encoded_input = tokenizer(text, return_tensors='tf')<br>
# output = model(encoded_input)

# %%
def encode(texts):
    input_id = []
    token_type_id = []
    attention_mask = []
    for text in texts:
        dictIn = tokenizer.encode_plus(text, max_length=128, pad_to_max_length=True,truncation=True)
        # dictIn = tokenizer(text, return_tensors='tf')
        input_id.append(dictIn['input_ids'])
        token_type_id.append(dictIn['token_type_ids'])
        attention_mask.append(dictIn['attention_mask'])
    return np.array(input_id), np.array(token_type_id), np.array(attention_mask)

# %%
input_id = Input(shape=(128,), dtype=tf.int64)
mask_id = Input(shape=(128,), dtype=tf.int64)
seg_id = Input(shape=(128,), dtype=tf.int64)
key=bert_layer([input_id, mask_id, seg_id])
dense = tf.keras.layers.Dense(768, activation='relu')(key[0])
dense = tf.keras.layers.Dense(256, activation='relu')(dense)
txt_repr = Dropout(0.4)(dense)
txt_repr

# %%
list_urls = []
list_Overallsentiment = []
images = []
sentiment = []
cnt = 0
list_Correctedtext = []


# %%
import pickle

with open('/Humor.pkl', 'rb') as f:
  mynewlist13 = pickle.load(f)

with open('/Valence.pkl', 'rb') as f:
  mynewlist24 = pickle.load(f)
with open('/Arousal.pkl', 'rb') as f:
  mynewlist25 = pickle.load(f)
with open('/train_text.pkl', 'rb') as f:
  mynewlist26 = pickle.load(f)
with open('/train_imgs.pkl', 'rb') as f:
  mynewlist27 = pickle.load(f)

# %%
print(len(mynewlist24))
print(len(mynewlist25))
print(len(mynewlist26))
print(len(mynewlist27))

# %%
print(len(mynewlist24))
training_set = pd.DataFrame( {'Text':mynewlist26,'Images': mynewlist27,'Humor':mynewlist13,'Valence':mynewlist24,'Arousal':mynewlist25})

# %%
print(len(training_set.Arousal))
print(len(training_set.Text))
print(len(training_set.Images))
print(len(training_set.Humor))
print(len(training_set.Valence))
train,test= train_test_split(training_set,test_size=0.20,random_state=123, shuffle=False)

X_train= train["Text"]
X_test= test["Text"]

# %%
y_train_Humor=train["Humor"].values
y_test_Humor=test["Humor"].values

# %%
y_train_Valence=train["Valence"].values
y_test_Valence=test["Valence"].values

# %%
y_train_Arousal=train["Arousal"].values
y_test_Arousal=test["Arousal"].values
print(len(y_train_Humor))
print(len(y_test_Humor))
print(len(y_train_Valence))
print(len(y_test_Valence))
print(len(y_train_Arousal))
print(len(y_test_Arousal))

# %%
model = VGG19(weights="imagenet", include_top=False)
for layer in model.layers:
    layer.trainable = False
model.summary()

# %%
t_imgss = np.vstack(train.Images)
# v_imgss = np.vstack(val_imgs)
train_img_features = model.predict(t_imgss, batch_size=32)
print(" train features : ", train_img_features.shape)
te_imgss = np.vstack(test.Images)
test_img_features = model.predict(te_imgss, batch_size=32)
print(" train features : ", test_img_features.shape)

trainshape = []
for i in range(len(train_img_features)):
    trainshape.append(np.reshape(train_img_features[i], (49, 512)))
trainshape = np.array(trainshape)
testshape = []
for i in range(len(test_img_features)):
    testshape.append(np.reshape(test_img_features[i], (49, 512)))
testshape = np.array(testshape)

# %%
print("shape of 1 img train ftrs", trainshape.shape)
print("shape of 1 img test ftrs", testshape.shape)
image_input = Input(shape=([trainshape.shape[1], trainshape.shape[2]]))
print(image_input)
image_reshape = Reshape((49, 512, 1))(image_input)
print("imgggggg ip", image_reshape)
print("imgggggg ip", image_reshape[6])

print("\n \n This is for Humor")
label_encoder1 = LabelEncoder()
values = array(y_train_Humor)
print("train_senti", values[1:10])
train_enc = label_encoder1.fit_transform(values)
print("train_integer_encoded", train_enc[1:5])
train_lbl = to_categorical(train_enc)
print(train_lbl[1:10])
label_encoder2=LabelEncoder()
values_test=array(y_test_Humor)
print("test_senti",values_test[1:10])
test_enc=label_encoder2.fit_transform(values_test)
print("test_integer_encoded",test_enc[1:5])
test_lbl=to_categorical(test_enc)
# print("test lbl",test_lbl[1:50])
print("len of tst_lbl",test_lbl[1:5])

# %%
print("\n \n This is for Valence")
label_encoder3 = LabelEncoder()
values2 = array(y_train_Valence)
print("train_senti", values2[1:10])
train_enc_2 = label_encoder3.fit_transform(values2)
print("train_integer_encoded", train_enc_2[1:5])
train_lbl_2 = to_categorical(train_enc_2)
# print(train_lbl[1:50])
print(train_lbl_2[1:10])
label_encoder4=LabelEncoder()
values_test2=array(y_test_Valence)
print("test_senti",values_test2[1:5])
test_enc2=label_encoder4.fit_transform(values_test2)
print("test_integer_encoded",test_enc2[1:5])
test_lbl2=to_categorical(test_enc2)
# print("test lbl",test_lbl[1:50])
print("len of tst_lbl",test_lbl2[1:5])

# %%
print("\n \n This is for Level Arousal")
label_encoder5 = LabelEncoder()
values3 = array(y_train_Arousal)
print("train_senti", values3[1:10])
train_enc_3 = label_encoder5.fit_transform(values3)
print("train_integer_encoded", train_enc_3[1:5])
train_lbl_3 = to_categorical(train_enc_3)
# print(train_lbl[1:50])
print(train_lbl_3[1:10])

# %%
label_encoder6=LabelEncoder()
values_test3=array(y_test_Arousal)
print("test_senti",values_test3[1:5])
test_enc3=label_encoder6.fit_transform(values_test3)
print("test_integer_encoded",test_enc3[1:5])
test_lbl3=to_categorical(test_enc3)
# print("test lbl",test_lbl[1:50])
print("len of tst_lbl",test_lbl3[1:5])

# %%
print(len(y_train_Humor))
h_train,h_test=[],[]
for i in range(len(y_train_Humor)):
  if (y_train_Humor[i]==1 or y_train_Humor[i]==2):
    h_train.append(1)
  else : h_train.append(0)
print(len(h_train))
print(len(y_test_Humor))
for i in range(len(y_test_Humor)):
  if (y_test_Humor[i]==1 or y_test_Humor[i]==2):
    h_test.append(1)
  else : h_test.append(0)
print(len(h_test))

# %%
print(len(y_train_Valence))
v_train,v_test=[],[]
for i in range(len(y_train_Valence)):
  if (y_train_Valence[i]==1 or y_train_Valence[i]==2):
    v_train.append(0)
  else : v_train.append(1)
print(len(v_train))
print(len(y_test_Valence))
for i in range(len(y_test_Valence)):
  if (y_test_Valence[i]==1 or y_test_Valence[i]==2):
    v_test.append(0)
  else : v_test.append(1)
print(len(v_test))

# %%
print(len(y_train_Arousal))
a_train,a_test=[],[]
for i in range(len(y_train_Arousal)):
  if (y_train_Arousal[i]==1 or y_train_Arousal[i]==2):
    a_train.append(0)
  else : a_train.append(1)
print(len(a_train))
print(len(y_test_Arousal))
for i in range(len(y_test_Arousal)):
  if (y_test_Arousal[i]==1 or y_test_Arousal[i]==2):
    a_test.append(0)
  else : a_test.append(1)
print(len(a_test))

# %%
print("\n \n This is for Level Humor/non-humor")
label_encoder13 = LabelEncoder()
values7 = array(h_train)
print("train_senti", values7[1:10])
train_enc_7 = label_encoder13.fit_transform(values7)
print("train_integer_encoded", train_enc_7[1:5])
train_lbl_7 = to_categorical(train_enc_7)
# print(train_lbl[1:50])
print(train_lbl_7[1:10])

# %%
label_encoder14=LabelEncoder()
values_test7=array(h_test)
print("test_senti",values_test7[1:5])
test_enc7=label_encoder14.fit_transform(values_test7)
# print("Emotion1_encoded",test_enc7)
test_lbl7=to_categorical(test_enc7)
# print("test lbl",test_lbl[1:50])
print("Emotion_1_lbl",test_lbl7[1:5])
print(len(test_lbl7))

# %% [markdown]
# %%

# %%
print("\n \n This is for Level Valence/Non-valence")
label_encoder15 = LabelEncoder()
values8 = array(v_train)
print("train_senti", values8[1:10])
train_enc_8 = label_encoder15.fit_transform(values8)
print("train_integer_encoded", train_enc_8[1:5])
train_lbl_8 = to_categorical(train_enc_8)
# print(train_lbl[1:50])
print(train_lbl_8[1:10])

# %%
label_encoder16=LabelEncoder()
values_test8=array(v_test)
print("test_senti",values_test8[1:5])
test_enc8=label_encoder16.fit_transform(values_test8)
# print("Emotion1_encoded",test_enc8)
test_lbl8=to_categorical(test_enc8)
# print("test lbl",test_lbl[1:50])
print("Emotion_1_lbl",test_lbl8[1:5])
print(len(test_lbl8))

# %%
print("\n \n This is for Level Arousal/Non-Arousal")
label_encoder17 = LabelEncoder()
values9 = array(a_train)
print("train_senti", values9[1:10])
train_enc_9 = label_encoder17.fit_transform(values9)
print("train_integer_encoded", train_enc_9[1:5])
train_lbl_9 = to_categorical(train_enc_9)
# print(train_lbl[1:50])
print(train_lbl_9[1:10])

# %%
label_encoder18=LabelEncoder()
values_test9=array(a_test)
print("test_senti",values_test9[1:5])
test_enc9=label_encoder18.fit_transform(values_test9)
# print("Emotion1_encoded",test_enc9)
test_lbl9=to_categorical(test_enc9)
# print("test lbl",test_lbl[1:50])
print("Emotion_1_lbl",test_lbl9[1:5])
print(len(test_lbl9))

# %%
class AttLayer(Layer):
    def __init__(self, **kwargs):
        super(AttLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel',shape=(input_shape[-1],),initializer='random_normal',trainable=True)
        super(AttLayer, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x, mask=None):
        eij = K.tanh(K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1))
        ai = K.exp(eij)
        weights = ai / K.expand_dims(K.sum(ai, axis=1), 1)
        weighted_input = x * K.expand_dims(weights, 2)
        return K.sum(weighted_input, axis=1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# %%
num_filters = 150  # 128+
filter_sizes = [2, 3, 4]  # 234
drop = 0.5
batch_size = 36
epochs = 75
print(image_input.shape)
img_f = AttLayer()(image_input)
dropout_img = Dropout(drop)(img_f)
print(dropout_img.shape)

dense = tf.keras.layers.Dense(256, activation='relu')(dropout_img)
img_repr = Dropout(0.4)(dense)
img_repr

txt_repr1 = AttLayer()(txt_repr)
print(txt_repr1.shape)
concat = Concatenate(axis=1)([img_repr, txt_repr1])
print(concat)
Dense_layer2 = tf.keras.layers.Dense(128,activation='relu', name="Dense_layer2")(concat)

humor_task = tf.keras.layers.Dense(128,activation='relu', name="humor_task")(Dense_layer2)
humor = tf.keras.layers.Dense(2, activation='softmax', name="predictions_task1")(humor_task)
valence_task = tf.keras.layers.Dense(128,activation='relu', name="valence_task")(Dense_layer2)
valence= tf.keras.layers.Dense(2, activation='softmax', name="predictions_task2")(valence_task)
# #Task Specific Layer for Task 14
arousal_task = tf.keras.layers.Dense(128,activation='relu', name="arousal_task")(Dense_layer2)
arousal = tf.keras.layers.Dense(2, activation='softmax', name="predictions_task3")(arousal_task)
# %%
input_id_train, token_type_id_train, attention_mask_train = encode(X_train)
input_id_test, token_type_id_test, attention_mask_test = encode(X_test)

model_ensemble = Model(inputs=[input_id, mask_id, seg_id, image_input], outputs=[humor,valence,arousal])
adam = Adam(lr=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
es = EarlyStopping(monitor='val_predictions_task1_accuracy', mode='max', verbose=1, patience= 10)
model_ensemble.compile(optimizer=Adam(), loss={'predictions_task1':'binary_crossentropy','predictions_task2':'binary_crossentropy','predictions_task3':'binary_crossentropy'},\
            metrics={'predictions_task1':'accuracy','predictions_task2':'accuracy','predictions_task3':'accuracy'}, run_eagerly=True)
model_ensemble.summary()

# %%
history = model_ensemble.fit([input_id_train, token_type_id_train, attention_mask_train, trainshape],\
     [train_lbl_7,train_lbl_8,train_lbl_9],validation_split=0.10,batch_size=16,epochs=30,  callbacks=[es],verbose=1)


# %%
input_id_test, token_type_id_test, attention_mask_test = encode(X_test)

# %%
predictions_text = model_ensemble.predict([input_id_test, token_type_id_test, attention_mask_test, testshape])
y_class0=predictions_text[0]
y_class1=predictions_text[1]
y_class2=predictions_text[2]
print(predictions_text[0])

print(y_class0[2])

# %%
predicted1 = np.argmax(y_class0, axis=1)
predicted2 = np.argmax(y_class1, axis=1)
predicted3 = np.argmax(y_class2, axis=1)
class_rep1=classification_report(test_enc7, predicted1)
class_rep2=classification_report(test_enc8, predicted2)
class_rep3=classification_report(test_enc9, predicted3)
print("confusion matrix1",confusion_matrix(test_enc7, predicted1))
print("confusion matrix2",confusion_matrix(test_enc8, predicted2))
print("confusion matrix3",confusion_matrix(test_enc9, predicted3))
print(class_rep1)
print(class_rep2)
print(class_rep3)
print("precision_recall_fscore_support_micro",precision_recall_fscore_support(test_enc7, predicted1, average='micro'))
print("precision_recall_fscore_support_micro",precision_recall_fscore_support(test_enc8, predicted2, average='micro'))
print("precision_recall_fscore_support_micro",precision_recall_fscore_support(test_enc9, predicted3, average='micro'))

print("precision_recall_fscore_support_macro",precision_recall_fscore_support(test_enc7, predicted1, average='macro'))
print("precision_recall_fscore_support_macro",precision_recall_fscore_support(test_enc8, predicted2, average='macro'))
print("precision_recall_fscore_support_macro",precision_recall_fscore_support(test_enc9, predicted3, average='macro'))
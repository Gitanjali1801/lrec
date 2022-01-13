
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

# %% [markdown]
# !pip install bert-for-tf2

# %%
print(tf.__version__)

# %% [markdown]
# !pip install transformers

# %% [markdown]
# import bert

# %%
import tensorflow_hub as hub

# %%
import tensorflow as tf
import numpy as np
from transformers import TFRobertaModel, RobertaTokenizer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# %%
from transformers import AutoTokenizer, AutoModel
# from preprocessing import preprocess_image, preprocess_txt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Input, Dropout, Flatten
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image as keras_image
# from metrics import precision, recall, f1
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
#  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
epochs = epochs
metrics = metrics
batch_size = batch_size
plot_model_diagram = plot_model_diagram
summary = summary
seq_len = 42
# bert_layer = TFBertModel.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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

# %% [markdown]
# 

# %% [markdown]
# file_train = open(file1, 'r', encoding='utf-8')<br>
# # file_test = open(file2, 'r', encoding='latin-1')<br>
# # file_val = open(file3, 'r', encoding='latin-1')<br>
# # file_test_new = open(file4, 'r', encoding='latin-1')<br>
# data_train= pd.read_csv(file_train)<br>
# # data_test = pd.read_csv(file_test)<br>
# # data_val = pd.read_csv(file_val)<br>
# # data_test_new = pd.read_csv(file_test_new)<br>
# # data_train=data_train[:2000]<br>
# #%%<br>
# # data_train=data_train[:2000]<br>
# print(data_train.head(5))<br>
# # x_train_original = list(data_train_original["text"])<br>
# x_train = list(data_train["text"])<br>
# print(len(x_train))<br>
# # print("corrected text x[i]", x_train[10])

# %% [markdown]
# # y_train = list(data_train["Level1"])<br>
# # print(len(y_train))<br>
# # print("corrected text x[i]", y_train[10])

# %% [markdown]
# # data_train.fillna({'Level1':-1}, inplace=True)<br>
# # data_train.replace({'Level1':{0:0, 1:1}}, inplace=True)

# %% [markdown]
# print(data_train.columns)<br>
# # data_train=data_train[:500]<br>
# print(len(data_train))

# %% [markdown]
# print(data_train.columns)

# %% [markdown]
# from PIL import ImageFile<br>
# ImageFile.LOAD_TRUNCATED_IMAGES = True #Fear,Neglect,irritation,Rage,Disgust,Nervousness,Shame,Disappointment,Envy,Suffering,Sadness,Joy,Pride,Surprise<br>
# train_imgs = []<br>
# train_text = []<br>
# train_img_name = []<br>
# train_off=[]<br>
# train_level2=[]<br>
# Valence=[]<br>
# Arousal=[]<br>
# Sarcasm=[]<br>
# Humor=[]<br>
# Fear=[]<br>
# Neglect=[]<br>
# irritation=[]<br>
# Rage=[]<br>
# target_1, target_2 , target_3,target_4, target_5 , target_6,target_7=[],[],[],[],[],[],[]<br>
# # Disgust,Nervousness,Shame,Disappointment,Envy,Suffering,Sadness,Joy,Pride,Surprise=[],[],[],[],[],[],[],[],[],[]<br>
# Disgust,Nervousness,Shame,Disappointment,Envy,Suffering,Sadness,Joy,Pride,Surprise=[],[],[],[],[],[],[],[],[],[]<br>
# Valence=[]<br>
# Arousal=[]<br>
# path = "/content/drive/MyDrive/HELIOS_MATERIAL_IIT_PATNA/MEMES/memes_our_dataset_hindi/my_meme_data"<br>
# # path = "/DATA/gitanjali_2021cs03/code/meme_H_NH/my_dataset_work/my_meme_data"<br>
# for i in range(len(data_train)):<br>
#     img_name = data_train.Name.values[i]<br>
#     flag = 0<br>
#     for im in os.listdir(path):  # if image from actual data is present in img folder<br>
#         # print("im ::::::;",im)<br>
#         if img_name == im:<br>
#             # imagePath=path+"\\"+img_name<br>
#             imagePath = path + "/" + img_name<br>
#             image = load_img(imagePath, target_size=(224, 224))<br>
#             # image1 = img_to_array(image)<br>
#             # print(image1.shape)<br>
#             # image = np.expand_dims(image1, axis=0)<br>
#             # print(image.shape)<br>
#             # image = imagenet_utils.preprocess_input(image)<br>
#             # train_imgs.append(image)  # img ftr<br>
#             # train_text.append(data_train.text.values[i])  # text<br>
#             train_off.append(data_train.Level1.values[i])<br>
#             train_level2.append(data_train.Level2.values[i]) #<br>
#             # Valence.append(data_train.Valence.values[i])<br>
#             # Arousal.append(data_train.Arousal.values[i])<br>
#             Sarcasm.append(data_train.Sarcasm.values[i])<br>
#             Humor.append(data_train.Humor.values[i])<br>
#             Fear.append(data_train.Fear.values[i])<br>
#             Neglect.append(data_train.Neglect.values[i])<br>
#             irritation.append(data_train.irritation.values[i])<br>
#             Rage.append(data_train.Rage.values[i])<br>
#             Disgust.append(data_train.Disgust.values[i])<br>
#             Nervousness.append(data_train.Nervousness.values[i])<br>
#             Shame.append(data_train.Shame.values[i])<br>
#             Disappointment.append(data_train.Disappointment.values[i])<br>
#             Envy.append(data_train.Envy.values[i])<br>
#             Suffering.append(data_train.Suffering.values[i])<br>
#             Sadness.append(data_train.Sadness.values[i])<br>
#             Pride.append(data_train.Pride.values[i])<br>
#             Joy.append(data_train.Joy.values[i])<br>
#             # Surprise.append(data_train.Surprise.values[i])<br>
#             target_1.append(data_train.Level10_1.values[i])<br>
#             target_2.append(data_train.Level10_2.values[i])<br>
#             target_3.append(data_train.Level10_3.values[i])<br>
#             target_4.append(data_train.Level10_4.values[i])<br>
#             target_5.append(data_train.Level10_5.values[i])<br>
#             target_6.append(data_train.Level10_6.values[i])<br>
#             target_7.append(data_train.Level10_7.values[i])<br>
#             train_img_name.append(im)  # img name<br>
#             flag = 1<br>
#             break

# %% [markdown]
#     if (flag == 0):<br>
#         print("not found,:::: ", img_name)

# %% [markdown]
# file_name = "train_off.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(train_off, open_file)<br>
# open_file.close()<br>
# file_name = "train_level2.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(train_level2, open_file)<br>
# open_file.close()<br>
# file_name = "Disgust.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(Disgust, open_file)<br>
# open_file.close()<br>
# file_name = "Nervousness.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(Nervousness, open_file)<br>
# open_file.close()<br>
# file_name = "Shame.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(Shame, open_file)<br>
# open_file.close()<br>
# file_name = "Disappointment.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(Disappointment, open_file)<br>
# open_file.close()<br>
# file_name = "Envy.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(Envy, open_file)<br>
# open_file.close()<br>
# file_name = "Suffering.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(Suffering, open_file)<br>
# open_file.close()<br>
# file_name = "Sadness.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(Sadness, open_file)<br>
# open_file.close()<br>
# file_name = "Joy.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(Joy, open_file)<br>
# open_file.close()<br>
# file_name = "Pride.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(Pride, open_file)<br>
# open_file.close()<br>
# file_name = "Fear.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(Fear, open_file)<br>
# open_file.close()<br>
# file_name = "Neglect.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(Neglect, open_file)<br>
# open_file.close()<br>
# file_name = "irritation.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(irritation, open_file)<br>
# open_file.close()<br>
# file_name = "Rage.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(Rage, open_file)<br>
# open_file.close()

# %% [markdown]
# file_name = "target_1.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(target_1, open_file)<br>
# open_file.close()<br>
# file_name = "target_2.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(target_2, open_file)<br>
# open_file.close()<br>
# file_name = "target_3.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(target_3, open_file)<br>
# open_file.close()<br>
# file_name = "target_4.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(target_4, open_file)<br>
# open_file.close()<br>
# file_name = "target_5.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(target_5, open_file)<br>
# open_file.close()<br>
# file_name = "target_6.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(target_6, open_file)<br>
# open_file.close()<br>
# file_name = "target_7.pkl"<br>
# open_file = open(file_name, "wb")<br>
# pickle.dump(target_7, open_file)<br>
# open_file.close()

# %%
import pickle
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/Disappointment.pkl', 'rb') as f:
#   mynewlist1 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/Disgust.pkl', 'rb') as f:
#   mynewlist2 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/Envy.pkl', 'rb') as f:
#   mynewlist3 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/Fear.pkl', 'rb') as f:
#   mynewlist4 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/Joy.pkl', 'rb') as f:
#   mynewlist5 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/Neglect.pkl', 'rb') as f:
#   mynewlist6 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/Nervousness.pkl', 'rb') as f:
#   mynewlist7 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/Pride.pkl', 'rb') as f:
#   mynewlist8 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/Rage.pkl', 'rb') as f:
#   mynewlist9 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/Sadness.pkl', 'rb') as f:
#   mynewlist10 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/Shame.pkl', 'rb') as f:
#   mynewlist11 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/Suffering.pkl', 'rb') as f:
  # mynewlist12 = pickle.load(f)
with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/Humor.pkl', 'rb') as f:
  mynewlist13 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/irritation.pkl', 'rb') as f:
#   mynewlist14 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/target_1.pkl', 'rb') as f:
#   mynewlist15 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/target_2.pkl', 'rb') as f:
#   mynewlist16 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/target_3.pkl', 'rb') as f:
#   mynewlist17 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/target_4.pkl', 'rb') as f:
#   mynewlist18 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/target_5.pkl', 'rb') as f:
#   mynewlist19 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/target_6.pkl', 'rb') as f:
  # mynewlist20 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/target_7.pkl', 'rb') as f:
  # mynewlist21 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/train_level2.pkl', 'rb') as f:
  # mynewlist22 = pickle.load(f)
# with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/train_off.pkl', 'rb') as f:
  # mynewlist23 = pickle.load(f)
with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/Valence.pkl', 'rb') as f:
  mynewlist24 = pickle.load(f)
with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/Arousal.pkl', 'rb') as f:
  mynewlist25 = pickle.load(f)
with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/train_text.pkl', 'rb') as f:
  mynewlist26 = pickle.load(f)
with open('/home1/gitanjali/code/My_annotation_guideline/npy_arrays_my_data/train_imgs.pkl', 'rb') as f:
  mynewlist27 = pickle.load(f)

# %% [markdown]
# print(len(mynewlist1))<br>
# print(len(mynewlist2))<br>
# print(len(mynewlist3))<br>
# print(len(mynewlist4))<br>
# print(len(mynewlist5))<br>
# print(len(mynewlist6))<br>
# print(len(mynewlist7))<br>
# print(len(mynewlist8))<br>
# print(len(mynewlist9))<br>
# print(len(mynewlist10))<br>
# print(len(mynewlist11))<br>
# print(len(mynewlist12))<br>
# print(len(mynewlist13))<br>
# print(len(mynewlist14))<br>
# print(len(mynewlist15))<br>
# print(len(mynewlist16))<br>
# print(len(mynewlist17))<br>
# print(len(mynewlist18))<br>
# print(len(mynewlist19))<br>
# print(len(mynewlist20))<br>
# print(len(mynewlist21))<br>
# print(len(mynewlist22))<br>
# print(len(mynewlist23))

# %%
print(len(mynewlist24))
print(len(mynewlist25))
print(len(mynewlist26))
print(len(mynewlist27))
mynewlist13=mynewlist13[:4000]
mynewlist24=mynewlist24[:4000]
mynewlist25=mynewlist25[:4000]
mynewlist26=mynewlist26[:4000]
mynewlist27=mynewlist27[:4000]

# %%
print(len(mynewlist24))
print(len(mynewlist25))
print(len(mynewlist26))
print(len(mynewlist27))
print(len(mynewlist13))

# %% [markdown]
# training_set = pd.DataFrame( {'Text':train_text,'Images': train_imgs,'Sarcasm':Sarcasm,\<br>
#     'Fear':Fear,'Neglect':Neglect,'irritation':irritation,'Rage':Rage,\<br>
#     'Disgust':Disgust,'Nervousness':Nervousness,'Shame':Shame,'Disappointment':Disappointment,\<br>
#         'Envy':Envy,'Suffering':Suffering,'Sadness':Sadness,'Joy':Joy,'Pride':Pride,\<br>
#     'Target1': target_1, 'Target2' : target_2, 'Target3' : target_3, 'Target4' : target_4, 'Target5' : target_5, \<br>
#         'Target6' : target_6, 'Target7' : target_7})

# %%
print(len(mynewlist24))
training_set = pd.DataFrame( {'Text':mynewlist26,'Images': mynewlist27,'Humor':mynewlist13,'Valence':mynewlist24,'Arousal':mynewlist25})

# %%
print(len(training_set.Arousal))
print(len(training_set.Text))
print(len(training_set.Images))
print(len(training_set.Humor))
print(len(training_set.Valence))
# print(mynewlist[1:50])

# %%
train,test= train_test_split(training_set,test_size=0.20,random_state=123, shuffle=False)

# %% [markdown]
# print(len(train_text))<br>
# print(len(train_imgs))<br>
# print(len(attention_mask))<br>
# print(len(Valence))

# %%
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
# val_image_features = model.predict(v_imgss, batch_size=32)
print(" train features : ", train_img_features.shape)
# print(" val features : ", val_image_features.shape)
te_imgss = np.vstack(test.Images)
# v_imgss = np.vstack(val_imgs)
test_img_features = model.predict(te_imgss, batch_size=32)
# val_image_features = model.predict(v_imgss, batch_size=32)
print(" train features : ", test_img_features.shape)
# print(" val features : ", val_image_features.shape)

# %% [markdown]
# ##############################################<br>
# img_in = Input(shape=(64, 224, 3))<br>
# img_out = vgg(img_in)<br>
# print(img_out.shape)

# %%
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
# print("shape of 1 img val ftrs", valshape.shape)
print("shape of 1 img test ftrs", testshape.shape)
# print("shape of 1 img test data ftrs", testdatashape_new.shape)
image_input = Input(shape=([trainshape.shape[1], trainshape.shape[2]]))
print(image_input)
image_reshape = Reshape((49, 512, 1))(image_input)
# image_input = Input(shape=(trainshape.shape[0],))
print("imgggggg ip", image_reshape)
print("imgggggg ip", image_reshape[6])

# %%
print("\n \n This is for Humor")
label_encoder1 = LabelEncoder()
values = array(y_train_Humor)
print("train_senti", values[1:10])
train_enc = label_encoder1.fit_transform(values)
print("train_integer_encoded", train_enc[1:5])
train_lbl = to_categorical(train_enc)
# print(train_lbl[1:50])
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
# conv_img_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], 512), padding='valid', kernel_initializer='normal', activation='relu')(image_reshape)
# conv_img_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], 512), padding='valid', kernel_initializer='normal',activation='relu')(image_reshape)
# conv_img_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], 512), padding='valid', kernel_initializer='normal',activation='relu')(image_reshape)
# maxpool_img_0 = MaxPool2D(pool_size=(49 - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_img_0)
# maxpool_img_1 = MaxPool2D(pool_size=(49 - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_img_1)
# maxpool_img_2 = MaxPool2D(pool_size=(49 - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_img_2)
# print(maxpool_img_0)
# concatenated_img_tensor = Concatenate(axis=1)([maxpool_img_0, maxpool_img_1, maxpool_img_2])
# print('HEllo',concatenated_img_tensor.shape)
# concatenated_img_tensor = Reshape((3, 150))(concatenated_img_tensor)
# print(concatenated_img_tensor.shape)
# print("Before attention ",concatenated_img_tensor.shape)
# print("Before attention ",concatenated_img_tensor[5])
# img_f = AttLayer()(concatenated_img_tensor)
print(image_input.shape)
img_f = AttLayer()(image_input)
dropout_img = Dropout(drop)(img_f)
print(dropout_img.shape)

# %% [markdown]
# flat = Flatten()(trainshape)<br>
# dense = tf.keras.layers.Dense(2742, activation='relu')(flat)

# %%
dense = tf.keras.layers.Dense(256, activation='relu')(dropout_img)
img_repr = Dropout(0.4)(dense)
img_repr

# %% [markdown]
# Commented out IPython magic to ensure Python compatibility.<br>
# Load the TensorBoard notebook extension<br>
# %load_ext tensorboard

# %%
# import datetime, os 
# logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# %%
txt_repr1 = AttLayer()(txt_repr)
print(txt_repr1.shape)
concat = Concatenate(axis=1)([img_repr, txt_repr1])
print(concat)
Dense_layer2 = tf.keras.layers.Dense(128,activation='relu', name="Dense_layer2")(concat)

# %% [markdown]
# nter_class Module Valence and Humor: 0 or 1

# %%
humor_task = tf.keras.layers.Dense(128,activation='relu', name="humor_task")(Dense_layer2)
humor = tf.keras.layers.Dense(2, activation='softmax', name="predictions_task1")(humor_task)
valence_task = tf.keras.layers.Dense(128,activation='relu', name="valence_task")(Dense_layer2)
valence= tf.keras.layers.Dense(2, activation='softmax', name="predictions_task2")(valence_task)
# #Task Specific Layer for Task 14
arousal_task = tf.keras.layers.Dense(128,activation='relu', name="arousal_task")(Dense_layer2)
arousal = tf.keras.layers.Dense(2, activation='softmax', name="predictions_task3")(arousal_task)

# %% [markdown]
# #loss={'predictions_task1':'binary_crossentropy','predictions_task3':'binary_crossentropy'}

# %% [markdown]
# # dense = tf.keras.layers.Dense(128, activation='relu')(concat)<br>
# # out = tf.keras.layers.Dense(4, activation='softmax')(dense)<br>
# model_new = Model(inputs=[input_id, mask_id, seg_id, image_input], outputs=valence)<br>
# # model_new.compile(loss='categorical_crossentropy', optimizer=Adam(2e-5),<br>
# #             metrics=['accuracy', precision, recall, f1]) if metrics else model_new.compile(<br>
# # loss='categorical_crossentropy', optimizer=Adam(2e-5), metrics=['accuracy'])<br>
# adam = Adam(learning_rate=3e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)<br>
# es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience= 10)<br>
# model_new.compile(loss={'predictions_task2':'binary_crossentropy'}, optimizer=adam,<br>
#             metrics=['accuracy', precision, recall, f1]) if metrics else model_new.compile(<br>
# loss={'predictions_task2':'binary_crossentropy'}, optimizer=adam, metrics=['accuracy'])

# %% [markdown]
# plot_model(model) #if plot_model_diagram else None<br>
# model_new.summary() #if summary else None

# %%
input_id_train, token_type_id_train, attention_mask_train = encode(X_train)
input_id_test, token_type_id_test, attention_mask_test = encode(X_test)
# image_data = train_df['img'].apply(preprocess_image)
# image_data = train_df['img'].apply(preprocess_image)
# eval_data = [input_id, token_type_id, attention_mask,image_input]

# %%
# print(input_id_train[1])

# %% [markdown]
# history = model_new.fit([input_id_train, token_type_id_train, attention_mask_train, trainshape],<br>
#                            train_lbl_8,validation_split=0.15,batch_size=32,epochs=30,  callbacks=[tensorboard_callback,es])

# %% [markdown]
# predictions_text = model_new.predict([input_id_test, token_type_id_test, attention_mask_test, testshape])<br>
# predicted1 = np.argmax(predictions_text, axis=1)<br>
# # predicted2 = np.argmax(y_class1, axis=1)<br>
# # predicted3 = np.argmax(y_class2, axis=1)<br>
# # predicted4 = np.argmax(y_class3, axis=1)<br>
# # predicted5 = np.argmax(y_class4, axis=1)<br>
# # predicted6 = np.argmax(y_class5, axis=1)<br>
# # score = model_ensemble.evaluate([test_sequence, testshape], [test_lbl,test_lbl2,test_lbl3,test_lbl4,test_lbl5,test_lbl6], verbose=1)<br>
# # print("Test Score:", score[0])<br>
# # print("Test Accuracy:", score[1])<br>
# # print('Accuracy: %f' % (accuracy*100))<br>
# # print("loss", loss)<br>
# target_names1 = ['0','1','2']<br>
# target_names2 = ['1','2','3','4']<br>
# target_names3 = ['1','2','3','4']<br>
# target_names4 = ['1','2','3','4']<br>
# target_names5 = ['0','1','2']<br>
# target_names5 = ['0','1','2']<br>
# class_rep1=classification_report(test_enc8, predicted1)<br>
# # class_rep2=classification_report(test_enc2, predicted2, target_names=target_names2)<br>
# # class_rep3=classification_report(test_enc3, predicted3, target_names=target_names3)<br>
# # class_rep4=classification_report(test_enc4, predicted4, target_names=target_names4)<br>
# # class_rep5=classification_report(test_enc5, predicted5, target_names=target_names5)<br>
# # class_rep6=classification_report(test_enc6, predicted6, target_names=target_names5)<br>
# print("confusion matrix1",confusion_matrix(test_enc8, predicted1))<br>
# # print("confusion matrix2",confusion_matrix(test_enc2, predicted2))<br>
# # print("confusion matrix3",confusion_matrix(test_enc3, predicted3))<br>
# # print("confusion matrix4",confusion_matrix(test_enc4, predicted4))<br>
# # print("confusion matrix5",confusion_matrix(test_enc5, predicted5))<br>
# # print("confusion matrix6",confusion_matrix(test_enc6, predicted6))<br>
# print(class_rep1)<br>
# # print(class_rep2)<br>
# # print(class_rep3)

# %% [markdown]
# # model_new2 = Model(inputs=[input_id, mask_id, seg_id, image_input], outputs=humor)<br>
# model_new2 = Model(inputs=[input_id, mask_id, seg_id, image_input], outputs=humor)<br>
# # adam = Adam(learning_rate=3e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)<br>
# # es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience= 10)<br>
# model_new2.compile(loss={'predictions_task1':'binary_crossentropy'}, optimizer=adam,<br>
#             metrics=['accuracy', precision, recall, f1]) if metrics else model_new2.compile(<br>
# loss={'predictions_task1':'binary_crossentropy'}, optimizer=adam, metrics=['accuracy'])

# %% [markdown]
# history = model_new2.fit([input_id_train, token_type_id_train, attention_mask_train, trainshape],<br>
#                            train_lbl_7,validation_split=0.15,batch_size=32,epochs=30,  callbacks=[tensorboard_callback,es])

# %%
# from numba import cuda 
# device = cuda.get_current_device()
# device.reset()

# %%
model_ensemble = Model(inputs=[input_id, mask_id, seg_id, image_input], outputs=[humor,valence,arousal])
# model_ensemble = Model(inputs=[inputs, image_input], outputs=off_predictions_task)
adam = Adam(lr=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
es = EarlyStopping(monitor='val_predictions_task1_accuracy', mode='max', verbose=1, patience= 10)
# es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience= 10)
model_ensemble.compile(optimizer=Adam(), loss={'predictions_task1':'binary_crossentropy','predictions_task2':'binary_crossentropy','predictions_task3':'binary_crossentropy'},\
            metrics={'predictions_task1':'accuracy','predictions_task2':'accuracy','predictions_task3':'accuracy'}, run_eagerly=True)
# model_ensemble.compile(optimizer=Adam(), loss={'predictions_task1':'binary_crossentropy'},\
#             metrics={'predictions_task1':['accuracy',mean_pred]})
# model_ensemble.compile(optimizer=adam, loss={'cnn_t1':'categorical_crossentropy','cnn_t2':'categorical_crossentropy','cnn_t3':'categorical_crossentropy','cnn_t4':'categorical_crossentropy','cnn_t5':'categorical_crossentropy','cnn_t6':'categorical_crossentropy'}, metrics={'cnn_t1':'accuracy','cnn_t2':'accuracy','cnn_t3':'accuracy','cnn_t4':'accuracy','cnn_t5':'accuracy','cnn_t6':'accuracy'})
# class_weights = {0: 0.85, 1: 1.4}
# model_ensemble.compile(optimizer=adam, loss={'cnn_t6':'categorical_crossentropy'}, metrics={'cnn_t6':['accuracy',mean_pred]})
model_ensemble.summary()

# %%
history = model_ensemble.fit([input_id_train, token_type_id_train, attention_mask_train, trainshape],\
     [train_lbl_7,train_lbl_8,train_lbl_9],validation_split=0.10,batch_size=16,epochs=30,  callbacks=[es],verbose=1)

# %% [markdown]
# print("\n \n This is for Level1")<br>
# label_encoder1 = LabelEncoder()<br>
# values = array(Valence)<br>
# print("train_senti", values[1:10])<br>
# train_enc = label_encoder1.fit_transform(values)<br>
# print("train_integer_encoded", train_enc[1:5])<br>
# train_lbl = to_categorical(train_enc)<br>
# # print(train_lbl[1:50])<br>
# print(train_lbl[1:10])

# %% [markdown]
# def labelencoder(labels):<br>
#     new_label = np.zeros((len(labels), 2))<br>
#     # new_label = np.zeros((len(labels), 3))<br>
#     for i, label in enumerate(labels):<br>
#         if label == 0:<br>
#             # new_label[i] = [0, 1, 1]<br>
#             new_label[i] = [0, 1]<br>
#         elif label == 1:<br>
#             # new_label[i] = [0, 1, 0]<br>
#             new_label[i] = [1,0]<br>
#         # elif label == 2:<br>
#         #     new_label[i] = [1, 0, 0]

# %% [markdown]
#     return new_label<br>
# labels = labelencoder(train_df['label'])<br>
# history = model_new.fit([input_id, token_type_id, attention_mask, trainshape],<br>
#                             train_lbl,<br>
#                             validation_split=0.10,<br>
#                             batch_size=32,<br>
#                             epochs=3,  callbacks=[tensorboard_callback])

# %%
input_id_test, token_type_id_test, attention_mask_test = encode(X_test)

# %%
predictions_text = model_ensemble.predict([input_id_test, token_type_id_test, attention_mask_test, testshape])
# y_class0=predictions_text
y_class0=predictions_text[0]
y_class1=predictions_text[1]
y_class2=predictions_text[2]
# y_class3=predictions_text[3]
# y_class4=predictions_text[4]
# y_class5=predictions_text[5]

# %% [markdown]
# %

# %%
print(predictions_text[0])

# %%
print(y_class0[2])

# %%
predicted1 = np.argmax(y_class0, axis=1)
predicted2 = np.argmax(y_class1, axis=1)
predicted3 = np.argmax(y_class2, axis=1)
# predicted4 = np.argmax(y_class3, axis=1)
# predicted5 = np.argmax(y_class4, axis=1)
# predicted6 = np.argmax(y_class5, axis=1)
# score = model_ensemble.evaluate([test_sequence, testshape], [test_lbl,test_lbl2,test_lbl3,test_lbl4,test_lbl5,test_lbl6], verbose=1)
# print("Test Score:", score[0])
# print("Test Accuracy:", score[1])
# print('Accuracy: %f' % (accuracy*100))
# print("loss", loss)
target_names1 = ['0','1','2']
target_names2 = ['1','2','3','4']
target_names3 = ['1','2','3','4']
target_names4 = ['1','2','3','4']
target_names5 = ['0','1','2']
target_names5 = ['0','1','2']
class_rep1=classification_report(test_enc7, predicted1)
class_rep2=classification_report(test_enc8, predicted2)
class_rep3=classification_report(test_enc9, predicted3)
# class_rep4=classification_report(test_enc4, predicted4, target_names=target_names4)
# class_rep5=classification_report(test_enc5, predicted5, target_names=target_names5)
# class_rep6=classification_report(test_enc6, predicted6, target_names=target_names5)
print("confusion matrix1",confusion_matrix(test_enc7, predicted1))
print("confusion matrix2",confusion_matrix(test_enc8, predicted2))
print("confusion matrix3",confusion_matrix(test_enc9, predicted3))
# print("confusion matrix4",confusion_matrix(test_enc4, predicted4))
# print("confusion matrix5",confusion_matrix(test_enc5, predicted5))
# print("confusion matrix6",confusion_matrix(test_enc6, predicted6))
print(class_rep1)
print(class_rep2)
print(class_rep3)
print("precision_recall_fscore_support_micro",precision_recall_fscore_support(test_enc7, predicted1, average='micro'))
print("precision_recall_fscore_support_micro",precision_recall_fscore_support(test_enc8, predicted2, average='micro'))
print("precision_recall_fscore_support_micro",precision_recall_fscore_support(test_enc9, predicted3, average='micro'))

print("precision_recall_fscore_support_macro",precision_recall_fscore_support(test_enc7, predicted1, average='macro'))
print("precision_recall_fscore_support_macro",precision_recall_fscore_support(test_enc8, predicted2, average='macro'))
print("precision_recall_fscore_support_macro",precision_recall_fscore_support(test_enc9, predicted3, average='macro'))


# %%
from tensorflow.keras.layers import Lambda
# computing cosine similarity 
def cosine_similarity(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return K.batch_dot(x, y, axes=-1)

# %%
def cos_sim_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
# print(type(humor_task))
# import tensorflow.Tensor.eval as eval
# humor_task_numpy = humor_task.tensorflow.Tensor.eval(session=tf.compat.v1.Session())
# print(type(humor_task_numpy))

# %% [markdown]
# #Cosine Similarity<br>
# s = tf.keras.losses.cosine_similarity(humor_task, valence_task)<br>
# print("Cosine Similarity:",s)

# %%
import tensorflow as tf
# tf.enable_eager_execution()
tf.executing_eagerly()

# %%
def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

# %%
def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

# %%
distance_hum_val = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([humor, valence])
print(distance_hum_val.shape)
distance_hum_aro = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([humor, arousal])
print(distance_hum_aro.shape)
distance_aro_val = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([arousal, valence])
print(distance_aro_val.shape)

# %%
concat_cosine=Concatenate(axis=1)([distance_hum_val,distance_hum_aro,distance_aro_val])
print(concat_cosine.shape)

# %%
# distance = Lambda(cosine_similarity,output_shape=cos_sim_output_shape)([humor_task, valence_task])
# print(distance.shape)

# %% [markdown]
# dotted = tf.keras.layers.Dot(axes=-1, normalize=True)([humor_task, valence_task])<br>
# print(dotted.shape, dotted)

# %% [markdown]
# #Cosine Similarity<br>
# s = tf.keras.losses.cosine_similarity(humor_task, valence_task)<br>
# print("Cosine Similarity:",s)<br>
# print(s)

# %%
max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2,
   strides=1, padding='valid')
arousal_task_expand= tf.expand_dims(arousal_task, axis=-1)
arousal_task_pool=max_pool_1d(arousal_task_expand)
print(arousal_task_pool.shape)

# %%
max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2,
   strides=1, padding='valid')
humor_task_expand= tf.expand_dims(humor_task, axis=-1)
humor_task_pool=max_pool_1d(humor_task_expand)
print(humor_task_pool.shape)

max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2,
   strides=1, padding='valid')
valence_task_expand= tf.expand_dims(valence_task, axis=-1)
valence_task_pool=max_pool_1d(valence_task_expand)
print(valence_task_pool.shape)

# %%
concatenated_pool = Concatenate(axis=1)([arousal_task_expand, humor_task_pool, valence_task_pool])
print(concatenated_pool.shape)
flat_pool = Flatten()(concatenated_pool)
print(flat_pool.shape)

# %%
final_concat = Concatenate(axis=1)([flat_pool, concat_cosine])
print(final_concat.shape)

# %%
humor_task_final = tf.keras.layers.Dense(128,activation='relu', name="humor_task_final")(final_concat)
humor_final = tf.keras.layers.Dense(3, activation='softmax', name="predictions_task_final1")(humor_task_final)
valence_task_final = tf.keras.layers.Dense(128,activation='relu', name="valence_task_final")(final_concat)
valence_final= tf.keras.layers.Dense(4, activation='softmax', name="predictions_task_final2")(valence_task_final)
# #Task Specific Layer for Task 14
arousal_task_final = tf.keras.layers.Dense(128,activation='relu', name="arousal_task_final")(final_concat)
arousal_final = tf.keras.layers.Dense(4, activation='softmax', name="predictions_task_final3")(arousal_task_final)

# %%
model_ensemble_final = Model(inputs=[input_id, mask_id, seg_id, image_input], outputs=[humor_final,valence_final,arousal_final])
# model_ensemble = Model(inputs=[inputs, image_input], outputs=off_predictions_task)
adam = Adam(lr=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
es = EarlyStopping(monitor='val_predictions_task_final1_accuracy', mode='max', verbose=1, patience= 10)
# es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience= 10)
model_ensemble_final.compile(optimizer=Adam(), loss={'predictions_task_final1':'binary_crossentropy','predictions_task_final2':'binary_crossentropy','predictions_task_final3':'binary_crossentropy'},\
            metrics={'predictions_task_final1':'accuracy','predictions_task_final2':'accuracy','predictions_task_final3':'accuracy'}, run_eagerly=True)
# model_ensemble.compile(optimizer=Adam(), loss={'predictions_task1':'binary_crossentropy'},\
#             metrics={'predictions_task1':['accuracy',mean_pred]})
# model_ensemble.compile(optimizer=adam, loss={'cnn_t1':'categorical_crossentropy','cnn_t2':'categorical_crossentropy','cnn_t3':'categorical_crossentropy','cnn_t4':'categorical_crossentropy','cnn_t5':'categorical_crossentropy','cnn_t6':'categorical_crossentropy'}, metrics={'cnn_t1':'accuracy','cnn_t2':'accuracy','cnn_t3':'accuracy','cnn_t4':'accuracy','cnn_t5':'accuracy','cnn_t6':'accuracy'})
# class_weights = {0: 0.85, 1: 1.4}
# model_ensemble.compile(optimizer=adam, loss={'cnn_t6':'categorical_crossentropy'}, metrics={'cnn_t6':['accuracy',mean_pred]})
model_ensemble_final.summary()

# %%
history_final = model_ensemble_final.fit([input_id_train, token_type_id_train, attention_mask_train, trainshape],\
     [train_lbl,train_lbl_2,train_lbl_3],validation_split=0.10,batch_size=16,epochs=3,  callbacks=[es],verbose=1)

# %% [markdown]
# from datetime import datetime<br>
# import io<br>
# import itertools<br>
# from packaging import version

# %%
predictions_text = model_ensemble_final.predict([input_id_test, token_type_id_test, attention_mask_test, testshape])
# y_class0=predictions_text
y_class0=predictions_text[0]
y_class1=predictions_text[1]
y_class2=predictions_text[2]
# y_class3=predictions_text[3]
# y_class4=predictions_text[4]
# y_class5=predictions_text[5]
predicted1 = np.argmax(y_class0, axis=1)
predicted2 = np.argmax(y_class1, axis=1)
predicted3 = np.argmax(y_class2, axis=1)
# predicted4 = np.argmax(y_class3, axis=1)
# predicted5 = np.argmax(y_class4, axis=1)
# predicted6 = np.argmax(y_class5, axis=1)
# score = model_ensemble.evaluate([test_sequence, testshape], [test_lbl,test_lbl2,test_lbl3,test_lbl4,test_lbl5,test_lbl6], verbose=1)
# print("Test Score:", score[0])
# print("Test Accuracy:", score[1])
# print('Accuracy: %f' % (accuracy*100))
# print("loss", loss)
target_names1 = ['0','1','2']
target_names2 = ['1','2','3','4']
target_names3 = ['1','2','3','4']
target_names4 = ['1','2','3','4']
target_names5 = ['0','1','2']
target_names5 = ['0','1','2']
class_rep1=classification_report(test_enc, predicted1)
class_rep2=classification_report(test_enc2, predicted2)
class_rep3=classification_report(test_enc3, predicted3)
# class_rep4=classification_report(test_enc4, predicted4, target_names=target_names4)
# class_rep5=classification_report(test_enc5, predicted5, target_names=target_names5)
# class_rep6=classification_report(test_enc6, predicted6, target_names=target_names5)
print("confusion matrix1",confusion_matrix(test_enc, predicted1))
print("confusion matrix2",confusion_matrix(test_enc2, predicted2))
print("confusion matrix3",confusion_matrix(test_enc3, predicted3))
# print("confusion matrix4",confusion_matrix(test_enc4, predicted4))
# print("confusion matrix5",confusion_matrix(test_enc5, predicted5))
# print("confusion matrix6",confusion_matrix(test_enc6, predicted6))
print(class_rep1)
print(class_rep2)
print(class_rep3)

print("precision_recall_fscore_support_micro",precision_recall_fscore_support(test_enc, predicted1, average='micro'))
print("precision_recall_fscore_support_micro",precision_recall_fscore_support(test_enc2, predicted2, average='micro'))
print("precision_recall_fscore_support_micro",precision_recall_fscore_support(test_enc3, predicted3, average='micro'))

print("precision_recall_fscore_support_macro",precision_recall_fscore_support(test_enc, predicted1, average='macro'))
print("precision_recall_fscore_support_macro",precision_recall_fscore_support(test_enc2, predicted2, average='macro'))
print("precision_recall_fscore_support_macro",precision_recall_fscore_support(test_enc3, predicted3, average='macro'))


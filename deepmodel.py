# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:11:07 2019

@author: 振振
"""
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers
#带标签的训练数据集
df = pd.read_csv('data/labeledTrainData.tsv',delimiter='\t')
#去掉id列
df = df.drop(['id'],axis=1)
stop_words=set(stopwords.words('english'))
lemmatizer=WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text
df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))

max_features=6000
tokenizer = Tokenizer(num_words=max_features) #处理的最大单词数量
tokenizer.fit_on_texts(df['Processed_Reviews']) #使用一系列文档来生成token词典
list_tokenized_train = tokenizer.texts_to_sequences(df['Processed_Reviews']) #将多个文档转换为word下标的向量形式

maxlen=130
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen) #序列长度补全，大于maxlen的截短，小于的补0
y = df['sentiment']

embed_size = 128 #嵌入维度
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
epochs = 3
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

df_test = pd.read_csv('data/testData.tsv',header=0, delimiter="\t", quoting=3)
df_test['review'] = df_test.review.apply(lambda x: clean_text(x))
df_test['sentiment'] = df_test['id'].map(lambda x: 1 if int(x.strip('"').split('_')[1])>=5 else 0)
y_test = df_test['sentiment']
list_sentences_test = df_test["review"]
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
prediction = model.predict(X_te)
y_pred = (prediction > 0.5)
print('train_data_shape')
print(X_t.shape)
print('test_shape')
print(X_te.shape)
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
print(confusion_matrix(y_pred, y_test))











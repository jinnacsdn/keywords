# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:26:38 2018
此程序大概用时一个半小时
@author: jinna
""
pip install gensim

import json
import re
import pickle
import time
import os
from gensim.corpora import Dictionary
from nltk.stem import SnowballStemmer
start=time.clock()
stemmer=SnowballStemmer('english')
stopwords=open('./dict/en_stopwords.txt','r',encoding='utf8').readlines()
stopwords=[word.strip(' \n') for word in stopwords]
hindi_stopwords=open('./dict/hindi_stopwords.txt','r',encoding='utf8').readlines()
hindi_stopwords=[word.strip(' \n') for word in hindi_stopwords]
stopwords.extend(hindi_stopwords)  #将英语和印地语停止词合并     
rootdir = 'video_info'
list_file = os.listdir(rootdir) #列出文件夹下所有的目录与文件
data_list=[]
url_list=[]
num_list=[]
for i in range(len(list_file)):
    path = os.path.join(rootdir,list_file[i])
    if os.path.isfile(path):
       f= open(path,'r',encoding='utf8')
       for line in f:
           line=line.strip('\n')
           if line:
              json1=json.loads(line)
              if json1.get('countries') and json1.get('id') and json1.get('langs') and json1.get('title') and ('in' in [w.lower() for w in json1['countries']]  or 'india' in [w.lower() for w in json1['countries']]) and ('en' in json1['langs']): 
                 if json1.get('description'):
                    data_list.append(json1["title"]+' '+json1['description'])
                 else:
                    data_list.append(json1["title"])
                 if json1.get('source_url'):
                    url_list.append(json1['source_url'])
                 else:
                    url_list.append([])
                 num_list.append(json1['id'])
train_list=[]
final_string_list=[]
final_url_list=[]
final_num_list=[]
for i in range(len(data_list)):
    line=data_list[i]
    line=line.lower()
    if re.search('subscribe@',line):
       line=line.split('subscribe@',1)[0]
    if re.search('subscribe',line):
       line=line.split('subscribe',1)[0]
    if re.search('http://',line):
       line=line.split('http://',1)[0]
    if re.search('www.',line):
       line=line.split('www.',1)[0]
    line=line.replace('shroff\u2019s','shroff') 
    line=line.rstrip(' ')
    line_list=line.split(' ')
    new_list=[]
    for word in line_list:
        word=word.strip(' ')
        if re.search('^[a-z0-9]+$',word):
           #word=stemmer.stem(word)        $提取单词词干
           new_list.append(word)
    if new_list:
       final_string_list.append(line)
       train_list.append([w for w in new_list if w not in stopwords])
       final_url_list.append(url_list[i])
       final_num_list.append(num_list[i])
dictionary=Dictionary(train_list)
dictionary.filter_extremes(no_below=20, no_above=0.1) #ignore words that appear in less than 20 documents or more than 10% documents
dictionary.compactify
corpus=[dictionary.doc2bow(text) for text in train_list]
dict_corpus=[dictionary,corpus,final_url_list,final_num_list,final_string_list,train_list]
with open('dict_corpus.pkl','wb') as f_dump:
     pickle.dump(dict_corpus,f_dump)
f_dump.close()

elapsed=time.clock()-start
print("Time used:",elapsed)

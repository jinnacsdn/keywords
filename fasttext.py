import fastText as ft
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import smart_open
punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~。，"""
FAST_TEXT_MODEL_PATH='/data/tanggp/fasttext_100.model.bin'
model=ft.load_model(FAST_TEXT_MODEL_PATH)




def word_sentence(sentence):
    sentor_vetor=model.get_sentence_vector(sentence).reshape(1, -1)

    sentence = re.sub(r'[{}]+'.format(punctuation), ' ', sentence)
    sentence = ' '.join([t.strip() for t in sentence.split(' ') if t.strip() != ''])

    words=list(set([w  for w  in sentence.lower().split(' ') if len(w)>2]))
    np_shape=len(words)
    # word_vect[0,:]=sentor_vetor
    word_vect=np.zeros((np_shape,100))

    for i,word in enumerate(words):
        word_vect[i,:]= model.get_word_vector(word)

    cos=cosine_similarity(word_vect,sentor_vetor).reshape(1,-1)[0,:]
    cos=np.fabs(cos)

    idx=np.argsort(-cos)
    score=np.sort(-cos)
    score=-score
    score=[str(round(s, 2)) for s in score]
    word_result=[]
    for i in idx:
        word_result.append(words[i])
    return score,word_result

def get_items(items):
    valid_count = 0
    all_line = 0
    has_country = 0
    import json
    result = []
    for line in items:
        all_line += 1
        if isinstance(line,str):
            line_dict = json.loads(line)
        else:
            line_dict=line
        if 'countries' not in line_dict.keys():
            continue
        has_country += 1
        if 'india' in [c.lower() for c in line_dict['countries']] or 'in' in [c.lower() for c in
                                                                              line_dict['countries']]:
            line_dict['countries']='india'
            valid_count += 1
            result.append(line_dict)

    return  result,has_country,all_line,valid_count

def  get_data_txt(file_name):
    with smart_open.smart_open(file_name, encoding='utf8') as f:
        items = f.readlines()

    result, has_country, all_line, valid_count = get_items(items)

    print('file {} total count is {}, has_country is {}, valid line is {}'.format(file_name, all_line,has_country,valid_count))
    return result

def save_txt(result,out_path_4column):
    with smart_open.smart_open(out_path_4column, 'w', encoding='utf8') as f:
        for r in result:
            f.write(r+ '\n')

def predict_label_txt(input_path,save_path):
    sql_dict = get_data_txt(input_path)
    result=[]
    count=0
    for sq in sql_dict:
        try:
            count+=1
            # if count>10:
            #     continue

            sentence=sq.get('title')+' '+sq.get('source_user')+' '+' '.join(sq.get('tags',[]))+sq.get('description')
            score, word_result = word_sentence(sentence)
            # print(word_result)
            # print(score)
            print(word_result)

            sentence =sq.get('id') +'\x01'+';'.join(word_result[:4])+ '\x01' +sq.get('title') + '\x01' + sq.get('source_user') + ' \x01' +' '.join(sq.get('tags',[]))+ ' \x01'+sq.get('source_url')+ ' \x01' +';'.join(word_result)+' \x01' +';'.join(score)
            result.append(sentence)
        except:
            count-=1

    save_txt(result,save_path)
    print(count)

def w_e(file_name):
    input_path='/data/tanggp/video_info/datepart=20180528/'+file_name
    save_path='/data/tanggp/key_words/'+file_name
    predict_label_txt(input_path, save_path)
import time
t1=time.time()
file_name='002999_0'
w_e(file_name)
t2=time.time()
print('finish time {}'.format(t2-t1))

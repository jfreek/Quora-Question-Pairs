# -*- coding: UTF-8 -*-
import pandas as pn
import re
from sklearn.cluster import KMeans
from gensim.models import word2vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def replace_text(text, replace_list, replace_by):
    if replace_list:
        replace_list = list(set(replace_list))
        for i in xrange(len(replace_list)):
            text = text.replace(replace_list[i], replace_by.format(replace_list[i]))
    return text


def clean_text(tset, to_unicode=True):
    tset = tset.lower()
    # undesirable chars out!
    to_del = re.findall(r"[#$'()|?]", tset, re.IGNORECASE)
    tset = replace_text(text=tset, replace_list=to_del, replace_by=" ")
    to_dspace = re.findall(r"[^\w\s.,:;\-\\]", tset, re.IGNORECASE)
    tset = replace_text(text=tset, replace_list=to_dspace, replace_by=" {0} ")
    if to_unicode and type(tset) != unicode:
        tset = tset.decode('utf8', 'ignore')
    tset = re.sub(r"\s{2,}", " ", tset)
    return tset


train_df = pn.read_csv('/home/jfreek/workspace/tmp/train.csv')
import time

t1 = time.time()
question_list = train_df["question1"].tolist() + train_df["question2"].tolist()
question_list = list(set(question_list))
question_list = [clean_text(tset=question) for question in question_list if str(question) != "nan"]
t2 = time.time()
total = t2-t1
print str(total)

tokenized_questions = [word_tokenize(question) for question in question_list]

# w2v parameters
parallel_workers = 7
vector_dimensionality = 300
min_word_count = 10
windows_size = 2
downsampling = 1e-3
skipgram = 1
h_softmax = 1
# train the model skipgram
t0 = time.time()
model_skg = word2vec.Word2Vec(sentences=tokenized_questions, sg=skipgram, workers=parallel_workers,
                              size=vector_dimensionality, min_count=min_word_count,
                              window=windows_size, sample=downsampling)
del tokenized_questions
t1 = time.time()
total = t1 - t0

print "********************MODEL trained: " + str(total) + "********************"

model_skg.init_sims(replace=True)
# save the model for later use. You can load it later using Word2Vec.load()
# Name Format: vectordimension_mincount_windowsize_downsampling_skipgram(CBoW)_hsampling
model_path = "/home/jfreek/workspace/w2v_models/"
model_skg.save(model_path + "300_10_2_e-3_sg_hs")

print "********************MODEL saved********************"

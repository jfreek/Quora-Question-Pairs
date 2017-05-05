# -*- coding: UTF-8 -*-
import pandas as pd
import re
import pickle
from sklearn.cluster import KMeans
from gensim.models import word2vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy


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
    to_space = re.findall(r"[^\w\s.,:;\-\\]", tset, re.IGNORECASE)
    tset = replace_text(text=tset, replace_list=to_space, replace_by=" {0} ")
    if to_unicode and type(tset) != unicode:
        tset = tset.decode('utf8', 'ignore')
    tset = re.sub(r"\s{2,}", " ", tset)
    return tset


def clean_stop_words(stop_words_list, wordlist):
    new_wordlist = []
    for word in wordlist:
        if word not in stop_words_list:
            new_wordlist.append(word)
    return new_wordlist


def get_percentage(list1, list2):
    t = len(list1) + len(list2)
    count = 0
    for item in list1:
        if list2:
            if item in list2:
                count += 1
                list2.remove(item)
        else:
            break
    percent = float(2*count)/t
    return percent


class Word2vecFunctions:
    def __init__(self):
        self.tmp_path = '/home/jfreek/workspace/tmp/'
        self.model_path = "/home/jfreek/workspace/w2v_models/"
        self.cluster_path = "/home/jfreek/workspace/w2v_clusters/"

    def data_prep(self, sw=False, checkpoint=False):
        # read training data
        train_df = pd.read_csv(self.tmp_path+'train.csv')
        # ********** clean text **********
        question_list = train_df["question1"].tolist() + train_df["question2"].tolist()
        question_list = list(set(question_list))
        question_list = [clean_text(tset=question) for question in question_list if str(question) != "nan"]
        # ********** tokenize text **********
        file_name = "tokens.p"
        tokenized_questions = [word_tokenize(question) for question in question_list]
        # ********** stop words **********
        if sw:
            file_name = "sw_"+file_name
            stop_words = set(stopwords.words('english'))
            for i in xrange(len(tokenized_questions)):
                tokenized_questions[i] = clean_stop_words(stop_words_list=stop_words, wordlist=tokenized_questions[i])
        # ********** checkpoint **********
        if checkpoint:
            with open(self.tmp_path+file_name, 'w') as f:
                pickle.dump(tokenized_questions, f)
            return tokenized_questions
        else:
            return tokenized_questions

    def w2v_model(self, tokens, parallel_workers=7, min_word_count=5, windows_size=2):
        # ********** train the model **********
        model_skg = word2vec.Word2Vec(sentences=tokens, sg=1, workers=parallel_workers,
                                      size=300, min_count=min_word_count,
                                      window=windows_size, sample=1e-3)
        # ********** save the model **********
        param = str(min_word_count)+"_"+str(windows_size)
        model_skg.init_sims(replace=True)
        # Name Format: vectordimension_mincount_windowsize_downsampling_skipgram(CBoW)_hsampling
        model_skg.save(self.model_path + "quora_300_{params}_e-3_sg".format(params=param))
        print "********************MODEL saved********************"

    def kmeans_clustering(self, param, cluster_size=10):
        # ********** Clusters! **********
        # Load the model
        w2v_model = word2vec.Word2Vec.load(self.model_path + "quora_300_{params}_e-3_sg".format(params=param))
        # set the list of words in the vocab in vector format
        word_vectors = w2v_model.wv.syn0
        # number of clusters
        num_clusters = len(w2v_model.wv.vocab) / cluster_size
        # initalize a k-means object and use it to extract centroids
        kmeans_clustering = KMeans(n_clusters=num_clusters)
        print "********************cluster initialized********************"
        # assignment of cluster for each word
        idx = kmeans_clustering.fit_predict(word_vectors)
        print "********************cluster assignment********************"
        # create a Word:Index dictionary
        word_centroid_map = dict(zip(w2v_model.wv.index2word, idx))
        print "********************cluster created********************"
        # Convert to DF
        clusters = pd.DataFrame.from_dict(data=word_centroid_map, orient='index')
        clusters.columns = ['cluster']
        clusters['words'] = clusters.index
        clusters.reset_index(drop=True, inplace=True)
        # data frame file into clusters folder
        clusters.to_pickle(self.cluster_path + "quora_300_{params}_e-3_sg_kmeans_{k}".format(params=param,
                                                                                             k=str(cluster_size)))


class FindDuplicates:
    def __init__(self):
        self.nlp = spacy.load('en')
        self.tmp_path = '/home/jfreek/workspace/tmp/'
        self.cluster_path = "/home/jfreek/workspace/w2v_clusters/quora_300_5_2_e-3_sg_kmeans_10"
        self.clusters = pd.read_pickle(self.cluster_path)

    def word_tag(self, question):
        """
        Tags words with question id, pos, lemma and w2v cluster
        :return: df with all words and all its tags
        """
        question = question.lower()
        # check text type and converting to unicode
        if type(question) != unicode:
            question = question.decode('utf8')
        # tag process
        quest = self.nlp(question)
        df = pd.DataFrame()
        for word in quest:
            try:
                context = self.clusters[self.clusters['words'] == word.text]['cluster'].iloc[0]
                context = str(context)
            except:
                context = None
            temp = pd.DataFrame({'word': word.text, 'lemma': word.lemma_,
                                 'pos': word.pos_, 'context': context}, index=[0])
            df = df.append(temp)
        df.reset_index(drop=True, inplace=True)
        return df

    def similarity_percentage(self, df1, df2):
        variables = ['context', 'pos', 'lemma', 'subject']
        df = pd.DataFrame(columns=variables, index=[0])
        for var in variables:
            if var == 'subject':
                df_list1 = df1[((df1['pos'] == 'NOUN') | (df1['pos'] == 'PRON')) & (df1['context'].notnull())]['context'].tolist()
                df_list2 = df2[((df2['pos'] == 'NOUN') | (df2['pos'] == 'PRON')) & (df2['context'].notnull())]['context'].tolist()
                p = get_percentage(list1=df_list1, list2=df_list2) if df_list1 and df_list2 else 0
                df[var] = p
            else:
                df_list1 = df1[df1[var].notnull()][var].tolist()
                df_list2 = df2[df2[var].notnull()][var].tolist()
                p = get_percentage(list1=df_list1, list2=df_list2) if df_list1 and df_list2 else 0
                df[var] = p
        return df

    def nn_filter(self):
        pass


def main():
    # ********** Train and create word2vec model **********
    wf = Word2vecFunctions()
    tokens = wf.data_prep(checkpoint=True)
    wf.w2v_model(tokens=tokens)
    wf.kmeans_clustering(param='5_2', cluster_size=10)
    # ********** DEV find duplicates **********
    fd = FindDuplicates()
    train_df = pd.read_csv(fd.tmp_path+'train.csv')
    # dev pipeline
    df = pd.DataFrame(index=[0])
    for row in train_df[['id', 'question1', 'question2', 'is_duplicate']].values:
        df_1 = fd.word_tag(row[1])
        df_2 = fd.word_tag(row[2])
        temp = fd.similarity_percentage(df1=df_1, df2=df_2)
        temp['id'] = row[0]
        temp['is_duplicate'] = row[3]
        df = df.append(temp)
    df.reset_index(drop=True, inplace=True)
    # ********** Find Duplicates pipeline **********
    df = pd.DataFrame(index=[0])
    for row in train_df[['id', 'question1', 'question2', 'is_duplicate']].values:
        df_1 = fd.word_tag(row[1])
        df_2 = fd.word_tag(row[2])
        temp = fd.similarity_percentage(df1=df_1, df2=df_2)
        # filter part HERE
        df = df.append(temp)
        df['id'] = row[0]
    df.reset_index(drop=True, inplace=True)

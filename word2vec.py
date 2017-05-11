# -*- coding: UTF-8 -*-
import re
import pickle
from sklearn.cluster import KMeans
from gensim.models import word2vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd


def replace_text(text, replace_list, replace_by):
    """
    Replaces items in replace_list by items in replace_by, from a text.
    :param text: str
    :param replace_list: list 
    :param replace_by: str
    :return: new text: str
    """
    if replace_list:
        replace_list = list(set(replace_list))
        for i in xrange(len(replace_list)):
            text = text.replace(replace_list[i], replace_by.format(replace_list[i]))
    return text


def clean_text(tset, to_unicode=True):
    """
    Replaces undesirable characters and transform to unicode if needed. 
    :param tset: str
    :param to_unicode:bool 
    :return: clean text: str
    """
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
    """
    Function to assist in the exclusion of stop words.
    :param stop_words_list: words to del:list
    :param wordlist: words to filter:list
    :return: new word list: list
    """
    new_wordlist = []
    for word in wordlist:
        if word not in stop_words_list:
            new_wordlist.append(word)
    return new_wordlist


class Word2vecFunctions:
    """
    All functions to prepare data, train a word2vec model and classify words in clusters.
    """
    def __init__(self):
        self.tmp_path = '/home/jfreek/workspace/tmp/'
        self.model_path = "/home/jfreek/workspace/models/"
        self.cluster_path = "/home/jfreek/workspace/w2v_clusters/"

    def data_prep(self, sw=False, checkpoint=False):
        """
        Prepares, cleans and gives format to the data for word2vec training input.
        :param sw: true if we want to del stop words: bool
        :param checkpoint: true if we want to save the tokens as a checkpoint: bool
        :return: word tokens: list
        """
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
        """
        Trains a word2vec model and saves it for future use.
        :param tokens: word tokens: list
        :param parallel_workers: number of processors to use: int 
        :param min_word_count: minimum number of occurrences of a word to be included in vocab: int
        :param windows_size: number of context words before and after the central word: int 
        :return: saves word2vec model
        """
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
        """
        Creates clusters of words using word2vec vectors and k-means algorithm.
        :param param: parameters used in word2vec model as an identifier in clusters names: str
        :param cluster_size: number of words in each cluster: int
        :return: saves clusters data frame as pickle
        """
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
        pickle.dump(word_centroid_map, open(self.cluster_path + "quora_300_{params}_e-3_sg_kmeans_{k}_dict.p".format(params=param, k=str(cluster_size)), "wb"))


def main():
    # **********To train and create word2vec model **********
    wf = Word2vecFunctions()
    tokens = wf.data_prep(checkpoint=True)
    wf.w2v_model(tokens=tokens)
    wf.kmeans_clustering(param='5_2', cluster_size=10)

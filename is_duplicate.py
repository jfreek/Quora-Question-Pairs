# -*- coding: UTF-8 -*-
import time
import re
import pickle
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from gensim.models import word2vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import spacy
from joblib import Parallel, delayed


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


def get_percentage(list1, list2):
    """
    Calculates the similarity of two lists (sequence matcher style)
    :param list1: list
    :param list2: list
    :return: percentage: float
    """
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


def get_relevance(df, columns):
    """
    Calculates the mean value of a variable when is equals to 1 and 0. 
    It helps to check how much a variable changes in each case, giving a sense of relevance.
    :param df: data frame
    :param columns: list of columns0
    :return: fields and its ave mean: dict
    """
    d = {}
    for column in columns:
        d[column+'_0'] = df[df['is_duplicate'] == 0][column].mean()
        d[column + '_1'] = df[df['is_duplicate'] == 1][column].mean()
    return d


def dev_pipeline(row):
    """
    Steps for development of model.
    tags words and finds similarities for further check. 
    :param row: data frame row: pandas series
    :return: data frame
    """
    fd = FindDuplicates()
    dl_1 = fd.word_tag(row[1])
    dl_2 = fd.word_tag(row[2])
    df_1 = pd.DataFrame(dl_1)
    df_2 = pd.DataFrame(dl_2)
    temp = fd.similarity_percentage(df1=df_1, df2=df_2)
    temp['id'] = row[0]
    temp['is_duplicate'] = row[3]
    return temp


class Word2vecFunctions:
    """
    All functions to prepare data, train a word2vec model and classify words in clusters.
    """
    def __init__(self):
        self.tmp_path = '/home/jfreek/workspace/tmp/'
        self.model_path = "/home/jfreek/workspace/w2v_models/"
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
        # Convert to DF
        # clusters = pd.DataFrame.from_dict(data=word_centroid_map, orient='index')
        # clusters.columns = ['cluster']
        # clusters['words'] = clusters.index
        # clusters.reset_index(drop=True, inplace=True)
        # # data frame file into clusters folder
        # clusters.to_pickle(self.cluster_path + "quora_300_{params}_e-3_sg_kmeans_{k}".format(params=param,
        #                                                                                      k=str(cluster_size)))


class FindDuplicates:
    """
    Class with functions to identify duplicate questions like nobody else!
    """
    def __init__(self):
        self.nlp = spacy.load('en')
        self.tmp_path = '/home/jfreek/workspace/tmp/'
        self.cluster_path = "/home/jfreek/workspace/w2v_clusters/quora_300_5_2_e-3_sg_kmeans_10_dict.p"
        self.clusters = pickle.load(open(self.cluster_path, "rb"))

    def word_tag(self, question):
        """
        Tags words with question id, pos, lemma and w2v cluster
        :param question: question to tag: str
        :return: A list of dictionaries with all words and all its tags: list
        """
        question = question.lower()
        # check text type and converting to unicode
        if type(question) != unicode:
            question = question.decode('utf8')
        # tag process
        quest = self.nlp(question)
        # df = pd.DataFrame()
        tags_list = []
        for word in quest:
            try:
                context = self.clusters[word.text]
                context = str(context)
            except:
                context = None
            temp = {'word': word.text, 'lemma': word.lemma_, 'pos': word.pos_, 'context': context}
            tags_list.append(temp)
        return tags_list

    def similarity_percentage(self, df1, df2):
        """
        Calculates similarity of two questions by comparing the tags.
        :param df1: data frame of question 1 and tags
        :param df2: data frame of question 2 and tags
        :return: data frame with percentages: data frame
        """
        variables = ['context', 'lemma', 'noun']
        df = pd.DataFrame(columns=variables, index=[0])
        for var in variables:
            if var == 'noun':
                df_list1 = df1[(df1['pos'] == 'NOUN') & (df1['context'].notnull())]['context'].tolist()
                df_list2 = df2[(df2['pos'] == 'NOUN') & (df2['context'].notnull())]['context'].tolist()
                p = get_percentage(list1=df_list1, list2=df_list2) if df_list1 and df_list2 else 0
                df[var] = p
            # elif var == 'pron':
            #     df_list1 = df1[(df1['pos'] == 'PRON') & (df1['context'].notnull())]['context'].tolist()
            #     df_list2 = df2[(df2['pos'] == 'PRON') & (df2['context'].notnull())]['context'].tolist()
            #     p = get_percentage(list1=df_list1, list2=df_list2) if df_list1 and df_list2 else 0
            #     df[var] = p
            # elif var == 'verb':
            #     df_list1 = df1[(df1['pos'] == 'VERB') & (df1['context'].notnull())]['context'].tolist()
            #     df_list2 = df2[(df2['pos'] == 'VERB') & (df2['context'].notnull())]['context'].tolist()
            #     p = get_percentage(list1=df_list1, list2=df_list2) if df_list1 and df_list2 else 0
            #     df[var] = p
            # elif var == 'adj':
            #     df_list1 = df1[(df1['pos'] == 'ADJ') & (df1['context'].notnull())]['context'].tolist()
            #     df_list2 = df2[(df2['pos'] == 'ADJ') & (df2['context'].notnull())]['context'].tolist()
            #     p = get_percentage(list1=df_list1, list2=df_list2) if df_list1 and df_list2 else 0
            #     df[var] = p
            else:
                df_list1 = df1[df1[var].notnull()][var].tolist()
                df_list2 = df2[df2[var].notnull()][var].tolist()
                p = get_percentage(list1=df_list1, list2=df_list2) if df_list1 and df_list2 else 0
                df[var] = p
        return df

    def nn_filter(self):
        pass


def main():
    # **********To train and create word2vec model **********
    wf = Word2vecFunctions()
    tokens = wf.data_prep(checkpoint=True)
    wf.w2v_model(tokens=tokens)
    wf.kmeans_clustering(param='5_2', cluster_size=10)

    # ********** DEV find duplicates **********
    fd = FindDuplicates()
    train_df = pd.read_csv(fd.tmp_path+'train.csv')
    train_df.dropna(inplace=True)
    train_df = train_df[:10000]

    # dev pipeline Parallel style ======================================
    # t0 = time.time()
    # temp = Parallel(n_jobs=7)(delayed(dev_pipeline)(row)
    #                         for row in train_df[['id', 'question1', 'question2', 'is_duplicate']].values)
    # t1 = time.time()
    # total = t1 - t0
    # print "total time: " + str(total)
    # df = pd.concat(temp)
    # df.reset_index(drop=True, inplace=True)
    # ============================================================

    # dev pipeline Cavernicola style =============================
    t0 = time.time()
    df = pd.DataFrame()
    for row in train_df[['id', 'question1', 'question2', 'is_duplicate']].values:
        dl_1 = fd.word_tag(row[1])
        dl_2 = fd.word_tag(row[2])
        df_1 = pd.DataFrame(dl_1)
        df_2 = pd.DataFrame(dl_2)
        temp = fd.similarity_percentage(df1=df_1, df2=df_2)
        temp['id'] = row[0]
        temp['is_duplicate'] = row[3]
        df = df.append(temp)
    df.reset_index(drop=True, inplace=True)
    t1 = time.time()
    total = t1 - t0
    print "total time: " + str(total)
    # ============================================================
    # Logistic Regression
    X = df[['lemma', 'noun']].values
    y = df['is_duplicate'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=6)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    # get average accuracy
    result = logreg.score(X_test, y_test)
    print result
    # predict 0 or 1
    y_pred = logreg.predict(X_test)
    confusion_m = confusion_matrix(y_test, y_pred)
    print confusion_m
    print(classification_report(y_test, y_pred))
    # to save model
    filename = '/home/jfreek/workspace/models/lr_model_test.sav'
    pickle.dump(logreg, open(filename, 'wb'))
    # to load
    lr_model = pickle.load(open(filename, 'rb'))
    # get probabilities
    probs = logreg.predict_proba(X_test)
    prob_df = pd.DataFrame()
    for prob in probs:
        temp = pd.DataFrame({'is_duplicate': prob[1]}, index=[0])
        prob_df = prob_df.append(temp)
    prob_df.reset_index(drop=True, inplace=True)
    prob_df['id'] = df['id']

    # ********** Find Duplicates pipeline **********
    df = pd.DataFrame()
    for row in train_df[['id', 'question1', 'question2', 'is_duplicate']].values:
        df_1 = fd.word_tag(row[1])
        df_2 = fd.word_tag(row[2])
        temp = fd.similarity_percentage(df1=df_1, df2=df_2)
        # filter part HERE
        temp['id'] = row[0]
        df = df.append(temp)
    df.reset_index(drop=True, inplace=True)

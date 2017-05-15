# -*- coding: UTF-8 -*-
import time
# from difflib import SequenceMatcher
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from pandas import DataFrame, read_csv, concat
import spacy
from joblib import Parallel, delayed

# Global Variables:
cluster_path = "/home/jfreek/workspace/w2v_clusters/quora_300_5_2_e-3_sg_kmeans_10_dict.p"
model_path = "/home/jfreek/workspace/models/wikipedia_glove_300_dict"
nlp = spacy.load('en')
clusters = pickle.load(open(cluster_path, "rb"))
model_dict = pickle.load(open(model_path, "rb"))


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
    # sm = SequenceMatcher(None, list1, list2)
    # return sm.ratio()


def word_tag(question, to_unicode=True):
    """
    Tags words with question id, pos, lemma and w2v cluster
    :param question: question to tag: str
    :param to_unicode: True if you want text transformed to unicode: bool
    :return: A list of dictionaries with all words and all its tags: list
    """
    question = question.title()
    # check text type and converting to unicode
    if type(question) != unicode and to_unicode:
        question = question.decode('utf8')
    # tag process
    quest = nlp(question)
    tags_list = []
    for word in quest:
        try:
            context = clusters[word.text.lower()]
            context = str(context)
        except:
            context = None
        temp = {'word': word.text, 'lemma': word.lemma_, 'pos': word.pos_, 'ner': word.ent_type_, 'context': context}
        tags_list.append(temp)
    return tags_list


def similarity_percentage(df1, df2):
    """
    Calculates similarity of two questions by comparing the tags.
    :param df1: data frame of question 1 and tags
    :param df2: data frame of question 2 and tags
    :return: data frame with percentages: data frame
    """
    variables = ['context', 'lemma', 'noun']
    df = DataFrame(columns=variables, index=[0])
    for var in variables:
        if var == 'noun':
            df_list1 = df1[(df1['pos'] == 'NOUN') & (df1['context'].notnull())]['context'].tolist()
            df_list2 = df2[(df2['pos'] == 'NOUN') & (df2['context'].notnull())]['context'].tolist()
            p = get_percentage(list1=df_list1, list2=df_list2) if df_list1 and df_list2 else 0
            df[var] = p
        else:
            df_list1 = df1[df1[var].notnull()][var].tolist()
            df_list2 = df2[df2[var].notnull()][var].tolist()
            p = get_percentage(list1=df_list1, list2=df_list2) if df_list1 and df_list2 else 0
            df[var] = p
    return df


def resultant_vector():
    pass


def vector_similarity():
    pass


def log_regression(df, x_variables, y_variables, path):
    """
    Saves a logistic regression model.
    :param df: data frame with variables
    :param x_variables: list of columns to consider: list
    :param y_variables: column to fit with: str
    :param filename: name of model
    :return: saves model as pickle
    """
    X = df[x_variables].values
    y = df[y_variables].values
    logreg = LogisticRegression()
    logreg.fit(X, y)
    # to save model
    pickle.dump(logreg, open(path, 'wb'))


def dev_pipeline(row):
    """
    Steps for development of model.
    tags words and finds similarities for further check. 
    :param row: data frame row: pandas series
    :return: data frame
    """
    dl_1 = word_tag(row[0])
    dl_2 = word_tag(row[1])
    df_1 = DataFrame(dl_1)
    df_2 = DataFrame(dl_2)
    temp = similarity_percentage(df1=df_1, df2=df_2)
    return temp


def main():
    tmp_path = '/home/jfreek/workspace/tmp/'
    models_path = '/home/jfreek/workspace/models/lr_model.sav'

    # ********** DEV find duplicates **********
    train_df = read_csv(tmp_path+'train.csv')
    train_df.dropna(inplace=True)
    train_df.reset_index(inplace=True, drop=True)
    train_df = train_df[:100000]

    # dev pipeline PARALLEL:
    t0 = time.time()
    temp = Parallel(n_jobs=7)(delayed(dev_pipeline)(row) for row in train_df[['question1', 'question2']].values)
    t1 = time.time()
    total = t1 - t0
    print "total time: " + str(total)
    df = concat(temp)
    del temp
    df.reset_index(drop=True, inplace=True)

    # Logistic Regression TEST
    df['is_duplicate'] = train_df['is_duplicate']
    df['id'] = train_df['id']
    del train_df
    X = df[['lemma', 'noun']].values
    y = df['is_duplicate'].values
    logreg = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=6)
    logreg.fit(X_train, y_train)
    # get average accuracy
    result = logreg.score(X_test, y_test)
    # predict 0 or 1 Conf Matrix
    y_pred = logreg.predict(X_test)
    confusion_m = confusion_matrix(y_test, y_pred)
    # show results:
    print result
    print confusion_m
    print(classification_report(y_test, y_pred))

    # Logistic Regression all data IF TEST LOOKS GOOD:
    log_regression(df=df, x_variables=['lemma', 'noun'], y_variables='is_duplicate', path=models_path)

    # ********** Find Duplicates pipeline **********
    test_df = read_csv(tmp_path+'test.csv')

    # PARALLEL:
    t0 = time.time()
    temp = Parallel(n_jobs=7)(delayed(dev_pipeline)(row) for row in test_df[['question1', 'question2']].values)
    df = concat(temp)
    del temp
    df.reset_index(drop=True, inplace=True)
    t1 = time.time()
    total = t1 - t0
    print "total time: " + str(total)

    df['test_id'] = test_df['test_id']
    del test_df
    logreg = pickle.load(open(models_path, 'rb'))

    # get probabilities
    X_test = df[['lemma', 'noun']].values
    probs = logreg.predict_proba(X_test)
    prob_df = DataFrame(data=probs[0:, 1:], columns=['is_duplicate'])
    prob_df['test_id'] = df['test_id']
    del df

    # save df with requested format
    prob_df.to_csv(path_or_buf=tmp_path+'results_test', header=['test_id', 'is_duplicate'],
                                              columns=['test_id', 'is_duplicate'], index=None, sep=',', mode='w')

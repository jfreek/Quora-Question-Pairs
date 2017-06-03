# -*- coding: UTF-8 -*-
import time
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from pandas import DataFrame, read_csv, concat
from numpy import sum
import spacy
from joblib import Parallel, delayed

# Global Variables:
model_path = "/home/jfreek/workspace/models/wikipedia_glove_300_dict"
model_dict = pickle.load(open(model_path, "rb"))
nlp = spacy.load('en')
cluster_path = "/home/jfreek/workspace/w2v_clusters/quora_300_5_2_e-3_sg_kmeans_10_dict.p"
clusters = pickle.load(open(cluster_path, "rb"))


def context_similarity(list1, list2, vectors=True):
    if vectors:
        v1 = sum(list1, axis=0)
        v2 = sum(list2, axis=0)
        sim = cosine_similarity(v1, v2)[0][0]
    else:
        t = len(list1) + len(list2)
        count = 0
        for item in list1:
            if list2:
                if item in list2:
                    count += 1
                    list2.remove(item)
            else:
                break
        sim = float(2*count)/t
    return sim


def word_tag(question, vectors=True):
    """
    Tags words with question id, pos, lemma and w2v cluster
    :param question: question to tag: str
    :param to_unicode: True if you want text transformed to unicode: bool
    :return: A list of dictionaries with all words and all its tags: list
    """
    # question = ' '.join([word[0].upper() + word[1:] for word in question.split(' ')])
    # check text type and converting to unicode
    if type(question) != unicode:
        question = question.decode('utf8')
    # tag process
    quest = nlp(question)
    tags_list = []
    for word in quest:
        try:
            if vectors:
                context = model_dict[word.text.lower()]
            else:
                context = clusters[word.text.lower()]
                context = str(context)
        except:
            if vectors:
                context = [0]*300
            else:
                context = None
        temp = {'word': word.text, 'pos': word.pos_, 'ner': word.ent_type_, 'context': context}
        tags_list.append(temp)
    return tags_list


def similarity_percentage(df1, df2, vectors=True):
    """
    Calculates similarity of two questions by comparing the tags.
    :param df1: data frame of question 1 and tags
    :param df2: data frame of question 2 and tags
    :return: data frame with percentages: data frame
    """
    pos_var = ['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'NUM', 'PROPN', 'PRON']
    ner_var = ['PERSON', 'NORP', 'FACILITY', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE',
               'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
    variables = pos_var + ner_var
    df = DataFrame(columns=variables, index=[0])
    # set indices to take values after the first verb
    verbs1 = df1[df1.pos == 'VERB']
    verbs2 = df2[df2.pos == 'VERB']
    i = verbs1.index[0] if not verbs1.empty else None
    j = verbs2.index[0] if not verbs2.empty else None
    # get context (vectors or clusters) & similarities
    for var in variables:
        if var in pos_var:
            context_list1 = df1[i+1:][(df1['pos'] == var)]['context'].tolist() if i else df1[(df1['pos'] == var)]['context'].tolist()
            context_list2 = df2[j+1:][(df2['pos'] == var)]['context'].tolist() if j else df2[(df2['pos'] == var)]['context'].tolist()
        else:
            context_list1 = df1[(df1['ner'] == var)]['context'].tolist()
            context_list2 = df2[(df2['ner'] == var)]['context'].tolist()
        # get the vector similarity
        if context_list1 and context_list2:
            p = context_similarity(list1=context_list1, list2=context_list2, vectors=vectors)
        elif not context_list1 and not context_list2:
            p = 1
        else:
            p = 0.0
        df[var] = p
    return df


def dev_pipeline(row):
    """
    Steps for development of model.
    tags words and finds similarities for further check. 
    :param row: data frame row: pandas series
    :return: data frame
    """
    vectors = True
    dl_1 = word_tag(question=row[0], vectors=vectors)
    dl_2 = word_tag(question=row[1], vectors=vectors)
    df_1 = DataFrame(dl_1)
    df_2 = DataFrame(dl_2)
    temp = similarity_percentage(df1=df_1, df2=df_2, vectors=vectors)
    return temp


def main():
    tmp_path = '/home/jfreek/workspace/tmp/'
    models_path = '/home/jfreek/workspace/models/mlp_model'

    # ********** TRAIN find duplicates **********
    train_df = read_csv(tmp_path+'train.csv')
    train_df.dropna(inplace=True)
    train_df.reset_index(inplace=True, drop=True)
    train_df = train_df[:4000]

    # Foreplay Pipeline PARALLEL:
    t0 = time.time()
    temp = Parallel(n_jobs=7)(delayed(dev_pipeline)(row) for row in train_df[['question1', 'question2']].values)
    t1 = time.time()
    total = t1 - t0
    print "total time: " + str(total)
    df = concat(temp)
    del temp
    df.reset_index(drop=True, inplace=True)
    df['is_duplicate'] = train_df['is_duplicate']
    df['id'] = train_df['id']
    del train_df

    # Initialize Classifier & Input values
    clf = MLPClassifier(solver='sgd', activation='logistic', alpha=1e-5, hidden_layer_sizes=(26,), random_state=1)
    X = df[['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'NUM', 'PROPN', 'PRON', 'PERSON', 'NORP', 'FACILITY', 'FAC', 
            'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 
            'QUANTITY', 'ORDINAL', 'CARDINAL']].values
    y = df['is_duplicate'].values

    del df

    # Check MLPClassifier
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=6)
    clf.fit(x_train, y_train)
    # get average accuracy
    result = clf.score(x_test, y_test)
    # predict 0 or 1 Conf Matrix
    y_pred = clf.predict(x_test)
    confusion_m = confusion_matrix(y_test, y_pred)
    # show results:
    print result
    print confusion_m
    print(classification_report(y_test, y_pred))    

    # Train MLPClassifier Full Data:
    clf.fit(X, y)
    
    # save classifier
    pickle.dump(clf, open(models_path, 'wb'))

    # ********** TEST Find Duplicates **********
    # Load Testing Data & Classifier
    test_df = read_csv(tmp_path+'test.csv')
    clf = pickle.load(open(models_path, 'rb'))

    # Foreplay Pipeline PARALLEL:
    t0 = time.time()
    temp = Parallel(n_jobs=7)(delayed(dev_pipeline)(row) for row in test_df[['question1', 'question2']].values)
    t1 = time.time()
    total = t1 - t0
    print "total time: " + str(total)
    df = concat(temp)
    del temp
    df.reset_index(drop=True, inplace=True)
    df['test_id'] = test_df['test_id']
    del test_df

    # get probabilities
    X_test = df[['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'NUM', 'PROPN', 'PRON', 'PERSON', 'NORP', 'FACILITY', 'FAC', 
        'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 
        'QUANTITY', 'ORDINAL', 'CARDINAL']].values
    probs = clf.predict_proba(X_test)
    prob_df = DataFrame(data=probs[0:, 1:], columns=['is_duplicate'])
    prob_df['test_id'] = df['test_id']
    del df

    # save df with requested format
    prob_df.to_csv(path_or_buf=tmp_path+'results_test', header=['test_id', 'is_duplicate'],
                                              columns=['test_id', 'is_duplicate'], index=None, sep=',', mode='w')

if __name__ == '__main__':
    main()

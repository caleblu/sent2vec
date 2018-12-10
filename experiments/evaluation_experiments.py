import numpy as np
import re
from nltk.tokenize import TweetTokenizer
from sklearn.metrics.pairwise import cosine_similarity as cos
from scipy.stats import pearsonr, spearmanr
import sent2vec
import random
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def sort_length_embedding_mr(sents, labels, model):
    """
    sents: array, n sentences
    labels: array, label of every movie review
    model: embedding model
    return emb: embedding vectors of sorted sentences
           sorted_labels
           length
    """
    dict_length = {}
    tknzr = TweetTokenizer()
    n = len(sents)
    tokenized_sents = []
    sentences = []
    for i in range (n):
        string = sents[i]
        string = re.sub(r'[^\w\s]','',string)
        sent_list = tknzr.tokenize(string)
        dict_length[i] = len(sent_list)
        tokenized_sents.append(' '.join(sent_list).lower())
    
    sorted_by_value = sorted(dict_length.items(), key=lambda kv: kv[1])
    
    sorted_sents = []
    sorted_labels = []
    length = []
    for item in sorted_by_value:
        sorted_sents.append(tokenized_sents[item[0]])
        sorted_labels.append(labels[item[0]])
        length.append(item[1])
    emb = model.embed_sentences(sorted_sents)
    return emb, sorted_labels, length

# logistic regression to classify the movie review
def nestedCV(X, Y, Cs, innercv, outercv):
    """
    Nested Cross Validation to select the best hyperparameters
    and evaluate the logistic regression model.
    :param X: n by d array, input features
    :param Y: n by 1 array, labels
    :param Cs: List or Array of candidates parameters for penalty in LR
    :param innercv: int, fold of the inner cross validation
    :param outercv: int, fold of the outer cross validation
    :return: average score of cross validation
    """
    clf_inner = GridSearchCV(estimator=LogisticRegression(), param_grid=Cs, cv=innercv)
    clf_inner.fit(X, Y)
    C_best = clf_inner.best_params_['C']
    clf_outer = LogisticRegression(C=C_best)
    scores = cross_val_score(clf_outer, X, Y, cv=outercv)
    return scores.mean()

def conduct_lr(x, y):
    # classify the movie reviews and see the accuracy
    sc = StandardScaler()
    x_std = sc.fit_transform(x)

    # create penalty coefficients candidates in logistic regression
    C_candidates = dict(C=np.arange(5, 10, 1))

    # nested CV for logistic regression
    score = nestedCV(x_std, y, C_candidates, 3, 3)
    return score

def sort_length_embedding_sts(sents1, sents2, labels, model):
    """
    sents: array, n sentences
    labels: array, label of every movie review
    model: embedding model
    return emb: embedding vectors of sorted sentences
           sorted_labels
           length
    """
    dict_length = {}
    tknzr = TweetTokenizer()
    n = len(sents1)
    tokenized_sents1 = []
    tokenized_sents2 = []

    for i in range (n):
        string1 = sents1[i]
        string2 = sents2[i]
        string1 = re.sub(r'[^\w\s]','',string1)
        string2 = re.sub(r'[^\w\s]','',string2)
        sent_list1 = tknzr.tokenize(string1)
        sent_list2 = tknzr.tokenize(string2)
        dict_length[i] = (len(sent_list1)+len(sent_list2))/2.0
        tokenized_sents1.append(' '.join(sent_list1).lower())
        tokenized_sents2.append(' '.join(sent_list2).lower())
    
    sorted_by_value = sorted(dict_length.items(), key=lambda kv: kv[1])
    
    sorted_sents1 = []
    sorted_sents2 = []
    sorted_labels = []
    length = []
    
    for item in sorted_by_value:
        sorted_sents1.append(tokenized_sents1[item[0]])
        sorted_sents2.append(tokenized_sents2[item[0]])
        sorted_labels.append(labels[item[0]])
        length.append(item[1])
    emb1 = model.embed_sentences(sorted_sents1)
    emb2 = model.embed_sentences(sorted_sents2)
    
    return emb1, emb2, sorted_labels, length
# evaluate STS using cosine similarity
# and compare the results with the gold standard.
# sentsets: sentence datasets:
#           deft-forum, deft-news, headlines, images, OnWM, tweet-news
def STS_eval(sentset, model,data_path):
    """
    Evaluate the similarities of 
    :param sentset: string, sentence dataset
    :param model: sentence embedding model
    :return: cosine similarity, of all pairs of sentences
             pearson & spearman coefficients compared to gold standard
    """
    sent_file = open(data_path + 'sts-en-test-gs-2014/STS.input.'+sentset+'.txt')
    sent_data = sent_file.readlines()
    sent_file.close()
    gs_file = open(data_path + 'sts-en-test-gs-2014/STS.gs.'+sentset+'.txt')
    gs_data = np.array(gs_file.readlines(), dtype=float)
    gs_file.close()
    splited_sent = []
    n = len(sent_data)
    for i in range(n):
        splited_sent.append(re.split(r'\t+', sent_data[i]))
    splited_sent = np.array(splited_sent)
    sent_1 = splited_sent[:,0]
    sent_2 = splited_sent[:,1]
    x_1, x_2, y, ls = sort_length_embedding_sts(sent_1, sent_2, gs_data, model)
    
    s1 = x_1[:81]
    s2 = x_2[:81]
    y1 = y[:81]
    c1 = []
    
    s1_2 = x_1[81:162]
    s2_2 = x_2[81:162]
    y2 = y[81:162]
    c2 = []
    
    s1_3 = x_1[162:227]
    s2_3 = x_2[162:227]
    y3 = y[162:227]
    c3 = []
    
    s1_4 = x_1[227:]
    s2_4 = x_2[227:]
    y4 = y[227:]
    c4 = []
    
    
    
    pearsons = []
    spearmanrs = []
    

    for i in range(len(s1)):
        v1 = s1[i]
        v2 = s2[i]
        cos_i = cos([v1], [v2])
        c1.append(cos_i[0][0])
    pearsons.append(pearsonr(c1, y1)[0])
    spearmanrs.append(spearmanr(c1, y1)[0])
    
    for i in range(len(y2)):
        v1 = s1_2[i]
        v2 = s2_2[i]
        cos_i = cos([v1], [v2])
        c2.append(cos_i[0][0])
    pearsons.append(pearsonr(c2, y2)[0])
    spearmanrs.append(spearmanr(c2, y2)[0])
    
    for i in range(len(y3)):
        v1 = s1_3[i]
        v2 = s2_3[i]
        cos_i = cos([v1], [v2])
        c3.append(cos_i[0][0])
    pearsons.append(pearsonr(c3, y3)[0])
    spearmanrs.append(spearmanr(c3, y3)[0])
    
    for i in range(len(y4)):
        v1 = s1_4[i]
        v2 = s2_4[i]
        cos_i = cos([v1], [v2])
        c4.append(cos_i[0][0])
    pearsons.append(pearsonr(c4, y4)[0])
    spearmanrs.append(spearmanr(c4, y4)[0])
        
    
    return pearsons, spearmanrs
    
    

def get_similarity(t1,t2,model):
    tknzr = TweetTokenizer()
    t1 = ' '.join(tknzr.tokenize(t1)).lower()
    t2 = ' '.join(tknzr.tokenize(t2)).lower()
    emb = model.embed_sentences([t1,t2])
#     print(emb.shape)
#     pearson = pearsonr(emb[0,:],emb[1,:])[0]
#     spearman = spearmanr(emb[0,:],emb[1,:])[0]
#     return np.round(pearson,3),np.round(spearman,3),np.round((pearson + spearman)/2.0,3)
    emb_1 = np.expand_dims(emb[0,:], axis=0)    
    emb_2 = np.expand_dims(emb[1,:], axis=0)
    cos = cosine_similarity(emb_1,emb_2)
    return np.round(cos[0][0],3)

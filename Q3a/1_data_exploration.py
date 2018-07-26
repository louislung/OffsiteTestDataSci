
# coding: utf-8

# # Data exploration

# In[63]:


import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import re, sys, os, math, jieba
import pandas as pd
import numpy as np
from sklearn import feature_extraction
from sklearn.preprocessing import LabelEncoder  
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import xgboost as xgb


# In[39]:


data_train = pd.read_csv('data/offsite-tagging-training-set.csv', index_col='id')
data_test = pd.read_csv('data/offsite-tagging-test-set.csv', index_col='id')
#print(data_train.isnull().sum())
#print(data_test.isnull().sum())
data_train.dropna(how="all",inplace=True)
data_test.dropna(how="all",inplace=True)
data_train['tags_en'] = data_train.tags
data_train['tags_en'][data_train.tags == '足球'] = 'Football'
data_train['tags_en'][data_train.tags == '梁振英'] = '689'
data_train['tags_en'][data_train.tags == '美國大選'] = 'USA election'
data_train.describe()


# In[40]:


data_test.describe()


# In[41]:


pd.DataFrame({'cnt':data_train.tags.value_counts(),'%':data_train.tags.value_counts()/data_train.tags.count()})


# In[42]:


#clean html tags
def remove_html_tags(text):
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

data_train['text_clean'] = data_train['text'].apply(remove_html_tags)
data_test['text_clean'] = data_test['text'].apply(remove_html_tags)
data_train['log_len'] = data_train.text.apply(len).apply(math.log)
data_train['log_len_clean'] = data_train.text_clean.apply(len).apply(math.log)
data_test['log_len'] = data_test.text.apply(len).apply(math.log)
data_test['log_len_clean'] = data_test.text_clean.apply(len).apply(math.log)


# In[43]:


fig = plt.figure(figsize=(18,6), dpi=1600) 
alpha=alpha_scatterplot = 0.2 
alpha_bar_chart = 0.55
prop = matplotlib.font_manager.FontProperties(fname="SimHei.ttf")
#plt.text(0.5, 0.5, s='测试', fontproperties=prop)
#plt.show()
plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
data_train.text_clean.apply(len).apply(math.log)[data_train.tags == '足球'].plot(kind='kde')
data_train.text_clean.apply(len).apply(math.log)[data_train.tags == '梁振英'].plot(kind='kde')
data_train.text_clean.apply(len).apply(math.log)[data_train.tags == '美國大選'].plot(kind='kde')
# plots an axis lable
plt.xlabel("Log(Length of Cleaned Text )")
plt.title("Text Length Distribution per Tags")
# sets our legend for our graph.
plt.legend(('足球', '梁振英','美國大選'),loc='best',prop=prop)


# In[44]:


data_train['log_len'] = data_train.text.apply(len).apply(math.log)
data_train['log_len_clean'] = data_train.text_clean.apply(len).apply(math.log)
data_train.boxplot('log_len',by='tags_en')
data_train.boxplot('log_len_clean',by='tags_en')


# In[45]:


data_train['text_clean'] = data_train['text_clean'].apply(lambda x: " ".join(jieba.cut(x)))
data_test['text_clean'] = data_test['text_clean'].apply(lambda x: " ".join(jieba.cut(x)))
"""
#CountVectorizer
vectorizer_cnt = CountVectorizer(tokenizer=lambda x: x.split(), stop_words='english', min_df=0.05, max_df=0.8)
X = vectorizer_cnt.fit_transform(data_train.text_clean)
vectorizer_cnt.get_feature_names()
"""
#TfidfVectorizer
vectorizer_tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(), min_df=0.05, max_df=0.7)
tfidf = vectorizer_tfidf.fit_transform(data_train.text_clean)
indices_tfidf = np.argsort(vectorizer_tfidf.idf_)[::-1]
features_tfidf = vectorizer_tfidf.get_feature_names()
top_features_tfidf = [features_tfidf[i] for i in indices_tfidf]
print('Total {0} features found. (Ordered by idf in ascending order)'.format(len(top_features_tfidf)))
print(top_features_tfidf)


# In[46]:


stop_words = ['nbsp','雖然','可能','可以','不過','不會','沒有']
top_features_tfidf = [x for x in top_features_tfidf if len(x) > 1 and not re.match('^[0-9\.%, ]*$',x) and x not in stop_words]
print('Total {0} features after cleaning. (Ordered by idf in ascending order)'.format(len(top_features_tfidf)))
print(top_features_tfidf)


# In[58]:


vocab_tfidf = dict(zip(top_features_tfidf,range(len(top_features_tfidf))))


# ## Build Model

# In[90]:


le = LabelEncoder()
train_Y = le.fit_transform(data_train['tags'].values)

#TfidfVectorizer as features
vectorizer_tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(), vocabulary=vocab_tfidf)
train_X = vectorizer_tfidf.fit_transform(data_train.text_clean)
test_X = vectorizer_tfidf.transform(data_test.text_clean)
#features_tfidf = vectorizer_tfidf.get_feature_names()
#unique, counts = np.unique(test_tfidf.getnnz(1), return_counts=True)
#print(dict(zip(unique, counts)))


params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.9,
        'objective': 'multi:softmax',
        'num_class': 3,
        #'silent': 0,
        'colsample_bytree': 0.9
}
num_boost_round = 500
early_stopping_rounds = 50

d_train = xgb.DMatrix(train_X, train_Y)
d_test = xgb.DMatrix(test_X)
cv_model = xgb.cv(params = params, dtrain = d_train, num_boost_round = num_boost_round, early_stopping_rounds = early_stopping_rounds, 
               nfold=10, verbose_eval = 20, seed=1234)

num_boost_round = cv_model['test-merror-mean'].idxmin()
final_model = xgb.train(params = params, dtrain = d_train, num_boost_round = num_boost_round,
               verbose_eval = 20, evals = [(d_train,'train')])

xgb_pred = list(final_model.predict(xgb.DMatrix(train_X)))
xgb_pred = [int(x) for x in xgb_pred]
correct = np.where(le.inverse_transform(xgb_pred) == data_train.tags,1,0)
output = pd.DataFrame({'id': data_train.index, 'target':data_train.tags, 'pred_target': le.inverse_transform(xgb_pred), 'text': data_train.text})
output.to_csv("{}-foldCV_sub.csv".format(K), index=False)





xgb_pred = list(final_model.predict(d_test))
xgb_pred = [int(x) for x in xgb_pred]
output = pd.DataFrame({'id': data_test.index, 'target': le.inverse_transform(xgb_pred), 'text': data_test.text})
output.to_csv("{}-foldCV_sub.csv".format(K), index=False)


# In[96]:


set(preds)


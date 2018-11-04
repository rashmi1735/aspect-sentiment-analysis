
# coding: utf-8

# In[1]:


from stanfordcorenlp import StanfordCoreNLP
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.corenlp import CoreNLPDependencyParser

import numpy as np
import pandas as pd
import re


# In[148]:


train_df = pd.read_csv('project_2_train/data_1_train.csv')
train_df.columns = ['example_id', 'text', 'aspect_term', 'term_location', 'class']


# In[149]:


def raw_parser(sentence, aspect_term):
    parse_view = dependency_parser.raw_parse(sentence)
    dep = next(parse_view)
    parse_list = list(dep.triples())
    df = pd.DataFrame(parse_list)
    df.columns = ['0','dep','2']
    df['left'] = (list(zip(*df['0'] ))[0])
    df['right'] = (list(zip(*df['2'] ))[0])
    df['left-tag'] = (list(zip(*df['0'] ))[1])
    df['right-tag'] = (list(zip(*df['2'] ))[1])
    df = df.drop(columns = ['0','2'])
    df = df.loc[df['dep'].isin(['dobj','advmod', 'amod', 'neg', 'nsubj'])]
    result = (df.loc[((df['left'] == aspect_term) | (df['right'] == aspect_term)), ['left','right']]).values.tolist()
    return result


# In[150]:


nlp = StanfordCoreNLP(r'C:\Users\parag\Downloads\stanford-corenlp-full-2018-02-27')
path_to_jar = 'C:/Users/parag/Downloads/stanford-parser-full-2018-02-27/stanford-parser.jar'
path_to_models_jar = 'C:/Users/parag/Downloads/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar'
dependency_parser = dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')#(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
train_df['text'] = train_df['text'].apply(lambda str : str.replace("[comma]", ","))
#train_df['text'] = train_df['text'].apply(lambda str : str.replace("-", " "))

#train_df['text'] = train_df['text'].apply(lambda str : str.lower())

stop = [' a ', ' an ', ' the ', ' A ', ' An ', ' The ', '-']
big_regex = re.compile('|'.join(map(re.escape, stop)))
#print(train_df['text'])
train_df['text_with_aspect'] = train_df['text'].apply(lambda x : big_regex.sub(" ", x))

print(train_df['text_with_aspect'][7])
train_df['text_with_aspect'] = train_df.apply(lambda x: x['text_with_aspect'].replace(x['aspect_term'],(x['aspect_term']).replace(" ","-")), 
                                  axis = 1)
#train_df['result'] = train_df['text_with_aspect'].apply(lambda x : raw_parser(x))

print(train_df['text_with_aspect'][7])
# print(train_df['aspect_term'][26])

# print(train_df['aspect_term'])
train_df['aspect_term'] = train_df['aspect_term'].apply(lambda str : str.replace(" ", "-"))
# print(train_df['aspect_term'])



# In[23]:


#a = train_df[:10]
#a['result'] = a.apply(lambda x : x['text_with_aspect']) 
#c = a.apply(lambda x : raw_parser(x['text_with_aspect'], x['aspect_term']), axis = 1)
train_df['text_with_aspect'][506]


# In[153]:


keywords = dependency_parser.raw_parse_sents(train_df.loc[: ,'text_with_aspect'])
list_triples = [list(next(x).triples()) for x in keywords]


# In[21]:


def req_key(parse_list, aspect_term):    
    df = pd.DataFrame(parse_list)
    df.columns = ['0','dep','2']
    df['left'] = (list(zip(*df['0'] ))[0])
    df['right'] = (list(zip(*df['2'] ))[0])
    df['left-tag'] = (list(zip(*df['0'] ))[1])
    df['right-tag'] = (list(zip(*df['2'] ))[1])
    df = df.drop(columns = ['0','2'])
    df = df.loc[df['dep'].isin(['dobj','advmod', 'amod', 'neg', 'nsubj'])]
    result = (df.loc[((df['left'] == aspect_term) | (df['right'] == aspect_term)), ['left','right']]).values.tolist()
    return result


# In[64]:


train_df['temp'] = list_triples
list_triples[7]
train_df['temp'][7]


# In[130]:


train_df['result'] = train_df.apply(lambda x : req_key(x['temp'], x['aspect_term']), axis = 1)


# In[26]:


req_key(train_df['temp'][506], train_df['aspect_term'][506])


# In[29]:


len(list_triples)


# In[74]:


print(train_df['temp'][15])
print(train_df['text_with_aspect'][15])
print(train_df['aspect_term'][15])


# In[72]:


train_df[train_df['temp'].isnull()].index.tolist()


# In[131]:


train_df[train_df['result'].str.len() == 0].index.tolist()
train_df[train_df['result'].str.len() == 0]['temp']

train_df = train_df.drop(train_df[train_df.result.str.len() == 0].index)


# In[87]:


train_df['temp'].shape


# In[78]:


3602 - 1212


# In[132]:


train_df['result'] = train_df['result'].apply(lambda x : [item for sublist in x for item in sublist])
print(train_df['result'][0])


# In[133]:


train_df['result'] = train_df['result'].apply(lambda x : list(set(x)))
print(train_df['result'][0])


# In[105]:


set(['hello', 'how'])


# In[134]:


print(train_df['result'][0])
train_df['result'] = train_df['result'].apply(lambda x : " ".join(x))
train_df['result']


# In[135]:


from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp

vectorizer = CountVectorizer(analyzer = "word",
                            tokenizer = None,
                            preprocessor = None,
                            stop_words = None,
                            min_df = 1)
sample = train_df[['result', 'aspect_term']]


features = sp.hstack(sample.apply(lambda col: vectorizer.fit_transform(col))) #no polarity of aspect taken into account yet!
features = features.toarray()
train_df = train_df.reset_index(drop=True)


# In[141]:


train_df.shape[0]


# In[143]:


import math

train_end = math.floor(0.8*(train_df.shape[0]))
train_features = features[:train_end]
test_features = features[train_end:]

print(len(train_features))
print(len(test_features))
print(train_end)
y = np.array(train_df.loc[: train_end - 1 , 'class'])


# In[ ]:


estimators = np.array([40,50,60,70,80,90,100])
depths = np.array([30,35,40,45,50,55,60,65])

model = RandomForestClassifier()
grid1 = GridSearchCV(estimator=model, param_grid = dict(max_depth = depths, n_estimators = estimators), cv = 10, refit=True)
grid1.fit(train_features, y)
predictions = grid1.predict(test_features)
print(((predictions == train_df.loc[ train_end :  , 'class' ]).sum()) / predictions.size)
print(classification_report(train_df.loc[ train_end :  , 'class' ], predictions)) #, target_names=target_names


# In[ ]:


predictions_train = grid1.predict(train_features)

print(((predictions_train == train_df.loc[: train_end - 1 , 'class' ]).sum()) / predictions_train.size)


# In[147]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

forest = RandomForestClassifier()
#forest = GaussianNB()
forest = forest.fit(train_features, train_df.loc[ : train_end - 1 , 'class'])
result = forest.predict(train_features)
print(((result == train_df.loc[: train_end - 1  , 'class']).sum()) / result.size)


import pandas as pd
import numpy as np
import dateutil.parser
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import gensim
from gensim.utils import simple_preprocess
from nltk.stem.wordnet import WordNetLemmatizer 
import string
import gensim.corpora as corpora
from nltk import word_tokenize, pos_tag, pos_tag_sents
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from readability import Readability 
stopwords.ensure_loaded()  

#this function remove some strings that are not really part of the actual article content
def content_cleaning(data_df,content_col):
    #clean out urls
    data_df[content_col] = data_df[content_col].replace(regex=True,to_replace='((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)',value='')
    #clean out html
    data_df[content_col] = data_df[content_col].replace(regex=True,to_replace='<(?!3).*?>|<(?!^[A-Z]).*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});',value='')
    #clean out usernames
    data_df[content_col] = data_df[content_col].replace(regex=True,to_replace='/@\w+/', value='')
    #clean out photography by
    data_df[content_col] = data_df[content_col].replace(regex=True,to_replace='(?<=[\.\?](?!.*[\.\?]))[ ]+[P]hotography\sby\s.*', value='')                                                                                                       
    #clean out art by
    data_df[content_col] = data_df[content_col].replace(regex=True,to_replace='(?<=[\.\?](?!.*[\.\?]))[ ]+[A]rt\sby\s.*', value='')                                                                                                       
    #clean out photograph by
    data_df[content_col] = data_df[content_col].replace(regex=True,to_replace='(?<=[\.\?](?!.*[\.\?]))[ ]+[P]hotograph\sby\s.*', value='')                                                                                                       
    #clean out photo by
    data_df[content_col] = data_df[content_col].replace(regex=True,to_replace='(?<=[\.\?](?!.*[\.\?]))[ ]+[P]hoto\sby\s.*', value='')                                                                                                       
    #clean out poster by
    data_df[content_col] = data_df[content_col].replace(regex=True,to_replace='(?<=[\.\?](?!.*[\.\?]))[ ]+[P]oster\sby\s.*', value='')                                                                                                       
    #'0' tags to blanks
    if 'tags' in data_df.columns:
        data_df['tags'] = data_df['tags'].fillna('').astype(str)
    
    return data_df

def date_cleaning(data_df,time_col,years):
    #convert string dates to datetime format and get year and year-month
    data_df[time_col] = data_df[time_col].astype(str).apply(lambda x: dateutil.parser.parse(x))
    data_df['year'] = pd.DatetimeIndex(data_df[time_col]).year
    data_df['year_month'] = data_df[time_col].dt.to_period('M')
    data_df = data_df.loc[~data_df.loc[:,'year'].isin(years)]
    
    return data_df

#function for lemmatization
def lemmatize(doc,stop_words):
    lemma = WordNetLemmatizer()
    exclude = set(string.punctuation) 
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop_words])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized



#function further removes some words not needed for topic_modeling and stores output in new columns, where listcol is headlines, topics and tags
def topicmodeling_cleaning(data_df,listcol,stop_words):
    listcol_new = [x + '_tm' for x in listcol]
    for col in listcol:
        data_df.loc[:,col+'_tm']=data_df.loc[:,col]
    for col in listcol_new:
      data_df.loc[:,col] = data_df.loc[:,col].apply(lambda x: " ".join(x.lower() for x in x.split()))                         #change to lowercase
      data_df.loc[:,col] = data_df.loc[:,col].replace(regex=True,to_replace='((?<=[a-z])[A-Z]|[A-Z](?=[a-z]))',value=' $1')   #Newline spaces
      data_df.loc[:,col] = data_df.loc[:,col].str.replace('[^\w\s]',' ')                                                      #removing punctuations
      data_df.loc[:,col] = data_df.loc[:,col].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))          #removing stopwords
      data_df.loc[:,col] = data_df.loc[:,col].str.replace('\d+', '')                                                          #removing numerics
      data_df.loc[:,col] = data_df.loc[:,col].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')
    return data_df,listcol_new

#this function tokenizes
def split_text_lemma(data_df,listcol_new,stop_words):
    data_df.loc[:,'topic'] = data_df.loc[:,listcol_new].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    data_df.loc[:,'topic'] = data_df.loc[:,'topic'].astype(str).apply(lambda x: lemmatize(x,stop_words).split())
    return data_df



def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        # deacc=True removes punctuations       

def bigram_get(data_df):
    data_words = list(sent_to_words(data_df.loc[:,'topic']))
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod

def bigrams_apply(data_df,bigram_mod):
    data_df.loc[:,'topic'] = [bigram_mod[doc] for doc in data_df.loc[:,'topic']]
    return data_df

def word_remove(data_df):
    
    least = list((pd.Series(' '.join(data_df.loc[:,'topic'].apply(lambda x: ' '.join(x))).split()).value_counts()[-26000:]).index)
    most = list((pd.Series(' '.join(data_df.loc[:,'topic'].apply(lambda x: ' '.join(x))).split()).value_counts()[:30]).index)
    list_out = least + most
    i=0
    topic_new = []

    while i <len(data_df):
        print(len(data_df.iloc[i]['topic']))
        print(len([ x for x in data_df.iloc[i]['topic'] if x not in list_out]))
        # df['topic_new'] = [ x for x in data_df.iloc[i]['topic'] if x not in list_out]
        topic_new.append([ x for x in data_df.iloc[i]['topic'] if x not in list_out])
        i = i+1
    data_df.loc[:,'topic'] = topic_new
    return data_df



#use only for training!!!!!! do not call for preprocessing of input.
def dictionary(data_df):
    id2word = corpora.Dictionary(data_df.loc[:,'topic'])      # Create dictionary
    id2word.save('./models/dictionary.sav')
    return id2word

def doc_term_matrix(data_df,dic):
    texts = data_df.loc[:,'topic']                           # Create corpus
    doc_term_matrix = [dic.doc2bow(text) for text in texts] # Apply Term Frequency
    return doc_term_matrix

def sentiment_score(data_df,col):
    analyser = SentimentIntensityAnalyzer()
    i=0 
    headline_s = [ ] 

    while (i<len(data_df)):
        h = analyser.polarity_scores(data_df.iloc[i][col])
        headline_s.append(h['compound'])
        i = i+1

    headline_s = np.array(headline_s)
    data_df.loc[:,'VADER_Score_'+col] = headline_s
    
    #Assigning score categories and logic
    i = 0

    predicted_value_content = [ ] #empty series to hold our predicted values

    while(i<len(data_df)):
        if ((data_df.iloc[i]['VADER_Score_'+col] >= 0.7)):
            predicted_value_content.append('positive')
            i = i+1
        elif ((data_df.iloc[i]['VADER_Score_'+col] > 0) & (data_df.iloc[i]['VADER_Score_'+col] < 0.7)):
            predicted_value_content.append('neutral')
            i = i+1
        elif ((data_df.iloc[i]['VADER_Score_'+col] <= 0)):
            predicted_value_content.append('negative')
            i = i+1

    data_df.loc[:,'predicted_sentiment_'+col] = predicted_value_content


    return data_df

def readability(data_df,col):
    i=0 #counter
    few_words=0
    flesch_kincaid_score = [ ]  
    flesch_kincaid_grade_level = [ ]

    for i in range(0,len(data_df)):
        if len(data_df.iloc[i][col].split())>=100:
          fk_score = Readability(data_df.iloc[i][col]).flesch_kincaid().score
          fk_gl = Readability(data_df.iloc[i][col]).flesch_kincaid().grade_level
          flesch_kincaid_score.append(fk_score)
          flesch_kincaid_grade_level.append(fk_gl)

        else:
          flesch_kincaid_score.append(np.nan)
          flesch_kincaid_grade_level.append(np.nan)
          few_words= few_words+1


    #converting sentiment values to numpy for easier usage

    flesch_kincaid_score = np.array(flesch_kincaid_score)
    flesch_kincaid_grade_level = np.array(flesch_kincaid_grade_level)

    data_df.loc[:,'fk_score'] = flesch_kincaid_score
    data_df.loc[:,'fk_grade_level'] = flesch_kincaid_grade_level
    return data_df

def article_counts(data_df,listcol):
    for col in listcol:
      data_df['word_count_' + col]=data_df[col].apply(lambda x: len(str(x).split(" ")))
      data_df['char_count_' + col]=data_df[col].map(str).apply(len)


    return data_df

#nouns,verbs,adjectives,only
def pos_tagging(data_df,listcol):
    pronouns = ['PRP','PRP$']
    for col in listcol:
        data_df.loc[:,col+'_nouns'] = data_df.loc[:,col].apply(lambda x : [token for token, pos in pos_tag(word_tokenize(x)) if pos.startswith('N')])
        data_df.loc[:,col+'_nouns_count'] = data_df.loc[:,col+'_nouns'].apply(lambda x: len(x))
        data_df.loc[:,col+'_verbs'] = data_df.loc[:,col].apply(lambda x : [token for token, pos in pos_tag(word_tokenize(x)) if pos.startswith('V')])
        data_df.loc[:,col+'_verbs_count'] = data_df.loc[:,col+'_verbs'].apply(lambda x: len(x))
        data_df.loc[:,col+'_adj'] = data_df.loc[:,col].apply(lambda x : [token for token, pos in pos_tag(word_tokenize(x)) if pos.startswith('J')])
        data_df.loc[:,col+'_adj_count'] = data_df.loc[:,col+'_adj'].apply(lambda x: len(x))
        data_df.loc[:,col+'_pronouns'] = data_df.loc[:,col].apply(lambda x : [token for token, pos in pos_tag(word_tokenize(x)) if pos in pronouns])
        data_df.loc[:,col+'_pronouns_count'] = data_df.loc[:,col+'_pronouns'].apply(lambda x: len(x))

    return data_df

def format_topics_sentences(ldamodel, corpus, data_df):
    # Init output
    sent_topics_df = pd.DataFrame()
    texts=data_df['topic']
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)



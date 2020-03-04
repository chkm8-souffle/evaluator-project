import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import streamlit as st
import textprep as tp
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

stopwords.ensure_loaded()  

nltk.download('stopwords')

#header
st.write("""
<style>
.header {
  padding-top: 50px;
  padding-bottom: 50px;
  width:10000 px;
  text-align: center;
  background: orange;
  color: white;
  box-sizing: border-box;
  box-shadow: 10px 10px 5px black;
}
</style>
<div class="header">
<h1><b><font size="+4">ART-E: ARTicle-Evaluator</font></b></h1>
</div>""",unsafe_allow_html=True)



st.write("\n")

#Subheader
st.markdown("### Your aid in writing content for your target audience.")


#body + letsgetthatbread
st.write("""<style>
body {
  background-image: url('https://images.unsplash.com/photo-1580637250481-b78db3e6f84b?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1489&q=80'); 
  opacity:0.9;
  background-repeat: no-repeat;
  background-size: cover;
  color: black;


}
</style>
<i>Let's get that bread.</i>""",unsafe_allow_html=True)

# Text input
headline = st.text_area("Headline of the Article", 'Input your title here.')
content = st.text_area("Content of the Article", 'Input your content here.',)

#submit button
btn = st.button("Submit.")

#when u click the button..
if btn:
	#balloons
    st.balloons()

    #loading
    st.spinner(text='In progress...')

    #create dataframe out of input
    df_input = pd.DataFrame([[headline,content]], columns=['title', 'content'])
    
    #load stopwords
    stop_words = tp.stopwords.words('english')
    stop_words.extend(['akin','aking','yung','lang''ako','alin','am','amin','aming','ang','ano','anumang','apat','at','atin','ating','ay','bababa','bago','bakit','bawat','bilang','dahil','dalawa','dapat','din','dito','doon','gagawin','gayunman','ginagawa','ginawa','ginawang','gumawa','gusto','habang','hanggang','hindi','huwag','iba','ibaba','ibabaw','ibig','ikaw','ilagay','ilalim','ilan','inyong','pic_twitter','com','ako','lang','mag','naman','mo','isa','isang','itaas','ito','iyo','iyon','iyong','ka','kahit','kailangan','kailanman','kami','kanila','kanilang','kanino','kanya','kanyang','kapag','kapwa','karamihan','katiyakan','katulad','kaya','kaysa','ko','kong','kulang','kumuha','kung','laban','lahat','lamang','likod','lima','maaari','maaaring','maging','mahusay','makita','marami','marapat','masyado','may','mayroon','mga','minsan','mismo','mula','muli','na','nabanggit','naging','nagkaroon','nais','nakita','namin','napaka','narito','nasaan','ng','ngayon','ni','nila','nilang','nito','niya','niyang','noon','o','pa','paano','pababa','paggawa','pagitan','pagkakaroon','pagkatapos','palabas','pamamagitan','panahon','pangalawa','para','paraan','pareho','pataas','pero','pumunta','pumupunta','sa','saan','sabi','sabihin','sarili','sila','sino','siya','tatlo','tayo','tulad','tungkol','una','walang'])
    
    #data cleaning to get corpus for topic probabilities
    df = tp.content_cleaning(df_input,'content')
    listcol = ['title','content']
    result_tm_cleaning = tp.topicmodeling_cleaning(df,listcol,stop_words)
    listcol_tm = result_tm_cleaning[1]
    df = result_tm_cleaning[0]
    df = tp.split_text_lemma(df,listcol_tm,stop_words)
    bigram_mod = tp.bigram_get(df)
    df = tp.bigrams_apply(df,bigram_mod)
    
    #other features
    df = tp.article_counts(df,listcol)
    df = tp.sentiment_score(df,'title')
    df = tp.sentiment_score(df,'content')
    df = tp.readability(df,'content')
    df = tp.pos_tagging(df,listcol)
    
    # Load dictionary 
    dic = corpora.Dictionary.load('./models/dictionary.sav') 
    
    # Convert list of words to document term array
    corpus = tp.doc_term_matrix(df,dic)
    
    # Load LDA model
    lda_model = gensim.models.ldamodel.LdaModel.load('./models/lda_model.sav')
    
    # Get document topic probabilities
    lists = lda_model.get_document_topics(corpus, minimum_probability=0.0)
    document_topic_dist = pd.DataFrame([i for i in lists]).apply(lambda x: [i[1] for i in x])
    df = pd.concat([df, document_topic_dist], axis = 1)
    
    #dominant topics
    df_topic_sents_keywords = tp.format_topics_sentences(lda_model,corpus,df)
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic.drop(columns=['Text'],axis=1,inplace=True)
    df = pd.concat([df, df_dominant_topic], axis = 1)

    #get columns for regression
    drop_columns= ['title','fk_grade_level','content','content_tm','title_tm','topic','Document_No','Dominant_Topic','Topic_Perc_Contrib','Keywords','predicted_sentiment_title', 'predicted_sentiment_content', 'title_nouns', 'title_verbs', 'title_adj', 'title_pronouns', 'content_nouns', 'content_verbs', 'content_adj', 'content_pronouns']
    df_regression = df.drop(columns=drop_columns, axis=1)
    

    # Load regression model
    regression_model1 = joblib.load('./models/xgboost_model1.sav')
    regression_model2 = joblib.load('./models/xgboost_model2.sav')
    # predict on output document topic
    if df['word_count_content'][0]<10 or df['word_count_title'][0]<3:
    	prediction_target=0
    	prediction_general=0
    else:
    	prediction_target = regression_model1.predict(df_regression)
    	prediction_general = regression_model2.predict(df_regression)


    #topic interpretation: states if good or not based on insights we got from 2016-2019 data
    if df['Dominant_Topic'].values==0:
    	label='Events-Local'
    	topic_interpretation ="This topic is just average for the target age group."
    elif df['Dominant_Topic'].values==1:
    	label='TV & Movies-Local'
    	topic_interpretation ="This topic is just average for the target age group."
    elif df['Dominant_Topic'].values==2:
    	label='Holidays'
    	topic_interpretation ="This topic is not popular for the target age group."
    elif df['Dominant_Topic'].values==3:
    	label='Food'
    	topic_interpretation ="This topic is not popular for the target age group."
    elif df['Dominant_Topic'].values==4:
    	label='Style-Fashion'
    	topic_interpretation ="This topic is just average for the target age group."
    elif df['Dominant_Topic'].values==5:
    	label='Adulting'
    	topic_interpretation ="This topic ranks high for the target age group."
    elif df['Dominant_Topic'].values==6:
    	label='TV & Movies-International'
    	topic_interpretation ="This topic ranks high for the target age group."
    elif df['Dominant_Topic'].values==7:
    	label='Politics-Duterte Presidency'
    	topic_interpretation ="This topic is just average for the target age group."
    elif df['Dominant_Topic'].values==8:
    	label='Culture'
    	topic_interpretation ="This topic is just average for the target age group."
    elif df['Dominant_Topic'].values==9:
    	label='Music'
    	topic_interpretation ="This topic ranks high for the target age group."
    elif df['Dominant_Topic'].values==10:
    	label='Technology'
    	topic_interpretation ="This topic ranks poor for the target age group."
    elif df['Dominant_Topic'].values==11:
    	label='Way of Living'
    	topic_interpretation ="This topic ranks high for the target age group."
    elif df['Dominant_Topic'].values==12:
    	label='Adulting-Professional Life'
    	topic_interpretation ="This topic ranks high for the target age group."
    elif df['Dominant_Topic'].values==13:
    	label='Nightlife'
    	topic_interpretation ="This topic ranks high for the target age group."
    elif df['Dominant_Topic'].values==14:
    	label='Video Games'
    	topic_interpretation ="This topic ranks high for the target age group."
    


    #rename labels of topics
    df = df.rename(columns={0: 'Events-Local', 1: 'Tv & Movies-Local', 2: 'Holidays', 3: 'Food', 4: 'Style-Fashion', 5: 'Adulting', 6: 'TV & Movies-International', 7: 'Politics-Duterte Presidency', 8: 'Culture', 9: 'Music', 10: 'Technology', 11: 'Way of Living', 12: 'Adulting-Professional Life', 13: 'Nightlife', 14: 'Video Games'})
    
    #base probability of which any prob >= to it will be considered related topics
    df['compare']=0.05
    df['top_topic_list']=df[['Events-Local','Tv & Movies-Local','Holidays','Food','Style-Fashion','Adulting','TV & Movies-International','Politics-Duterte Presidency','Culture','Music','Technology','Way of Living','Adulting-Professional Life','Nightlife', 'Video Games']].gt(df['compare'],0).apply(lambda x: ', '.join(x.index[x]),axis=1)
    
    #Topic Detected
    st.write(f"""<b> <font size="+3">Dominant Topic: {label} </font></b>""",unsafe_allow_html=True)
    st.write(f"""<b> <font size="+1">All related topics: {df['top_topic_list'][0]} </font></b>""",unsafe_allow_html=True)
    st.write(f"""<i> <font size="-2">{topic_interpretation} </font></i>""",unsafe_allow_html=True)
    
    #Predicted number of views
    if df['word_count_content'][0]<10 or df['word_count_title'][0]<3:
    	st.write("""<font color="red">Please input proper headline and content </font>""",unsafe_allow_html=True)
    else:
    	st.write(f"""<b> <font size="+1">Predicted Target views is {int(prediction_target)}.</font></b>""",unsafe_allow_html=True)
    	st.write(f"""<b> <font size="+1">Predicted General views is {int(prediction_general)}.</font></b>""",unsafe_allow_html=True)
    	st.write('\n______________________________________________________\n')
    
    #Headline features
    st.write(f"""<b> <font size="+3">About your Headline</font></b>""",unsafe_allow_html=True)
    st.write('\n_____________________________________________________\n')
    st.write(f"Headline Word Count:{df['word_count_title'][0]}")
    st.write(f"Headline Character Count:{df['char_count_title'][0]}")
    st.write(f"Headline Sentiment:{df['predicted_sentiment_title'][0]}")
    st.write('\n______________________________________________________\n')
    

    #Content Features
    st.write(f"""<b> <font size="+3">About your Content</font></b>""",unsafe_allow_html=True)
    st.write(f"Content Word Count:{df['word_count_content'][0]}")
    st.write(f"Content Character Count:{df['char_count_content'][0]}")
    st.write("Content Sentiment:",df['predicted_sentiment_content'][0])
   
    st.write('\n______________________________________________________\n')
    
    #Readability
    st.write(f"""<b> <font size="+3">Grade Level</font></b>""",unsafe_allow_html=True)
    st.write(f"""<i> <font size="-2">What grade level is your text? How readable is it?</font></i>""",unsafe_allow_html=True)
    st.write(f"""<i> <font size="-10">Grade Levels 8-11 works best for the target age group.</font></i>""",unsafe_allow_html=True)
    st.write("Content Flesch Kincaid Grade Level:",df['fk_grade_level'][0])
    st.write('\n______________________________________________________\n')
    

    #Content Summary and keywords
    st.write(f"""<b> <font size="+3">Summarizing your Content</font></b>""",unsafe_allow_html=True)
    st.write(f"""<i> <font size="-2">What is it about?</font></i>""",unsafe_allow_html=True)
  	
    options_list = keywords(' '.join(df['content']+df['title']),words=15,split=True,deacc=True, pos_filter=('NN'))
    st.write("Keywords for tags:", options_list+ [str(label)])
    st.write("Content Summary:")
    st.write(f"{summarize(' '.join(df['content']),word_count=50)}")
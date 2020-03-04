import pandas as pd
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
from unidecode import unidecode
import matplotlib.pyplot as plt


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
	    coherence_values = []
	    model_list = []
	    for num_topics in range(start, limit, step):
	        model =  gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics = num_topics , random_state = 100,update_every = 1, chunksize = 100, passes = 10, alpha = 'auto',per_word_topics=True) 
	        model_list.append(model)
	        coherencemodel = CoherenceModel(model=model, texts=texts,dictionary=dictionary, coherence='c_v')
	        coherence_values.append(coherencemodel.get_coherence()) 
	        print(coherence_values)
	    return model_list,coherence_values

def main():
	df = pd.read_pickle('./data/training_dataset.pickle')
	dic = corpora.Dictionary.load('./models/dictionary.sav') 
	# Convert list of words to document term array
	corpus = tp.doc_term_matrix(df,dic)
	texts =df['topic']	

	model_list, coherence_values = compute_coherence_values(dictionary=dic,corpus=corpus,texts=df['topic'],start=2, limit=40,step=6)

	
	limit=40; start=2; step=6;
	x = range(start, limit, step)
	plt.plot(x, coherence_values)
	plt.xlabel('Num Topics')
	plt.ylabel('Coherence score')
	plt.legend(('coherence_values'), loc='best')
	plt.show()

if __name__ == "__main__":
    main()
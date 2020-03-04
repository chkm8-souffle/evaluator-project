import gensim
import pandas as pd
import textprep as tp
from gensim.models import CoherenceModel
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import streamlit as st
import textprep as tp
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

if __name__ == '__main__':

	df = pd.read_pickle('./data/training_dataset.pickle')
	df.to_pickle('./data/training_dataset.pickle')
	dic = tp.dictionary(df)
	corpus = tp.doc_term_matrix(df,dic)

	lda_model = gensim.models.ldamodel.LdaModel.load('./models/lda_model.sav')


	#load age data
	print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
	
	coherence_model_lda = CoherenceModel(model=lda_model, texts=df['topic'], dictionary=dic, coherence='c_v')
	coherence_lda = coherence_model_lda.get_coherence()
	
	print('\nCoherence Score: ', coherence_lda)
	
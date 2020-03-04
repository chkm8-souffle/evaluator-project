import gensim
import pandas as pd
import textprep as tp
from gensim.models import CoherenceModel

if __name__ == '__main__':

	df = pd.read_pickle('./data/training_dataset.pickle')
	df.to_pickle('./data/training_dataset.pickle')
	dic = tp.dictionary(df)
	corpus = tp.doc_term_matrix(df,dic)

	lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dic, num_topics = 15, random_state = 100, update_every = 1, chunksize = 100, passes = 10, alpha = 'auto', per_word_topics=True) 
	# Here we selected 27 topics
	lda_model.save('./models/lda_model.sav')
	#load age data
	print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
	
	coherence_model_lda = CoherenceModel(model=lda_model, texts=df['topic'], dictionary=dic, coherence='c_v')
	coherence_lda = coherence_model_lda.get_coherence()
	
	print('\nCoherence Score: ', coherence_lda)
	
	lists = lda_model.get_document_topics(corpus, minimum_probability=0.0)
	document_topic_dist = pd.DataFrame([i for i in lists]).apply(lambda x: [i[1] for i in x])
	df = pd.concat([df, document_topic_dist], axis = 1)

	df_topic_sents_keywords = tp.format_topics_sentences(lda_model,corpus,df)
	df_dominant_topic = df_topic_sents_keywords.reset_index()
	df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
	df_dominant_topic.drop(columns=['Text'],axis=1,inplace=True)
	df = pd.concat([df, df_dominant_topic], axis = 1)

	#print topics
	with open('./data/topics_generated.txt', 'w') as f:
		print(lda_model.show_topics(num_topics=15, log=False, formatted=True, num_words=100),file=f)

	df.to_pickle('./data/complete_dataset.pickle')
	df.to_csv('./data/complete_dataset.csv',encoding='utf-8',index=False)
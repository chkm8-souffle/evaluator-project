import pandas as pd
import textprep as tp
import gaprep as ga
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


#website data
website_df= pd.read_csv('./magcrawl/scrapeData/websitedata.csv',encoding ='utf-8')

stop_words = tp.stopwords.words('english')
stop_words.extend(['akin','aking','yung','lang''ako','alin','am','amin','aming','ang','ano','anumang','apat','at','atin','ating','ay','bababa','bago','bakit','bawat','bilang','dahil','dalawa','dapat','din','dito','doon','gagawin','gayunman','ginagawa','ginawa','ginawang','gumawa','gusto','habang','hanggang','hindi','huwag','iba','ibaba','ibabaw','ibig','ikaw','ilagay','ilalim','ilan','inyong','pic_twitter','com','ako','lang','mag','naman','mo','isa','isang','itaas','ito','iyo','iyon','iyong','ka','kahit','kailangan','kailanman','kami','kanila','kanilang','kanino','kanya','kanyang','kapag','kapwa','karamihan','katiyakan','katulad','kaya','kaysa','ko','kong','kulang','kumuha','kung','laban','lahat','lamang','likod','lima','maaari','maaaring','maging','mahusay','makita','marami','marapat','masyado','may','mayroon','mga','minsan','mismo','mula','muli','na','nabanggit','naging','nagkaroon','nais','nakita','namin','napaka','narito','nasaan','ng','ngayon','ni','nila','nilang','nito','niya','niyang','noon','o','pa','paano','pababa','paggawa','pagitan','pagkakaroon','pagkatapos','palabas','pamamagitan','panahon','pangalawa','para','paraan','pareho','pataas','pero','pumunta','pumupunta','sa','saan','sabi','sabihin','sarili','sila','sino','siya','tatlo','tayo','tulad','tungkol','una','walang'])
df = tp.content_cleaning(website_df,'content')
years = [2016, 2019]
df = tp.date_cleaning(df,'published_time',years)

listcol = ['title','content','tags']
result_tm_cleaning = tp.topicmodeling_cleaning(df,listcol,stop_words)
listcol_tm = result_tm_cleaning[1]
df = result_tm_cleaning[0]

df = tp.split_text_lemma(df,listcol_tm,stop_words)
bigram_mod = tp.bigram_get(df)
df = tp.bigrams_apply(df,bigram_mod)

df = tp.word_remove(df)

df = tp.article_counts(df,listcol)
df = tp.sentiment_score(df,'title')
df = tp.sentiment_score(df,'content')
df = tp.readability(df,'content')
df = tp.pos_tagging(df,listcol)


#get GA data
age1824 = pd.read_csv('./data/targetGAData.csv')
age1824=age1824.rename(columns={'Pageviews':'18-24'})
age2534 = pd.read_csv('./data/2534GAData.csv')
age2534=age2534.rename(columns={'Pageviews':'25-34'})
age3544 = pd.read_csv('./data/3544GAData.csv')
age3544=age3544.rename(columns={'Pageviews':'35-44'})
age4554 = pd.read_csv('./data/4554GAData.csv')
age4554=age4554.rename(columns={'Pageviews':'45-54'})
age5564 = pd.read_csv('./data/5564GAData.csv')
age5564=age5564.rename(columns={'Pageviews':'55-64'})
age65 = pd.read_csv('./data/65GAData.csv')
age65=age65.rename(columns={'Pageviews':'65+'})
general = pd.read_csv('./data/AllGAData.csv')

age1824 = ga.clean_ga_data(age1824)
age2534 = ga.clean_ga_data(age2534)
age3544 = ga.clean_ga_data(age3544)
age4554 = ga.clean_ga_data(age4554)
age5564 = ga.clean_ga_data(age5564)
age65 = ga.clean_ga_data(age65)
general = ga.clean_ga_data(general)


df = pd.merge(df,general,  on='url', how='inner')
df_list = [age1824,age2534,age3544,age4554,age5564,age65]
for n in df_list:
    df = pd.merge(df,n,  on='url', how='left')

df.to_pickle('./data/training_dataset.pickle',protocol=4)
df.to_csv('./data/training_dataset.csv',encoding='utf-8',index=False)
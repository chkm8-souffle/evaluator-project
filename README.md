# Article Evaluator

[Repository removes all confidential data sources. Code is shows a skeletal view of the process used to create the model.]

### Description

The tool predicts the page views of an article based its title and content.
Built with a machine learning backend, the tool predicts the page views based on:
- topic group it is associated with
- word count
- character count
- part of speech counts (# of nouns/verbs/adjectives/pronouns)
- FK Score/ FK Grade Level:[A metric on assessing difficulty of text](https://github.com/cdimascio/py-readability-metrics)
- Sentiment

### Data

Article Data was scraped using Scrapy. Scrapy spider code is included in this repository. To tailorfit to selected website, please update website and xpaths.

To crawl a website, you must install Scrapy and run the spider inside the project directory using the following command:
```console
foo@bar:~$ Scrapy crawl spider
```

 
### Model

Gensim LDA was used to model the topics of the articles.
XGBoost was used to predict the page views.

### Deployment Through Web Application

Streamlit was used to deploy the app.


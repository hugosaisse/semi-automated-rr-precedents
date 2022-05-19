from os.path import join
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from string import punctuation

def createCountry(text, countries):
    
    foundCountries = []
    
    if isinstance(text, str):
        for country in countries['Name']:
            p = re.compile(country)
            for m in p.finditer(text):
                foundCountries.append((m.start(), m.group()))
        foundCountries.sort(reverse=False)
        if len(foundCountries) > 0:
            return foundCountries[0][1]
        else:
            return 'N.A.'
    
    else:
        return 'N.A.'

def cleaner(text):
    # Remove duplicated spaces, tabs, new lines, carriage returns and so on.
    text = " ".join(text.split())
    
    # Converts all the strings to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text

def tokenizer_stemmer(text):

    stemmer = nltk.stem.RSLPStemmer()
    stoplist = set(stopwords.words('english') + list(punctuation))

    token_list = nltk.word_tokenize(text, language='english')
    token_list = [token for token in token_list if re.search(r"(?u)\b[0-9a-zÀ-ÿ-]{3,}\b", token)]
    token_list = [token for token in token_list if token not in stoplist]
    token_list = [stemmer.stem(token) for token in token_list]
    listToStr = ' '.join([str(elem) for elem in token_list])
    
    return listToStr

def cosineSimilarities(nGramRangeDict, df, column, projectRoot):

    for key, value in nGramRangeDict.items():

        print(f'Creating TF-IDF vectorizer for ngramrange={str(value)}...')
        
        vect = TfidfVectorizer(lowercase=False, max_df=0.8, ngram_range=value)
        
        tfidf = vect.fit_transform(df[column])
            
        print(f'Calculating cosine similarity for {key}grams ...')

        cossim = pd.DataFrame(data=cosine_similarity(tfidf),
                              index=df.index,
                            columns=df.index)
        
        path = join(projectRoot, 'Results', f'tfidf_{key}grams_{column}.csv')

        cossim.to_csv(path)

def jaccard_similarity(topic_1, topic_2):
    """
    Derives the Jaccard similarity of two topics

    Jaccard similarity:
    - A statistic used for comparing the similarity and diversity of sample sets
    - J(A,B) = (A ∩ B)/(A ∪ B)
    - Goal is low Jaccard scores for coverage of the diverse elements
    """
    intersection = set(topic_1).intersection(set(topic_2))
    union = set(topic_1).union(set(topic_2))
                    
    return float(len(intersection))/float(len(union))
# ================================================================================
# ================================================================================
#  FUNCTIONS FOR THE PREPROCESSING SCIPT – CALCULATING META DATA AND CLEANING TEXT
# ================================================================================
# ================================================================================

import re
from nltk.corpus import stopwords
from sentida import Sentida

# Function for finding long words (more than 6 characters)
def long_words_counter(article):
    counter = 0
    words = article.split()
    for x in words:
        if len(x) >= 7:
            counter += 1
    return (counter)

# Counts words in article
def word_counter(article):
    return len([words for words in article.split()])

# Function for counting punctuation
def punctuation_counter(article):
    count = sum([1 for char in article if char in ['.','!','?']])
    return count

def stopword_counter(article):
    stop = (stopwords.words('danish')) # load stopwords
    return len([x for x in article.split() if x in stop])

def character_counter(article):
    return len(article)

def remove_numerical_characters(corpus):
    pattern = '[0-9]'
    list = [re.sub(pattern, '', i) for i in corpus]
    return list

def remove_html_elements(corpus):
    if corpus != None:
        html = re.compile(r'<.*?>')
        html = html.sub(r' ',corpus)
        html = html.replace("&nbsp;", " ")
        html = html.replace("&amp;", "og")
        return html

def lowercasing(corpus):
    if corpus != None:
        corpus = corpus.lower()
    return corpus

def sentiment_analysis(article):
    SV = Sentida()
    return SV.sentida(text = article, 
                      output = 'mean', 
                      normal = True, 
                      speed = 'normal')

def remove_punctuation(corpus):
    if corpus != None:
        corpus = re.sub(r'[^\w\s]', '', corpus)
    return corpus

def remove_extra_spaces(input_string):
    output_string = []
    space_flag = False # Flag to check if spaces have occurred

    if input_string != None:        
        for index in range(len(input_string)):
        
            if input_string[index] != ' ':
                if space_flag == True:
                    if (input_string[index] == '.'
                            or input_string[index] == '?'
                            or input_string[index] == ','):
                        pass
                    else:
                        output_string.append(' ')
                    space_flag = False
                output_string.append(input_string[index])
            elif input_string[index - 1] != ' ':
                space_flag = True

    return ''.join(output_string)


def count_ci_words(string):
    count = 0
    with open('constructive_words/ci_wordlist.txt', 'r') as f:
        ci_words = f.read().splitlines()

        # replace _ with space in ci_words
        ci_words = [word.replace("_", " ") for word in ci_words]

        # remove all "" from all strings in ci_words
        ci_words = [word.replace('"', '') for word in ci_words]
    for word in ci_words:
        if word in string:
            count += 1
    return count
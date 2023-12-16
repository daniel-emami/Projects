from datetime import datetime
import time
import pandas as pd
import requests



# ============================================================
# ============================================================
#           FUNCTIONS FOR THE API_REQUEST SCRIPT
# ============================================================
# ============================================================


# Concatenates the article body before and after a textbox (it is split up into multiple parts when a textbox is present)
# Is used inside the api_request function
def get_article_body(content):
    article_body = ''
    for dictionaries in content:
        if dictionaries['type'] == 'Text':
            article_body += dictionaries['content']['html'] + ' '
            
    return article_body

# Gets the textbox from the article body
# Is used inside the api_request function
def get_textbox(content):
    textbox = ''
    for dictionaries in content:
        if dictionaries['type'] == 'Textbox':
            textbox += dictionaries['content']['body'] + ' '
            
    return textbox

def api_request(url, pagenumber):
    '''
    This function takes in a url and a pagenumber and returns a dataframe with the following columns:
    article_id, article_title, article_sub_header, article_url, article_date, article_author, article_body, article_textbox

    Parameters
    ----------
    url : str
        url to the API.
    pagenumber : int
        How many pages to crawl

    Returns
    -------
    articles_df : dataframe
        dataframe with the following columns:
        article_id, article_title, article_sub_header, article_url, article_date, article_author, article_body, article_textbox 

    '''
    
    articles_df = pd.DataFrame()
    for page in range(0, int(pagenumber)): # if pagenumber = 1, then it will only get the first page
        response = requests.get(url+str(page))
        if response.status_code == 200:
            response = response.json()
        else:
            print("Error from server: " + str(response.content))
        for article in range(0,100): # There are 100 articles per page
            try: # Some articles are missing data, so we need to skip them

                # ID, URL, Title, Subheader, Author
                article_id = response['data'][article]['uuid'] # Getting the article id
                article_title = response['data'][article]['title'] # Getting the article title
                article_trumpet = response['data'][article]['trumpet'] # Getting the article trumpet
                if article_trumpet == None:
                    article_trumpet = ''
                article_sub_header = response['data'][article]['summary'] # Getting the article sub header
                article_url = response['data'][article]['canonical'] # Getting the article url
                article_author = []
                for authors in range(len(response['data'][article]['authors'])):
                    article_author.append(response['data'][article]['authors'][authors]['name'])

                # Date
                article_date = response['data'][article]['date_published_at'].split('T')[0] # Removing time from date 
                article_date = datetime.strptime(article_date, '%Y-%m-%d') # Converting the date to the format YYYY-MM-DD
                article_date = article_date.strftime('%Y-%m-%d')

                # Body & Textbox. Using functions because of a problem with the API having a list of dictionaries with the article body if a Textbox is present
                article_body = get_article_body(response['data'][article]['content'])
                article_textbox = get_textbox(response['data'][article]['content'])
                
                # Appending the data to the dataframe
                articles_df = articles_df.append({'article_id': article_id, 'article_title': article_trumpet + ' ' + article_title, 'article_sub_header': article_sub_header, 'article_url': article_url, 'article_date': article_date, 'article_author': article_author, 'article_body': article_body, 'article_textbox': article_textbox}, ignore_index=True) # , 'article_body': article_body || , 'article_textbox': article_textbox
            
            except:
                pass
        time.sleep(10)
    return articles_df



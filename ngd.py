import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import math, sys, time, random, collections
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

from dotenv import load_dotenv
import os

load_dotenv()

# Now you can access your variables like this:
API_KEY = os.getenv('API_KEY')
CX = os.getenv('CX')


""" 
A python script to calculate Normalized Google Distance 

The Normalized Google Distance (NGD) is a semantic similarity measure, 
calculated based on the number of hits returned by Google for a set of 
keywords. If keywords have many pages in common relative to their respective, 
independent frequencies, then these keywords are thought to be semantically 
similar. 

If two search terms w1 and w2 never occur together on the same web 
page, but do occur separately, the NGD between them is infinite. 

If both terms always (and only) occur together, their NGD is zero.
"""

def NGD(w1, w2, lan='en'):
  """
  Returns the Normalized Google Distance between two queries.

  Params:
   w1 (str): word 1
   w2 (str): word 2
  Returns:
   NGD (float)
  """
  N_en = 51930000000.0 # Number of results for "the", proxy for total pages
  N_fa = 4710000000.0 # Number of results for "و", proxy for total pages
  
  if lan == 'fa':
    N = N_fa
  else: 
    N = N_en

  N = math.log(N,2) 
  if w1 != w2:
    f_w1 = math.log(number_of_results(w1),2)
    f_w2 = math.log(number_of_results(w2),2)
    f_w1_w2 = math.log(number_of_results(w1+" "+w2),2)
    NGD = (max(f_w1,f_w2) - f_w1_w2) / (N - min(f_w1,f_w2))
    print(" >> ",w1,f_w1,"\t",w2,f_w2,"\tboth:",f_w1_w2,"\t NGD:",NGD)
    return NGD
  else: 
    return 0
 
def calculate_NGD(w1, w2, n_retries=10,lan='en'):
  """ 
  Attempt to calculate NGD. 

  We will attempt to calculate NGD, trying `n_retries`. (Sometimes Google throws
  captcha pages. But we will just wait and try again). Iff all attempts fail, 
  then we'll return NaN for this pairwise comparison. 

  Params: 
    w1 (str): word 1 
    w2 (str): word 2
    retries (int): Number of attempts to retry before returning NaN 
  Returns:
    if succesful:
      returns NGD
    if not succesful:
      returns np.NaN
  """

  for attempt in range(n_retries):
    try:
      return NGD(w1, w2,lan)
    except Exception as e:
      #print("Trying again...")
      print(e)
  else: 
    print("Sorry. We tried and failed. Returning NaN.")
    return np.NaN

def pairwise_NGD(element_list, retries=10,lan='en'):
  """Compute pairwise NGD for a list of terms"""
  distance_matrix = collections.defaultdict(dict) # init a nested dict
  for i in element_list:
    sleep(10, 15)
    for j in element_list:
      try: # See if we already calculated NGD(j, i)
        #print("Searching for:", i, j)
        distance_matrix[i][j] = distance_matrix[j][i]
      except KeyError: # If not, calculate NGD(i, j)
        distance_matrix[i][j] = calculate_NGD(i, j, retries, lan)
  return distance_matrix

def pairwise_NGD_to_df(distances):
  """Returns a dataframe of pairwise NGD calculations"""
  df_data = {} 
  for i in distances:
    df_data[i] = [distances[i][j] for j in distances]
  df = pd.DataFrame(df_data)
  df.index = distances
  return df 


def sleep(alpha, beta):
  """Sleep for an amount of time in range(alpha, beta)"""
  rand = random.Random()
  time.sleep(rand.uniform(alpha, beta))

def number_of_results(text):
  """Returns the number of Google results for a given query."""
  headers = {'User-Agent': UserAgent().firefox}
  sleep(5, 10)
  r = requests.get("https://www.google.com/search?q={}".format(text.replace(" ","+")), headers=headers)
  soup = BeautifulSoup(r.text, "lxml") # Get text response
  res = soup.find('div', {'id': 'result-stats'}) # Find result string 
  return int(res.text.replace(",", "").split()[1]) # Return result int


def get_search_results_count(query):
    """
    Fetch the number of search results for a query using Google Custom Search API.
    
    Parameters:
    api_key (str): Your Google API key.
    cx (str): The Custom Search Engine ID.
    query (str): The search query.
    
    Returns:
    int: The number of search results.
    """
    

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': API_KEY,
        'cx': CX,
        'q': query,
        'num': 1  # We're only interested in the result count, not the actual results
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    #print(data)
    #print(query, int(data['searchInformation']['totalResults']))
    return int(data['searchInformation']['totalResults'])


def vis_heatmap(matrix):
  # Convert defaultdict to pandas DataFrame
  df = pd.DataFrame(matrix)

  # Create the heatmap using seaborn
  plt.figure(figsize=(8, 6))
  sns.heatmap(df, annot=True, cmap='Reds', linewidths=.5)

  # Add labels and title
  plt.title('Heatmap of Similarities')
  plt.xlabel('')
  plt.ylabel('')

  # Display the heatmap
  plt.show()

def get_random_words(n=2):
  word_list = []
  dic = pd.read_csv('english_dictionary_extended.csv')[['word']].drop_duplicates().reset_index(drop=True)
  for i in range(n):
    rand = int(random.Random().uniform(0,len(dic))) 
    word_list.append(dic.iloc[rand][0])
  #print(word_list)
  return word_list


if __name__ == "__main__":
  print("This is a script for calculating NGD.")

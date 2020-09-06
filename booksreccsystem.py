# This is a book recommendation system based on a few inputs 
#In this system I am recommending the books based on the titles.
#However, you can also recommend the books based on the user ratings which consists of integer values. 
#I have only displayed the most basic way 
# Also most of the print statements have been hashed out

#importing libraries 
import numpy as np
import pandas as pd


#Reading in the data 
#Although I have loaded all the files I am only making use of two of them in this one.
#You can recommend books on various other attributes, I jus so happened to choose one of them
Books = pd.read_csv('books.csv')
ratings = pd.read_csv('ratings.csv')
book_tags = pd.read_csv('book_tags.csv')
tags = pd.read_csv('tags.csv')
to_read = pd.read_csv('to_read.csv')
#print(Books.head())
#print(Books.shape)

#to check if there are any missing values use isnull().sum gives the columnwise sum of missing values 
total = Books.isnull().sum().sort_values(ascending=False) 
#print(total)

#merging all tags datasets 
merge_tags = pd.merge(book_tags , tags, how='inner' , left_on= 'tag_id' , right_on= 'tag_id')
#print(merge_tags)

#making use of NLP techniques such as Count vectorization as it converts textual data into a matrix 
# You can also use Tfid vectorizer. 
#Tfid vectorizer returns float while count vectorizer returns ints 
'''
Example of count vectorization
set = ['The', 'Cat' ,'Is' , 'Out' , 'Of' 'The' , Bag']
Output = The Cat Is Out Of Bag
          2  1    1  1   1  1
'''

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
count_vec_matrix = count_vec.fit_transform(Books['authors'])
cos_sim = cosine_similarity(count_vec_matrix, count_vec_matrix)

#print(cos_sim)
'''
#extracting valuable data from the file 
book_titles = Books[['book_id' , 'title']]
#print(book_titles)
'''
#This builds a 1D array
titles = Books['title']
indices = pd.Series(Books.index, index=Books['title'])
#print(indices.shape)

#to get the recommendations we need to define a function that will give recommendations based on the cos similarity
def author_rec(title):
    index = indices[title]

    simulation_score = list(enumerate(cos_sim[index]))
    simulation_score = sorted(simulation_score , key=lambda x: x[1] , reverse=True)
    simulation_score = simulation_score[1:21]
    book_indices = [i[0] for i in simulation_score]
    final = titles.iloc[book_indices]
    return final
print(author_rec('The Great Gatsby').head(20)) #Prints out book recommendations with similar titles 


#recommending books based on the tags added on the books 
#since i have already used book_tags, I will use another function to merge here 
tags_to_books = pd.merge(Books, merge_tags, left_on='book_id', right_on='goodreads_book_id', how='inner')
#print(tags_to_books)

count_vec2 = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
count_vec_matrix2 = count_vec2.fit_transform(tags_to_books['tag_name'].head(10000))
cos_sim2 = cosine_similarity(count_vec_matrix2, count_vec_matrix2)

titles1 = Books['title']
indices1 = pd.Series(Books.index, index=Books['title'])

# Function that get book recommendations based on the cosine similarity score of books tags
def tags_recommendations(title):
    idx = indices1[title]
    simulation_scores = list(enumerate(cos_sim2[idx]))
    simulation_scores = sorted(simulation_scores, key=lambda x: x[1], reverse=True)
    simulation_scores = simulation_scores[1:21]
    book_indices = [i[0] for i in simulation_scores]
    return titles.iloc[book_indices]
print(tags_recommendations('The Great Gatsby').head(20))    

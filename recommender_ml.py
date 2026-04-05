import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

print("Data load ho raha hai...")
books = pd.read_csv('Books.csv', low_memory=False)

print("Data clean ho raha hai...")
books = books[['ISBN', 'Book-Title', 'Book-Author', 
               'Year-Of-Publication', 'Publisher', 
               'Image-URL-M']].dropna()

books['Book-Title'] = books['Book-Title'].str.lower()
books['Book-Author'] = books['Book-Author'].str.lower()
books['Publisher'] = books['Publisher'].str.lower()

books['tags'] = (books['Book-Title'] + ' ' + 
                 books['Book-Author'] + ' ' + 
                 books['Publisher'])

books = books.drop_duplicates(subset='Book-Title')
books = books.head(5000)
books = books.reset_index(drop=True)

print("ML Model ban raha hai...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['tags'])

similarity = cosine_similarity(tfidf_matrix)

print("Model save ho raha hai...")
pickle.dump(books, open('books_data.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("\nML Model ready hai!")
print(f"Total books processed: {len(books)}")

def recommend(book_name):
    book_name = book_name.lower()
    
    if book_name not in books['Book-Title'].values:
        print(f"Book nahi mili: {book_name}")
        return
    
    idx = books[books['Book-Title'] == book_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:6]
    
    print(f"\n'{book_name}' jaisi 5 books:")
    print("-" * 40)
    for i, (index, score) in enumerate(scores, 1):
        title = books.iloc[index]['Book-Title'].title()
        author = books.iloc[index]['Book-Author'].title()
        print(f"{i}. {title} — {author}")

recommend("the da vinci code")
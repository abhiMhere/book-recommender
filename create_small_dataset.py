import pandas as pd

books = pd.read_csv('Books.csv', low_memory=False)
books = books[['ISBN', 'Book-Title', 'Book-Author', 
               'Year-Of-Publication', 'Publisher', 
               'Image-URL-M']].dropna()
books = books.drop_duplicates(subset='Book-Title')
books = books.head(1000)
books.to_csv('books_small.csv', index=False)
print(f"Small dataset ready! Total: {len(books)} books")
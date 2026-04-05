import pandas as pd

print("Books data load ho raha hai...")
books = pd.read_csv('Books.csv', low_memory=False)
print(f"Total Books: {len(books)}")
print(books[['Book-Title', 'Book-Author', 'Year-Of-Publication']].head(3))

print("\nRatings data load ho raha hai...")
ratings = pd.read_csv('Ratings.csv')
print(f"Total Ratings: {len(ratings)}")

print("\nUsers data load ho raha hai...")
users = pd.read_csv('Users.csv')
print(f"Total Users: {len(users)}")

print("\nSab data sahi hai!")

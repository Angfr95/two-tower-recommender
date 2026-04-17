import pandas as pd

def load_data():
    ratings = pd.read_csv(
        'data/ml-1m/ratings.dat',
        sep='::',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )

    movies = pd.read_csv(
        'data/ml-1m/movies.dat',
        sep='::',
        names=['movie_id', 'title', 'genres'],
        engine='python',
        encoding='latin-1'
    )

    users = pd.read_csv(
        'data/ml-1m/users.dat',
        sep='::',
        names=['user_id', 'gender', 'age', 'occupation', 'zip'],
        engine='python'
    )

    print(f"Ratings : {len(ratings)}")
    print(f"Films   : {len(movies)}")
    print(f"Users   : {len(users)}")
    print("\nExemple rating :")
    print(ratings.head(3))
    print("\nExemple film :")
    print(movies.head(3))

    return ratings, movies, users

if __name__ == '__main__':
    load_data()
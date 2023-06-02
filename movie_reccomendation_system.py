import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate

# Load the MovieLens dataset
ratingsDF = pd.read_csv('ml-data-set/ratings.csv')
moviesDF = pd.read_csv('ml-data-set/movies.csv')

# Preprocess the data
user_ids = ratingsDF['userId'].unique()
user2idx = {user_id: i for i, user_id in enumerate(user_ids)}
movie_ids = ratingsDF['movieId'].unique()
movie2idx = {movie_id: i for i, movie_id in enumerate(movie_ids)}
ratingsDF['userId'] = ratingsDF['userId'].map(user2idx)
ratingsDF['movieId'] = ratingsDF['movieId'].map(movie2idx)
num_users = len(user_ids)
num_movies = len(movie_ids)


movieIndex = ratingsDF.groupby("movieId").count().sort_values(by= \
"rating",ascending=False)[0:1000].index
ratingsDF2 = ratingsDF[ratingsDF.movieId.isin(movieIndex)]
ratingsDF2.count()

userIndex = ratingsDF2.groupby("userId").count().sort_values(by= \
"rating",ascending=False).sample(n=1000, random_state=2018).index
ratingsDF3 = ratingsDF2[ratingsDF2.userId.isin(userIndex)]
ratingsDF3.count()

movies = ratingsDF3.movieId.unique()
moviesDF = pd.DataFrame(data=movies,columns=['originalMovieId'])
moviesDF['newMovieId'] = moviesDF.index+1

users = ratingsDF3.userId.unique()
usersDF = pd.DataFrame(data=users,columns=['originalUserId'])
usersDF['newUserId'] = usersDF.index+1

ratingsDF3 = ratingsDF3.merge(moviesDF,left_on='movieId', \
right_on='originalMovieId')
ratingsDF3.drop(labels='originalMovieId', axis=1, inplace=True)

ratingsDF3 = ratingsDF3.merge(usersDF,left_on='userId', \
right_on='originalUserId')
ratingsDF3.drop(labels='originalUserId', axis=1, inplace=True)

# Split the data into training and testing sets
train_data, test_data = train_test_split(ratingsDF3, test_size=0.2)

# Define input layers
user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))

# Embedding layers
user_embedding = Embedding(num_users, 50)(user_input)
movie_embedding = Embedding(num_movies, 50)(movie_input)

# Flatten the embeddings
user_flatten = Flatten()(user_embedding)
movie_flatten = Flatten()(movie_embedding)

# Concatenate user and movie embeddings
concat = Concatenate()([user_flatten, movie_flatten])

# Fully connected layers
fc1 = Dense(128, activation='relu')(concat)
output = Dense(1, activation='linear')(fc1)

# Create the model
model = Model(inputs=[user_input, movie_input], outputs=output)

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', run_eagerly=True)

# Train the model
model.fit([np.array(train_data['userId']), np.array(train_data['movieId'])], np.array(train_data['rating']), epochs=10, batch_size=32)

# Evaluate the model
mse = model.evaluate([np.array(test_data['userId']), np.array(test_data['movieId'])], np.array(test_data['rating']))
print('Mean Squared Error:', mse)

# Generate recommendations for a user
user_id = 42  # Example user ID
user_movies = ratingsDF3[ratingsDF3['userId'] == user2idx[user_id]]
user_movies['predicted_rating'] = model.predict([np.array(user_movies['userId']), np.array(user_movies['movieId'])])
user_movies = user_movies.sort_values(by='predicted_rating', ascending=False)
recommended_movies = user_movies[['movieId', 'predicted_rating']].drop_duplicates()

# Print recommended movies
print('Recommended Movies:')
for _, row in recommended_movies.iterrows():
    movie_id = row['movieId']
    predicted_rating = row['predicted_rating']
    movie_name = moviesDF[moviesDF['movieId'] == movie_id]['title'].iloc[0]
    print(f'Movie ID: {movie_id}, Predicted Rating: {predicted_rating}, Movie Title: {movie_name}')

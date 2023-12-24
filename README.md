# MovieLens Movie Recommendation System
This project demonstrates building a movie recommendation system using the MovieLens dataset and Keras, implementing both a collaborative filtering neural network-based approach and an item-based collaborative filtering method.

## Dataset
The MovieLens dataset used in this project is available at [MovieLens Dataset](https://grouplens.org/datasets/movielens/). It includes movie ratings by users.

## Movie Recommendation using Keras
### Neural Network Model
The system employs a neural network with the following architecture:

* **Input Layer**: Takes movie and user vectors as input.
* **Embedding Layer**: Converts movie and user indices into vectors.
* **Hidden Layers**: Two fully connected layers with 128 and 32 neurons respectively.
* **Output Layer**: Produces the predicted rating for a user-movie pair.
The model is trained using the Adam optimizer and mean squared error loss for 20 epochs.

### Making Recommendations
The system predicts ratings for unwatched movies for a specific user, sorts them, and recommends the top-rated movies.

## Item-based Collaborative Filtering
This section uses item-based collaborative filtering to recommend movies based on similarity.

**Steps**
* **Data Preparation**: Pivot the data to create a movie-user matrix.
* **Nearest Neighbors Algorithm**: Use cosine similarity and k-nearest neighbors to find similar movies.
* **Recommendation Generation**: Generate movie recommendations for a given movie by finding similar items.

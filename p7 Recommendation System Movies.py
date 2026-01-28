#p7
# Developing a recommendation system using collaborative filtering or deep learning approaches.
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df1 = pd.read_csv(r'/content/movies.csv')
df2 = pd.read_csv(r'/content/ratings.csv')

# Merge movies and ratings dataframes
df = df2.merge(df1, left_on='movieId', right_on='movieId', how='left')

# Drop unnecessary columns
del df['timestamp']
del df['genres']

# Create user-movie rating matrix
user_movie_matrix = pd.pivot_table(df, values='rating', index='movieId', columns='userId')
user_movie_matrix = user_movie_matrix.fillna(0)

# User-based collaborative filtering - compute user-user Pearson correlation matrix
user_user_matrix = user_movie_matrix.corr(method='pearson')

# Extract top 10 similar users for user with userId=2
top_similar_users = user_user_matrix.loc[2].sort_values(ascending=False).head(10)

# Prepare dataframe of similar users with similarity scores
df_2 = pd.DataFrame(top_similar_users).reset_index()
df_2.columns = ['userId', 'similarity']

# Remove the user itself from similar users
df_2 = df_2[df_2['userId'] != 2]

# Merge similar users with their ratings and movie info
final_df = df_2.merge(df, on='userId', how='left')

# Compute weighted score
final_df['score'] = final_df['similarity'] * final_df['rating']

# Get movies already watched by user 2
watched_df = df[df['userId'] == 2]

# Remove movies already watched by user 2 from recommendations
final_df = final_df[~final_df['movieId'].isin(watched_df['movieId'])]

# Recommend top 10 movies based on weighted score
recommended_df = final_df.sort_values(by='score', ascending=False)['title'].head(10).reset_index(drop=True)

print("Top 10 Recommended Movies for User 2:")
print(recommended_df)

# Plot distribution of ratings
plt.figure(figsize=(8, 6))
sns.histplot(df['rating'], bins=10, kde=True)
plt.title('Distribution of User-Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


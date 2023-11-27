
import numpy as np
import pandas as pd 

from sklearn.metrics.pairwise import cosine_similarity

# Taking input from User
dataset_index = int(input("Please enter user item dataset index : "))
movie_id = int(input("Enter Movie_ID : "))
picked_userid = int(input("Enter User_ID  : "))
neighborhood_size = int(input("Enter Neighbourhood size  : "))


# Taking Data/file input
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1', parse_dates=True)

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u{}.base'.format(dataset_index), sep='\t', names=r_cols,
                      encoding='latin-1')

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')

# Merge movies, ratings and users
movie_ratings = pd.merge(movies, ratings)
df = pd.merge(movie_ratings, users)
# print(df.head(10))

# Data Preprocessing
# Dropping all the columns that are not really needed
df.drop(df.columns[[3,4,7]], axis=1, inplace=True)
ratings.drop( "unix_timestamp", inplace = True, axis = 1 )
movies.drop(movies.columns[[3,4]], inplace = True, axis = 1 )
# print(df.info())


##############################################################
# ----------------------- USER BASED CF ---------------------#
##############################################################
# rating matrix
ratings_matrix = ratings.pivot_table(index=['user_id'],columns=['movie_id'],values='rating').fillna(0)
# print(ratings_matrix.head())

# Calculating Cosine similarity
sim = cosine_similarity(ratings_matrix)
sim[np.isnan(sim)] = 0
np.fill_diagonal(sim, 0)
similar_mindset = list(enumerate(sim[picked_userid-1]))
sorted_similar = sorted(similar_mindset, key=lambda x: x[1], reverse=True)[0:neighborhood_size]
# print(sim)
# print("Sorted similarities--", sorted_similar)

# Average neighbor similarities
neighbour_avg = []
for neighbour, similarity in sorted_similar:
    np_row = ratings_matrix.iloc[neighbour].to_numpy()
    neighbour_avg.append(np_row[np.where(np_row > 0)].mean())
res = pd.DataFrame(sorted_similar)
res['mean'] = neighbour_avg
# print(res)

# Loop through users and implemented Formula(given in slides)
rating = ratings_matrix.loc[:, ratings_matrix.columns != movie_id].iloc[picked_userid - 1].to_numpy()
rating = rating[np.where(rating > 0)].mean()    # considering positive integer ratings only
# print("hi -",rating)
rating_numerator = 0
for i, data in res.iterrows():
    rating_numerator += (data[1]*(ratings_matrix.iloc[int(data[0])][movie_id]-data['mean']))    # applied Formula
# print(rating_numerator)
rating = rating + (rating_numerator/res[1].sum())
print("\n\t------ USER BASED CF ------")
print(f'The average movie rating for user {picked_userid} and {movie_id} is {rating:.2f}')


##############################################################
# ----------------------- ITEM BASED CF ---------------------#
##############################################################
# Item rating matrix
movie_features = ratings.pivot_table(index=['movie_id'],columns=['user_id'],values='rating').fillna(0)

# Cosine similarity calculation
sim1 = cosine_similarity(movie_features)
sim1[np.isnan(sim1)] = 0
np.fill_diagonal(sim1, 0)
similar_movies = list(enumerate(sim1[movie_id-1]))
sorted_similar1 = sorted(similar_movies, key=lambda x: x[1], reverse=True)[0:neighborhood_size]
# print(sim1)
# print("Sorted similar--", sorted_similar1)

# Average neighbor similarities
neighbour_avg1 = []
for neighbour, similarity in sorted_similar1:
    np_row = movie_features.iloc[neighbour].to_numpy()
    neighbour_avg1.append(np_row[np.where(np_row > 0)].mean())
res1 = pd.DataFrame(sorted_similar1)
res1['mean'] = neighbour_avg1
# print(res1)

# Loop through users and implemented Formula(given in slides for item-based similarities)
rating1 = movie_features.loc[:, movie_features.columns != picked_userid].iloc[movie_id-1].to_numpy()
rating1 = rating1[np.where(rating1 > 0)].mean()
rating_numerator1 = 0
# print("hh:",rating1)
for i, data in res1.iterrows(): # numpy iterate rows
    rating_numerator1 += (data[1]*(movie_features.iloc[int(data[0])][picked_userid]-data['mean']))
rating1 = rating1 + (rating_numerator1/res1[1].sum())
# print(rating1)

print("\n\t------ ITEM BASED CF ------")
print(f'The average movie rating for user {picked_userid} and {movie_id} is {rating1:.2f}')



"""OUTPUT

/usr/bin/python3 /Users/parag/Documents/Intro-to-Data-Science/IDS_HW5_Q3/Q3_CollaborativeFiltering.py 
Please enter user item dataset index : 1
Enter Movie_ID : 12
Enter User_ID  : 7
Enter Neighbourhood size  : 10

	------ USER BASED CF ------
The average movie rating for user 7 and 12 is 2.84

	------ ITEM BASED CF ------
The average movie rating for user 7 and 12 is 2.98

Process finished with exit code 0

"""

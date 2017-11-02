# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:28:40 2017

@author: s.jayanthi
"""
    
def import_data():
    import pandas as pd
    import pickle
    
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    rat = pd.read_csv('u.data', sep='\t', names=r_cols, encoding='latin-1')
    
    # Making all ratings=1
    rat['rating'] = [1]*rat.shape[0]
    rat.drop("unix_timestamp",inplace=True, axis=1)
    
    #Saving ratings data in a pickle file
    with open(r"ratings.pickle", "wb") as output_file:
        pickle.dump(rat, output_file)
    
    i_cols = ['movie_id', 'movie_title' ,'release_date','video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
     'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items_df = pd.read_csv('u.item', sep='|', names=i_cols, encoding='latin-1')
    with open(r"u_item.pickle", "wb") as output_file:
        pickle.dump(items_df, output_file)
        
    u_cols = ['user id', 'age', 'gender', 'occupation', 'zip code']
    users_df = pd.read_csv('u.user', sep='|', names=u_cols, encoding='latin-1')
    with open(r"u_user.pickle", "wb") as output_file:
        pickle.dump(users_df, output_file)        
    
    return
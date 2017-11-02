# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:28:12 2017

@author: s.jayanthi
"""

'''
def movies_watched(user_id):
    import pandas as pd
    import numpy as np
    import cPickle 
    
    with open(r"ratings.pickle", "rb") as input_file:
        rat = pickle.load(input_file)
    rat_mat = rat.as_matrix(columns=None)  
    with open(r"u_item.pickle", "rb") as input_file:
        items_df = pickle.load(input_file)
    
    select_user_id = user_id
    # Converting to user vs item matrix 943x1682
    mat = rat.pivot(index = 'user_id', columns ='movie_id', values = 'rating').as_matrix(columns=None).astype(np.float)
    watched_df = pd.DataFrame(columns={'movie_id', 'movie_title'})
    # Iteration across all movies
    for ind in range(0,mat[select_user_id-1,:].shape[0]):
        if(mat[select_user_id-1][ind]==1):
            watched_df = watched_df.append({'movie_id':ind+1,'movie_title':items_df.loc[ind,'movie_title']}, ignore_index=True)
    
    watched_df = watched_df.to_json(orient='records')
    return watched_df
    
'''
# In[ ]:
'''
def give_genre_freq(select_user_id, retrieve_top=20):
    
    import pickle
    import numpy as np
    import pandas as pd
    #Reading rat dataframe
    with open(r"ratings.pickle", "rb") as input_file:
        rat = pickle.load(input_file)
    
    #Initial definitions
    genre_indices_for_mat = range(6,24)
    round_off_to = 2
    
    # Obtainng required matrices 
    mat = rat.pivot(index = 'user_id', columns ='movie_id', values = 'rating').as_matrix(columns=None).astype(np.float)
    with open(r"u_item.pickle", "rb") as input_file:
        items_df = pickle.load(input_file)
    items_df_mat = items_df.as_matrix()
    # Obtaining feature matrices
    with open(r"user_feat.pickle", "rb") as input_file:
        user_feat = pickle.load(input_file)
    with open(r"item_feat.pickle", "rb") as input_file:
        item_feat = pickle.load(input_file)

    # Defining the final values to be returned in json format
    answer_df = pd.DataFrame(columns={'movie_id', 'movie_title', 'pred_rating'})
    user_genre_freq = np.array( [0]*len(genre_indices_for_mat), dtype='float64' ) # 18 genres available in the database
    pred_genre_freq = np.array( [0]*len(genre_indices_for_mat), dtype='float64' ) # 18 genres available in the database
    
    # Iteration across all movies and obtaining user_genre_freq
    for ind in range(0,mat[select_user_id-1,:].shape[0]):
        if( mat[select_user_id-1][ind]!=1):
            # Compute dot product
            pred_rat = user_feat[select_user_id-1,:].dot(item_feat[ind])
            answer_df = answer_df.append({'movie_id':ind+1,'movie_title':items_df.loc[ind,'movie_title'], 'pred_rating':np.abs(1-pred_rat)}, ignore_index=True)
        else:
            #The 'UNKNOWN' Column
            if(items_df_mat[ind][5]!=1): 
                user_genre_freq += np.asfarray( items_df_mat[ind,genre_indices_for_mat] ) # Compute genre_frequency of this user
    user_genre_freq /= user_genre_freq.sum()  
    user_genre_freq = np.round_(user_genre_freq,round_off_to)
    return_this_df = pd.DataFrame(user_genre_freq, index=['Action', 'Adventure',  'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'], columns=['user_genre_freq'])
    #user_genre_freq.sort_values('freq',ascending=False, inplace=True)

    # Obtaining only the top n predicted values and counting genre frequencies in them: pred_genre_freq
    answer_df.sort_values('pred_rating',ascending=True, inplace=True)
    save_as_json = answer_df.iloc[0:retrieve_top,:]
    for id in save_as_json['movie_id']:
        pred_genre_freq += np.asfarray(items_df_mat[id-1,genre_indices_for_mat])
    pred_genre_freq /= pred_genre_freq.sum()  
    pred_genre_freq = np.round_(pred_genre_freq,round_off_to)
    return_this_df['pred_genre_freq'] = pred_genre_freq 
    
    # Transposing the matrix
    return_this_df = return_this_df.transpose()
    #print "GENRE FREQUENCIES: ", '\n', return_this_df
    
    return_this_df = return_this_df.to_json(orient='index')
    import json, codecs
    json_file = "genre_frequencies.json" 
    json.dump(return_this_df, codecs.open(json_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    #print 'JSON file **genre_frequencies** saved success...Printing file data...', '\n', return_this_df

    return return_this_df
'''

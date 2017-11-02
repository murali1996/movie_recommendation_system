# -*- coding: utf-8 -*-
"""
@author: s.jayanthi
"""

#Import required libraries
import pandas as pd
import pickle
from import_movielens_data import import_data
from evaluate_funcs import evaluate
import sys
    
    
def give_rec(rat, user_id, retrieve_top=5):  
    import numpy as np
    # Evaluating result for the given user
    print ('\n', "User ID Selected: ", user_id, '\n')
    # Converting to user vs item matrix 943x1682
    mat = rat.pivot(index = 'user_id', columns ='movie_id', values = 'rating').as_matrix(columns=None).astype(np.float)
    #Loading data
    with open(r"u_item.pickle", "rb") as input_file:
        items_df = pickle.load(input_file, encoding='bytes')
    #items_df_mat = items_df.as_matrix()
    # Obtaining feature matrices
    #with open(r"user_feat.pickle", "rb") as input_file:
        #user_feat = pickle.load(input_file, encoding='bytes')
    with open(r"item_feat.pickle", "rb") as input_file:
        item_feat = pickle.load(input_file, encoding='bytes')
    '''
    # Saving the unrated movie pred values for this user
    # Also... Saving the user's genre selection in percentages
    genre_indices_for_mat = range(6,24)
    round_off_to = 2
    answer_df = pd.DataFrame(columns={'movie_id', 'movie_title', 'pred_rating'})
    user_genre_freq = np.array( [0]*18, dtype='float64' ) # 18 genres available in the database
    # Iteration across all movies
    for ind in range(0,mat[user_id-1,:].shape[0]):
        if( mat[user_id-1][ind]!=1):
            # Compute dot product
            pred_rat = user_feat[user_id-1,:].dot(item_feat[ind])
            answer_df = answer_df.append({'movie_id':ind+1,'movie_title':items_df.loc[ind,'movie_title'], 'pred_rating':np.abs(1-pred_rat)}, ignore_index=True)
        else:
            #The 'UNKNOWN' Column
            if(items_df_mat[ind][5]!=1): 
                user_genre_freq += np.asfarray( items_df_mat[ind,genre_indices_for_mat] ) # Compute genre_frequency of this user
    user_genre_freq /= user_genre_freq.sum()  
    user_genre_freq = np.round_(user_genre_freq,round_off_to)
    user_genre_freq = pd.DataFrame(user_genre_freq, index=['Action', 'Adventure',  'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'], columns=['freq'])
    #user_genre_freq.sort_values('freq',ascending=False, inplace=True)

    # Printing the top values in descending order of priority and saving to json file
    answer_df.sort_values('pred_rating',ascending=True, inplace=True)
    save_as_json = answer_df.iloc[0:retrieve_top,:]
    for id in save_as_json['movie_id']:
        user_genre_freq[str(id)] = np.array(items_df_mat[id-1][genre_indices_for_mat])
    print ("USER GENRE FREQUENCY: ", '\n', user_genre_freq )
    save_as_json = save_as_json.to_json(orient='records')
    
    import json, codecs
    json_file = "output.json" 
    json.dump(save_as_json, codecs.open(json_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    print ('JSON file saved success...Printing file data...', '\n', save_as_json)
    '''
    # Computing a similarity matrix between the items and use it in computing CF memory-based learning
    item_sim_mat = np.array([[0.0]*rat['movie_id'].unique().size]*rat['movie_id'].unique().size, dtype='float64')
    for row in range(0,item_feat.shape[0]):
        for col in range(row,item_feat.shape[0]):
            item_sim_mat[row][col] = item_sim_mat[col][row] = (item_feat[row,:].dot(item_feat[col,:]))/ (np.linalg.norm(item_feat[row,:])*np.linalg.norm(item_feat[col,:]))
    #Fetch top "retrieve_top" movieids and movie names for the user
    #temp = pd.DataFrame(np.array(range(1,user_feat.shape[0]+1)),columns = ['USER_ID'])
    # Converting to user vs item matrix 943x1682
    mat = rat.pivot(index = 'user_id', columns ='movie_id', values = 'rating').as_matrix(columns=None).astype(np.float)    
    to_pred_ind = np.argwhere(np.isnan(mat[user_id-1,:]))
    to_pred_ind = to_pred_ind[:,0]
    already_watched = np.argwhere(~np.isnan(mat[user_id-1,:]))
    already_watched = already_watched[:,0]
    result = pd.DataFrame(columns=['VAL','ITEM_ID','MOVIE_NAME']) #Item index has base 1
    index = 0
    for ind in to_pred_ind: #base 0
        val = 0
        for those in already_watched: #base 0
            val += mat[user_id-1,those]*(item_sim_mat[ind,those])
        result.loc[index,'VAL'] = val
        result.loc[index,'ITEM_ID'] = ind+1
        result.loc[index,'MOVIE_NAME'] = items_df.loc[items_df.index[items_df['movie_id']==ind+1],'movie_title'].values[0]
        index+=1
    result.sort_values('VAL',ascending=False,inplace=True)
    result = result.set_index(np.arange(len(result.index)))
    result = result.iloc[0:retrieve_top,:]
    
    # 18 genres available in the database
    genre_indices_for_mat = range(6,24)
    #round_off_to = 2
    user_genre_freq = np.zeros(shape=[1,len(genre_indices_for_mat)], dtype='float32')
    pred_genre_freq = np.zeros(shape=[1,len(genre_indices_for_mat)], dtype='float32')
    for those in already_watched: #base 0
        user_genre_freq += np.array(items_df.iloc[items_df.index[items_df['movie_id']==those+1], genre_indices_for_mat])
    #user_genre_freq /= already_watched.shape[0]
    user_genre_freq /= np.amax(user_genre_freq)
    for ind in range(0,retrieve_top):
        item_id = result.loc[ind,'ITEM_ID'] #base 1
        pred_genre_freq += np.array(items_df.iloc[items_df.index[items_df['movie_id']==item_id], genre_indices_for_mat])
    #pred_genre_freq /= retrieve_top
    pred_genre_freq /= np.amax(pred_genre_freq)
    
    # Bar chart demo with pairs of bars grouped for easy comparison.
    # http://matplotlib.org/examples/pylab_examples/barchart_demo.html
    import numpy as np
    import matplotlib.pyplot as plt
    n_groups = len(genre_indices_for_mat)
    this1 = tuple(user_genre_freq.reshape(1, -1)[0])
    this2 = tuple(pred_genre_freq.reshape(1, -1)[0])

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4
    plt.bar(index, this1, bar_width, alpha=opacity, color='b', label='Watched')
    plt.bar(index + bar_width, this2, bar_width, alpha=opacity, color='r', label='Predicted')
    plt.xlabel('Genres')
    plt.ylabel('Avg. responses')
    plt.title('Movie Predictions')
    #plt.xticks(index + bar_width / 2, ('Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'))
    plt.legend()
    plt.tight_layout()
    plt.show()    

    return result #save_as_json


def main_func(user_id=10):   #Take 10 as default if no input is given    
    #Import data in pickle files
    import_data()
    
    #Give stats
    #movies_watched(user_id)
    #give_genre_freq(user_id,20)    
    
    #Import ratings info
    with open(r"ratings.pickle", "rb") as input_file:
        rat = pickle.load(input_file, encoding='bytes')
    #rat_mat = rat.as_matrix(columns=None)   

    #Setting hyper-parameters
    
    #Length of feature vectors and feature matrix
    n_features=15
    max_users = rat['user_id'].unique().size
    max_items = rat['movie_id'].unique().size
    
    #initialization of feature vectors: from a normal gaussian
    gaus_mean=0
    gaus_var=0.75
    
    #Learning rate
    alpha = 200
    lbd = 0.02
    
    #Set error threshold. Gradient descent stops once diff between
    #two consecutive error goes below this value        
    error_threshold = 0.000005
    
    #Fetch how many movies to show at the end
    retrieve_top=5
    
    #Number of iterations
    num_iter = 0
    
    # Should continue from prevoious saved model? And later save this model?
    continue_from_prev_model = 1
    save_model = 0
    

    # Training the data
    evaluate(rat, max_users, max_items, n_features, gaus_mean, gaus_var, num_iter, alpha, lbd, error_threshold, continue_from_prev_model, save_model)
    
    # kfold_evaluate(rat, max_users, max_items, n_features, gaus_mean, gaus_var, num_iter, alpha, continue_from_prev_model, save_model)
    # num_iter-> NUmber of iterations per fold
    # kfold_evaluate(rat, max_users, max_items, n_features, gaus_mean, gaus_var, 100, alpha, 0, 0)
 
    return give_rec(rat,user_id,retrieve_top)

if __name__=="__main__":
    if len(sys.argv)<2:
        print(len(sys.argv))
        #raise Exception('Not sufficient number of inputs')
    user_id = sys.argv[1]
    print( main_func(user_id) )




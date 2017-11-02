# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:24:30 2017

@author: s.jayanthi
"""
def init_func(rat, max_users, max_items, n_features, gaus_mean, gaus_var, continue_from_prev_model):
    
    import numpy as np
    import pickle
    
    #Initialize feature vector values
    if(continue_from_prev_model==0): 
        user_feat = np.random.normal(loc=gaus_mean, scale=gaus_var, size=(max_users,n_features))
        item_feat = np.random.normal(loc=gaus_mean, scale=gaus_var, size=(max_items,n_features)) 
    elif(continue_from_prev_model==1):
        with open(r"user_feat.pickle", "rb") as input_file:
            user_feat = pickle.load(input_file, encoding='bytes')
        with open(r"item_feat.pickle", "rb") as input_file:
            item_feat = pickle.load(input_file, encoding='bytes')
            
    #Initialize error values
    prev_error = float(-1)
    curr_error = float(-1)
    return user_feat, item_feat, prev_error, curr_error

def compute_cost(rat, user_feat, item_feat):  # Comput cost for linear regression  
    rat = rat.as_matrix()
    J_cost = 0
    error_norm_factor = (1.0 / (2 * rat.shape[0]))
    for row in range(0,rat.shape[0]):
        user_id = rat[row][0]-1 #Making 0 base so as to acess the id feat. in user_feat matrix
        item_id = rat[row][1]-1
        pred = user_feat[user_id].dot(item_feat[item_id])
        actl = rat[row][2]
        errorSq = (pred-actl)**2
        J_cost += errorSq
    print("Squared Error between actual values and predicted values: ", J_cost)
    J_cost = error_norm_factor * J_cost
    return J_cost


def gradient_descent(rat, user_feat, item_feat, alpha, lbd): # Alpha is the learning rate
    new_user_feat = user_feat
    new_item_feat = item_feat
     
    for user_id in rat['user_id'].unique():
        temp = rat.loc[rat['user_id']==user_id,:]
        temp = temp.as_matrix()
        error_for_user = 0
        error_norm_factor = (1.0 / (2 * rat.shape[0]))
        for row_index in range(0,temp.shape[0]):
            user_id_temp = temp[row_index][0]-1 #Making 0 base so as to acess the id feat. in user_feat matrix
            item_id_temp = temp[row_index][1]-1
            pred = user_feat[user_id_temp].dot(item_feat[item_id_temp])
            actl = temp[row_index][2]
            error_for_user += error_norm_factor*(pred-actl)*item_feat[item_id_temp]
        error_for_user += error_norm_factor*lbd*user_feat[user_id-1]
        #print("!! ",error_for_user, " \n ",lbd*user_feat[user_id-1])
        new_user_feat[user_id-1] -= alpha*error_for_user 
        
    for item_id in rat['movie_id'].unique():
        temp = rat.loc[rat['movie_id']==item_id,:]
        temp = temp.as_matrix()
        error_for_item = 0
        error_norm_factor = (1.0 / (2 * rat.shape[0]))
        for row_index in range(0,temp.shape[0]):
            user_id_temp = temp[row_index][0]-1 #Making 0 base so as to acess the id feat. in user_feat matrix
            item_id_temp = temp[row_index][1]-1
            pred = user_feat[user_id_temp].dot(item_feat[item_id_temp])
            actl = temp[row_index][2]
            error_for_item += error_norm_factor*(pred-actl)*user_feat[user_id_temp]
        error_for_item += error_norm_factor*lbd*item_feat[item_id-1]
        #print("!! ",error_for_item, " \n ",lbd*item_feat[item_id-1])
        new_item_feat[item_id-1] -= alpha*error_for_item
        
    user_feat = new_user_feat
    item_feat = new_item_feat
    #print(user_feat,'\n',item_feat)
    return user_feat, item_feat
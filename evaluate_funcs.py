# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:26:15 2017

@author: s.jayanthi
"""
def evaluate(rat, max_users, max_items, n_features, gaus_mean, gaus_var, num_iter, alpha, lbd, error_threshold, continue_from_prev_model, save_model):
    
    from my_funcs import init_func, compute_cost, gradient_descent
    import pickle

    # Initialization      
    user_feat, item_feat, prev_error, curr_error = init_func(rat, max_users, max_items, n_features, gaus_mean, gaus_var, continue_from_prev_model)
    print ("User and Item feature matrices initialized successfully")
    
    #Iterations for finding stable point
    for itr in range(0,num_iter):
        error = compute_cost(rat, user_feat, item_feat)
        print("Iteration: ", itr+1, " and the normalized loss ", error)
        
        # Check if saturation occured
        if(float(prev_error)==-1 and float(curr_error)==-1):
            prev_error = curr_error = error
            user_feat, item_feat = gradient_descent(rat, user_feat, item_feat, alpha, lbd)
        else:
            curr_error = error
            err_diff = abs(curr_error-prev_error)
            if(err_diff<=error_threshold):
                break
            else:
                prev_error = curr_error
                user_feat, item_feat = gradient_descent(rat, user_feat, item_feat, alpha, lbd)
    
    if(save_model==1):
        with open(r"user_feat.pickle", "wb") as output_file:
            pickle.dump(user_feat, output_file)
        with open(r"item_feat.pickle", "wb") as output_file:
            pickle.dump(item_feat, output_file)   
        print ("Pickle Save Successful :)"  )      
    else:
        print ("Training Complete and Result Unsaved :(")
    return n_features, user_feat, item_feat


'''
def kfold_evaluate(rat, max_users, max_items, n_features, gaus_mean, gaus_var, num_iter, alpha, continue_from_prev_model, save_model):
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    import pandas as pd
    import math
    
    #num_iter-> NUmber of iterations per fold
    #From Input
    rat_mat = rat.as_matrix(columns=None) 
    # Converting to user vs item matrix 943x1682
    mat = rat.pivot(index = 'user_id', columns ='movie_id', values = 'rating').as_matrix(columns=None).astype(np.float)

    #Some initializations
    n_splits = 5
    print ("Evaluating " , n_splits, "-fold cross validation...")

    # train-test kfold split
    skf = StratifiedKFold(n_splits=n_splits)
    order_by = rat_mat[:,0] # Stratified based on this argument
    skf.get_n_splits(rat_mat, order_by) 

    #Iterations
    fold_number = 0
    for train_index, test_index in skf.split(rat_mat, order_by):
        #Print
        fold_number+=1
        print ("Fold Number: ", fold_number)

        # print "TRAIN:", train_index, "TEST:", test_index
        X_train, X_test = rat_mat[train_index,:], rat_mat[test_index,:]  

        # X_train min. #of ratings by each user
        mn = np.unique((X_train[X_train[:,0]==1,:])[:,1]).size
        for i in np.unique(X_train[:,0]):
            if( np.unique((X_train[X_train[:,0]==i,:])[:,1]).size <mn):
                mn=np.unique((X_train[X_train[:,0]==i,:])[:,1]).size
        print ("X_train USERS UNIQUE", np.unique(X_train[:,0]).size,"X_train ITEMS UNIQUE", np.unique(X_train[:,1]).size)
        #print "X_train min. #of ratings by each user", mn 

        # X_test min. #of ratings by each user
        mn = np.unique((X_test[X_test[:,0]==1,:])[:,1]).size
        for i in np.unique(X_test[:,0]):
            if( np.unique((X_test[X_test[:,0]==i,:])[:,1]).size <mn):
                mn=np.unique((X_test[X_test[:,0]==i,:])[:,1]).size
        print ( "X_test USERS UNIQUE", np.unique(X_test[:,0]).size,"X_test ITEMS UNIQUE", np.unique(X_test[:,1]).size)
        #print "X_test min. #of ratings by each user", mn
        
        df_for_train = pd.DataFrame(X_train, columns=['user_id', 'movie_id', 'rating'])
        df_for_test = pd.DataFrame(X_test, columns=['user_id', 'movie_id', 'rating'])

        #Regression fitting
        #(rat, max_users, max_items, n_features, gaus_mean, gaus_var, num_iter, alpha, continue_from_prev_model, save_model)
        n_features, user_feat, item_feat = evaluate(df_for_train, max_users, max_items, n_features, gaus_mean, gaus_var, num_iter, alpha, continue_from_prev_model, save_model)

        #Predicting the recommendations for each user
        mean_avg_precision = 0.0
        for user_id in range(0,rat['user_id'].unique().size):
            pred_array = np.empty((1,3))
            compare2 = (df_for_test.loc[df_for_test['user_id']==(user_id+1),'movie_id']).as_matrix()
            
            #Obtaing the prediction list including unknown+data_in_test movies
            for item_id in range(0,np.size(mat[user_id,:])):
                if(mat[user_id][item_id]!=1 | df_for_test.loc[ ((df_for_test['user_id']==(user_id+1)) & (df_for_test['movie_id']==(item_id+1))),:].shape[0]>0):
                     pred_array = np.append( pred_array, [[user_id+1, item_id+1, user_feat[user_id].dot(item_feat[item_id])]], axis=0)
            pred_array = np.delete(pred_array, (0), axis=0) 
            
            #Sort by ratings
            col_index_for_sort = 2 
            pred_array=pred_array[np.argsort(-pred_array[:,col_index_for_sort])]
            #pred_array = pred_array[pred_array[:,-2].argsort()] 
                
            #Hold the column of movie IDs and compare with compare 1
            compare1 = pred_array[range(0,np.size(compare2)),1].astype(dtype='int') #Take top df_for_test.shape[0] predictions' movie ids
            mean_avg_precision += float( (len( set(compare1) & set(compare2)))/np.size(compare2) )
            print ("User ID: ",user_id , "Precision: ", float( (len( set(compare1) & set(compare2)))/np.size(compare2) ))
            
            if(user_id==1):
                print (pred_array,[[compare1, compare2]])
        
        #Find average precision
        mean_avg_precision/=rat['user_id'].unique().size
        print ("MEAN AVG PRECISION: ", mean_avg_precision)
        
        #Error computation
        testError_rmse = 0.0
        testError_mae = 0.0
        #metric_undef = 0.0
        for row in range(0,df_for_test.shape[0]):
            user_id = df_for_test.loc[row,'user_id']-1
            item_id = df_for_test.loc[row,'movie_id']-1
            pred = user_feat[user_id].dot(item_feat[item_id])
            actu = df_for_test.iloc[row,2]
            er = (actu-pred)
            testError_rmse += er **2
            testError_mae += abs(er)
            #if (abs(er)<0.2):
                #metric_undef+=1
        #metric_undef = float(metric_undef/df_for_test.shape[0])
        testError_rmse /= df_for_test.shape[0]
        math.sqrt(testError_rmse)
        testError_mae /= df_for_test.shape[0]
        print ("Fold Number: ", fold_number)
        print ("TEST ERRORS: ", "RMSE: ", testError_rmse, "MAE: ", testError_mae) #, "metric_undef: ", metric_undef
        
        #[EDIT: These var are not available]The last folds user_feat and item_feat will be saved as final save_user_feat and save_item_feat resp.
        print ("********************************************", '\n')
        
        # Take mean of feat vectros in all folds and return that
        #return n_features, user_feat, item_feat
    return
'''
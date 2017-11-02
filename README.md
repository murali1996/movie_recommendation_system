# reco_system1
A recommendation engine built using movies-lens 100k dataset.  Given a user id, the system obtains movie features for all the movies in the dataset (from the already-trained-and-saved latent features obtained from user-item watchlist and optimizing a loss function by back propogation)  and finds a list of best movies in the interest of the user based on his watch-history using a cosine similarity. 

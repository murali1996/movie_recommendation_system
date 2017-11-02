**Recommendation Engine**
--------------------------
A recommendation engine built using movies-lens 100k dataset.
Given a user id, the system obtains movie features for all the movies in the dataset (from the already-trained-and-saved latent features obtained from user-item watchlist and optimizing a loss function by back propogation) and finds a list of best movies in the interest of the user based on his watch-history using a cosine similarity. 

Unlike other popular movie recommendations obtained from the ratings given by a user, this system only requires data of whether a given movie is watched by the user or not. 

**Requirements**
----------------
- python 3.5 or above


**To run the code**
----------
- git clone https://github.com/murali1996/reco_system1.git
- cd reco_system1
- To edit the user-id in main_regression file
- python main_regression <userID>

- Ex: python main_regression 902
- The UserIDs range [1,1680]

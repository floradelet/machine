# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
import math
from contextlib import contextmanager


import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import BernoulliRBM
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))


def load_from_csv(path, delimiter=',', manual=False):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """

    res=pd.read_csv(path, delimiter=delimiter).values.squeeze()
    if manual:
        print('start')
        length=res.shape[1]-1
        wrong=res[pd.isnull(res[:,length])]
        print(wrong)
    return res


def build_rating_matrix(user_movie_rating_triplets):
    """
    Create the rating matrix from triplets of user and movie and ratings.

    A rating matrix `R` is such that `R[u, m]` is the rating given by user `u`
    for movie `m`. If no such rating exists, `R[u, m] = 0`.

    Parameters
    ----------
    user_movie_rating_triplets: array [n_triplets, 3]
        an array of trpilets: the user id, the movie id, and the corresponding
        rating.
        if `u, m, r = user_movie_rating_triplets[i]` then `R[u, m] = r`

    Return
    ------
    R: sparse csr matrix [n_users, n_movies]
        The rating matrix
    """
    
    """
    rows = np.array(user_movie_rating_triplets[:, 0:3])
    cols = np.array(user_movie_rating_triplets[:, 3:22])
    training_ratings = np.array(user_movie_rating_triplets[:, 22])
    """    
    rows = user_movie_rating_triplets[:, 0]
    cols = user_movie_rating_triplets[:, 1]
    training_ratings = user_movie_rating_triplets[:, 2]
    return sparse.coo_matrix((training_ratings, (rows, cols))).tocsr()


def create_learning_matrices(rating_matrix, user_movie_pairs):
    """
    Create the learning matrix `X` from the `rating_matrix`.

    If `u, m = user_movie_pairs[i]`, then X[i] is the feature vector
    corresponding to user `u` and movie `m`. The feature vector is composed
    of `n_users + n_movies` features. The `n_users` first features is the
    `u-th` row of the `rating_matrix`. The `n_movies` last features is the
    `m-th` columns of the `rating_matrix`

    In other words, the feature vector for a pair (user, movie) is the
    concatenation of the rating the given user made for all the movies and
    the rating the given movie receive from all the user.

    Parameters
    ----------
    rating_matrix: sparse matrix [n_users, n_movies]
        The rating matrix. i.e. `rating_matrix[u, m]` is the rating given
        by the user `u` for the movie `m`. If the user did not give a rating for
        that movie, `rating_matrix[u, m] = 0`
    user_movie_pairs: array [n_predictions, 2]
        If `u, m = user_movie_pairs[i]`, the i-th raw of the learning matrix
        must relate to user `u` and movie `m`

    Return
    ------
    X: sparse array [n_predictions, n_users + n_movies]
        The learning matrix in csr sparse format
    """
    # Feature for users
    rating_matrix = rating_matrix.tocsr()
    user_features = rating_matrix[user_movie_pairs[:, 0]]

    # Features for movies
    rating_matrix = rating_matrix.tocsc()
    movie_features = rating_matrix[:, user_movie_pairs[:, 1]].transpose()

    X = sparse.hstack((user_features, movie_features))
    return X.tocsr()


def make_submission(y_predict, user_movie_ids, file_name='submission',
                    date=True):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    y_predict: array [n_predictions]
        The predictions to write in the file. `y_predict[i]` refer to the
        user `user_ids[i]` and movie `movie_ids[i]`
    user_movie_ids: array [n_predictions, 2]
        if `u, m = user_movie_ids[i]` then `y_predict[i]` is the prediction
        for user `u` and movie `m`
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """

    # Naming the file
    if date:
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    # Writing into the file
    with open(file_name, 'w') as handle:
        handle.write('"USER_ID_MOVIE_ID","PREDICTED_RATING"\n')
        for (user_id, movie_id), prediction in zip(user_movie_ids,
                                                 y_predict):

            if np.isnan(prediction):
                raise ValueError('The prediction cannot be NaN')
            line = '{:d}_{:d},{}\n'.format(user_id, movie_id, prediction)
            handle.write(line)
    return file_name

# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

def convert_movie_gender(gender_array):
    gender = []
    for i in range(5,len(gender_array)):
        if gender_array[i] == 1:
            gender += [i]
    return sum(gender)

def convert_date(date):
    if pd.isnull(date):
        return 0
    return int(date[7:11])

def convert_user_gender(user_gender):
    if user_gender == 'M':
        return 0
    elif user_gender == 'F':
        return 1
    return 2
    
def convert_occupation(occupation):
    if occupation == 'artist':
        return 0
    elif occupation == 'administrator':
        return 1
    elif occupation == 'educator':
        return 2
    elif occupation == 'student':
        return 3
    elif occupation == 'librarian':
        return 4
    elif occupation == 'scientist':
        return 5
    elif occupation == 'doctor':
        return 6
    elif occupation == 'none':
        return 7
    elif occupation == 'technician':
        return 8
    elif occupation == 'other':
        return 9
    elif occupation == 'executive':
        return 10
    elif occupation == 'engineer':
        return 11
    elif occupation == 'programmer':
        return 12
    elif occupation == 'writer':
        return 13
    elif occupation == 'marketing':
        return 14
    elif occupation == 'healthcare':
        return 15
    elif occupation == 'retired':
        return 16
    elif occupation == 'lawyer':
        return 17
    elif occupation == 'salesman':
        return 18
    elif occupation == 'entertainment':
        return 19
    elif occupation == 'homemaker':
        return 20
    return 21 
    
if __name__ == '__main__':
    prefix = 'data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_train.csv'))
    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))
    training_user_param = load_from_csv(os.path.join(prefix, 'data_user.csv'))
    training_movie_param = load_from_csv(os.path.join(prefix, 'data_movie.csv'))
    
    print(training_movie_param[1,:])
    print(training_movie_param[5,:])
    
    # Parameters    
    user_age_movie_title = np.zeros((len(training_user_movie_pairs), 2))
    user_age_movie_date = np.zeros((len(training_user_movie_pairs), 2))
    user_age_movie_gender = np.zeros((len(training_user_movie_pairs), 2))
    user_gender_movie_title = np.zeros((len(training_user_movie_pairs), 2))
    user_gender_movie_date = np.zeros((len(training_user_movie_pairs), 2))
    user_gender_movie_gender = np.zeros((len(training_user_movie_pairs), 2))
    user_occupation_movie_title = np.zeros((len(training_user_movie_pairs), 2))
    user_occupation_movie_date = np.zeros((len(training_user_movie_pairs), 2))
    user_occupation_movie_gender = np.zeros((len(training_user_movie_pairs), 2))

    for i in range(len(training_user_movie_pairs)):
        
        gender = convert_movie_gender(training_movie_param[training_user_movie_pairs[i][1]-1])
    
        user_age_movie_title[i][0] = training_user_param[training_user_movie_pairs[i][0]-1][1]
        user_age_movie_title[i][1] = training_movie_param[training_user_movie_pairs[i][1]-1][0]
                                  
        user_age_movie_date[i][0] = training_user_param[training_user_movie_pairs[i][0]-1][1]
        user_age_movie_date[i][1] = convert_date(training_movie_param[training_user_movie_pairs[i][1]-1][2])
        
        user_age_movie_gender[i][0] = training_user_param[training_user_movie_pairs[i][0]-1][1]
        user_age_movie_gender[i][1] = gender
        
        user_gender_movie_title[i][0] = convert_user_gender(training_user_param[training_user_movie_pairs[i][0]-1][2])
        user_gender_movie_title[i][1] = training_movie_param[training_user_movie_pairs[i][1]-1][0]
        
        user_gender_movie_date[i][0] = convert_user_gender(training_user_param[training_user_movie_pairs[i][0]-1][2])
        user_gender_movie_date[i][1] = convert_date(training_movie_param[training_user_movie_pairs[i][1]-1][2])
        
        user_gender_movie_gender[i][0] = convert_user_gender(training_user_param[training_user_movie_pairs[i][0]-1][2])
        user_gender_movie_gender[i][1] = gender
        
        user_occupation_movie_title[i][0] = convert_occupation(training_user_param[training_user_movie_pairs[i][0]-1][3])
        user_occupation_movie_title[i][1] = training_movie_param[training_user_movie_pairs[i][1]-1][0]
        
        user_occupation_movie_date[i][0] = convert_occupation(training_user_param[training_user_movie_pairs[i][0]-1][3])
        user_occupation_movie_date[i][1] = convert_date(training_movie_param[training_user_movie_pairs[i][1]-1][2])
        
        user_occupation_movie_gender[i][0] = convert_occupation(training_user_param[training_user_movie_pairs[i][0]-1][3])
        user_occupation_movie_gender[i][1] = gender
    
    user_movie_rating_triplets = np.hstack((training_user_movie_pairs,
                                      training_labels.reshape((-1, 1))))
    training_user_age_movie_title = np.hstack((user_age_movie_title,
                                      training_labels.reshape((-1, 1))))
    training_user_age_movie_date = np.hstack((user_age_movie_date,
                                      training_labels.reshape((-1, 1))))
    training_user_age_movie_gender = np.hstack((user_age_movie_gender,
                                      training_labels.reshape((-1, 1))))
    training_user_gender_movie_title = np.hstack((user_gender_movie_title,
                                      training_labels.reshape((-1, 1))))
    training_user_gender_movie_date = np.hstack((user_gender_movie_date,
                                      training_labels.reshape((-1, 1))))
    training_user_gender_movie_gender = np.hstack((user_gender_movie_gender,
                                      training_labels.reshape((-1, 1))))
    training_user_occupation_movie_title = np.hstack((user_occupation_movie_title,
                                      training_labels.reshape((-1, 1))))
    training_user_occupation_movie_date = np.hstack((user_occupation_movie_date,
                                      training_labels.reshape((-1, 1))))
    training_user_occupation_movie_gender = np.hstack((user_occupation_movie_gender,
                                      training_labels.reshape((-1, 1))))
    # Build the learning matrix
    rating_matrix = build_rating_matrix(user_movie_rating_triplets)
    rating_matrix_age_title = build_rating_matrix(training_user_age_movie_title)
    rating_matrix_age_date = build_rating_matrix(training_user_age_movie_date)
    rating_matrix_age_gender = build_rating_matrix(training_user_age_movie_gender)
    rating_matrix_gender_title = build_rating_matrix(training_user_gender_movie_title)
    rating_matrix_gender_date = build_rating_matrix(training_user_gender_movie_date)
    rating_matrix_gender_gender = build_rating_matrix(training_user_gender_movie_gender)
    rating_matrix_occupation_title = build_rating_matrix(training_user_occupation_movie_title)
    rating_matrix_occupation_date = build_rating_matrix(training_user_occupation_movie_date)
    rating_matrix_occupation_gender = build_rating_matrix(training_user_occupation_movie_gender)
    
    X_ls = create_learning_matrices(rating_matrix, training_user_movie_pairs)
    X_ls_age_title = create_learning_matrices(rating_matrix_age_title, training_user_age_movie_title)
    X_ls_age_date = create_learning_matrices(rating_matrix_age_date, training_user_age_movie_date)
    X_ls_age_gender = create_learning_matrices(rating_matrix_age_gender, training_user_age_movie_gender)
    X_ls_gender_title = create_learning_matrices(rating_matrix_gender_title, training_user_gender_movie_title)
    X_ls_gender_date = create_learning_matrices(rating_matrix_gender_date, training_user_gender_movie_date)
    X_ls_gender_gender = create_learning_matrices(rating_matrix_gender_gender, training_user_gender_movie_gender)
    X_ls_occupation_title = create_learning_matrices(rating_matrix_occupation_title, training_user_occupation_movie_title)
    X_ls_occupation_date = create_learning_matrices(rating_matrix_occupation_date, training_user_occupation_movie_date)
    X_ls_occupation_gender = create_learning_matrices(rating_matrix_occupation_gender, training_user_occupation_movie_gender)

    # Build the model
    y_ls = training_labels
    start = time.time()
    
    layer = 3
    iterNN = 17000
    lri = 0.00001
    alphaNN = 0.01
    model = MLPRegressor(hidden_layer_sizes=(layer,),
                         activation='logistic',
                         solver='adam',
                         learning_rate='adaptive',
                         max_iter=iterNN,
                         learning_rate_init=lri,
                         alpha=alphaNN)
    model_age_title = model
    model_age_date = model
    model_age_gender = model
    model_gender_title = model
    model_gender_date = model
    model_gender_gender = model
    model_occupation_title = model
    model_occupation_date = model
    model_occupation_gender = model
    
    with measure_time('Training'):
        """
        print('Training...')
        model_age_title.fit(X_ls_age_title, y_ls)
        print("model")
        model_age_date.fit(X_ls_age_date, y_ls)
        print("model")
        model_age_gender.fit(X_ls_age_gender, y_ls)
        print("model")
        model_gender_title.fit(X_ls_gender_title, y_ls)
        print("model")
        model_gender_date.fit(X_ls_gender_date, y_ls)
        print("model")
        model_gender_gender.fit(X_ls_gender_gender, y_ls)
        print("model")
        model_occupation_title.fit(X_ls_occupation_title, y_ls)
        print("model")
        model_occupation_date.fit(X_ls_occupation_date, y_ls)
        print("model")
        model_occupation_gender.fit(X_ls_occupation_gender, y_ls)
        print("model")
        """
        model.fit(X_ls, y_ls)
        
    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

    test_user_age_movie_title = np.zeros((len(test_user_movie_pairs), 2))
    test_user_age_movie_date = np.zeros((len(test_user_movie_pairs), 2))
    test_user_age_movie_gender = np.zeros((len(test_user_movie_pairs), 2))
    test_user_gender_movie_title = np.zeros((len(test_user_movie_pairs), 2))
    test_user_gender_movie_date = np.zeros((len(test_user_movie_pairs), 2))
    test_user_gender_movie_gender = np.zeros((len(test_user_movie_pairs), 2))
    test_user_occupation_movie_title = np.zeros((len(test_user_movie_pairs), 2))
    test_user_occupation_movie_date = np.zeros((len(test_user_movie_pairs), 2))
    test_user_occupation_movie_gender = np.zeros((len(test_user_movie_pairs), 2))
    
    for i in range(len(test_user_movie_pairs)):   
  
        gender = convert_movie_gender(training_movie_param[test_user_movie_pairs[i][1]-1])
        
        test_user_age_movie_title[i][0] = training_user_param[test_user_movie_pairs[i][0]-1][1]
        test_user_age_movie_title[i][1] = training_movie_param[test_user_movie_pairs[i][1]-1][0]
        
        test_user_age_movie_date[i][0] = training_user_param[test_user_movie_pairs[i][0]-1][1]
        test_user_age_movie_date[i][1] = convert_date(training_movie_param[test_user_movie_pairs[i][1]-1][2])
        
        test_user_age_movie_gender[i][0] = training_user_param[test_user_movie_pairs[i][0]-1][1]
        test_user_age_movie_gender[i][1] = gender
        
        test_user_gender_movie_title[i][0] = convert_user_gender(training_user_param[training_user_movie_pairs[i][0]-1][2])
        test_user_gender_movie_title[i][1] = training_movie_param[test_user_movie_pairs[i][1]-1][0]
        
        test_user_gender_movie_date[i][0] = convert_user_gender(training_user_param[training_user_movie_pairs[i][0]-1][2])
        test_user_gender_movie_date[i][1] = convert_date(training_movie_param[test_user_movie_pairs[i][1]-1][2])
        
        test_user_gender_movie_gender[i][0] = convert_user_gender(training_user_param[training_user_movie_pairs[i][0]-1][2])
        test_user_gender_movie_gender[i][1] = gender
        
        test_user_occupation_movie_title[i][0] = convert_occupation(training_user_param[training_user_movie_pairs[i][0]-1][3])
        test_user_occupation_movie_title[i][1] = training_movie_param[test_user_movie_pairs[i][1]-1][0]
        
        test_user_occupation_movie_date[i][0] = convert_occupation(training_user_param[training_user_movie_pairs[i][0]-1][3])
        test_user_occupation_movie_date[i][1] = convert_date(training_movie_param[test_user_movie_pairs[i][1]-1][2])
        
        test_user_occupation_movie_gender[i][0] = convert_occupation(training_user_param[training_user_movie_pairs[i][0]-1][3])
        test_user_occupation_movie_gender[i][1] = gender
        
    # Build the prediction matrix
    X_ts = create_learning_matrices(rating_matrix, test_user_movie_pairs)
    X_ts_age_title = create_learning_matrices(rating_matrix_age_title, test_user_age_movie_title)
    X_ts_age_date = create_learning_matrices(rating_matrix_age_date, test_user_age_movie_date)
    X_ts_age_gender = create_learning_matrices(rating_matrix_age_gender, test_user_age_movie_gender)
    X_ts_gender_title = create_learning_matrices(rating_matrix_gender_title, test_user_gender_movie_title)
    X_ts_gender_date = create_learning_matrices(rating_matrix_gender_date, test_user_gender_movie_date)
    X_ts_gender_gender = create_learning_matrices(rating_matrix_gender_gender, test_user_gender_movie_gender)
    X_ts_occupation_title = create_learning_matrices(rating_matrix_occupation_title, test_user_occupation_movie_title)
    X_ts_occupation_date = create_learning_matrices(rating_matrix_occupation_date, test_user_occupation_movie_date)
    X_ts_occupation_gender = create_learning_matrices(rating_matrix_occupation_gender, test_user_occupation_movie_gender)

    # Predict
    print("Prediction")
    y_pred = model.predict(X_ts)
    y_pred_age_title = model_age_title.predict(X_ts_age_title)
    y_pred_age_date = model_age_date.predict(X_ts_age_date)
    y_pred_age_gender = model_age_gender.predict(X_ts_age_gender)
    y_pred_gender_title = model_gender_title.predict(X_ts_gender_title)
    y_pred_gender_date = model_gender_date.predict(X_ts_gender_date)
    y_pred_gender_gender = model_gender_gender.predict(X_ts_gender_gender)
    y_pred_occupation_title = model_occupation_title.predict(X_ts_occupation_title)
    y_pred_occupation_date = model_occupation_date.predict(X_ts_occupation_date)
    y_pred_occupation_gender = model_occupation_gender.predict(X_ts_occupation_gender)
    
    y_pred_matrix = np.zeros((9,len(y_pred_age_title)))
    """
    y_pred_matrix[0] = y_pred_age_title
    y_pred_matrix[1] = y_pred_age_date
    y_pred_matrix[2] = y_pred_age_gender
    y_pred_matrix[3] = y_pred_gender_title
    y_pred_matrix[4] = y_pred_gender_date
    y_pred_matrix[5] = y_pred_gender_gender
    y_pred_matrix[6] = y_pred_occupation_title
    y_pred_matrix[7] = y_pred_occupation_date
    y_pred_matrix[8] = y_pred_occupation_gender
    new_y_pred = (y_pred + y_pred_matrix.mean(0))/2
    """
    #(y_pred - y_pred_matrix.mean(0))/10 + y_pred
    
    # Making the submission file
    #fname = make_submission(y_pred_matrix.mean(0), test_user_movie_pairs, 'toy_example')
    fname2 = make_submission(y_pred, test_user_movie_pairs, 'toy_exampleTrue') 
    #fname3 = make_submission(new_y_pred, test_user_movie_pairs, 'toy_exampleFinal') 
    print('Submission file "{}" successfully written'.format(fname2))

    
    """
    Autre idee => faire plusieurs fois le modele et prendre la moyenne 
    mauvaise idee ? => remet au centre
    travailler sur la deviation
    
    
    
    
    """
    
    
    
    
    
    
    
    

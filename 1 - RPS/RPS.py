# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import random
import numpy as np
import pandas as pd
from tensorflow import keras


moves = ['R', 'P', 'S']
ideal_response = {'R': 'P', 'P': 'S', 'S': 'R'}

df_train_x = None
df_train_y = None
model = None
hlen = 5
hentries = 20

use_markov_chain = True
pair_keys = ['RR', 'RP', 'RS', 'PR', 'PP', 'PS', 'SR', 'SP', 'SS']
matrix = {}
memory = 0.9
my_history = []

def player(prev_play, opponent_history=[]):

    guess = random.choice(moves)

  
    if use_markov_chain == True:
        global matrix, my_history
        if prev_play == '':
            for pair_key in pair_keys:
                matrix[pair_key] = {'R': 1 / 3,
                                    'P': 1 / 3,
                                    'S': 1 / 3}
            opponent_history = []
            my_history = []
        else:
            opponent_history.append(prev_play)

        if len(my_history) >= 2:
            prev_pair = my_history[-2] + opponent_history[-2]
            
            for rps_key in matrix[prev_pair]:
                matrix[prev_pair][rps_key] = memory * matrix[prev_pair][rps_key]
            matrix[prev_pair][prev_play] += 1

            last_pair = my_history[-1] + opponent_history[-1]

            if max(matrix[last_pair].values()) != min(matrix[last_pair].values()):
                prediction = max([(v, k) for k, v in matrix[last_pair].items()])[1]
                guess = ideal_response[prediction]


        my_history.append(guess)

    if use_markov_chain == False:
        global df_train_x, df_train_y, model

        if prev_play == '':
            df_train_x = pd.DataFrame()
            df_train_y = pd.DataFrame()
            model = keras.Sequential([
                keras.layers.Dense(hlen, input_shape=(hlen,)),
                keras.layers.Dense(3, activation='softmax')
            ])
            model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
            opponent_history = []

        else:
            opponent_history.append(moves.index(prev_play))

        if len(opponent_history) > hlen:
            df_train_x = df_train_x.append(pd.Series(opponent_history[-(hlen+1):-1]), ignore_index=True).astype('int8')
            df_train_y = df_train_y.append(pd.Series(opponent_history[-1]), ignore_index=True).astype('int8')

        
        if len(opponent_history) >= (hlen+hentries):
            model.fit(df_train_x, df_train_y, epochs=5, verbose=0)
            df_test_x = pd.DataFrame([opponent_history[-hlen:]])
            predictions = model.predict([df_test_x])
            guess = ideal_response[moves[np.argmax(predictions[0])]]


    return guess
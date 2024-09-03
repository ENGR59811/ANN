# The City College of New York, City University of New York 
# Written by: Ricardo Valdez                                                                             
# August, 2020
# SLEEP STUDY EXAMPLE
# An ANN to predict the exam grade for a student based on the number of hours 
# slept and studied for the exam
# This example problem is from: https://www.youtube.com/watch?v=bxe2t-v8xrs

import numpy as np
import pandas as pd
import time
from tensorflow import keras

def print_weights(weights):
    # weights = model.get_weights();
    print('\n******* WEIGHTS OF ANN *******\n') 
    for i in range(int(len(weights)/2)):
        print('Weights W%d:\n' %(i), weights[i*2])
        print('Bias b%d:\n' %(i), weights[(i*2)+1])
#END print_weights()

#% ANN TRAINING
print('\n')
print('*********************************************************************')     
print('**  WELCOME TO GRADE PREDICTIONS USING ARTIFICIAL NEURAL NETWORKS  **')
print('*********************************************************************')

# prompt user to train or load an ANN model
option_list = ['1','2']
option = ''
while option not in option_list:  
    print('\nOPTIONS:')
    print('1 - Train a new ANN model')
    print('2 - Load an existing model')
    
    option = input('\nSelect an option by entering a number: \n')
    if option not in option_list:
        message = 'Invalid input: Input must be one of the following - '
        print(message, option_list)
        time.sleep(2)
        
if option == '1':
    ## OPTION 1: TRAIN A NEW ANN MODEL
    train_data_file = 'study_data.csv'
    
    print('\n********* NOW TRAINING ANN USING', train_data_file,'*********')
    time.sleep(3)
    
    ## load the training data
    df = pd.read_csv(train_data_file)
    ## define input matrix X (get rid of column called exam_grade)
    X = np.array(df.drop(['exam_grade'], axis=1))
    ## define expected output matrix Y
    Y = np.array(df['exam_grade'])
    
    ## create a model for the ANN
    model = keras.Sequential()
    ## add a hidden layer that accepts 2 input features (hours studied/slept)
    ## the hidden layer has 3 neurons.
    ## Dense means every neuron in the layer connects to every neuron in the
    ## previous layer.
    model.add(keras.layers.Dense(3, activation='relu', input_shape=(2,)))
    # ## Add another hidden layer with 4 neurons to the ANN
    # model.add(keras.layers.Dense(4, activation='relu')
    ## add an output layer with a single output (exam grade)
    model.add(keras.layers.Dense(1, activation='linear'))
    
    ## set the optimization algorithm used for minimizing loss function
    ## use gradient descent (adam) to minimize error (loss)
    model.compile(optimizer='adam', loss='mean_squared_error')
    ## train the ANN model using 2000 iterations
    #model.fit(X, Y, epochs=10000)
    model.fit(X, Y, epochs=2000)
    
    print('\n\n********** ANN training complete **********\n\n')    
elif option == '2':
    ## OPTION 2: LOAD ANN MODEL FROM FILE
    
    message = 'Enter the file name of the ANN Model you want to load: \n'
    load_file = input(message)
    #load_file = input('It must be a .h5 file')
    
    ## if file name does not end with '.h5', add '.h5' to the file name
    if load_file[-3:] != '.h5':
        load_file += '.h5'
    ## load the ANN model from load_file
    model = keras.models.load_model(load_file)
    
    print('\n\n****** SUCCESSFULLY LOADED ANN MODEL FROM', load_file,'******')   
else:
    print('ERROR: INVALID OPTION SELECTED')
    ## raise an exception to terminate the program
    raise ValueError()

weights = model.get_weights();
print_weights(weights)

#% GRADE PREDICTION USING ANN
input('\n\n********** Press ENTER to start using the ANN **********\n\n')
finished = False
while not finished:
    ## prompt user for inputs
    studied = float(input('\n\nEnter number of hours studied: \n'))
    slept = float(input('Enter number of hours slept: \n'))
    
    # ## get ANN prediction (only element [0,0])
    # prediction = model.predict([[studied,slept]])[0,0]
    user_input = np.array([[studied,slept]])
    prediction = model.predict(user_input)
    
    ## restrict prediction between 0 and 100
    if prediction > 100:
        prediction = 100
    elif prediction < 0:
        prediction = 0
    else:
        pass

    ## display prediction
    print('\n*******************************')
    print('ANN Predicted Grade: ', int (prediction))
    print('*******************************')
    ## ask user if they would like to continue
    choice = ''
    while choice not in ['y','n']:
        choice = input('\n\nWould you like to continue? (y/n): \n')
        if choice == 'y':
            pass
        elif choice == 'n':
            finished = True
        else:
            print("Invalid input: Input must be 'y' or 'n'")
    #END WHILE
#END WHILE

# ask user if they would like to save the ANN model
choice = ''
while choice not in ['y','n']:
    choice = input('\n\nWould you like to save the ANN model? (y/n): \n')
    if choice == 'y':
        save_name = input('\n\nEnter a name for the save file: \n')
        ## if file name does not end with '.h5', add '.h5' to the file name
        if save_name[-3:] != '.h5':
            save_name += '.h5'
        model.save(save_name)
        print('\n\n')
        print('***** ANN MODEL SUCCESSFULLY SAVED AS '+save_name+' *****')
    elif choice == 'n':
        pass
    else:
        print("Invalid input: Input must be 'y' or 'n'")
#END WHILE

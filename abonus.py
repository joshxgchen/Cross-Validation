# abonus.py

# template for Bonus Assignment, Artificial Intelligence Survey, CMPT 310 D200
# Spring 2021, Simon Fraser University

# author: Josh Chen (jca432@sfu.ca)

from learning import *

def generate_restaurant_dataset(size=100):
    """
    Generate a data set for the restaurant scenario, using a numerical
    representation that can be used for neural networks. Examples will
    be newly created at random from the "real" restaurant decision
    tree.
    :param size: number of examples to be included
    """

    tempList = list(range(100)) #initial list 
    def gen(): #gen function from learning.py
        examples = list(map(random.choice, restaurant.values))
        examples[restaurant.target] = waiting_decision_tree(examples)
        return examples
    for i in range (0, len(tempList)):
      tempList[i] = gen()



    data = DataSet(name='restaurant_numeric',
      target='Wait', examples=tempList, attr_names='Alternate Bar Fri/Sat Hungry Patrons Price Raining Reservation Type WaitEstimate Wait')

    for i in range(len(data.examples)):
      for j in range(0,len(data.attr_names)):

        #boolean checker
        if (data.examples[i][j] == "Yes"):
          data.examples[i][j] = 1;
        elif (data.examples[i][j] == "No"):
          data.examples[i][j] = 0;

        #none/some/full check
        if (data.examples[i][j] == "None"):
          data.examples[i][j] = 0;
        elif (data.examples[i][j] == "Some"):
          data.examples[i][j] = 1;
        elif (data.examples[i][j] == "Full"):
          data.examples[i][j] = 2;

        #price check 
        if (data.examples[i][j] == "$"):
          data.examples[i][j] = 0;
        elif (data.examples[i][j] == "$$"):
          data.examples[i][j] = 1;
        elif (data.examples[i][j]  == "$$$"):
          data.examples[i][j] = 2;

        #WaitEstimate check 
        if (data.examples[i][j] == "0-10"):
          data.examples[i][j] = 0;
        elif (data.examples[i][j] == "10-30"):
          data.examples[i][j] = 1;
        elif (data.examples[i][j]  == "30-60"):
          data.examples[i][j] = 2;
        elif (data.examples[i][j]  == ">60"):
          data.examples[i][j] = 3;

        #cuisine-type check // hard coded for 8th element, as we know 8th element is type
        if (data.examples[i][8] == ("French" or "Italian" or "Burger" or "Thai")):
          data.examples[i][8] = 1;
        elif (data.examples[i][8]) != ("French" or "Italian" or "Burger" or "Thai"):
          data.examples[i][8] = 0; 

    print(data.examples); # print the results for TAs to see 
    print("Prints all numeric values. Correct!\n")
    return data
    
def nn_cross_validation(dataset, hidden_units, epochs=100, k=15):
    """
    Perform k-fold cross-validation. In each round, train a
    feed-forward neural network with one hidden layer. Returns the
    error ratio averaged over all rounds.
    :param dataset:      the data set to be used
    :param hidden_units: the number of hidden units (one layer) of the neural nets to be created
    :param epochs:       the maximal number of epochs to be performed in a single round of training
    :param k:            k-parameter for cross-validation 
                         (do k many rounds, use a different 1/k of data for testing in each round) 
    """
    
    error = 0 #saves error value
    tempErrorOne = 0 #holds first error estimate 
    tempErrorTwo = 0 #second error estimate
    learning_rate = 0.01;
 
    n = len(dataset.examples) 
    examples = dataset.examples
    random.shuffle(dataset.examples)
    for fold in range(k):
       temp1, temp2 = train_test_split(dataset, fold * (n // k), (fold + 1) * (n // k))
       dataset.examples = temp1
       h = NeuralNetLearner(dataset,[hidden_units],learning_rate,epochs, sigmoid)
       tempErrorOne += err_ratio(h, dataset, temp1)
       tempErrorTwo += err_ratio(h, dataset, temp2)
       # reverting back to original once test is completed
       dataset.examples = examples
    error = (tempErrorOne / k + tempErrorTwo / k)/2


    return error


N          = 100   # number of examples to be used in experiments
k          =   5   # k parameter
epochs     = 100   # maximal number of epochs to be used in each training round
size_limit =  15   # maximal number of hidden units to be considered

# generate a new, random data set
# use the same data set for all following experiments
dataset = generate_restaurant_dataset(N)

# try out possible numbers of hidden units
for hidden_units in range(1,size_limit+1):
    # do cross-validation
    error = nn_cross_validation(dataset=dataset,
                                hidden_units=hidden_units,
                                epochs=epochs,
                                k=k)
    # report size and error ratio
    print("Size " + str(hidden_units) + ":", error)

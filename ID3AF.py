# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:36:15 2020

@author: aina
"""

import pandas as pd
import numpy as np
from functools import reduce
import timeit
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import sys

#Import the dataset and define the feature as well as the target datasets / columns#
dataset = pd.read_csv('mammofold1.data',
                      names=["Biraids","Age","Shape","Margin","Density","class",])#Import all columns omitting the fist which consists the names of the animals

# We drop the animal names since this is not a good feature to split the data on
#dataset = dataset.drop('Biraids', axis=1)

###################

def entropy(target_col, isPrint):
    result = {"entropy":"", "RD":""}
    """
    Calculate the entropy of a dataset.
    The only parameter of this function is the target_col parameter which specifies the target column
    """
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
      #RELATION DEGREE FOR AF
    global RD
    s = ([counts[i] for i in range(len(elements))])
    RD = abs(s[0] - sum(s[1:]))
    if isPrint == True:
        """
         print("elements:", elements)
         #Value of RD
         print("Relation Degree:", RD)
    #END RELATION DEGREE
        """
    result["entropy"] = entropy
    result["RD"] = RD

    return result


################### 

def InfoGain(data, split_attribute_name, target_name, isprint):
    target_name = "class"
    result = {"Information_Gain":"", "AF":""}
    """
    Calculate the information gain of a dataset. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
    """    

    #Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name], True)["entropy"]
    
    ##Calculate the entropy of the dataset
    
    #Calculate the values and the corresponding counts for the split attribute 
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    """
    #start associationfunction
    pprint("================================================================")
    pprint("split_attribute_name:{}".format(split_attribute_name))
    print("vals:{}".format(vals))
    pprint("counts:{}".format(counts))
    pprint("sumCounts:{}".format(sum(counts)))
   
    pprint("Vals value:{}".format(m))
    """
    #assocvaluearray suppose to be a array form like assocvalue[i]. But it not save previous data of assocvalue? 
    m=len(vals)
     
    #Calculate the weighted entropy
   
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name], False)["entropy"] for i in range(len(vals))])
    #Calculate the information gain

    #Calculate the sum(RDs)
    SumRDs = np.sum([entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name], True)["RD"] for i in range(len(vals))])
    
    # pprint("SumRDs:{}".format(SumRDs))

    #Calculate the AF(split_attribute_name)
    AF = SumRDs / m
    
    Information_Gain = (total_entropy - Weighted_Entropy)# * SumRDs / m)
   
    #step 5
    Information_Gain = Information_Gain# * Normalization_Factor

    if isprint == True:
        """
        print("split_attribute_name:{}".format(split_attribute_name))
        print("Information_Gain:{}".format(Information_Gain))
        print("AF:{}".format(AF))
        """
    result["Information_Gain"] = Information_Gain
    result["AF"] = AF
 
    return result
       
###################

###################


def ID3(data,originaldata,features,target_attribute_name="class",parent_node_class = None):
    """
    ID3 Algorithm: This function takes five paramters:
    1. data = the data for which the ID3 algorithm should be run --> In the first run this equals the total dataset
 
    2. originaldata = This is the original dataset needed to calculate the mode target feature value of the original dataset
    in the case the dataset delivered by the first parameter is empty

    3. features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
    we have to remove features from our dataset --> Splitting at each node

    4. target_attribute_name = the name of the target attribute

    5. parent_node_class = This is the value or class of the mode target feature value of the parent node for a specific node. This is 
    also needed for the recursive call since if the splitting leads to a situation that there are no more features left in the feature
    space, we want to return the mode target feature value of the direct parent node.
    """   
    #Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#
    
    #If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
   
    #If the dataset is empty or treshold, return the mode target feature value in the original dataset
    elif len(data)==0:
         return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
          
    #If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
    #the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    #the mode target feature value is stored in the parent_node_class variable.
    
    elif len(features) ==0:
        return parent_node_class

       
    #If none of the above holds true, grow the tree!
    
    else:
        #Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
         
        #Select the feature which best splits the dataset
        
        #calculate the sumAF
        sumAF = np.sum([InfoGain(data,feature,target_attribute_name, True)["AF"] for feature in features])
        
        #calculate the Information gain list
        item_values = [InfoGain(data,feature,target_attribute_name, True)["Information_Gain"] for feature in features] #Return the information gain values for the features in the dataset
        
        #calculate the AF list
        AF_values = [InfoGain(data,feature,target_attribute_name, True)["AF"] for feature in features]

        np.seterr(divide='ignore', invalid='ignore')
        #calculate the V(k) list:[AF(A)/Sum(AF), AF(B)/Sum(AF), ..., AF(M)/Sum(AF)]
        AF_values = AF_values/sumAF

        """
        print("====================")
        print("Information_Gain:{}".format(item_values))
        print("sumAF:{}".format(sumAF))
        print("AF_values:{}".format(AF_values))
        """
        
        #calculate the new information gain list:[IG(A) * V(A), IG(B) * V(B), ..., IG(M) * V(M)]
        item_values = item_values*AF_values

        
        #print("item_values:{}".format(item_values))

        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index] 
        
        #Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        #gain in the first run
        tree = {best_feature:{}}
    
        #Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]
        
        #Grow a branch under the root node for each possible value of the root node feature
        
        for value in np.unique(data[best_feature]):
            value = value
            #Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()
            #Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!

            subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)
            
            #Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree
          
        return(tree)    
                
###################

###################

    
def predict(query,tree,default = 1):
    """
    Prediction of a new/unseen query instance. This takes two parameters:
    1. The query instance as a dictionary of the shape {"feature_name":feature_value,...}

    2. The tree 


    We do this also in a recursive manner. That is, we wander down the tree and check if we have reached a leaf or if we are still in a sub tree. 
    Since this is a important step to understand, the single steps are extensively commented below.

    1.Check for every feature in the query instance if this feature is existing in the tree.keys() for the first call, 
    tree.keys() only contains the value for the root node 
    --> if this value is not existing, we can not make a prediction and have to 
    return the default value which is the majority value of the target feature

    2. First of all we have to take care of a important fact: Since we train our model with a database A and then show our model
    a unseen query it may happen that the feature values of these query are not existing in our tree model because non of the
    training instances has had such a value for this specific feature. 
    For instance imagine the situation where your model has only seen animals with one to four
    legs - The "legs" node in your model will only have four outgoing branches (from one to four). If you now show your model
    a new instance (animal) which has for the legs feature the vale 5, you have to tell your model what to do in such a 
    situation because otherwise there is no classification possible because in the classification step you try to 
    run down the outgoing branch with the value 5 but there is no such a branch. Hence: Error and no Classification!
    We can address this issue with a classification value of for instance (999) which tells us that there is no classification
    possible or we assign the most frequent target feature value of our dataset used to train the model. Or, in for instance 
    medical application we can return the most worse case - just to make sure... 
    We can also return the most frequent value of the direct parent node. To make a long story short, we have to tell the model 
    what to do in this situation.
    In our example, since we are dealing with animal species where a false classification is not that critical, we will assign
    the value 1 which is the value for the mammal species (for convenience).

    3. Address the key in the tree which fits the value for key --> Note that key == the features in the query. 
    Because we want the tree to predict the value which is hidden under the key value (imagine you have a drawn tree model on 
    the table in front of you and you have a query instance for which you want to predict the target feature 
    - What are you doing? - Correct:
    You start at the root node and wander down the tree comparing your query to the node values. Hence you want to have the
    value which is hidden under the current node. If this is a leaf, perfect, otherwise you wander the tree deeper until you
    get to a leaf node. 
    Though, you want to have this "something" [either leaf or sub_tree] which is hidden under the current node
    and hence we must address the node in the tree which == the key value from our query instance. 
    This is done with tree[keys]. Next you want to run down the branch of this node which is equal to the value given "behind"
    the key value of your query instance e.g. if you find "legs" == to tree.keys() that is, for the first run == the root node.
    You want to run deeper and therefore you have to address the branch at your node whose value is == to the value behind key.
    This is done with query[key] e.g. query[key] == query['legs'] == 0 --> Therewith we run down the branch of the node with the
    value 0. Summarized, in this step we want to address the node which is hidden behind a specific branch of the root node (in the first run)
    this is done with: result = [key][query[key]]

    4. As said in the 2. step, we run down the tree along nodes and branches until we get to a leaf node.
    That is, if result = tree[key][query[key]] returns another tree object (we have represented this by a dict object --> 
    that is if result is a dict object) we know that we have not arrived at a root node and have to run deeper the tree. 
    Okay... Look at your drawn tree in front of you... what are you doing?...well, you run down the next branch... 
    exactly as we have done it above with the slight difference that we already have passed a node and therewith 
    have to run only a fraction of the tree --> You clever guy! That "fraction of the tree" is exactly what we have stored
    under 'result'.
    So we simply call our predict method using the same query instance (we do not have to drop any features from the query
    instance since for instance the feature for the root node will not be available in any of the deeper sub_trees and hence 
    we will simply not find that feature) as well as the "reduced / sub_tree" stored in result.

    SUMMARIZED: If we have a query instance consisting of values for features, we take this features and check if the 
    name of the root node is equal to one of the query features.
    If this is true, we run down the root node outgoing branch whose value equals the value of query feature == the root node.
    If we find at the end of this branch a leaf node (not a dict object) we return this value (this is our prediction).
    If we instead find another node (== sub_tree == dict objct) we search in our query for the feature which equals the value 
    of that node. Next we look up the value of our query feature and run down the branch whose value is equal to the 
    query[key] == query feature value. And as you can see this is exactly the recursion we talked about
    with the important fact that for each node we run down the tree, we check only the nodes and branches which are 
    below this node and do not run the whole tree beginning at the root node 
    --> This is why we re-call the classification function with 'result'
    """
    
    mosttargetfeature= np.unique(training_data["class"])[np.argmax(np.unique(training_data["class"],return_counts=True)[1])]
    
    #1.
    for key in list(query.keys()):
        if key in list(tree.keys()):
            #2.
            try:
                result = tree[key][query[key]] 
            except:
                pred= mosttargetfeature
                return pred
            #3.
            result = tree[key][query[key]]
            #4.
            if isinstance(result,dict):
                return predict(query,result)

            else:
                return result 
      
"""
Check the accuracy of our prediction.
The train_test_split function takes the dataset as parameter which should be divided into
a training and a testing set. The test function takes two parameters, which are the testing data as well as the tree model.
"""
###################

###################

def train_test_split(dataset):
    training_data = dataset.iloc[:456].reset_index(drop=True)  # We drop the index respectively relabel the index
    # starting form 0, because we do not want to run into errors regarding the row labels / indexes
    testing_data = dataset.iloc[456:].reset_index(drop=True)
    return training_data, testing_data


def kfold_split(n_samples, n_fold=10, random_state=np.nan):
    # set random seed
    if np.isnan(random_state):
        np.random.seed(random_state)

    # determine fold sizes
    fold_sizes = np.floor(n_samples / n_fold) * np.ones(n_fold, dtype=int)

    # check if there is remainder
    r = n_samples % n_fold

    # distribute remainder
    for i in range(r):
        fold_sizes[i] += 1

    # create fold indices
    train_indices = []
    test_indices = []

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + int(fold_size)
        test_mask = np.zeros(n_samples, dtype=np.bool)
        test_mask[start:stop] = True
        train_mask = np.logical_not(test_mask)

        train_indices.append(indices[train_mask])
        test_indices.append(indices[test_mask])

        current = stop

    return train_indices, test_indices


def plot_confusion_matrix(y_true, y_pred):
    # unique classes
    conf_mat = {}
    classes = np.unique(y_true)
    # C is positive class while True class is y_true or temp_true
    for c in classes:
        temp_true = y_true[y_true == c]
        temp_pred = y_pred[y_true == c]
        conf_mat[c] = {pred: np.sum(temp_pred == pred) for pred in classes}
    print("Confusion Matrix: \n", pd.DataFrame(conf_mat))

    # plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(data=pd.DataFrame(conf_mat), annot=True, cmap=plt.get_cmap("Blues"), fmt='d')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def calculate_metrics(y_true, y_pred):
    # convert to integer numpy array
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    pre_list = []
    rec_list = []
    f1_list = []
    # loop over unique classes
    for c in np.unique(y_true):
        # copy arrays
        temp_true = y_true.copy()
        temp_pred = y_pred.copy()

        # positive class
        temp_true[y_true == c] = '1'
        temp_pred[y_pred == c] = '1'

        # negative class
        temp_true[y_true != c] = '0'
        temp_pred[y_pred != c] = '0'

        # tp, fp and fn
        tp = np.sum(temp_pred[temp_pred == '1'] == temp_true[temp_pred == '1'])
        tn = np.sum(temp_pred[temp_pred == '0'] == temp_true[temp_pred == '0'])
        fp = np.sum(temp_pred[temp_pred == '1'] != temp_true[temp_pred == '1'])
        fn = np.sum(temp_pred[temp_pred == '0'] != temp_true[temp_pred == '0'])

        precision = tp / (tp + fp) * 100
        recall = tp / (tp + fn) * 100
        f1 = 2 * (precision * recall) / (precision + recall)

        pre_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1)
        print(
            "Class {}: Precision = {:0.3f}    Recall = {:0.3f}    F1-Score = {:0.3f}".format(c, precision, recall, f1))

    print("Average: Precision = {:0.3f}    Recall = {:0.3f}    F1-Score = {:0.3f}   Accuracy = {:0.3f}".
          format(np.mean(pre_list),
                 np.mean(rec_list),
                 np.mean(f1_list),
                 np.sum(y_pred == y_true)/y_pred.shape[0]*100))


def test(data, tree):
    # Create new query instances by simply removing the target feature column from the original dataset and
    # convert it to a dictionary
    queries = data.iloc[:, :-1].to_dict(orient="records")

    # Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"])

    # Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree, 1.0)

    return predicted["predicted"]


# pprint(tree)

n_fold = 10
#random_state=0,4
train_idx, test_idx = kfold_split(dataset.shape[0], n_fold=n_fold, random_state=0)
all_true = []
all_pred = []


for i in range(len(train_idx)):
    training_data, testing_data = dataset.iloc[train_idx[i]], dataset.iloc[test_idx[i]]

    tree = ID3(training_data, training_data, training_data.columns[:-1])

    y_pred = test(testing_data, tree)
    y_true = testing_data["class"]

    y_pred = np.array(y_pred).astype(str)
    y_true = np.array(y_true).astype(str)

    all_true.append(list(y_true))
    all_pred.append(list(y_pred))

    acc = (np.sum(y_true == y_pred) / y_true.shape[0])*100
    print("Fold-{}: Accuracy: {:.4f}".format(i + 1, acc))

all_true = [v for item in all_true for v in item]
all_pred = [v for item in all_pred for v in item]

# Calculate Overall Metrics
print("\nOverall Metrics")
# calculate precision, recall and f1-score
calculate_metrics(all_true, all_pred)
# plot confusion matrix
plot_confusion_matrix(np.array(all_true), np.array(all_pred))

"""
# to display in text:results.txt
with open('results-10.txt', 'w') as f:
    sys.stdout = f
    for i in range(len(train_idx)):
        training_data, testing_data = dataset.iloc[train_idx[i]], dataset.iloc[test_idx[i]]

        tree = ID3(training_data, training_data, training_data.columns[:-1])

        y_pred = test(testing_data, tree)
        y_true = testing_data["class"]

        y_pred = np.array(y_pred).astype(str)
        y_true = np.array(y_true).astype(str)

        all_true.append(list(y_true))
        all_pred.append(list(y_pred))

        print("----------------- Fold {} --------------".format(i+1))
        # print tree
        #pprint(tree)

        # calculate precision, recall and f1-score
        calculate_metrics(y_true, y_pred)

        # plot confusion matrix
        plot_confusion_matrix(y_true, y_pred)

    all_true = [v for item in all_true for v in item]
    all_pred = [v for item in all_pred for v in item]

    # Calculate Overall Metrics
    print("\n----------- Overall Metrics --------------")
    # calculate precision, recall and f1-score
    calculate_metrics(all_true, all_pred)
    # plot confusion matrix
    plot_confusion_matrix(np.array(all_true), np.array(all_pred))

"""
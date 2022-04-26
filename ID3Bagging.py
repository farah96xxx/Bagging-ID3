# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:38:55 2020

@author: aina
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 00:21:38 2020

@author: aina
"""
import pandas as pd
import numpy as np
import timeit
import scipy.stats as sps
# from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from pprint import pprint

start = timeit.default_timer()
#Import the dataset and define the feature as well as the target datasets / columns#
dataset = pd.read_csv('mammofold1.data',
                      names=["Biraids","Age","Shape","Margin","Density","class",])#Import all columns omitting the fist which consists the names of the animals

#We drop the id names since this is not a good feature to split the data on
#dataset=dataset.drop('id',axis=1)
###################


def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy


################### 

def InfoGain(data,split_attribute_name,target_name="class"):
    
    #Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])
    
    ##Calculate the entropy of the dataset
    
    #Calculate the values and the corresponding counts for the split attribute 
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    
    #Calculate the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    #Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain
       
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
    
    #If the dataset is empty, return the mode target feature value in the original dataset
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
         
        ################################################################################################################
        ############!!!!!!!!!Start the bagging!!!!!!!!#############
        ###############################################################################################################
        
        #only apply in RandomForest
        #features = np.random.choice(features,size=np.int(np.sqrt(len(features))),replace=False)
        
        #Select the feature which best splits the dataset
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
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
                

###########################################################################################################
###########################################################################################################

    
def predict(query,tree,default = 'benign'):
        
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)

            else:
                return result
            
###########################################################################################################
###########################################################################################################

def boot_strap(dataset):
    bootstrap_data = dataset.sample(frac=1).reset_index(drop=True)
    return bootstrap_data


###########################################################################################################
###########################################################################################################

#######Train the Bagging model###########

def Bagging_Train(dataset,number_of_Trees):
    #Create a list in which the single bagging are stored
    bagging_sub_tree = []
    
    #Create a number of n models
    for i in range(number_of_Trees):
        #Create a number of bootstrap sampled datasets from the original dataset 
        bootstrap_sample = dataset.sample(frac=1,replace=True)
        
        #Create a boot starp training data
        bootstrap_training_data = boot_strap(bootstrap_sample)

        #Grow a tree model for each of the training data
        #We implement the subspace sampling in the ID3 algorithm itself. Hence take a look at the ID3 algorithm above!
        bagging_sub_tree.append(ID3(bootstrap_training_data,bootstrap_training_data,bootstrap_training_data.drop(labels=['class'],axis=1).columns))
        
    return bagging_sub_tree

 
#######Predict a new query instance###########
def Bagging_Predict(query,bagging,default='benign'):
    predictions = []
    for tree in bagging:
        predictions.append(predict(query,tree,default))
    return sps.mode(predictions)[0][0]


#######Test the model on the testing data and return the accuracy###########
def Bagging_Test(data, bagging):
    pred = np.empty(shape=data.shape[0], dtype=data["class"].values.dtype)
    for i in range(len(data)):
        query = data.iloc[i,:].drop('class').to_dict()
        pred[i] = Bagging_Predict(query,bagging, default='benign')

    return pred


def kfold_split(n_samples, n_fold=10, random_state=np.nan):
    
    # set random seed// set not for reproducible
    if not np.isnan(random_state):
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

    return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.sum(y_pred == y_true) / y_pred.shape[0] * 100


n_fold = 10
n_trees = 100
train_idx, test_idx = kfold_split(dataset.shape[0], n_fold=n_fold,)
all_true = []
all_pred = []
av_pre = []
av_rec = []
av_f1 = []
av_acc = []

"""
for i in range(len(train_idx)):
        training_data, testing_data = dataset.iloc[train_idx[i]], dataset.iloc[test_idx[i]]

        tree = Bagging_Train(training_data, n_trees)
          
        y_pred = Bagging_Test(testing_data, tree)
        y_true = testing_data["class"]

        y_pred = np.array(y_pred).astype(str)
        y_true = np.array(y_true).astype(str)

        all_true.append(list(y_true))
        all_pred.append(list(y_pred))

        print("----------------- Fold {} --------------".format(i+1))

        # calculate precision, recall and f1-score
        p, r, f, a = calculate_metrics(y_true, y_pred)
        av_pre.append(p)
        av_rec.append(r)
        av_f1.append(f)
        av_acc.append(a)

        # plot confusion matrix
        plot_confusion_matrix(y_true, y_pred)

all_true = [v for item in all_true for v in item]
all_pred = [v for item in all_pred for v in item]

    # Calculate Overall/Average Metrics
print("\n----------- Overall Confusion Matrix --------------")
    # plot confusion matrix
plot_confusion_matrix(np.array(all_true), np.array(all_pred))
    
calculate_metrics(all_true, all_pred)

"""
with open('resultsbaggingid3-17.txt', 'w') as f:
    sys.stdout = f
    for i in range(len(train_idx)):
        training_data, testing_data = dataset.iloc[train_idx[i]], dataset.iloc[test_idx[i]]

        tree = Bagging_Train(training_data, n_trees)

        y_pred = Bagging_Test(testing_data, tree)
        y_true = testing_data["class"]

        y_pred = np.array(y_pred).astype(str)
        y_true = np.array(y_true).astype(str)

        all_true.append(list(y_true))
        all_pred.append(list(y_pred))

        print("----------------- Fold {} --------------".format(i+1))

        # calculate precision, recall and f1-score
        p, r, f, a = calculate_metrics(y_true, y_pred)
        av_pre.append(p)
        av_rec.append(r)
        av_f1.append(f)
        av_acc.append(a)

        # plot confusion matrix
        plot_confusion_matrix(y_true, y_pred)

    all_true = [v for item in all_true for v in item]
    all_pred = [v for item in all_pred for v in item]

    # Calculate Overall/Average Metrics
    print("\n----------- Overall Confusion Matrix --------------")
    # plot confusion matrix
    plot_confusion_matrix(np.array(all_true), np.array(all_pred))
    
    print("\n----------- Overall Matrix --------------")
    calculate_metrics(all_true, all_pred)
    # average metrics
    
    print("----------- Average KFold-Cross --------------")
    print("Average: Precision = {:0.3f}    Recall = {:0.3f}    F1-Score = {:0.3f}   Accuracy = {:0.3f}".
          format(np.mean(av_pre),
                 np.mean(av_rec),
                 np.mean(av_f1),
                 np.mean(av_acc)))
 
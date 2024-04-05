import math
import pandas as pd
import random
import time
from colorama import Fore, Back, Style

class DecisionTree:
    def __init__(self, dataset):
        self.tree = None
        self.dataset = dataset; # pandas dataframe containing the entire dataset

    def learn(self):
        # Learn the decision tree from the dataset
        self.tree = self.learnDecisionTree(self.dataset)

    def learnDecisionTree(self, examples, remove_attributes = False):
        # Decision tree learning algorithm. Target attribute must be the last column in the dataset, with name 'target'
        # I use recursive splitting based on information gain
        # I split the dataset based on the attribute with the maximum information gain and then recursively split the dataset
        # To handle both continuous attributes I split based on the threshold that gives the maximum information gain
        if examples.empty:# or len(examples) < 4:
            return Node(self.mostCommonTargetValue(self.dataset))
        attributes = examples.iloc[:, :-1] # list of attributes
        if examples['target'].nunique() == 1: #if all examples have the same classification
            return Node(examples['target'].unique()[0])
        if len(attributes.columns) == 0: #if there are no more attributes to split on, I return a leaf node with the most common target value
            return Node(self.mostCommonTargetValue(examples))
        A = self.maxGainAttribute(examples, attributes) # Attribute with the maximum information gain
        if self.isContinuous(A):
            threshold = self.findThreshold(examples, A)
            if threshold == None: # if the threshold is None, it means the attribute has only one value, so I return a leaf node with the most common target value
                return Node(self.mostCommonTargetValue(examples))
            else:
                tree = Node(A.name, threshold, isContinuous=True)
                subset_v = examples[examples[A.name] < threshold]
                if remove_attributes:
                    subset_v = subset_v.drop(A.name, axis=1)
                if not subset_v.empty:
                    tree.add_child(0, self.learnDecisionTree(subset_v))
                subset_v = examples[examples[A.name] >= threshold]
                if remove_attributes:
                    subset_v = subset_v.drop(A.name, axis=1)
                if not subset_v.empty:
                    tree.add_child(1, self.learnDecisionTree(subset_v))
        else:
            tree = Node(A.name)
            for v in examples[A.name].unique():
                subset_v = examples[examples[A.name] == v]
                if remove_attributes:
                    subset_v = subset_v.drop(A.name, axis=1)
                tree.add_child(v, self.learnDecisionTree(subset_v))
        return tree

    def maxGainAttribute(self, examples, attributes):
        # Returns the attribute with the maximum information gain
        # This will be used to split the dataset at each node of the decision tree
        maxGain = 0
        gain = 0
        datasetCost = self.entropy(self.dataset)
        bestAttribute = attributes.iloc[:,math.floor(random.random() * len(attributes.columns))] # I initialize the best attribute to a random attribute
        for at in attributes:
            a = attributes[at] # the dataframe iterator returns just the column name, so I have to get the column itself
            if self.isContinuous(a):
                threshold = self.findThreshold(examples, a)
                gain = self.informationGain(examples, a, datasetCost, threshold)
            else:
                gain = self.informationGain(examples, a, datasetCost)
            if gain > maxGain:
                maxGain = gain
                bestAttribute = a
        return bestAttribute

    def isContinuous(self, attribute):
        if attribute.dtype == float or self.dataset[attribute.name].nunique() >10:
            # if the attribute type is float it means it is continuous, I also consider it continuous if it has more than 10 unique values
            return True
        return False

    def entropy(self, dataset):
        entropy = 0
        for target in dataset['target'].unique():
            p = len(dataset[dataset['target'] == target]) / len(dataset)
            entropy -= ( p * math.log2(p) ) if p != 0 else 0 # to avoid log(0)
        return entropy

    def informationGain(self, examples, attribute, datasetCost, threshold = None):
        # Calculates the information gain of the dataset splitting based on the given attribute
        gain = datasetCost
        if threshold != None: # if the attribute is continuous or a threshold is given
            greater = examples[examples[attribute.name] >= threshold] # subset of the dataset where the attribute is greater or equal to the threshold
            less = examples[examples[attribute.name] < threshold] # subset of the dataset where the attribute is less than the threshold
            gain -= ( ( ( len(greater) / len(examples) ) * self.entropy(greater) ) + ( ( len(less) / len(examples) ) * self.entropy(less) ) )
        else:
            for v in examples[attribute.name].unique():
                Dv = examples[examples[attribute.name] == v] # subset of the dataset where the attribute has value v
                gain -= ( ( len(Dv) / len(examples) ) * self.entropy(Dv) )
        return gain
        
    def findThreshold(self, examples, attribute):
        # Finds the threshold that gives the maximum information gain, used when the attribute is continuous
        # It orders the values of the attribute and calculates the midpoint between two consecutive values with different target value as the candidate thresholds
        # Then it calculates the information gain for each threshold and returns the threshold that gives the maximum information gain
        orderedValues = examples.sort_values(attribute.name) # sort the values of the attribute in ascending order
        thresholds = []
        for i in range(1, len(orderedValues)):
            cx = orderedValues.iloc[i-1]['target']
            cy = orderedValues.iloc[i]['target']
            x = orderedValues.iloc[i-1][attribute.name]
            y = orderedValues.iloc[i][attribute.name]
            if (not math.isnan(x)) and (not math.isnan(y)) and cx != cy:
                t = ( x + y ) / 2
                if t not in thresholds:
                    thresholds.append(t)
        # I calculate the information gain for each threshold and return the threshold that gives the maximum information gain
        if len(thresholds) == 0:
            return None
        bestThreshold = random.choice(thresholds)
        maxGain = 0
        for th in thresholds:
            gain = self.informationGain(examples, attribute, self.entropy(examples), th)
            if gain > maxGain:
                maxGain = gain
                bestThreshold = th
        return bestThreshold

    def predict(self, example, log = False):
        # Predicts the target value of a single example
        node = self.tree
        while not node.isLeaf():
            if node.isContinuous:
                if log: 
                   print("Continuous) Node Attribute: ", node.attribute, ", Node Threshold: ",node.threshold, ", Example Value: ", example[node.attribute])
                   print("Children keys: ", node.children.keys())
                if example[node.attribute] < node.threshold:
                    node = node.children[0]
                    if log:
                        print("Less than threshold")
                else:
                    node = node.children[1]
                    if log:
                        print("Greater than or equal to threshold")
                        print()
            else:
                # TODO: I should handle the case where the attribute value is not in the tree or the attribute is not in the example
                if example[node.attribute] not in node.children.keys():
                    return None
                if log: 
                    print("Not Continuous) Node Attribute: ", node.attribute, ", Example Value: ", example[node.attribute])
                    print("Children keys: ", node.children.keys())
                    print("Children attributes: ", [c.attribute for c in node.children.values()])
                node = node.children[example[node.attribute]]
                if log:
                    print("Pick child: ", node.attribute)
                    print()
                if node == None:
                    return None
        if log:
            if node.attribute == example['target']:
                print(Fore.GREEN, end="")
            else:
                print(Fore.RED, end="")
            print("LEAF NODE: ", node.attribute, ", Example Target: ", example['target'])
            print(Style.RESET_ALL)
        return node.attribute
    
    def predictAll(self, examples):
        # Predicts the target values of all examples in the dataset
        # Return a list of the predicted target values
        predictions = []
        for i in range(len(examples)):
            prediction = self.predict(examples.iloc[i])
            predictions.append(prediction)
        return predictions

    def mostCommonTargetValue(self, dataset):
        # Returns the most common target value in the dataset
        # This will be used when the tree cannot be split further
        return dataset['target'].value_counts().idxmax()

class Node:
    def __init__(self, attribute, threshold = None, isContinuous = False):
        self.attribute = attribute # attribute to split on (column name)
        # If it's a leaf node, it will be the target value
        # I know a node is a Leaf node if it has no children
        self.threshold = threshold # threshold for continuous attributes (if attribute is continuous)
        self.isContinuous = isContinuous # flag to indicate if the attribute is continuous
        self.children = {} # dictionary to store the child nodes, keys are the assignments to the attribute
    
    def isLeaf(self):
        return len(self.children) == 0

    def add_child(self, assignment, node):
        self.children[assignment] = node # add a child node, with the assignment as the key. For continuous attributes, the assignment will be 0 or 1


class DecisionTreeClassifier:
    def __init__(self, dataset, label_column):
        self.tree = None
        self.dataset = dataset
        self.dataset['target'] = label_column
    
    def train(self, k_folds, log = False):
        # I Split the dataset into k folds, for each fold: I use k-1 folds for training and the remaining fold for validation
        folds = []
        for i in range(k_folds):
            folds.append(pd.DataFrame())
        # I split the dataset based on the target values to make sure each fold has the same distribution of target values
        for target in self.dataset['target'].unique():
            targetValues = self.dataset[self.dataset['target'] == target]
            n = math.floor(len(targetValues) / k_folds)
            # I distribute the target values into the folds
            for i in range(k_folds):
                maxIndex = (i + 1 ) * n - 1 if i != k_folds - 1 else len(targetValues) - 1
                folds[i] = pd.concat( [ folds[i],targetValues.iloc[i*n:maxIndex] ] )
        
        if log:
           print("Fold sizes: ", [len(f) for f in folds], " out of ", len(self.dataset), " total dataset size")

        absoluteErrors = []
        relativeErrors = []
        validationSetSizes = []
        originalValues = []
        predictionValues = []
        for i in range(k_folds):
            if k_folds == 1:
                validationSet = self.dataset
                trainingSet = self.dataset
            else:
                validationSet = folds[i]
                
                trainingSet = pd.concat(folds[:i] + folds[i+1:]) # Join all folds except the validation set
            validationSetSizes.append(len(validationSet))
            print("fold",i+1,": Learning")
            if log:
                print("Training set ",i+1," [size:",len(trainingSet),"] ")
                print("Validation set ",i+1," [size:",len(validationSet),"] ")
            tree = DecisionTree(trainingSet)
            if log:
                print(i+1,'/',k_folds,") Learning...")
            t = time.time()
            tree.learn()
            print("Learning time: ", round(time.time() - t,2), " seconds")
            if log:
                print(i+1,'/',k_folds,") Learning done")
            predictions = tree.predictAll(validationSet)
            if log:
                print(i+1,'/',k_folds,") Original target values: ")
                print(validationSet['target'].tolist())
                print(i+1,'/',k_folds,") Predictions done: ")
                print(predictions)
            originalValues.append(validationSet['target'].tolist())
            predictionValues.append(predictions)
            absoluteErrors.append(self.calculateError(validationSet['target'], predictions))
            if log:
                print("Errors : ", absoluteErrors[i], " out of ", len(validationSet))
            relativeErrors.append(absoluteErrors[i] / len(validationSet))
            if log:
                print("Relative error : ", relativeErrors[i]*100, "%")
        return absoluteErrors, relativeErrors, validationSetSizes, originalValues, predictionValues
    
    def calculateError(self, true_values, predicted_values):
        return sum(true_values != predicted_values)
    
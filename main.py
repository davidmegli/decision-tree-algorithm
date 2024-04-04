from ucimlrepo import fetch_ucirepo
import pandas as pd
import decision_tree as dt  
import math
import numpy as np
from colorama import Fore, Back, Style

print("Welcome to Decision Tree Classifier by David Megli")
print("This program uses the UCI Machine Learning Repository to fetch datasets and train a decision tree classifier using k-fold cross validation.")

while(True):
    th1 = 0.5
    th2 = 5
    print("Please select a dataset from the list below: ")
    print("1. Iris")
    print("2. Breast Cancer")
    print("3. Wine")
    print("4. Custom Dataset")
    print("5. Exit")

    id = int(input())
    if id == 5:
        break
    if id == 1:
        id=53
    elif id == 2:
        id=17
    elif id == 3:
        id=109
    elif id == 4:
        print("Please enter the dataset id of a UCI Machine Learning repository: ")
        id = int(input())
    else:
        print("Invalid selection")
        continue
    print("\nFetching dataset...")
    dataset = fetch_ucirepo(id=id)
    if dataset == None:
        print("Invalid dataset id")
        continue
    X = dataset.data.features
    y = dataset.data.targets
    print("\nDataset Name: ",Fore.YELLOW, dataset.metadata.name, Style.RESET_ALL)
    print("\nSummary: \n", Fore.YELLOW, dataset.metadata.additional_info.summary, Style.RESET_ALL)
    
    dcc = dt.DecisionTreeClassifier(X, y)

    print("\nData:\n", X)
    print("\nPlease choose a value for the k-fold cross validation: ")
    k = int(input())
    absErr, relErr, valSetSize, originalVals, predictVals = dcc.train(k, log=False)
    for i in range(k):
        print("\nFold ",i+1,") Original Values: ", end="")
        for j in range(len(originalVals[i])):
            if(originalVals[i][j] != predictVals[i][j]):
                print(Fore.RED, end="")
            else:
                print(Fore.GREEN, end="")
            print(originalVals[i][j], end=", ")
            print(Style.RESET_ALL, end="")
        print("\nFold ",i+1,") Predicted Values: ", end="")
        for j in range(len(predictVals[i])):
            if(originalVals[i][j] != predictVals[i][j]):
                print(Fore.RED, end="")
            else:
                print(Fore.GREEN, end="")
            print(predictVals[i][j], end=", ")
            print(Style.RESET_ALL, end="")
        print()
        print()
    print('Dataset: ',dataset.metadata.name,",",k,'fold cross validation stratified results:')
    for i in range(k):
        print(i+1,") Errors: ",end="")

        percentage = round(relErr[i]*100,2)
        if percentage < th1:
            print(Fore.GREEN, end="")
        elif percentage < th2:
            print(Fore.YELLOW, end="")
        else:
            print(Fore.RED, end="")
        print(absErr[i],"/",valSetSize[i],end="")
        print(Style.RESET_ALL, end="")
        print("  Percentage Error: ",end="")
        if percentage < th1:
            print(Fore.GREEN, end="")
        elif percentage < th2:
            print(Fore.YELLOW, end="")
        else:
            print(Fore.RED, end="")
        print(percentage,"%")
        print(Style.RESET_ALL, end="")
    print()

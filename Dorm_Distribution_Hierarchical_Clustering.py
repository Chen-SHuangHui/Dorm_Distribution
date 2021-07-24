# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 11:11:50 2021

@author: Chen Shuanghui

This script can distribute sutudents based on their characteristics,
by the method of Hierarchical Clustering
"""

import argparse;
import pandas as pd;
import random;
from sklearn.cluster import AgglomerativeClustering

def createDataset(*seqs,num,seqnames=None):
    df = pd.DataFrame()
    for i in range(len(seqs)):
        feature = []
        for j in range(num):
            feature.append(random.choice(seqs[i]))
        df["feature" + str(i+1)] = feature
    if seqnames:
        df.columns = seqnames
    df.index = ["student%s" % x for x in range(1,num+1)]
    return df

def convertDtype(rawdf):
    df = rawdf.copy()
    for i in range(df.shape[1]):
        if df.iloc[:,i].dtype == "object":
            df.iloc[:,i] = pd.Categorical(df.iloc[:,i]).codes
    return df

def normalize(rawdf):
    df = rawdf.copy()
    for i in range(df.shape[1]):
        FeatureRange = max(df.iloc[:,i])- min(df.iloc[:,i])
        if FeatureRange == 0:
            df.iloc[:,i] = 0.5
        else:
            df.iloc[:,i] = (df.iloc[:,i]- min(df.iloc[:,i])) / FeatureRange
    return df

def weightFeature(rawdf,dt):
    df = rawdf.copy()
    for i in pd.Series(dt).index:
        df[i] = df[i]*dt[i]
    return df

def varianceComment(dataset,label):  #data: students' information; label: results of distribution
    dorm_IDs = label.Dorm_ID[label.Dorm_ID.duplicated()]
    errs = []
    for i in dorm_IDs:
        dorm = label.loc[label.Dorm_ID == i,:]
        students = dorm.Student_ID
        df = dataset.loc[students,:]
        err = sum(df.sem(axis=0))/df.shape[0]
        errs.append(err)
    return sum(errs)

# Clustering
def dormCluster(df,size): 
    if df.shape[0] % size == 0:
        dorm_num = df.shape[0] // size
    else:
        dorm_num = df.shape[0] // size + 1 
    ac = AgglomerativeClustering(n_clusters=dorm_num, affinity='euclidean', linkage='average')
    clustering = ac.fit(df.values)
    result = pd.DataFrame({"Student_ID":df.index,"Dorm_ID":clustering.labels_})
    return result

def randomDistribute(names,size):   #names: student ID; num: student number in a dorm
    l = []
    if len(names) % size == 0:
        dorm_num = len(names) // size
        dorm_IDs = list(range(0,dorm_num))*size
    else:
        dorm_num = len(names) // size + 1
        dorm_IDs = list(range(0,dorm_num - 1))*size + [dorm_num]*(len(names) % size)     
    while dorm_IDs:
        DID = random.choice(dorm_IDs)
        dorm_IDs.remove(DID)
        l.append(DID)
    result = pd.DataFrame({"Student_ID":names,"Dorm_ID":l})
    return result

def main():
    parser = argparse.ArgumentParser(description='Input the student information and dormitory size.')
    parser.add_argument("--input","-i", type=str, help='Student information file.')
    parser.add_argument("--out","-o", type=str, help='Prefix of distribution result file.')
    parser.add_argument("--size","-s", type=int, help='Dormitory size.')
    parser.add_argument("--weight","-w", required = False, type=str, \
                        help='Weight of features. Format: Feature1:num1;Feature2:num2... (no blank)')
    args = parser.parse_args()
    
    #size
    dorm_size = args.size
        
    #student information
    dataset = pd.read_csv(args.input);
    dataset = convertDtype(dataset)
     #weight
    if args.weight:
        weight_str = args.weight
        weight_dt = dict()
        for s in weight_str.split("/"):
            key = s.split(",")[0]
            value = eval(s.split(",")[1])
            weight_dt[key] = value
        dataset = weightFeature(normalize(dataset),weight_dt)
    else:
        dataset = normalize(dataset)
   
    #Clustering
    result = dormCluster(dataset,dorm_size)
    
    result.to_csv(args.out + ".csv")
    #for test
    #print(result) 
    print("Done.")
    
if __name__ == '__main__':
    main()
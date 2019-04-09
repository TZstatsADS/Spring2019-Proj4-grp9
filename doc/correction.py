# -*- coding: utf-8 -*-
"""
Correction sequence
"""

from pyxdameraulevenshtein import damerau_levenshtein_distance
from nltk.metrics.distance import edit_distance
import py_common_subseq.py_common_subseq as CS #Need to change xrange() as range()
import math
import numpy as np
import pandas as pd
import jieba
import copy
import os
import sys
sys.path.append("..")
from lib.functions import project4 as p4 

#==============================================================================
lexicon1 = set(pd.read_csv("../output/Detection_Group/group1.csv").dictionary)
lexicon2 = set(pd.read_csv("../output/Detection_Group/group2.csv").dictionary)
lexicon3 = set(pd.read_csv("../output/Detection_Group/group3.csv").dictionary)
lexicon4 = set(pd.read_csv("../output/Detection_Group/group4.csv").dictionary)
lexicon5 = set(pd.read_csv("../output/Detection_Group/group5.csv").dictionary)
#==============================================================================
dictionary = pd.read_csv("../output/onegram.csv")
Dictionary = dictionary.set_index('word').T.to_dict("index")['freq']
#==============================================================================
#five_gram_dictionary = pd.read_csv("../output/5-gram.csv")
#Five_gram_dictionary = five_gram_dictionary.set_index('5_gram').T.to_dict("index")['freq']
#==============================================================================
three_gram_dictionary = pd.read_csv("../output/3-gram.csv")
Three_gram_dictionary = three_gram_dictionary.set_index('3_gram').T.to_dict("index")['freq']
#==============================================================================
#five_gram_dictionary_c = pd.read_csv("../output/relaxed1.df.csv")
#Five_gram_dictionary_x = five_gram_dictionary_c.set_index('5_gram').T.to_dict("index")['freq']
#five_gram_dictionary_c = pd.read_csv("../output/relaxed2.df.csv")
#Five_gram_dictionary_x.update(five_gram_dictionary_c.set_index('5_gram').T.to_dict("index")['freq'])
#five_gram_dictionary_c = pd.read_csv("../output/relaxed3.df.csv")
#Five_gram_dictionary_x.update(five_gram_dictionary_c.set_index('5_gram').T.to_dict("index")['freq'])
#five_gram_dictionary_c = pd.read_csv("../output/relaxed4.df.csv")
#Five_gram_dictionary_x.update(five_gram_dictionary_c.set_index('5_gram').T.to_dict("index")['freq'])
#five_gram_dictionary_c = pd.read_csv("../output/relaxed5.df.csv")
#Five_gram_dictionary_x.update(five_gram_dictionary_c.set_index('5_gram').T.to_dict("index")['freq'])
#==============================================================================
three_gram_dictionary_c = pd.read_csv("../output/relaxed1.3g.df.csv")
Three_gram_dictionary_x = three_gram_dictionary_c.set_index('3_gram').T.to_dict("index")['freq']
three_gram_dictionary_c = pd.read_csv("../output/relaxed2.3g.df.csv")
Three_gram_dictionary_x.update(three_gram_dictionary_c.set_index('3_gram').T.to_dict("index")['freq'])
three_gram_dictionary_c = pd.read_csv("../output/relaxed3.3g.df.csv")
Three_gram_dictionary_x.update(three_gram_dictionary_c.set_index('3_gram').T.to_dict("index")['freq'])



#error_detection = pd.read_csv("../output/orc5.csv")
#Error_Detection = error_detection.loc[error_detection.TF==False]

Whole_text = pd.DataFrame()
dirpath = '../output/ForCorrection/forcorrection/'
for root, dirs, files in os.walk(dirpath):
    for file in files:
        file_context = pd.read_csv(dirpath+file)
        if i <= 9:
            file_context['Group'] = [1]*len(file_context)
        elif i <= 36:
            file_context['Group'] = [2]*len(file_context)
        elif i <= 39:
            file_context['Group'] = [3]*len(file_context)
        elif i <= 66:
            file_context['Group'] = [4]*len(file_context)
        elif i <= 96:
            file_context['Group'] = [5]*len(file_context)
        Whole_text = Whole_text.append(file_context)
     
        
        
#Error_Detection = Whole_text.loc[Whole_text.TF==False]
#Whole_text = pd.read_csv('../output/ForCorrection/forcorrection/group1_00000005.csv')
#Whole_text["Group"] = 1
training = Whole_text.sample(frac=0.8)
test = Whole_text.append(training)
test = test.drop_duplicates(keep=False)
training = training.loc[training.TF==False]

Output = pd.DataFrame()

for L in range(len(training)):
    print(L)
    We = training.iloc[L,5]
    output_piece = pd.DataFrame()
    for Threshold in range(20):
        Candidates = p4.candidate_search(Dictionary, We, Threshold)
        if len(Candidates) >= 10:
            n = 0
            Candidates2 = {}
            for key in Candidates:
                Candidates2[key] = Candidates[key]
                if n >= 9:
                    break
                else:
                    n = n+1
            break
    if len(Candidates) < 10:
        continue
            
        
    Candidates = Candidates2
   
    dist_score = p4.distance_score(Candidates, We, Threshold)
    a1=0.25
    a2=0.25
    a3=0.25
    a4=0.25
    simi_score = p4.similarity_score(Candidates, We, a1, a2, a3, a4)
    
    pop_score = p4.popularity_score(Candidates)
    
    if training.iloc[L,12] == 1:
        Lexicon = lexicon1
    elif training.iloc[L,12] == 2:
        Lexicon = lexicon2
    elif training.iloc[L,12] == 3:
        Lexicon = lexicon3
    elif training.iloc[L,12] == 4:
        Lexicon = lexicon4
    elif training.iloc[L,12] == 5:
        Lexicon = lexicon5

    exis_score = p4.existance_score(Candidates, Lexicon)



# =============================================================================
#     five_gram_e = Error_Detection.loc[Error_Detection.word==We]
#     five_gram_list = []
#     for i in range(4):
#         five_gram_list.append(five_gram_e.iloc[0,3+i])
#     five_gram_list.append(We)
#     for i in range(4):
#         five_gram_list.append(five_gram_e.iloc[0,7+i])
#     Five_Gram_E = []
#     for i in range(5):
#         five_gram_string = ' '.join(five_gram_list[i:i+5])
#         Five_Gram_E.append(five_gram_string)
# =============================================================================
    
#        three_gram_e = Error_Detection.loc[Error_Detection.word==We]
    three_gram_list = []
    for i in range(5):
        if pd.isnull(training.iloc[L,3+i]):
            three_gram_list.append(" ")
        else:
            three_gram_list.append(training.iloc[L,3+i])
        
#    three_gram_list.append(We)
#    for i in range(2):
#        three_gram_list.append(three_gram_e.iloc[0,7+i])
    Three_Gram_E = []
    for i in range(3):
        three_gram_string = ' '.join(three_gram_list[i:i+3])
        Three_Gram_E.append(three_gram_string)
                  
    
    exat_pop_score = p4.exact_popularity_score(Candidates, We, Three_Gram_E, Three_gram_dictionary)
            
    relax_pop_score = p4.relaxed_popularity_score(Candidates, We, Three_Gram_E, Three_gram_dictionary_x)
    
    output_piece["We"] = [We]*len(Candidates)
    output_piece["Wc"] = Candidates.keys()
    output_piece["x1"] = dist_score.values()
    output_piece["x2"] = simi_score.values()
    output_piece["x3"] = pop_score.values()
    output_piece["x4"] = exis_score.values()
    output_piece["x5"] = exat_pop_score.values()
    output_piece["x6"] = relax_pop_score.values()
    
    label_list = []
    for i in range(len(Candidates)):
        if output_piece.iloc[i,1] == training.iloc[L,10]:
            label_list.append(1)
        else:
            label_list.append(0)
    
    output_piece["lab"] = label_list
    Output = Output.append(output_piece)

weight = len(Output) / sum(Output.lab)
Output['Weight'] = Output['lab']*weight

from sklearn.ensemble import AdaBoostRegressor

m = AdaBoostRegressor()

X = Output.drop(["We","Wc", "lab", "Weight"],axis=1)
y = Output.lab
m.fit(X,y,sample_weight=Output.Weight)


for L in range(len(test)):
    if test.iloc[L,11] == False:
        We = test.iloc[L,5]
        output_piece = pd.DataFrame()
        for Threshold in range(20):
            Candidates = p4.candidate_search(Dictionary, We, Threshold)
            if len(Candidates) >= 10:
                break

       
        dist_score = p4.distance_score(Candidates, We, Threshold)
        
        a1=0.25
        a2=0.25
        a3=0.25
        a4=0.25
        simi_score = p4.similarity_score(Candidates, We, a1, a2, a3, a4)
        
        pop_score = p4.popularity_score(Candidates)
        
        if training.iloc[L,12] == 1:
            Lexicon = lexicon1
        elif training.iloc[L,12] == 2:
            Lexicon = lexicon2
        elif training.iloc[L,12] == 3:
            Lexicon = lexicon3
        elif training.iloc[L,12] == 4:
            Lexicon = lexicon4
        elif training.iloc[L,12] == 5:
            Lexicon = lexicon5
        
        exis_score = p4.existance_score(Candidates, Lexicon)
    
    
    
    # =============================================================================
    #     five_gram_e = Error_Detection.loc[Error_Detection.word==We]
    #     five_gram_list = []
    #     for i in range(4):
    #         five_gram_list.append(five_gram_e.iloc[0,3+i])
    #     five_gram_list.append(We)
    #     for i in range(4):
    #         five_gram_list.append(five_gram_e.iloc[0,7+i])
    #     Five_Gram_E = []
    #     for i in range(5):
    #         five_gram_string = ' '.join(five_gram_list[i:i+5])
    #         Five_Gram_E.append(five_gram_string)
    # =============================================================================
        
    #        three_gram_e = Error_Detection.loc[Error_Detection.word==We]
        three_gram_list = []
        for i in range(5):
            if pd.isnull(training.iloc[L,3+i]):
                three_gram_list.append(" ")
            else:
                three_gram_list.append(training.iloc[L,3+i])
    #    three_gram_list.append(We)
    #    for i in range(2):
    #        three_gram_list.append(three_gram_e.iloc[0,7+i])
        Three_Gram_E = []
        for i in range(3):
            three_gram_string = ' '.join(three_gram_list[i:i+3])
            Three_Gram_E.append(three_gram_string)
                      
        
        exat_pop_score = p4.exact_popularity_score(Candidates, We, Three_Gram_E, Three_gram_dictionary)
                
        relax_pop_score = p4.relaxed_popularity_score(Candidates, We, Three_Gram_E, Three_gram_dictionary_x)
        
        output_piece["We"] = [We]*len(Candidates)
        output_piece["Wc"] = Candidates.keys()
        output_piece["x1"] = dist_score.values()
        output_piece["x2"] = simi_score.values()
        output_piece["x3"] = pop_score.values()
        output_piece["x4"] = exis_score.values()
        output_piece["x5"] = exat_pop_score.values()
        output_piece["x6"] = relax_pop_score.values()
        
        label_list = []
        for i in range(len(Candidates)):
            if output_piece.iloc[i,1] == test.iloc[L,10]:
                label_list.append(1)
            else:
                label_list.append(0)
        
        output_piece["lab"] = label_list
        
        X_pred = output_piece.drop(["We","Wc", "lab"],axis=1)
    
        redict_piece = m.predict(X_pred).tolist()
        redict_piece.index(max(redict_piece))
        output_piece["pred"] = m.predict(X_pred)
        output_piece.pred.index(max(output_piece.pred))


#------------------------------

# =============================================================================
# import numpy as np
# import os
# from collections import Counter
# import re, string
# from string import digits
# 
# #file_object = open('../data/ground_truth/group1_00000005.txt')
# 
# TEXT = str()
# dirpath = '../data/ground_truth/'
# for root, dirs, files in os.walk(dirpath):
#     for file in files:
#         file_object = open(dirpath+file)
#         try:
#             file_context = file_object.read()
#         finally:
#             file_object.close()
#         text = file_context.lower()
#         exclude = set(string.punctuation)
#         text = ''.join(ch for ch in text if ch not in exclude)
#         remove_digits = str.maketrans('', '', digits)
#         text = text.translate(remove_digits)
# 
#         TEXT = TEXT + text
# 
# 
# data_ = jieba.cut(TEXT)
# data = dict(Counter(data_))
# 
# data['rch']
# =============================================================================
# =============================================================================
# We = "eve"
# Wc = "King"
# test = ["apple bee car dog eve", "bee car dog eve fat", "car dog eve fat get", "dog eve fat get hi", "eve fat get hi ill"]
# grams_e = []
# grams_C_X = []
# for grams in test:
#     grams_e.append(grams.replace(We, Wc, 1))
# for i in range(5):
#     five_gram_s = list(jieba.cut(grams_e[i]))
#     for k in range(5):
#         if -i+4==k:
#             continue
#         else:
#             five_gram_s_copy = copy.deepcopy(five_gram_s)
#             five_gram_s_copy[2*k] = "*"
#             gram_c_x = "".join(five_gram_s_copy)
#             grams_C_X.append(gram_c_x)
#             
# print(grams_C_X)
# =============================================================================
# Feature
Feature = pd.DataFrame()
Feature["We"] = ["rah"]*10 + ["abc"]*10
Feature["Y"] = [0]*6+[1]+[0]*8+[1]+[0]*4
We_df = pd.DataFrame()
length = len(Candidates)
We_df["We"] = [We]*10

We_df["dist_score"] = 
# resample to balance label 1 and label 0
label_1 = Feature.loc[Feature.Y == 1]
for i in range(2):
    label_1 = label_1.append(label_1)
Feature = Freature.append(label_1)


CS.find_common_subsequences("qweert", "qwwert")

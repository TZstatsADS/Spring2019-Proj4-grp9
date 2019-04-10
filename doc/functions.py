# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:18:26 2018

@author: Chenghao
"""
from pyxdameraulevenshtein import damerau_levenshtein_distance
from nltk.metrics.distance import edit_distance
import math
import pandas as pd
import jieba
import copy


# Candidate search


def candidate_search(Dictionary, We, threshold):
    candidate = {}
    for Wc in Dictionary:
        dist = damerau_levenshtein_distance(Wc, We)
        if dist <= threshold:
            candidate[Wc] = Dictionary[Wc]

    return(candidate)

# --------------------------------------
# Feature scoring
# Levenshtein edit distance


def distance_score(candidates, We, threshold):
    Score = {}
    for Wc in candidates:
        score = 1 - edit_distance(Wc, We, substitution_cost=1,
                                  transpositions=False)/(threshold + 1)
        Score[Wc] = score

    return(Score)

# String similarity
def similarity_score(candidates, We, a1=0.25, a2=0.25, a3=0.25, a4=0.25):
    Score = {}
    for Wc in candidates:
        common_subsequences = find_common_subsequences(Wc, We)
        lcs = sorted(common_subsequences, key=lambda x: len(x))[-1]

        IniLetter = We[0]
        EndLetter = We[-1]
        MidLetter = We[math.ceil(len(We)/2)]

            # LCS_1
        common_subseq_IntLetter = set([])
        for W in common_subsequences:
            if W.startswith(IniLetter):
                common_subseq_IntLetter.add(W)

        if len(common_subseq_IntLetter) == 0:
            lcs1 = ''
        else:
            lcs1 = sorted(common_subseq_IntLetter, key=lambda x: len(x))[-1]

            # LCS_z
        common_subseq_EndLetter = set([])
        for W in common_subsequences:
            if W.endswith(EndLetter):
                common_subseq_EndLetter.add(W)

        if len(common_subseq_EndLetter) == 0:
            lcsz = ''
        else:
            lcsz = sorted(common_subseq_EndLetter, key=lambda x: len(x))[-1]

            # LCS_n
        common_subseq_MidLetter = set([])
        for W in common_subsequences:
            if W.startswith(MidLetter):
                common_subseq_MidLetter.add(W)

        if len(common_subseq_MidLetter) == 0:
            lcsn = ''
        else:
            lcsn = sorted(common_subseq_MidLetter, key=lambda x: len(x))[-1]

        denom = len(Wc) + len(We)

            # original paper
        nlcs = (2*len(lcs))/denom
        nmnlcs1 = (2*len(lcs1))/denom
        nmnlcsn = (2*len(lcsn))/denom
        nmnlcsz = (2*len(lcsz))/denom
        score = a1*nlcs + a2*nmnlcs1 + a3*nmnlcsn + a4*nmnlcsz
        Score[Wc] = score

    return(Score)

# Language popularity
def popularity_score(candidates):
    Score = {}
    denom = max(candidates.values())
    for Wc in candidates:
        score = candidates[Wc]/denom
        Score[Wc] = score
        
    return(Score)

# Lexicon existance
def existance_score(candidates, lexicon): # lexicon is a set {candidate1, candidate2}
    Score = {}
    for Wc in candidates:
        score = int(Wc in lexicon)
        Score[Wc] = score
        
    return(Score)
        
# Exact-context popularity
def exact_popularity_score(candidates, We, five_gram_E, five_gram_dic): # five_gram_E is 5-gram with We, five_gram_dic is dictionary {5 gram w/ Wc: frequency}
    Score = {}
    Numer = {}
    for Wc in candidates:
        five_gram_C = []
        for five_gram in five_gram_E:
            five_gram_C.append(five_gram.replace(We, Wc, 1))
        numer = 0
        for five_gram_c in five_gram_C:
            if five_gram_c in five_gram_dic:
                numer = numer + five_gram_dic[five_gram_c]
            else:
                numer = numer
        Numer[Wc] = numer
# Here Denom may be zero, check it
# 5-gram is too lang to get the good result, should try 3-gram
    Denom = max(Numer.values())
    for Wc in candidates:
        if Denom == 0:
            Score[Wc] = 0
        else:
            Score[Wc] = Numer[Wc]/Denom
            
    return(Score)
        
# Relaxed-context popularity
def relaxed_popularity_score(candidates, We, five_gram_E, five_gram_dic_X):
    Score = {}
    Numer = {}
    for Wc in candidates:
        five_gram_C = []
        grams_C_X = []
        for five_gram in five_gram_E:
            five_gram_C.append(five_gram.replace(We, Wc, 1))
# 3-gram               
        for i in range(3):
            five_gram_s = list(jieba.cut(five_gram_C[i]))
            for k in range(3):
                if -i+2==k:
                    continue
                else:
                    five_gram_s_copy = copy.deepcopy(five_gram_s)
                    five_gram_s_copy[2*k] = "*"
                    gram_c_x = "".join(five_gram_s_copy)
                    grams_C_X.append(gram_c_x)
# 5-gram                        
# =============================================================================
        numer = 0
        for five_gram_c_x in grams_C_X:
            if five_gram_c_x in five_gram_dic_X:
                numer = numer + five_gram_dic_X[five_gram_c_x]
            else:
                numer = numer
    Numer[Wc] = numer
        
    Denom = max(Numer.values())
    for Wc in candidates:
        if Denom == 0:
            Score[Wc] = 0
        else:
            Score[Wc] = Numer[Wc]/Denom
        
    return(Score)
 
def find_common_subsequences(seq_1, seq_2, sep='', empty_val=''):

    if len(seq_1) < len(seq_2):
        new_seq_1 = seq_2
        seq_2 = seq_1
        seq_1 = new_seq_1

    seq_1_len = len(seq_1)
    seq_2_len = len(seq_2)
    seq_1_len_plus_1 = seq_1_len + 1
    seq_2_len_plus_1 = seq_2_len + 1
    

    subseq_last_row = [set([empty_val])] * seq_2_len_plus_1
    subseq_current_row = [set([empty_val])] + [set()] * seq_2_len

    for row in range(1, seq_1_len_plus_1):

        for col in range(1, seq_2_len_plus_1):

            if seq_1[row-1] == seq_2[col-1]:
                diagonal_cell_value = subseq_last_row[col - 1]
                matched_element = seq_1[row-1]
                new_cell_value = add_matched_element(matched_element,
                    diagonal_cell_value, sep)
            else:
                above_set = subseq_last_row[col]
                left_set = subseq_current_row[col-1]
                new_cell_value = above_set.union(left_set)
            subseq_current_row[col] = new_cell_value

        subseq_last_row = subseq_current_row
        subseq_current_row = [set([empty_val])] + [set()] * seq_2_len

    return subseq_last_row[seq_2_len]

def add_matched_element(element, target_set, sep):
    new_elements = map(lambda x: x + sep + element, target_set)
    return target_set.union(new_elements)

def three_gram(error_text_row):
    three_gram_list = []
    for i in range(5):
        if pd.isnull(error_text_row[3+i]):
            three_gram_list.append(" ")
        else:
            three_gram_list.append(error_text_row[3+i])

    three_grams = []
    for i in range(3):
        three_gram_string = ' '.join(three_gram_list[i:i+3])
        three_grams.append(three_gram_string)
        
    return(three_grams)

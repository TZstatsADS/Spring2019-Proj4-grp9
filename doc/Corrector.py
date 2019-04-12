import os
import numpy as np
import pandas as pd
import math
from pyxdameraulevenshtein import damerau_levenshtein_distance
from nltk.metrics.distance import edit_distance


class Corrector():

    def __init__(self):
        # working directory of 'Spring2019-Proj4-grp9'
        pwd = os.path.dirname(os.getcwd())
        # set the working directory of processed data: Correction
        Correction_wd = os.path.join(pwd, "output", "Correction")

        self.whole_text = pd.read_pickle(os.path.join(Correction_wd, 'whole_text.pkl'))
        self.error_text = self.whole_text.loc[self.whole_text.SAME == False].drop_duplicates(keep=False)

        self.lexicon = np.load(os.path.join(Correction_wd, 'lexicon.npy')).item()
        self.dictionary = np.load(os.path.join(Correction_wd, 'dictionary.npy')).item()
        self.dictionary_exact = np.load(os.path.join(Correction_wd, 'dictionary_exact.npy')).item()
        self.dictionary_relaxed = np.load(os.path.join(Correction_wd, 'dictionary_relaxed.npy')).item()

    def candidate_search(self, We, threshold):
        candidate = {}
        for Wc in self.dictionary:
            dist = damerau_levenshtein_distance(Wc, We)
            if dist <= threshold:
                candidate[Wc] = self.dictionary[Wc]

        return (candidate)

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

            if seq_1[row - 1] == seq_2[col - 1]:
                diagonal_cell_value = subseq_last_row[col - 1]
                matched_element = seq_1[row - 1]
                new_cell_value = add_matched_element(matched_element,
                                                     diagonal_cell_value, sep)
            else:
                above_set = subseq_last_row[col]
                left_set = subseq_current_row[col - 1]
                new_cell_value = above_set.union(left_set)
            subseq_current_row[col] = new_cell_value

        subseq_last_row = subseq_current_row
        subseq_current_row = [set([empty_val])] + [set()] * seq_2_len

    return subseq_last_row[seq_2_len]

def add_matched_element(element, target_set, sep):
    new_elements = map(lambda x: x + sep + element, target_set)
    return target_set.union(new_elements)

def three_gram(error_text_row):
    three_grams = []
    for i in range(2, 5):
        three_gram_string = ' '.join(error_text_row[i:i + 3])
        three_grams.append(three_gram_string)
    return (three_grams)

def distance_score(candidates, We, threshold):
    Score = {}
    for Wc in candidates:
        score = 1 - edit_distance(Wc, We, substitution_cost=1,
                                  transpositions=False) / (threshold + 1)
        Score[Wc] = score
    return (Score)

# String similarity
def similarity_score(candidates, We, a1=0.25, a2=0.25, a3=0.25, a4=0.25):
    Score = {}
    for Wc in candidates:
        common_subsequences = find_common_subsequences(Wc, We)
        lcs = sorted(common_subsequences, key=lambda x: len(x))[-1]

        IniLetter = We[0]
        EndLetter = We[-1]
        MidLetter = We[math.ceil(len(We) / 2)]

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
        nlcs = (2 * len(lcs)) / denom
        nmnlcs1 = (2 * len(lcs1)) / denom
        nmnlcsn = (2 * len(lcsn)) / denom
        nmnlcsz = (2 * len(lcsz)) / denom
        score = a1 * nlcs + a2 * nmnlcs1 + a3 * nmnlcsn + a4 * nmnlcsz
        Score[Wc] = score

    return (Score)

# Language popularity
def popularity_score(candidates):
    Score = {}
    try:
        denom = max(candidates.values())
        for Wc in candidates:
            score = candidates[Wc] / denom
            Score[Wc] = score
    except:
        Score["error"]=0
        print("Catched")

    return (Score)


# Lexicon existance
def existance_score(candidates, lexicon):  # lexicon is a set {candidate1, candidate2}
    Score = {}
    for Wc in candidates:
        score = int(Wc in lexicon)
        Score[Wc] = score
    return (Score)


# Exact-context popularity
def exact_popularity_score(candidates, We, Ge, G_dict):
    Score = {}

    Numerator = {}
    for Wc in candidates:
        numerator = 0
        Gc = [grams.replace(We, Wc, 1) for grams in Ge]
        for grams in Gc:
            if grams in G_dict:
                numerator += G_dict[grams]
        Numerator[Wc] = numerator

    Denominator = max(Numerator.values())
    for Wc in candidates:
        if Denominator == 0:
            Score[Wc] = 0
        else:
            Score[Wc] = Numerator[Wc] / Denominator

    return (Score)


# Relaxed-context popularity
def relaxed_popularity_score(candidates, We, Ge, G_dict):
    Score = {}
    Numerator = {}
    for Wc in candidates:
        Gc = [grams.replace(We, Wc, 1) for grams in Ge]
        Gc_relaxed = []
        for gram in Gc:
            split = gram.split()
            splitted = gram.split()
            for i in range(3):
                if splitted[i] != Wc:
                    splitted[i] = "*"
                    Gc_relaxed.append(" ".join(splitted))
                    splitted = split
        numerator = 0
        for grams in Gc_relaxed:
            if grams in G_dict:
                numerator += G_dict[grams]
        Numerator[Wc] = numerator

    Denominator = max(Numerator.values())
    for Wc in candidates:
        if Denominator == 0:
            Score[Wc] = 0
        else:
            Score[Wc] = Numerator[Wc] / Denominator

    return (Score)

#!/usr/bin/env python
#_*_coding:utf-8_*_


"""
question similarity features
"""

import numpy as np



def pre_process(word):
    """ remove the tail signal """
    word = word.strip()
    if word and word[-1].lower() not in "abcdefghijklmnopqrstuvwxyz0123456789":
        word = word[0:-1]
    if word and word[0].lower() not in "abcdefghijklmnopqrstuvwxyz0123456789":
        word = word[1:]
    return word


def run(string1, string2):
    """ main function """
    words1 = [pre_process(word) for word in string1.split()]
    words2 = [pre_process(word) for word in string2.split()]
    return sim_by_ed(words1, words2), sim_by_lcs(words1, words2), sim_by_interrogative(words1,
            words2)
    """
    print(string1, "--", string2)
    print("ed: ", sim_by_ed(words1, words2))
    print("lcs: ", sim_by_lcs(words1, words2))
    """


def sim_by_ed(words1, words2):
    """ edit distane """
    state = np.zeros([len(words1) + 1, len(words2) + 1])
    
    for i in range(len(words1) + 1):
        state[i][0] = i

    for j in range(len(words2) + 1):
        state[0][j] = j

    for i in range(len(words1)):
        for j in range(len(words2)):
            sig = 0 if words1[i] == words2[j] else 1
            state[i+1][j+1] = min(state[i+1][j] + 1, state[i][j+1] + 1, state[i][j] + sig)
    edit = state[len(words1)][len(words2)]
    return 1 - float(edit) / max(len(words1), len(words2))


def sim_by_lcs(words1, words2):
    """ edit distane """
    state = np.zeros([len(words1) + 1, len(words2) + 1])

    for i in range(len(words1)):
        for j in range(len(words2)):
            sig = 1 if words1[i] == words2[j] else 0
            state[i+1][j+1] = max(state[i+1][j], state[i][j+1], state[i][j] + sig)
    lcs = state[len(words1)][len(words2)]
    return 2 * float(lcs) / (len(words1) + len(words2))


def sim_by_interrogative(words1, words2):
    """ interrogative feature. 1 if both has else 0 """
    uni_words = ['what', 'whi', 'which', 'how', 'where', 'when']
    f = list()
    for i, word in enumerate(uni_words):
        for j in range(i, len(uni_words)):
            if (words1.count(word) > 0 and words2.count(uni_words[j]) > 0) or \
                    (words2.count(word) > 0 and words1.count(uni_words[j]) > 0): 
                f.append(1)
            else:
                f.append(0)
    return f;



if __name__ == '__main__':
    print(run('what are you doing', 'what are you doing'))

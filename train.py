#!/usr/bin/env python
#_*_coding:utf-8_*_


"""
data parse and train
"""

import similarity
import csv

def run1(txt):
    """ parse, feature, train """
    for line in open(txt):
        line = line.strip()
        label, ques1, ques2 = line.split('\t')
        ed, lcs = similarity.run(ques1, ques2)
        final_score = 0.5 * ed + 0.5 * lcs
        flag = 0
        if final_score > 0.5:
            flag = 1
        print("\t".join([label, ques1, ques2, str(ed), str(lcs), str(final_score), str(flag)]))


def run(txt):
    """ parse, feature, train """
    csvfile = open(txt, 'r')
    reader = csv.reader(csvfile)
    for data in reader:
        if reader.line_num == 1:
            continue
        label, ques1, ques2 = data
        ed, lcs = similarity.run(ques1, ques2)
        print(",".join([label, str(final_score)]))


if __name__ == '__main__':
    #run1('Quora-QP_dev_new.txt')
    run('com/test.csv')

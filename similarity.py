#!/usr/bin/env python
#_*_coding:utf-8_*_


"""
question similarity features
"""

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re



class TextPreProcessor(object):

    _stemmer = SnowballStemmer('english')

    def __init__(self):
        pass

    @staticmethod
    def clean_text(text):
        """
        Clean text
        :param text: the string of text
        :return: text string after cleaning
        """
        # unit
        text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)        # e.g. 4kgs => 4 kg
        text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)         # e.g. 4kg => 4 kg
        text = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', text)          # e.g. 4k => 4000
        text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
        text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)

        # acronym
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"what\'s", "what is", text)
        text = re.sub(r"What\'s", "what is", text)
        text = re.sub(r"\'ve ", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"i\'m", "i am ", text)
        text = re.sub(r"I\'m", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"c\+\+", "cplusplus", text)
        text = re.sub(r"c \+\+", "cplusplus", text)
        text = re.sub(r"c \+ \+", "cplusplus", text)
        text = re.sub(r"c#", "csharp", text)
        text = re.sub(r"f#", "fsharp", text)
        text = re.sub(r"g#", "gsharp", text)
        text = re.sub(r" e mail ", " email ", text)
        text = re.sub(r" e \- mail ", " email ", text)
        text = re.sub(r" e\-mail ", " email ", text)
        text = re.sub(r",000", '000', text)
        text = re.sub(r"\'s", " ", text)

        # spelling correction
        text = re.sub(r"ph\.d", "phd", text)
        text = re.sub(r"PhD", "phd", text)
        text = re.sub(r"pokemons", "pokemon", text)
        text = re.sub(r"pokémon", "pokemon", text)
        text = re.sub(r"pokemon go ", "pokemon-go ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" 9 11 ", " 911 ", text)
        text = re.sub(r" j k ", " jk ", text)
        text = re.sub(r" fb ", " facebook ", text)
        text = re.sub(r"facebooks", " facebook ", text)
        text = re.sub(r"facebooking", " facebook ", text)
        text = re.sub(r"insidefacebook", "inside facebook", text)
        text = re.sub(r"donald trump", "trump", text)
        text = re.sub(r"the big bang", "big-bang", text)
        text = re.sub(r"the european union", "eu", text)
        text = re.sub(r" usa ", " america ", text)
        text = re.sub(r" us ", " america ", text)
        text = re.sub(r" u s ", " america ", text)
        text = re.sub(r" U\.S\. ", " america ", text)
        text = re.sub(r" US ", " america ", text)
        text = re.sub(r" American ", " america ", text)
        text = re.sub(r" America ", " america ", text)
        text = re.sub(r" quaro ", " quora ", text)
        text = re.sub(r" mbp ", " macbook-pro ", text)
        text = re.sub(r" mac ", " macbook ", text)
        text = re.sub(r"macbook pro", "macbook-pro", text)
        text = re.sub(r"macbook-pros", "macbook-pro", text)
        text = re.sub(r" 1 ", " one ", text)
        text = re.sub(r" 2 ", " two ", text)
        text = re.sub(r" 3 ", " three ", text)
        text = re.sub(r" 4 ", " four ", text)
        text = re.sub(r" 5 ", " five ", text)
        text = re.sub(r" 6 ", " six ", text)
        text = re.sub(r" 7 ", " seven ", text)
        text = re.sub(r" 8 ", " eight ", text)
        text = re.sub(r" 9 ", " nine ", text)
        text = re.sub(r"googling", " google ", text)
        text = re.sub(r"googled", " google ", text)
        text = re.sub(r"googleable", " google ", text)
        text = re.sub(r"googles", " google ", text)
        text = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"the european union", " eu ", text)
        text = re.sub(r"dollars", " dollar ", text)

        # punctuation
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"/", " / ", text)
        text = re.sub(r"\\", " \ ", text)
        text = re.sub(r"=", " = ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"\.", " . ", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\"", " \" ", text)
        text = re.sub(r"&", " & ", text)
        text = re.sub(r"\|", " | ", text)
        text = re.sub(r";", " ; ", text)
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ( ", text)

        # symbol replacement
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r"₹", " rs ", text)      # 测试！
        text = re.sub(r"\$", " dollar ", text)

        # remove extra space
        text = ' '.join(text.split())

        return text

    @staticmethod
    def stem(df):
        """
        Process the text data with SnowballStemmer
        :param df: dataframe of original data
        :return: dataframe after stemming
        """
        df['question1'] = df.question1.map(lambda x: ' '.join(
            [TextPreProcessor._stemmer.stem(word) for word in
             nltk.word_tokenize(TextPreProcessor.clean_text(str(x).lower()))]))
        df['question2'] = df.question2.map(lambda x: ' '.join(
            [TextPreProcessor._stemmer.stem(word) for word in
             nltk.word_tokenize(TextPreProcessor.clean_text(str(x).lower()))]))
        return df


class PowerfulWord(object):
    @staticmethod
    def load_powerful_word(fp):
        powful_word = []
        f = open(fp, 'r')
        for line in f:
            subs = line.split('\t')
            word = subs[0]
            stats = [float(num) for num in subs[1:]]
            powful_word.append((word, stats))
        f.close()
        return powful_word

    @staticmethod
    def generate_powerful_word(data, subset_indexs):
        """
        计算数据中词语的影响力，格式如下：
            词语 --> [0. 出现语句对数量，1. 出现语句对比例，2. 正确语句对比例，3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. 双侧语句对正确比例]
        """
        words_power = {}
        train_subset_data = data.iloc[subset_indexs, :]
        for index, row in train_subset_data.iterrows():
            label = int(row['is_duplicate'])
            q1_words = str(row['question1']).lower().split()
            q2_words = str(row['question2']).lower().split()
            all_words = set(q1_words + q2_words)
            q1_words = set(q1_words)
            q2_words = set(q2_words)
            for word in all_words:
                if word not in words_power:
                    words_power[word] = [0. for i in range(7)]
                # 计算出现语句对数量
                words_power[word][0] += 1.
                words_power[word][1] += 1.

                if ((word in q1_words) and (word not in q2_words)) or ((word not in q1_words) and (word in q2_words)):
                    # 计算单侧语句数量
                    words_power[word][3] += 1.
                    if 0 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算单侧语句正确比例
                        words_power[word][4] += 1.
                if (word in q1_words) and (word in q2_words):
                    # 计算双侧语句数量
                    words_power[word][5] += 1.
                    if 1 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算双侧语句正确比例
                        words_power[word][6] += 1.
        for word in words_power:
            # 计算出现语句对比例
            words_power[word][1] /= len(subset_indexs)
            # 计算正确语句对比例
            words_power[word][2] /= words_power[word][0]
            # 计算单侧语句对正确比例
            if words_power[word][3] > 1e-6:
                words_power[word][4] /= words_power[word][3]
            # 计算单侧语句对比例
            words_power[word][3] /= words_power[word][0]
            # 计算双侧语句对正确比例
            if words_power[word][5] > 1e-6:
                words_power[word][6] /= words_power[word][5]
            # 计算双侧语句对比例
            words_power[word][5] /= words_power[word][0]
        sorted_words_power = sorted(words_power.items(), key=lambda d: d[1][0], reverse=True)
        return sorted_words_power

    @staticmethod
    def save_powerful_word(words_power, fp):
        f = open(fp, 'w')
        for ele in words_power:
            f.write("%s" % ele[0])
            for num in ele[1]:
                f.write("\t%.5f" % num)
            f.write("\n")
        f.close()

def pre_process(word):
    """ remove the tail signal """
    word = word.strip()
    if word and word[-1].lower() not in "abcdefghijklmnopqrstuvwxyz0123456789":
        word = word[0:-1]
    if word and word[0].lower() not in "abcdefghijklmnopqrstuvwxyz0123456789":
        word = word[1:]
    return word


def feature(string1, string2):
    """ main function """
    words1 = [pre_process(word.lower()) for word in string1.split()]
    words2 = [pre_process(word.lower()) for word in string2.split()]
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
    # print(feature('what are you doing', 'what are you doing'))
    # load train pandas csv
    data = pd.read_csv('../com/train.csv')
    data = TextPreProcessor.stem(data)
    subindex = np.arange(data.shape[0])
    words_power = PowerfulWord.generate_powerful_word(data, subindex)
    PowerfulWord.save_powerful_word(words_power, "power")

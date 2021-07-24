import os
import pandas as pd
import numpy as np
import re
class preProcess:
    def __init__(self):
        self.path = '../data/Big5/'
        self.data_path ='../data/input/'
        self.data = pd.DataFrame(self.getData())
        self.data.columns = ['AUTHID', 'STATUS']
        self.users = set(self.data['AUTHID'])
        self.nrc = pd.read_csv("../data/personality-detection-my-copy/Emotion_Lexicon.csv")
        self.nrc = self.nrc.iloc[:, 0:12]

    def getData(self):
        # return test data
        docs = []
        for r, d, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(self.path+file, 'r') as f:
                        for line in f:
                            div_line = line.strip().split(',', 1)
                            if len(div_line) > 1:
                                # if users val not null and username startswith @
                                if(div_line[0]!='AUTHID' and div_line[0].startswith('@')):
                                    docs.append(div_line)
        return docs

    def prepareData(self):

        # generate input data matrix using nrc words
        for user in self.users:
            ref_data = self.data[self.data.AUTHID == user]
            statuses = ref_data['STATUS']
            # print(statuses)`
            matrix_columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust', 'ext', 'neu', 'agr', 'con', 'opn']
            matrix = list()
            matrix_indv = [0] * 10
            # remove the unncecessary data & tokenize the word of the status
            words = list(statuses.apply(lambda x: re.sub(r'[^\w\s]', '', x).lower().split()))
            for word in words:
                # print(word)
                for indv_word in word:
                    status_nrc = self.nrc.loc[self.nrc.Words == indv_word]
                    for indx, row in status_nrc.iterrows():
                        if indx > 0:
                            temp_arr = np.array(row[1:11].values.tolist()).astype(int)
                            for i, val in enumerate(temp_arr):
                                if val == 1 and i < 10:
                                    # print(temp_arr[i])
                                    matrix_indv[i] += 1
                    matrix.append(matrix_indv)
            df = pd.DataFrame(matrix)

            # store the users data matrix to a csv file
            df.to_csv(self.data_path+re.sub(r'[^\w\s0-9]', '', user)+'.csv', header=None, index=False)

x = preProcess()
x.prepareData()
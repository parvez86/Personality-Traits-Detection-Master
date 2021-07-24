import re
import string
import pandas as pd
from nltk.corpus import stopwords
import numpy as np


class trainData:
    def __init__(self):
        self.stop_words = list(set(stopwords.words("english")))
        self.words = list()     # store the train dataset words
        self.nrc_dict = dict()  # store the nrc value of the presented words
        self.data = pd.DataFrame()
        self.nrc_path = "../data/personality-detection-my-copy/Emotion_Lexicon.csv"
        self.data_path = '../data/myPersonality/mypersonality_final.csv'

    def getValues(self):
        # get the nrc data from the file
        nrc = pd.read_csv(self.nrc_path, header=None, index_col=False)
        nrc = nrc.iloc[:, 0:11] # remove the unnecessary column
        # print(nrc)

        # store the presented train datasets nrc words with associated values
        for index, row in nrc.iterrows():
            if index > 0:
                temp_arr = np.array(row[1:].values.tolist()).astype(int)
                self.nrc_dict[row[0]] = temp_arr
            else:
                self.nrc_dict[row[0]] = row[1:].values.tolist()
            self.words.append(row[0])

        # get the train dataset and remove unnecessary columns
        self.data = pd.read_csv(self.data_path)
        self.data = self.data.iloc[:, 0:12]
        self.data = self.clean_data(self.data)
        # print(self.data)

    def clean_data(self, data):
        data = data.iloc[:, 0: 12]
        data['cEXT'] = data['cEXT'].apply(lambda x: 1 if x == 'y' else 0)
        data['cNEU'] = data['cNEU'].apply(lambda x: 1 if x == 'y' else 0)
        data['cAGR'] = data['cAGR'].apply(lambda x: 1 if x == 'y' else 0)
        data['cCON'] = data['cCON'].apply(lambda x: 1 if x == 'y' else 0)
        data['cOPN'] = data['cOPN'].apply(lambda x: 1 if x == 'y' else 0)
        return data.infer_objects()

    def getStatusProcessed(self):
        status = list()

        for indx, row in self.data.iterrows():
            status_indv = row['STATUS']
            attr = list()

            # clean data
            status_indv = re.sub(r"(?:@\S*|#\S*|http(?=.*://)\S*)", "", status_indv.rsplit("\n")[0].lower())
            status_indv = status_indv.replace("rt", "").rsplit("\n")[0]

            for word in status_indv.translate(str.maketrans('', '', string.punctuation)).split():
                if word in self.words and word not in self.stop_words:
                    attr.append(self.nrc_dict[word])
            status.append([sum(status_indvd) for status_indvd in zip(*attr)])
            # status.append(list(map(sum, zip(*attr))))


        ## keep only english status, and clean the .csv file
        label_delete = [i for i, v in enumerate(status) if not v]
        self.data.drop(label_delete, inplace=True)

        # update train dataset
        self.data.to_csv("../data/trainData/trainV1.csv", index=False, header=False)

        mat = list()

        for indx, row in self.data.iterrows():
            mat.append(status[indx]+row[2:].values.tolist())

        # write to file
        pd.DataFrame(mat).to_csv("../data/trainData/trainV2.csv", index=False, header=False)


x = trainData()
x.getValues()
x.getStatusProcessed()






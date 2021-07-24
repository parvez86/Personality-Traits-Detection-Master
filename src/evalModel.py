import os
import pandas as pd
import numpy as np
from pickle import loads
from src.trainModel import trainModel

from sklearn.impute import SimpleImputer

class evalModel:
    def __init__(self):
        self.path = "../data/input/"
        self.root = "../data/predict/"
        # for traits
        self.name = list()
        # to store classification model
        self.label_model = dict()
        # store regression "Yes" model
        self.modelYes = dict()
        # store regression "No" model
        self.modelNo = dict()

    def getData(self):
        # return test data
        docs = []
        for r, d, f in os.walk(self.path):
            for files in f:
                if files.endswith(".csv"):
                    docs.append(files)
        return docs

    def getModel(self):
        model = trainModel()
        self.name = model.name

        # store the best models for each traits
        self.label_model = model.trainModelLabel()
        model.saveModel()

        # store the best models for yes category for each traits
        self.modelYes = model.modelYes
        # store the best models for yes category for each traits
        self.modelNo = model.modelNo

    # apply trained model on test dataset
    def getTrainedData(self, user):
        # store the data file
        path = self.path + user
        df = pd.read_csv(path)
        # print(df.dtypes)

        # for missing values
        imputer = SimpleImputer(strategy='median')
        imputer.fit(df)
        df = pd.DataFrame(imputer.transform(df)).astype('int64')
        # print(df.dtypes)


        prediction = list()
        # pred = dict()
        for i, item in enumerate(self.name):
            pred = loads(self.label_model[item]).predict(df)
            prediction.append(pred)

        matrix = pd.concat([pd.DataFrame(df), pd.DataFrame(prediction).T], axis=1)
        matrix.columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust', 'ext', 'neu', 'agr', 'con', 'opn']

        return matrix

    def getRegressed(self, mat,  trait, status):

       sample = mat[mat[trait] == status]


       if sample.empty:
           return [0]
       else:
           sample = sample.iloc[:, 0:10]
           if status == 1:
               pred = loads(self.modelYes[trait]).predict(sample).tolist()
               return pred
           else:
               pred = loads(self.modelNo[trait]).predict(sample).tolist()
               return pred

    def getRated(self):
        self.getModel()
        data = self.getData()
        f = open('../data/predict/output.txt', 'a')
        for file in data:
            f.write("Processing classification validation user data:%s\n" % file)
            print("Processing classification validation user data:", file)
            matrix = self.getTrainedData(file)
            # matrix.to_csv('../data/predict/'+file)
            # store the best score
            score_dict = dict()
            # score_dict2 = dict()
            f.write("Processing regression validation user data:%s\n" % (file))
            print("Processing regression validation user data: ", file)
            for each in self.name:
                pred1 = self.getRegressed(matrix, each, 1)
                pred2 = self.getRegressed(matrix, each, 0)
                score = (np.nanmean(pred1)*len(pred1)+np.nanmean(pred2)*len(pred2))/(len(pred1) + len(pred2))
                score_dict[each] = score

            for k, v in score_dict.items():
                f.write('%s: %s\n' % (k, v))
                print(k, ":", v)


x = evalModel()

x.getRated()

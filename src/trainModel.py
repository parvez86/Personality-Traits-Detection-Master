import pandas as pd
from pickle import dumps
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

class trainModel:
    def __init__(self):
        self.data_path = "../data/trainData/trainV2.csv"
        self.data = pd.read_csv(self.data_path)
        self.model_name = {'KNN': 'K Nearest Neighbor', 'DT': 'Decision Tree', 'RF': 'Random Forest',
                           'GB': 'Gradient Boosting', 'SGD': 'Stochastic Gradient', 'ADB': 'AdaBoost',
                           'SVC': 'Support Vector Machine', 'SVR': 'Support Vector Machine'}
        self.label_col = [15, 16, 17, 18, 19]
        self.score_col = [10, 11, 12, 13, 14]
        self.train_col = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.modelYes = dict()
        self.modelNo = dict()
        self.name = ["ext", "neu", "agr", "con", "opn"]
        f = open('../data/predict/output.txt', 'w')
        f.close()
        ## the labeled traits are [cEXT, cNEU, cAGR, cCON, cOPN] -> [15, 16, 17, 18, 19]
        ## the scored traits are [sEXT, sNEU, sAGR, sCON, sOPN] -> [10, 11, 12, 13, 14]


    ################# Using Classifier Model for finding best trait ###############
    print("Classifier Model for finding best classification model for trait: ")

    def trainModelLabel(self):
        sample = self.data.iloc[:, self.train_col]
        # print(sample.shape)
        # best model for each category
        score = dict()
        f = open('../data/predict/output.txt', 'a')
        f.write("Classifier Model for finding best classification model trait: \n")
        for trait in self.label_col:
            result = dict()
            models = dict()
            label = self.data.iloc[:, trait]
            f.write("Processing traits: %s\n" % self.name[self.label_col.index(trait)])
            print("Processing traits: ", self.name[self.label_col.index(trait)])

            ##########################################################
            # K Nearest Neighbors Classifier
            clf = KNeighborsClassifier(n_neighbors=25)
            clf.fit(sample, label)
            scores = cross_val_score(clf, sample, label, cv=5)
            f.write("KNN Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
            print("KNN Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            # dumps return the value in form of byte-object
            result["KNN"] = scores.mean()
            models["KNN"] = dumps(clf)

            ############################################################
            # Decision Tree Classifier
            clf = DecisionTreeClassifier(criterion="entropy")
            clf.fit(sample, label)
            scores = cross_val_score(clf, sample, label, cv=5)
            f.write("Decision Tree Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
            print("Decision Tree Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            result["DT"] = scores.mean()
            models["DT"] = dumps(clf)

            ##########################################################
            # Random Forest Classifier
            clf = RandomForestClassifier(criterion='entropy', n_estimators=100)
            clf.fit(sample, label)

            scores = cross_val_score(clf, sample, label, cv=5)
            f.write("Random Forest Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
            print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            result["RF"] = scores.mean()
            models["RF"] = dumps(clf)

            ###########################################################
            # AdaBoost Classifier
            clf = AdaBoostClassifier(n_estimators=100)
            clf.fit(sample, label)

            scores = cross_val_score(clf, sample, label, cv=5)
            f.write("AdaBoost Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
            print("AdaBoost Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            result["ADB"] = scores.mean()
            models["ADB"] = dumps(clf)

            ############################################################
            # Gradient Boosting Classifier
            clf = GradientBoostingClassifier(criterion="mse", n_estimators=200)
            clf.fit(sample, label)

            scores = cross_val_score(clf, sample, label, cv=5)
            f.write("Gradient Boosting Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
            print("Gradient Boosting Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            result["GB"] = scores.mean()
            models["GB"] = dumps(clf)

            ##########################################################
            # Stochastic Gradient Descent Classifier
            clf = SGDClassifier(loss="log", penalty="elasticnet")
            clf.fit(sample, label)

            scores = cross_val_score(clf, sample, label, cv=5)
            f.write("Stochastic Gradient Descent(SGD) Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
            print("SGD Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            # dumps return the value in form of byte-object
            result["SGD"] = scores.mean()
            models["SGD"] = dumps(clf)

            ############################################################
            # Support Vector Machine
            clf = SVC(kernel='linear')
            clf.fit(sample, label)

            scores = cross_val_score(clf, sample, label, cv=5)
            f.write("Support Vector Machine(SVM) Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
            print("Support Vector Machine Classifier: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            result["SVC"] = scores.mean()
            models["SVC"] = dumps(clf)

            # find highest f1 score and store it to the output with associated model
            h_max = max(result, key=result.get)
            f.write("Best model for %s: %s\n\n" % (self.name[self.label_col.index(trait)], self.model_name[h_max]))
            print("Best model for %s:%s\n" % (self.name[self.label_col.index(trait)], self.model_name[h_max]))
            score[self.name[self.label_col.index(trait)]] = models[h_max]

        f.close()
        return score


    ################# Build Regresssion Model for calculatinf f1 score ###############
    print("Regression Model for getting score: ")
    def trainModelRegression(self, trait, status):
        f = open('../data/predict/output.txt', 'a')
        f.write("Regression Model for getting score of %s: \n"%(trait))
        sample = self.data.iloc[:, 0:10]
        score = self.data.iloc[:, self.score_col[self.name.index(trait)]]
        label = self.data.iloc[:, self.label_col[self.name.index(trait)]]

        df = pd.concat([sample, score, label], axis=1)
        df.columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust', 'score', 'label']

        sample = df[df['label'] == status]

        # Have a look
        label = sample.score
        sample = sample.iloc[:, self.train_col]

        # set evaluation metrics
        mse = make_scorer(mean_squared_error)
        score = dict()
        score_mean = dict()
        f.write("################# Regression Model ###############\n")
        print("################# Regression Model ###############")
        ################################################################
        reg_model = KNeighborsRegressor(weights="distance", algorithm="auto", n_jobs=-1)
        reg_model.fit(sample, label)

        score_cv = cross_val_score(reg_model, sample, label, cv=5, scoring=mse)
        f.write("KNN Regression MSE: %0.2f (+/- %0.2f)\n"%(score_cv.mean(), score_cv.std() * 2))
        print("KNN Regression MSE: %0.2f (+/- %0.2f)" % (score_cv.mean(), score_cv.std() * 2))
        score_mean["KNN"] = score_cv.mean()
        score["KNN"] = dumps(reg_model)

        ################################################################
        # Decision Tree Regression
        reg_model = DecisionTreeRegressor()
        reg_model.fit(sample, label)

        score_cv = cross_val_score(reg_model, sample, label, cv=5, scoring=mse)
        f.write("Decision Tree Regression MSE: %0.2f (+/- %0.2f)\n" % (score_cv.mean(), score_cv.std() * 2))
        print("Decision Tree Regression MSE: %0.2f (+/- %0.2f)" % (score_cv.mean(), score_cv.std() * 2))
        score_mean["DT"] = score_cv.mean()
        score["DT"] = dumps(reg_model)

        ################################################################
        # Random Forest Regression
        reg_model = RandomForestRegressor(n_estimators=20, n_jobs=-1)
        reg_model.fit(sample, label)

        score_cv = cross_val_score(reg_model, sample, label, cv=5, scoring=mse)
        f.write("Random Forest Regression MSE: %0.2f (+/- %0.2f)\n" % (score_cv.mean(), score_cv.std() * 2))
        print("Random Forest Regression MSE: %0.2f (+/- %0.2f)" % (score_cv.mean(), score_cv.std() * 2))
        score_mean["RF"] = score_cv.mean()
        score["RF"] = dumps(reg_model)

        ################################################################
        # AdaBoost Regression
        reg_model = AdaBoostRegressor()
        reg_model.fit(sample, label)

        score_cv = cross_val_score(reg_model, sample, label, cv=5, scoring=mse)
        f.write("AdaBoost Regression MSE: %0.2f (+/- %0.2f)\n" % (score_cv.mean(), score_cv.std() * 2))
        print("AdaBoost Regression MSE: %0.2f (+/- %0.2f)" % (score_cv.mean(), score_cv.std() * 2))
        score_mean["ADB"] = score_cv.mean()
        score["ADB"] = dumps(reg_model)

        ###############################################################
        # Gradient Boosting Regression
        reg_model = GradientBoostingRegressor(loss="huber", n_estimators=100)
        reg_model.fit(sample, label)

        score_cv = cross_val_score(reg_model, sample, label, cv=5, scoring=mse)
        f.write("Gradient Boosting Regression MSE: %0.2f (+/- %0.2f)\n" % (score_cv.mean(), score_cv.std() * 2))
        print("Gradient Boosting Regression MSE: %0.2f (+/- %0.2f)" % (score_cv.mean(), score_cv.std() * 2))
        score_mean["GB"] = score_cv.mean()
        score["GB"] = dumps(reg_model)

        ################################################################
        # Stochastic Gradient Regression
        reg_model = SGDRegressor(loss="epsilon_insensitive", penalty="l2")
        reg_model.fit(sample, label)

        score_cv = cross_val_score(reg_model, sample, label, cv=5, scoring=mse)
        f.write("Stochastic Gradient Descent(SGD) Regression MSE: %0.2f (+/- %0.2f)\n" % (score_cv.mean(), score_cv.std() * 2))
        print("SGD Regression MSE: %0.2f (+/- %0.2f)" % (score_cv.mean(), score_cv.std() * 2))
        score_mean["SGD"] = score_cv.mean()
        score["SGD"] = dumps(reg_model)

        ###############################################################
        # Support Vector Machine Regression
        reg_model = SVR(kernel="linear")
        reg_model.fit(sample, label)

        score_cv = cross_val_score(reg_model, sample, label, cv=5, scoring=mse)
        f.write("Support Vector Machine(SVM) Regression MSE: %0.2f (+/- %0.2f)\n" % (score_cv.mean(), score_cv.std() * 2))
        print("Support Vector Regression MSE: %0.2f (+/- %0.2f)" % (score_cv.mean(), score_cv.std() * 2))
        score_mean["SVR"] = score_cv.mean()
        score["SVR"] = dumps(reg_model)

        # find the best model (lowest MSE)
        h_min = min(score_mean, key=score_mean.get)
        f.write("Best Model: %s (mse: %0.2f)\n\n" % (self.model_name[h_min], score_mean[h_min]))
        print("Best Model: %s (mse: %0.2f)\n\n" % (self.model_name[h_min], score_mean[h_min]))
        f.close()
        return score[h_min]


    # Save the best fitting model
    # output: update model storage

    def saveModel(self):
        f = open('../data/predict/output.txt', 'a')
        for item in self.name:
            f.write("Processing regression model on trait: \n".format(item))
            print("Processing regression model on trait: ", item)
            f.write("Classified as yes!\n")
            print("Classified as yes!")
            self.modelYes[item] = self.trainModelRegression(item, 1)
            f.write("Classified as no!\n")
            print("Classified as no!")
            self.modelNo[item] = self.trainModelRegression(item, 0)
        f.close()

# Personality-Traits-Detection-Master
The project is a simple personality traits predictor. It shows different personality trait scores of several users from the given tweets. Here, it analyses 100 tweets of more than 25 users and shows their big5 personality traits score. It uses different kinds of machine learning algorithms to predict personality scores. The classification models are used to get the best model for each trait and the regression models are used to predict the personality scores. At first, the tweets are pre-processed and stored in the input folder. Then these data are used in the model to get the personality traits score. Each personality trait score is calculated with the associated best model from the analysis of the classification model. The outputs are shown in the output.txt file.

# Features
Predicting user's personality traits.

# Methods 
## Data pre-processing
  * Unigram
  * Word Frequency
  
## Classification
  * K Nearest Neighbor
  * Decission Tree
  * Random Forest
  * AdaBoost
  * Gradient Boosting
  * Stochastic Gradient Descent
  * Support Vector Machine

## Regression
  * K Nearest Neighbor
  * Decission Tree
  * Random Forest
  * AdaBoost
  * Gradient Boosting
  * Stochastic Gradient Descent
  * Support Vector Machine
  
 ## Model Selection
 For each traits:
  * **Classification model:** Model with the highest 5-fold cross score will be selected.
  * **Regression Model:** Model with the lowest 5-fold MSE will be selected.
  
 ## Files:
  * **preProcess.py:** Process the input raw tweets from the input files stored in Big5 folder.
  * **trainData.py:** Prepare the training data.
  * **trainModel.py:** Train the data using machine learning techniques.
    * Classification methods are aimed to predict best model for each traits' category.
    * Regression methods are aimed to calculate the personality traits score under each category.
  * **evalModel.py:** Evaluate the best model for each traits, get the personality score of each traits with the asssociated best model and shows the result in the output.txt file.
  
 ## Follow of scripts:
   1. **Prepare validation data:**
      preProcess.py
   1. **Prepare the train data:**
      trainData.py
   1. **Run the model:**
      evalModel.py
      
      
## Model Performance On Big5 Traits:  
* **Classifier Model for finding best classification model on each trait:** 
  * **Processing traits: Extraversion**
    * KNN Accuracy: 0.54 (+/- 0.02)
    * Decision Tree Accuracy: 0.55 (+/- 0.01)
    * Random Forest Accuracy: 0.54 (+/- 0.02)
    * AdaBoost Accuracy: 0.56 (+/- 0.02)
    * Gradient Boosting Accuracy: 0.56 (+/- 0.01)
    * Stochastic Gradient Descent(SGD) Accuracy: 0.55 (+/- 0.06)
    * Support Vector Machine(SVM) Accuracy: 0.57 (+/- 0.00)
    * Best model for ext: Support Vector Machine

  * **Processing traits: Neuroticism**
    * KNN Accuracy: 0.55 (+/- 0.03)
    * Decision Tree Accuracy: 0.60 (+/- 0.01)
    * Random Forest Accuracy: 0.60 (+/- 0.01)
    * AdaBoost Accuracy: 0.63 (+/- 0.00)
    * Gradient Boosting Accuracy: 0.62 (+/- 0.01)
    * Stochastic Gradient Descent(SGD) Accuracy: 0.61 (+/- 0.07)
    * Support Vector Machine(SVM) Accuracy: 0.63 (+/- 0.00)
    * Best model for neuroticism: Support Vector Machine
   
  * **Processing traits: Agreeableness**
    * KNN Accuracy: 0.50 (+/- 0.02)
    * Decision Tree Accuracy: 0.51 (+/- 0.03)
    * Random Forest Accuracy: 0.51 (+/- 0.03)
    * AdaBoost Accuracy: 0.53 (+/- 0.02)
    * Gradient Boosting Accuracy: 0.53 (+/- 0.02)
    * Stochastic Gradient Descent(SGD) Accuracy: 0.51 (+/- 0.05)
    * Support Vector Machine(SVM) Accuracy: 0.54 (+/- 0.00)
    * Best model for agreeableness: Support Vector Machine

  * **Processing traits: Conscientiousness**
    * KNN Accuracy: 0.52 (+/- 0.02)
    * Decision Tree Accuracy: 0.52 (+/- 0.01)
    * Random Forest Accuracy: 0.51 (+/- 0.02)
    * AdaBoost Accuracy: 0.53 (+/- 0.02)
    * Gradient Boosting Accuracy: 0.53 (+/- 0.02)
    * Stochastic Gradient Descent(SGD) Accuracy: 0.53 (+/- 0.03)
    * Support Vector Machine(SVM) Accuracy: 0.54 (+/- 0.00)
    * Best model for conscientiousness: Support Vector Machine

  * **Processing traits: Openness**
    * KNN Accuracy: 0.73 (+/- 0.02)
    * Decision Tree Accuracy: 0.70 (+/- 0.01)
    * Random Forest Accuracy: 0.73 (+/- 0.01)
    * AdaBoost Accuracy: 0.74 (+/- 0.01)
    * Gradient Boosting Accuracy: 0.74 (+/- 0.00)
    * Stochastic Gradient Descent(SGD) Accuracy: 0.74 (+/- 0.02)
    * Support Vector Machine(SVM) Accuracy: 0.74 (+/- 0.00)
    * Best model for openness: Support Vector Machine
    
* **Regression Model for getting personality traits score**
  * **Traits Extraversion: yes**
    * KNN Regression MSE: 0.20 (+/- 0.11)
     * Decision Tree Regression MSE: 0.22 (+/- 0.11)
     * Random Forest Regression MSE: 0.18 (+/- 0.11)
     * AdaBoost Regression MSE: 0.16 (+/- 0.11)
     * Gradient Boosting Regression MSE: 0.16 (+/- 0.11)
     * Stochastic Gradient Descent(SGD) Regression MSE: 0.16 (+/- 0.11)
     * Support Vector Machine(SVM) Regression MSE: 0.16 (+/- 0.11)
     * Best Model: AdaBoost (mse: 0.16)

  * **Traits Extraversion: no**
    * KNN Regression MSE: 0.49 (+/- 0.25)
    * Decision Tree Regression MSE: 0.47 (+/- 0.25)
    * Random Forest Regression MSE: 0.40 (+/- 0.26)
    * AdaBoost Regression MSE: 0.37 (+/- 0.23)
    * Gradient Boosting Regression MSE: 0.37 (+/- 0.28)
    * Stochastic Gradient Descent(SGD) Regression MSE: 0.41 (+/- 0.38)
    * Support Vector Machine(SVM) Regression MSE: 0.39 (+/- 0.37)
    * Best Model: AdaBoost (mse: 0.37)

  * **Traits Neuroticism: yes**
    * KNN Regression MSE: 0.34 (+/- 0.10)
    * Decision Tree Regression MSE: 0.34 (+/- 0.10)
    * Random Forest Regression MSE: 0.28 (+/- 0.09)
    * AdaBoost Regression MSE: 0.26 (+/- 0.08)
    * Gradient Boosting Regression MSE: 0.26 (+/- 0.11)
    * Stochastic Gradient Descent(SGD) Regression MSE: 0.28 (+/- 0.15)
    * Support Vector Machine(SVM) Regression MSE: 0.28 (+/- 0.14)
    * Best Model: Gradient Boosting (mse: 0.26)

  * **Traits Neuroticism: no**
     * KNN Regression MSE: 0.26 (+/- 0.03)
     * Decision Tree Regression MSE: 0.25 (+/- 0.01)
     * Random Forest Regression MSE: 0.22 (+/- 0.02)
     * AdaBoost Regression MSE: 0.19 (+/- 0.03)
     * Gradient Boosting Regression MSE: 0.20 (+/- 0.02)
     * Stochastic Gradient Descent(SGD) Regression MSE: 0.20 (+/- 0.02)
     * Support Vector Machine(SVM) Regression MSE: 0.20 (+/- 0.02)
     * Best Model: AdaBoost (mse: 0.19)

  * **Traits Agreeableness: yes**
    * KNN Regression MSE: 0.18 (+/- 0.12)
    * Decision Tree Regression MSE: 0.20 (+/- 0.11)
    * Random Forest Regression MSE: 0.17 (+/- 0.12)
    * AdaBoost Regression MSE: 0.15 (+/- 0.11)
    * Gradint Boosting Regression MSE: 0.15 (+/- 0.13)
    * Stochastic Gradient Descent(SGD) Regression MSE: 0.16 (+/- 0.14)
    * Support Vector Machine(SVM) Regression MSE: 0.15 (+/- 0.14)
    * Best Model: AdaBoost (mse: 0.15)


  * **Traits Agreeableness: no**
    * KNN Regression MSE: 0.18 (+/- 0.04)
    * Decision Tree Regression MSE: 0.19 (+/- 0.04)
    * Random Forest Regression MSE: 0.16 (+/- 0.04)
    * AdaBoost Regression MSE: 0.15 (+/- 0.04)
    * Gradient Boosting Regression MSE: 0.14 (+/- 0.05)
    * Stochastic Gradient Descent(SGD) Regression MSE: 0.15 (+/- 0.06)
    * Support Vector Machine(SVM) Regression MSE: 0.15 (+/- 0.05)
    * Best Model: Gradient Boosting (mse: 0.14)

  * **Traits Conscientiousness: yes**
    * KNN Regression MSE: 0.24 (+/- 0.09)
    * Decision Tree Regression MSE: 0.26 (+/- 0.10)
    * Random Forest Regression MSE: 0.22 (+/- 0.10)
    * AdaBoost Regression MSE: 0.19 (+/- 0.11)
    * Gradient Boosting Regression MSE: 0.20 (+/- 0.10)
    * Stochastic Gradient Descent(SGD) Regression MSE: 0.20 (+/- 0.09)
    * Support Vector Machine(SVM) Regression MSE: 0.20 (+/- 0.09)
    * Best Model: AdaBoost (mse: 0.19)

  * **Traits Conscientiousness: no**
    * KNN Regression MSE: 0.28 (+/- 0.15)
    * Decision Tree Regression MSE: 0.30 (+/- 0.16)
    * Random Forest Regression MSE: 0.26 (+/- 0.15)
    * AdaBoost Regression MSE: 0.23 (+/- 0.13)
    * Gradient Boosting Regression MSE: 0.23 (+/- 0.16)
    * Stochstic Gradient Descent(SGD) Regression MSE: 0.25 (+/- 0.22)
    * Support Vector Machine(SVM) Regression MSE: 0.25 (+/- 0.21)
    * Best Model: Gradient Boosting (mse: 0.23)

  * **Traits Openness: yes**
    * KNN Regression MSE: 0.13 (+/- 0.05)
    * Decision Tree Regression MSE: 0.13 (+/- 0.05)
    * Random Forest Regression MSE: 0.11 (+/- 0.06)
    * AdaBoost Regression MSE: 0.10 (+/- 0.06)
    * Gradient Boosting Regression MSE: 0.10 (+/- 0.06)
    * Stochastic Gradient Descent(SGD) Regression MSE: 0.11 (+/- 0.06)
    * Support Vector Machine(SVM) Regression MSE: 0.10 (+/- 0.06)
    * Best Model: AdaBoost (mse: 0.10)

  * **Traits Openness: no**
    * KNN Regression MSE: 0.31 (+/- 0.14)
    * Decision Tree Regression MSE: 0.35 (+/- 0.11)
    * Random Forest Regression MSE: 0.27 (+/- 0.10)
    * AdaBoost Regression MSE: 0.25 (+/- 0.08)
    * Gradient Boosting Regression MSE: 0.25 (+/- 0.11)
    * Stochastic Gradient Descent(SGD) Regression MSE: 0.27 (+/- 0.20)
    * Support Vector Machine(SVM) Regression MSE: 0.27 (+/- 0.22)
    * Best Model: Gradient Boosting (mse: 0.25)

## Sample Output
 * **Big5 traits score of user: ahti7860**
   * ext: 2.6734401571990154
   * neu: 2.0246302264349607
   * agr: 4.080308148148174
   * con: 2.784746022053077
   * opn: 4.443022222222222
 * **Big5 traits score of user: akideares**
   * ext: 2.5545974530018314
   * neu: 2.003784450843272
   * agr: 3.9437908496732215
   * con: 2.8206301708647925
   * opn: 4.139455844353271
  
 * **Big5 traits score of user: AlJazeera9762528**
   * ext: 2.3749773195876402
   * neu: 1.8393435980551167
   * agr: 3.6198148596321698
   * con: 2.5831282930519155
   * opn: 3.8450263157894824

 * **Big5 traits score of user: BillGates**
   * ext: 2.6262106609808096
   * neu: 2.0466279391424584
   * agr: 3.9576126984126963
   * con: 2.704762076953871
   * opn: 4.400479574570943

* **Big5 traits score of user: data:ClimateEnvoy**
   * ext: 2.6297293858575967
   * neu: 2.0493701108768074
   * agr: 3.9629153005464457
   * con: 3.0897803764098906
   * opn: 4.427285860655737
 

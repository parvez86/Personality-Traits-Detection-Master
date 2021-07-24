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

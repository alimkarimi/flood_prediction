import numpy as np

from xgboost import XGBClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


from dataloading_scripts.feature_builder import getFeatures, df_pos_neg

import os


class statistical_model():
    def __init__(self, data = df_pos_neg, model_type = 'svm'):
        self.num_cores = os.cpu_count()
        self.model_type = model_type
        self.test_ratio = 0.3
        self.train_ration = 1 - self.test_ratio
        X = data['feature_vector']
        y = data['label'] # these are either 'tp' or 'tn'. use LabelEncoder() to convert them to numerical format

        le = LabelEncoder()
        y = le.fit_transform(y)
        print('this is y!!!', y)
        # handle train and test split:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_ratio, random_state=42)

        # move X_train, X_test to numpy arrays. They seem to be in an awkward format for using pre-processing ops. 
        self.X_train = np.vstack(self.X_train.to_list())
        self.X_test = np.vstack(self.X_test.to_list())

    def fit_model(self, verbose=True):
        """
        This function instatiates SVM model with default parameters and fits to training data
        """
        if self.model_type == 'svm':
            self.model = SVC(n_jobs = self.num_cores) # instantiate SVM model. Specify n_jobs to be the number of cores available on 
            # machine's CPU. The goal of this is to parallelize the training.
        if self.model_type == 'xgb':
            self.model = XGBClassifier()
        if self.model_type == 'rf':
            self.model = RandomForestClassifier()

        self.scaler = StandardScaler() # instantiate scaler object to transform training features to have 0 mean and 1 variance.
        transformed_X_train = self.scaler.fit_transform(self.X_train) # fit and transform training data using scaler object
        if verbose:
            print(f"Starting to train {self.model_type}")
        self.model.fit(X = transformed_X_train, y = self.y_train) # find optimal SVM parameters using the given hyperparameters
        if verbose:
            print(f"Done training {self.model_type}")
    def test_model(self, verbose=False):
        """
        This function will test the fitted SVM model and compute the accuracy of predictions (correct predictions / total predictions).

        We first transform X_test using the transformation used for X_train, the use the svm.predict() method to generate predictions on the test dataset. 
        """
        print("starting testing")

        self.X_test = self.scaler.transform(self.X_test) # transform test data using transform for training data
        
        self.y_pred = self.model.predict(self.X_test) # run predictions on test set.
        if verbose:
            print("Precitions on X_test:", self.y_pred)
            print("Ground truth data", self.y_test)

        accuracy = accuracy_score(y_true = self.y_test, y_pred = self.y_pred)
        print(f"Accuracy for {self.model_type} trained on {self.X_train.shape[0]} data points is {accuracy}")

        true_pos_counter = 0
        false_pos_counter = 0
        false_negative_counter = 0

        # compute precision/recall metrics (true_pos / true_pos + false_neg). 

        for (pred, gt) in zip(self.y_pred, self.y_test):
            # iterate through predictions and ground truth to understand which predictions are true positives, false positives, and false negatives
            if gt == 1 and pred == 0:
                false_negative_counter += 1

            if gt == 1 and pred == 1:
                true_pos_counter+= 1
            
            if gt == 0 and pred == 1:
                false_pos_counter +=1
        
        # print out precision and recall metrics:

        if true_pos_counter + false_pos_counter == 0:
            print("Cannot compute precision because there are no TPs or FPs in dataset")
        else:
            print(f"Precision is TP / (TP + FP) which is {true_pos_counter / (true_pos_counter + false_pos_counter)}")

        if true_pos_counter + false_negative_counter == 0:
            print("Cannot generate recall because TP + FN = 0")
        else:
            print(f"Recall is TP / (TP + FN) which is {true_pos_counter / (true_pos_counter + false_negative_counter)}")


if __name__ == "__main__":
    model = statistical_model(data = df_pos_neg, model_type='xgb')

    model.fit_model()
    model.test_model()
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from dataloading_scripts.feature_builder import df_pos_neg

print(df_pos_neg)

class SVM_model():
    def __init__(self, data = df_pos_neg):
        self.train_ratio = 0.5
        self.test_ratio = 0.5
        X = data['feature_vector']
        y = data['label'] # these are either 'tp' or 'tn'. use LabelEncoder() to convert them to numerical format

        le = LabelEncoder()
        y = le.fit_transform(y)
        print('this is y!!!', y)
        # handle train and test split:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_ratio, random_state=None)

        # move X_train, X_test to numpy arrays. They seem to be in an awkward format for using pre-processing ops. 
        self.X_train = np.vstack(self.X_train.to_list())
        self.X_test = np.vstack(self.X_test.to_list())


    def fit_svm(self):
        """
        This function instatiates SVM model with default parameters and fits to training data
        """
        self.svm = SVC()

        # apply standard scalar pre-processing transform:
        # print("before standard scalar: ", self.X_train)
        # print("shape of values", self.X_train.shape)

        self.scaler = StandardScaler()
        transformed_X_train = self.scaler.fit_transform(self.X_train)
        #print("after standard scaler", transformed_X_train)
        print(self.y_train)

        self.svm.fit(X = transformed_X_train, y = self.y_train)
    
    def test_svm(self):
        """
        This function will test the fitted SVM model and compute the accuracy of predictions (correct predictions / total predictions).

        We need to transform X_test using the transformation used for X_train
        """

        self.X_test = self.scaler.transform(self.X_test)
        self.y_pred = self.svm.predict(self.X_test)
        print("Precitions on X_test:", self.y_pred)
        print("Ground truth data", self.y_test)

        accuracy = accuracy_score(y_true = self.y_test, y_pred = self.y_pred)
        print(f"Accuracy for SVM trained on {self.X_train.shape[0]} data points is {accuracy}")

        true_pos_counter = 0
        false_pos_counter = 0
        false_negative_counter = 0
        # compute precision (true_pos / true_pos + false_neg)
        for (pred, gt) in zip(self.y_pred, self.y_test):
            if gt == 1 and pred == 0:
                false_negative_counter += 1

            if gt == 1 and pred == 1:
                true_pos_counter+= 1
            
            if gt == 0 and pred == 1:
                false_pos_counter +=1
        
        if true_pos_counter + false_pos_counter == 0:
            print("Cannot compute precision because there are no TPs or FPs in dataset")
        else:
            print(f"Precision is TP / (TP + FP) which is {true_pos_counter / (true_pos_counter + false_pos_counter)}")

        if true_pos_counter + false_negative_counter == 0:
            print("Cannot generate recall because TP + FN = 0")
        else:
            print(f"Recall is TP / (TP + FN) which is {true_pos_counter / (true_pos_counter + false_negative_counter)}")


if __name__ == "__main__":
    svm = SVM_model(data = df_pos_neg)

    svm.fit_svm()
    svm.test_svm()


    
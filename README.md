# Flood Prediction
Flood prediction is a difficult problem. This repo represents code for the 2024 NeurIPS workshop proposal paper accepted to the workshop "Tackling Climate Change with ML"

Flood prediction is difficult because there are very sparse true positive observations. The proposal paper outlines a method to first establish a baseline in Kenya 
and then to leverage pretrained models to generate more training points. With these additional training points, deep learning theory suggests that higher order models 
(i.e, neural networks) can be more appropriately used.

# Initial Results:
All models trained on about one million data points (~700k train, ~300k test):

Accuracy for SVM trained on test dataset (total correct / total predictions) is 0.993821741506965
Cannot compute precision because there are no TPs or FPs in dataset
Recall is TP / (TP + FN) which is 0.0

Accuracy for XGBoost on test dataset is (total correct / total predictions) is 0.9954880492935584
Precision is TP / (TP + FP) which is 0.6576158940397351
Recall is TP / (TP + FN) which is 0.5402611534276387

Accuracy for rf trained on test dataset (total correct / total predictions) is 0.9962897321650407
Precision is TP / (TP + FP) which is 0.7442622950819672
Recall is TP / (TP + FN) which is 0.6085790884718498

| Model    | Accuracy          | Precision            | Recall               |
| -------- | ----------------- | -------------------- | -------------------- |
| SVM      | 0.993821741506965  | Cannot compute       | 0.0                  |
| XGBoost  | 0.9954880492935584 | 0.6576158940397351   | 0.5402611534276387   |
| Random Forest | 0.9962897321650407 | 0.7442622950819672   | 0.6085790884718498   |

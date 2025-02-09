# Flood Prediction
This repo represents code for the 2024 NeurIPS workshop proposal paper accepted to the workshop "Tackling Climate Change with ML". Hammed A Akande, Valerie Brosnon, David Quispe, Nicole Mong’are, and Asbina Baral have also contributed to the proposal. 

Flood prediction is difficult because there are very sparse true positive observations. The proposal paper outlines a method to first establish a baseline in Kenya 
and then to leverage pretrained models to generate more training points. With these additional training points, deep learning theory suggests that higher order models 
(i.e, neural networks) can be more appropriately used.

# How to gather climate and elevation features (predictors):
Climate features from ERA5 were downloaded using the script at flood_prediction/dataloading_scripts/download_era5.py . Kick that off by:

`cd flood_prediction`
`python3 -m dataloading_scripts.download_era5`

This will download relevant climate features, however, each year will be in a separate .nc file.

We want to aggegate all the files from 1996 - 2018 into single file to reduce management overhead.

To do this, run:

`python3 -m dataloading_scripts.merge_era5`

Climate features have now been downloaded, but if one wants to incorporate elevation data into the predictors, it will require getting that data from the USGS Earth Explorer. To do this, follow instruction on this YouTube video: https://www.youtube.com/watch?v=IdilpusxTHY&t=246s . Point the download to a folder called `flood_prediction/data/srtm` . The download should have a latitude bound of -5 to +5 and a longitude bound of 34 to 42.5. Make sure to download the data the 3arcsecond resolution data.

Once these steps have been completed, we are ready to prep features and run models.

# How to gather ground truth (flood / no flood per grid cell)
From the Darthmouth Flood Observatory, download the Flood Archive from https://floodobservatory.colorado.edu/temp/FloodArchive.txt . Place this file in `data/gt/`

The model building code (in the next section) will automatically use this for ground truth.

# How to run statistical models:

Running `python3 -m models.models` will kick off the statistical models (either XGBoost or RF). The actual model has to be specified in the code. We plan to add command line arguments for ease of use in the future. 

Note that if one wants to incorporate elevation features or adjust the number of negatives (non-flood examples or true negatives in the number of samples), one has to adjust the function definition in flood_prediction/dataloading_scripts/feature_builder.py .

Specifically, line 49 has the function defintion: 
```def getFeatures(df,target_cube = target_cube, predictor_vars = predictor_vars, append_dtm = True, num_neg_samples = 1000000, pos_feature_extraction=True):```

Adjust `append_dtm` to `True` if you want to incorporate DTM features. This will automatically resample SRTM data to the ERA5 grid in the background. 

Adjust `num_neg_samples` to you desired choice of negative samples. The paper results were run with 1 million negative samples. 

All models trained on about one million data points (~700k train, ~300k test):

# Initial Results:

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

# Results with DTM data
All models trained on about one million data points (~700k train, ~300k test):


Accuracy for XGBoost on test dataset is (total correct / total predictions) is 0.9962864194259023
Precision is TP / (TP + FP) which is 0.7017353579175705
Recall is TP / (TP + FN) which is 0.693833780160858

Accuracy for rf trained on test dataset (total correct / total predictions) is 0.9968661487751147
Precision is TP / (TP + FP) which is 0.7506819421713039
Recall is TP / (TP + FN) which is 0.7378016085790885

| Model    | Accuracy          | Precision            | Recall               |
| -------- | ----------------- | -------------------- | -------------------- |
| XGBoost  | 0.9962864194259023 |  0.7017353579175705  |  0.693833780160858  |
| Random Forest | 0.9968661487751147 |  0.7506819421713039  | 0.7378016085790885   |
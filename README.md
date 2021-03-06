# EEL5840-Project
* This project tries to classify 4 different types of bricks and non-brick object by image-based. The Edge Histogram Descriptor (EHD) extracts features for each image. Then, train Probabilistic Generative Classifier (PGC) to classify bricks.

## Parameter explanation
### Train.py
1. Initial parameter
```
CrossValidation = False # If true, do cross validation.
ShowBestTrainModel = True # If trues, showing best train model results by best Kfold results.
```
* Both CrossValidation and ShowBestTrainModel variables cannot be true at the same time.
* If wanted to show best train model results, the CrossValidation should be False.
* Otherwise, the best file will be coverd by process of cross validation.

2. The below variables is to input data:
```
Image = np.load('Images.npy') #import train figures. The shape is (X, 200, 200, 3). 
Labels = np.load('Labels.npy') #import train labels. The shape is (Y,).
```

3. The below variable is the output vector Y which is predicted lables.:
```
>>PredLabels
```

### Test.py
1.The below variables is to input data
```
Image = np.load('Images.npy') #import train figures. The shape is (X, 200, 200, 3). 
Labels = np.load('Labels.npy') #import train labels. The shape is (Y,).
```
2. The below variable is the output vector Y which is predicted lables.
```
>>PredLabels
```
* The study do filter before training model. Because the Git limits the file size (< 100 mb), the Images.npy and Labels.npy in the repository is not the filted data. We randomely select blind data to be Images.npy and Labels.npy.

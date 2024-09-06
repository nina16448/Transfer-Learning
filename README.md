# Plant Seedlings Classification

## Execution description
**Step 1.** Install requirements.txt
```
pip install -r requirements.txt
```
**Step 2.** Train the model
```
python train.py
```
**Step 3.** Test the model
```
python test.py
```

## Project Structure
```
├── result/                       # Stores training result images
├── weights/                      # Stores trained model weights
├── .gitignore                    # Specifies files and directories to ignore
├── README.md                     # Project description
├── dataloader.py                 # Used for loading datasets
├── model.py                      # Defines the model architecture
├── predictions.csv               # Contains the model's prediction results
├── requirements.txt              # Project dependencies
├── test.py                       # Script for testing the model
└── train.py                      # Script for training the model
```

## Experimental results

### Training accuracy
![Training_accuracy](https://hackmd.io/_uploads/B1NV_Od3A.png)

### Loss curve
![Loss_curve](https://hackmd.io/_uploads/rydS_Od2A.png)

### confusion matrix
![Confusion_Matrix](https://hackmd.io/_uploads/BJiGktuhR.png)

### Score on Kaggle
![image](https://hackmd.io/_uploads/H1mYudd3C.png)

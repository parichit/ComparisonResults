# Label Imputation benchmarks 

Data is uploaded to google drive. Data should be downloaded and placed inside "./data" directory.  https://drive.google.com/drive/folders/1xwkySE2M4cVS_7KhDci3vcBMbWtMf6_Q?usp=sharing

There are 2 steps in label Imputation benchmarking:
1. Applying Labeling algorithm
2. Training neural net on data with generated labels.

####To run labeling algorithm, run labeling.py file. 

example: python ../labeling.py --data_set modcloth --debug False --seed 0

```--data_set``` defines the dataset to run labeling algorithm on. Options supported are ['modcloth','amazon_review','amazon_vid'] . amazon_review runs for Amazon fashion review data. amazon_vid runs Amazon video game reviews data. 

```--debug``` flag True runs for a sample of data, considering only 1000 rows of data.

```--seed``` sets the randomisation seed. Should be an integer.

#### To train model on generated labels, run train_model.py file. 
example: python ../train_model.py --data_set modcloth --debug False --seed 0 --labeling_algorithm snorkel

```--data_set```  , ```--debug``` ,```--seed``` are the same as labeling.py file

```--labeling_algorithm``` flag defines the labels to use for training. supported options are ['snorkel','triplet']. 

#### Scripts folder has shell scripts which can be used to run all steps for each dataset. 

#### results will be written to a results.csv file with the following headers:
Dataset,Neural network,Labeling Algorithm,Epochs,Train acc,Test acc,Execution time (in secs)

All the dependencies are added to requirements.txt file. 

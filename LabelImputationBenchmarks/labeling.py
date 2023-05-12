import os
import argparse , sys
import numpy as np
import pandas as pd
import labeling_functions as lfs
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel as snorkel_lm
from snorkel.utils import probs_to_preds
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from snorkel.preprocess import preprocessor
from textblob import TextBlob
from snorkel.labeling import filter_unlabeled_dataframe
from flyingsquid.label_model import LabelModel as triplet_lm



parser = argparse.ArgumentParser()
# parser.add_argument('--lr', type = float, default = 0.001)
# parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--data_set', type = str, default = 'modcloth',choices=['modcloth','amazon_review','amazon_vid'])
parser.add_argument('--debug',type=bool,default=True)
parser.add_argument('--seed',type=int,default=0)

args = parser.parse_args()

def main():
    print(args)
    
    if(args.data_set=='modcloth'):
        df_train=pd.read_json("./data/renttherunway_final_data.json",lines=True)
        df_train=df_train[df_train.review_text.notnull()]
        df_train['review_text'] = df_train['review_text'].astype(str)
        df_test=pd.read_csv("./data/renttherunway_labeled_data.csv")
        Y_test=df_test.label.values
        labeling_functions=[lfs.textblob_polarity,lfs.vader,lfs.textblob_subjectivity,lfs.fit_val,lfs.rating,lfs.lf_contains_positive_word,lfs.lf_contains_negative_word,lfs.lf_contains_recommendation]

    elif(args.data_set=='amazon_review'):
        df_train=pd.read_json("./data/AMAZON_FASHION.json",lines=True)
        df_train=df_train[df_train.review_text.notnull()]
        df_test=pd.read_csv("./data/amazon_fashion_labeled_data.csv")
        df_train['review_text'] = df_train['review_text'].astype(str)
        labeling_functions=[lfs.vader,lfs.textblob_polarity,lfs.textblob_subjectivity,lfs.lf_positive,lfs.lf_negative,lfs.lf_contains_positive_word,lfs.lf_contains_negative_word,lfs.lf_contains_recommendation]
        Y_test=df_test.label.values
    elif(args.data_set=='amazon_vid'):
        df_train=pd.read_csv("./data/AmazonVideoGame_800000_train_data.csv",nrows=200000)
        # df_train=df_train[df_train.review_text.notnull()]
        df_test=pd.read_csv("./data/AmazonVideoGame_500_labeled_data.csv")
        df_train['review_text'] = df_train['review_text'].astype(str)
        labeling_functions=[lfs.vader,lfs.textblob_polarity,lfs.textblob_subjectivity,lfs.lf_positive,lfs.lf_negative,lfs.lf_contains_positive_word,lfs.lf_contains_negative_word,lfs.lf_contains_recommendation]
        Y_test=df_test.label.values
    # if(args.debug==True):
    #     df_train=df_train.sample(1000,random_state=args.seed)

    #generating label matrix
    if(os.path.exists('./data/label_matrix/'+args.data_set+"_L_train.npy")):
        print("Found Label Matrix generated, Using the same. Delete it to rerun. path:",'./data/label_matrix/'+args.data_set+"_L_train.npy")
        L_train=np.load('./data/label_matrix/'+args.data_set+"_L_train.npy")
        L_test=np.load('./data/label_matrix/'+args.data_set+"_L_test.npy")
    else:
        applier = PandasLFApplier(labeling_functions)
        L_train = applier.apply(df_train)
        L_test = applier.apply(df=df_test)

    print(LFAnalysis(L_train, labeling_functions).lf_summary())
    #calculating accuracy of each labeling function
    
    for lf_idx in range(len(labeling_functions)):
        print("arruracy of lf",lf_idx," = ",(np.sum(L_test[:,lf_idx]==Y_test))/Y_test.shape[0])
    np.save('./data/label_matrix/'+args.data_set+"_L_train",L_train)
    np.save('./data/label_matrix/'+args.data_set+"_L_test",L_test)
    #snorkel majority voting
    majority_model = MajorityLabelVoter()
    preds_train_majority = majority_model.predict(L=L_train)

    
    #snorkel labelmodel (slm)
    snorkel_label_model = snorkel_lm(cardinality=2, verbose=True)
    snorkel_label_model.fit(L_train=L_train, n_epochs=2000, log_freq=100, seed=123)
    probs_train_slm = snorkel_label_model.predict_proba(L=L_train)
    df_train_filtered_slm, probs_train_filtered_slm = filter_unlabeled_dataframe(
        X=df_train, y=probs_train_slm, L=L_train
    )
    preds_train_filtered_slm = probs_to_preds(probs=probs_train_filtered_slm)
    df_train_filtered_slm.to_pickle("./data/filtered_train/snorkel_filtered_"+args.data_set+".pkl")
    #generating labels using triplet
    m = len(labeling_functions)
    triplet_label_model = triplet_lm(m)


    #triplet expects abstain as 0 and neg as -1
    L_train_triplet=np.select([L_train == -1, L_train == 0], [0, -1], L_train)
    L_test_triplet=np.select([L_test == -1, L_test == 0], [0, -1], L_test)
    triplet_label_model.fit(L_train_triplet)
    preds_triplet_train = triplet_label_model.predict(L_train_triplet).reshape((L_train_triplet.shape[0],))
    preds_triplet_test=triplet_label_model.predict(L_test_triplet).reshape(Y_test.shape)
    preds_triplet_train[preds_triplet_train==-1]=0
    preds_triplet_test[preds_triplet_test==-1]=0

    accuracy = np.sum(preds_triplet_test == Y_test) / Y_test.shape[0]

    print('triplet Label model accuracy: {}%'.format(int(100 * accuracy)))

    majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
        "accuracy"
    ]
    print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

    snorkel_label_model_acc = snorkel_label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
        "accuracy"
    ]
    print(f"{'snorkel Label Model Accuracy:':<25} {snorkel_label_model_acc * 100:.1f}%")

    np.save('./data/generated_labels/snorkel_preds_'+args.data_set,preds_train_filtered_slm)
    np.save('./data/generated_labels/triplet_preds_'+args.data_set,preds_triplet_train)

    


if __name__=="__main__":
    main()

# Import necessary libraries
import os
import time
import csv
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--lr', type = float, default = 0.001)
# parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--data_set', type=str, default='modcloth',
                    choices=['modcloth', 'amazon_review', 'amazon_vid'])
parser.add_argument('--debug', type=bool, default=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--labeling_algorithm', type=str,
                    default='snorkel', choices=['snorkel', 'triplet'])
parser.add_argument('--epochs', type=int, default=30)
args = parser.parse_args()


def main():
    # print(args)
    if(args.data_set == 'modcloth'):
        if(args.labeling_algorithm == 'snorkel'):
            y_train = np.load(
                "./data/generated_labels/snorkel_preds_modcloth.npy")
            df_train = pd.read_pickle(
                "/nobackup/dichalla/task3/RA-Phase1Task3-DataLabeling/data/filtered_train/snorkel_filtered_modcloth.pkl")
        elif(args.labeling_algorithm == 'triplet'):
            df_train = pd.read_json(
                "./data/renttherunway_final_data.json", lines=True)
            y_train = np.load(
                "./data/generated_labels/triplet_preds_modcloth.npy")
        # if(args.debug==True):
        #     df_train=df_train.sample(1000,random_state=args.seed)
        # df_train=df_train[df_train.review_text.notnull()]
        # df_train['review_text'] = df_train['review_text'].astype(str)
        df_test = pd.read_csv("./data/renttherunway_labeled_data.csv")
        train_texts = df_train['review_text'].values
        test_texts = df_test['review_text'].values

        y_test = df_test.label.values
    elif(args.data_set == 'amazon_review'):
        if(args.labeling_algorithm == 'snorkel'):
            y_train = np.load(
                "./data/generated_labels/snorkel_preds_amazon_review.npy")
            df_train = pd.read_pickle(
                "RA-Phase1Task3-DataLabeling/data/filtered_train/snorkel_filtered_amazon_review.pkl")
        elif(args.labeling_algorithm == 'triplet'):
            df_train = pd.read_json("./data/AMAZON_FASHION.json", lines=True)
            y_train = np.load(
                "./data/generated_labels/triplet_preds_amazon_review.npy")

        # if(args.debug==True):
        #     df_train=df_train.sample(500,random_state=args.seed)
        df_train = df_train[df_train.review_text.notnull()]
        df_test = pd.read_csv("./data/amazon_fashion_labeled_data.csv")
        df_train['review_text'] = df_train['review_text'].astype(str)
        train_texts = df_train['review_text'].values
        test_texts = df_test['review_text'].values

        y_test = df_test.label.values
    elif(args.data_set == 'amazon_vid'):
        if(args.labeling_algorithm == 'snorkel'):
            y_train = np.load(
                "./data/generated_labels/snorkel_preds_"+args.data_set+".npy")
            df_train = pd.read_pickle(
                "RA-Phase1Task3-DataLabeling/data/filtered_train/snorkel_filtered_amazon_vid.pkl")
        elif(args.labeling_algorithm == 'triplet'):
            df_train = pd.read_csv(
                "./data/AmazonVideoGame_800000_train_data.csv", nrows=200000)
            y_train = np.load(
                "./data/generated_labels/triplet_preds_amazon_vid.npy")

        # if(args.debug==True):
        #     df_train=df_train.sample(500,random_state=args.seed)
        # df_train=df_train[df_train.review_text.notnull()]
        df_test = pd.read_csv("./data/AmazonVideoGame_500_labeled_data.csv")
        df_train['review_text'] = df_train['review_text'].astype(str)
        train_texts = df_train['review_text'].values
        test_texts = df_test['review_text'].values

        y_test = df_test.label.values

    # Tokenize the texts
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_texts)
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    # Pad sequences to a fixed length
    max_len = 100
    X_train = pad_sequences(train_sequences, maxlen=max_len)
    X_test = pad_sequences(test_sequences, maxlen=max_len)
    np.save("./data/tokenized_data/"+args.data_set+"X_train", X_train)
    np.save("./data/tokenized_data/"+args.data_set+"X_test", X_test)
    print("data saved!!")
    # Define the LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=32, input_length=max_len))
    model.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Train the model
    model_history = model.fit(
        X_train, y_train, batch_size=128, epochs=args.epochs, validation_data=(X_test, y_test))

    # Evaluate the model on test data
    score, acc = model.evaluate(X_test, y_test, batch_size=128)
    print('Test accuracy:', acc)
    hist_df = pd.DataFrame(model_history.history)
    print(hist_df.head())
    hist_df.to_csv("./results/"+args.data_set+"_"+args.labeling_algorithm)


def get_results_row(exec_time):
    if(args.data_set=="modcloth"):
        result_Dataset="ModCloth"
    elif(args.data_set=="amazon_review"):
        result_Dataset="Amazon fashion"
    elif(args.data_set=="amazon_vid"):
        result_Dataset="Amazon Video"
    result_Neural_network = "Resnet20"
    result_labeling_algorithm=args.labeling_algorithm
    result_Epochs = args.epochs
    ep_logs = pd.read_csv("./results/"+args.data_set+"_"+args.labeling_algorithm)
    result_train_acc = max(ep_logs["accuracy"])
    result_test_acc = max(ep_logs["val_accuracy"])
    return [result_Dataset, result_Neural_network, result_labeling_algorithm, result_Epochs, result_train_acc, result_test_acc, exec_time]


if __name__ == '__main__':
    start_time = time.time()
    main()
    exec_time = (time.time() - start_time)
    if (not os.path.exists('./results.csv')):
        with open("./results.csv", 'a') as results_csv:
            csvwriter = csv.writer(results_csv)
            csvwriter.writerow(['Dataset',	'Neural network', 'Labeling Algorithm',
                               'Epochs',	'Train acc',	'Test acc', 'Execution time (in secs)'])
    with open("./results.csv", 'a') as results_csv:
        csvwriter = csv.writer(results_csv)
        csvwriter.writerow(get_results_row(exec_time))

from sklearn.linear_model import LogisticRegression, RidgeClassifier
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import f1_score
from cogdl.datasets import build_dataset_from_name
from cogdl import experiment
from sklearn import svm
from sklearn.utils import shuffle as skshuffle
import scipy.sparse as sp
import time
import os
import cogdl.models.emb.netmf

# from
class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = sp.lil_matrix(probs.shape)

        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            for label in labels:
                all_labels[i, label] = 1
        return all_labels


def evaluate_node_embeddings_using_liblinear(features_matrix, label_matrix, num_shuffle, training_percent):
    if len(label_matrix.shape) > 1:
        labeled_nodes = np.nonzero(np.sum(label_matrix, axis=1) > 0)[0]
        features_matrix = features_matrix[labeled_nodes]
        label_matrix = label_matrix[labeled_nodes]

    # shuffle, to create train/test groups
    shuffles = []
    for _ in range(num_shuffle):
        shuffles.append(skshuffle(features_matrix, label_matrix))

    # score each train/test group
    all_results = {"micro": [], "macro": []}

    # for train_percent in training_percents:
    for shuf in shuffles:
        X, y = shuf

        training_size = int(training_percent * len(features_matrix))
        X_train = X[:training_size, :]
        y_train = y[:training_size, :]

        X_test = X[training_size:, :]
        y_test = y[training_size:, :]

        clf = TopKRanker(LogisticRegression(solver="liblinear"))
        clf.fit(X_train, y_train)

        # find out how many labels should be predicted
        top_k_list = y_test.sum(axis=1).astype(np.int).tolist()
        preds = clf.predict(X_test, top_k_list)
        result1 = f1_score(y_test, preds, average="micro")
        result2 = f1_score(y_test, preds, average="macro")

        all_results["micro"].append(result1)
        all_results["macro"].append(result2)

    return np.mean(all_results["micro"]), np.mean(all_results["macro"])


def get_embedding_and_evaluate(model_name, dataset_name, save_path, train_percent, average=5):
    t1 = time.time()

    clf_multilabel = OneVsRestClassifier(LogisticRegression())
    if dataset_name in ["flickr", "flickr-ne"]:
        dataset_name = "flickr-ne"
    if dataset_name in ["ppi", "ppi-ne"]:
        dataset_name = "ppi-ne"
    dataset = build_dataset_from_name(dataset_name)

    if not os.path.exists(save_path):
        if model_name == "deepwalk":
            experiment(model=model_name, dataset=dataset_name, training_percents=[0.9], save_emb_path=save_path,
                       walk_length=40, walk_num=80, window_size=10)
        elif model_name == "node2vec":
            experiment(model=model_name, dataset=dataset_name, training_percents=[0.9], save_emb_path=save_path,
                       walk_length=40, walk_num=80, window_size=10, p=1, q=1)
        elif model_name == "line":
            # experiment(model=model_name, dataset=dataset_name, training_percents=[0.9], save_emb_path=save_path,
            #            negative=1, order=2, walk_length=1, walk_num=1, dimension=128)
            experiment(model="netmf", dataset=dataset_name, training_percents=[0.9], save_emb_path=save_path,
                       window_size=1, rank=256, dimension=128, negative=1, is_large=False)
        elif model_name == "netmf":
            if dataset_name in ["blogcatalog", "ppi-ne"]:
                print(1)
                experiment(model=model_name, dataset=dataset_name, training_percents=[0.9], save_emb_path=save_path,
                           window_size=10, rank=256, dimension=128, negative=1, is_large=False)
            elif dataset_name in ["flickr", "flickr-ne"]:
                experiment(model=model_name, dataset="flickr-ne", training_percents=[0.9], save_emb_path=save_path,
                           window_size=2, rank=256, dimension=128, negative=1, is_large=False)
        elif model_name == "netsmf":
            # num_round = m
            #
            print(1)
            if dataset_name in ["blogcatalog", "ppi-ne"]:
                experiment(model=model_name, dataset=dataset_name, training_percents=[0.9], save_emb_path=save_path,
                           num_round=500, hidden_size=128, window_size=2, worker=14, negative=1)
            elif dataset_name in ["flickr", "flickr-ne"]:
                experiment(model=model_name, dataset="flickr-ne", training_percents=[0.9], save_emb_path=save_path,
                           num_round=50, hidden_size=128, window_size=2, worker=16)

    train_data = np.load(save_path)
    total_label = np.array(dataset.data.y)

    mi_sum = 0
    ma_sum = 0
    for i in range (10):
        f1_micro, f1_macro = evaluate_node_embeddings_using_liblinear(train_data, total_label, 1,
                                                                  training_percent=train_percent)
        mi_sum+=f1_micro
        ma_sum+=f1_macro
    f1_micro = mi_sum/10
    f1_macro = ma_sum/10

    # train_data = np.load(save_path)
    # total_label = np.array(dataset.data.y)

    # # print(train_data.shape)
    # # print(total_label.shape)

    # X_train, X_test, y_train, y_test = train_test_split(train_data, total_label, train_size=train_percent)

    # clf_multilabel.fit(X_train, y_train)
    # y_pred = clf_multilabel.predict(X_test)

    # # 计算micro f1 score
    # f1_macro_list = []
    # f1_micro_list = []
    # for i in range(average):
    #     f1_micro_list.append(f1_score(y_test, y_pred, average='micro'))
    #     f1_macro_list.append(f1_score(y_test, y_pred, average='macro'))

    # f1_micro = np.mean(f1_micro_list)
    # f1_macro = np.mean(f1_macro_list)

    t2 = time.time()

    t = t2 - t1
    return t, f1_micro, f1_macro


# ress = []
# model_names = [ "fednetsmf"]
# tasks = ["ppi-ne"]
# for model_name in model_names:
#     for task in tasks:
#         print(f"model:{model_name}, task:{task}")
#         percents = []
#         if task in ["blogcatalog", "ppi-ne"]:
#             percents = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#         else:
#             percents = [0.01, 0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
#         for percent in percents:
#             t, f1_micro, f1_macro = get_embedding_and_evaluate(model_name,task,f"./embedding/{model_name}_{task}.npy",percent)
#             print(model_name, task, percent, t, f1_micro, f1_macro)
#             ress.append(f"{model_name},{task},{percent},{t},{f1_micro},{f1_macro}\n")
#             with open("res.txt","a") as fw:
#                 fw.write(f"{model_name},{task},{percent},{t},{f1_micro},{f1_macro}\n")

def evaluate(time, model_name, task, party=2, total_communication=0):
    ress = []
    # model_names = [ "fednetsmf"]
    # tasks = ["ppi-ne"]
    # for model_name in model_names:
    #     for task in tasks:
    print(f"model:{model_name}, task:{task}")
    percents = []
    if task in ["blogcatalog", "ppi"]:
        percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        percents = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    for percent in percents:
        if party == 2:
            t, f1_micro, f1_macro = get_embedding_and_evaluate(model_name, task, f"./embedding/{model_name}_{task}.npy",
                                                               percent)
        else:
            t, f1_micro, f1_macro = get_embedding_and_evaluate(model_name, task,
                                                               f"./embedding/{model_name}_{task}_{party}.npy", percent)
        print(model_name, task, percent, time + t, f1_micro, f1_macro, party,total_communication)
        ress.append(f"{model_name},{task},{percent},{time + t},{f1_micro},{f1_macro},{party},{total_communication}\n")
        with open("res.txt", "a") as fw:
            fw.write(f"{model_name},{task},{percent},{time + t},{f1_micro},{f1_macro},{party},{total_communication}\n")


# res = f"{}"
# print(res)
if __name__ == '__main__':
    evaluate(1, "netsmf", "blogcatalog", 2)

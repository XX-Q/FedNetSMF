import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from networkx.convert_matrix import *

from scipy.sparse import *
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

def read_network_from_edgelist(path):
    with open(path, "r") as f:
        edges = f.readlines()
        edges = [e[:-1].split(" ") for e in edges]
    edges = np.array(edges).T
    A = coo_matrix(edges)
    return A

# def read_network_from_mat(path):

def dataset_splitter(path, total_node, num_part, is_directed=True):
    """
    random split dataset into parties
    :param path:
    :param num_part:
    :return:
    """

    with open(path, "r") as f:
        all_data = f.readlines()
    if is_directed:
        random.shuffle(all_data)

        splitted_dataset = np.array_split(all_data, num_part)
        total_edge = len(all_data)

        for i in range(num_part):
            single_part = list(splitted_dataset[i])
            with open(f"{path[:-4]}_{i}_{num_part}_{total_node}_{total_edge}.txt", "w") as f:
                f.writelines(single_part)
                print(f"Sub Network {path[:-4]}_{i} Splitted Successfully")
    else:
        # all_data = [(int(ad[:-1].split(" ")[0]), int(ad[:-1].split(" ")[1])) for ad in all_data]
        # G = nx.from_edgelist(all_data)
        # all_data = list(G.edges())
        # # print(all_data)
        # all_data = [f"{ad[0]} {ad[1]}\n" for ad in all_data]
        #
        # random.shuffle(all_data)
        #
        # splitted_dataset = np.array_split(all_data, num_part)
        # total_edge = len(all_data)
        # for i in range(num_part):
        #     single_part = list(splitted_dataset[i])
        #     with open(f"{path[:-4]}_{i}_{num_part}_{total_node}_{total_edge}.txt", "w") as f:
        #         f.writelines(single_part)
        #         print(f"Sub Network {path[:-4]}_{i} Splitted Successfully")
        # all_data_group = [x[:-1].split(" ") for x in all_data]
        # all_data_group_note = []
        # for i in range(len(all_data_group)):
        #     if all_data_group[i] in all_data_group_note:
        #         continue
        #     if [all_data_group[i][1],all_data_group[i][0]] in all_data_group_note:
        #         continue
        #     if all_data_group[i][1] == all_data_group[i][0]:
        #         all_data_group_note.append(all_data_group[i])
        #         continue
        #     # print([all_data_group[i][1],all_data_group[i][0]])
        #     all_data_group_note.append(all_data_group[i])
        #     all_data_group_note.append([all_data_group[i][1],all_data_group[i][0]])

        # all_data_group_note = [x[:-1].split(" ") for x in all_data]
        # for i in tqdm(all_data_group_note):
        #     all_data_group_note.remove([i[1], i[0]])
        #     # print(i)
        #     if i[1] == i[0]:
        #         continue

        all_data_group_note = set()
        all_data = [x[:-1].split(" ") for x in all_data]

        for line in tqdm(all_data):
            u, v = map(int, line)
            edge = tuple(sorted([u, v]))

            if edge not in all_data_group_note:
                all_data_group_note.add(edge)



        with open(f"{path[:-4]}_unique.txt", "w") as f:
            all_data_group_note = [f"{x[0]} {x[1]}\n" for x in all_data_group_note]
            f.writelines(all_data_group_note)

        print(len(all_data_group_note))
        random.shuffle(all_data_group_note)

        splitted_dataset = np.array_split(all_data_group_note, num_part)
        total_edge = len(all_data)

        for i in range(num_part):
            single_part = list(splitted_dataset[i])
            with open(f"{path[:-4]}_{i}_{num_part}_{total_node}_{total_edge}.txt", "w") as f:
                single_part = [x for x in single_part]
                single_part_reverse = [x[:-1].split(" ") for x in single_part]
                single_part_reverse = [f"{x[1]} {x[0]}\n" for x in single_part_reverse]
                f.writelines(single_part)
                f.writelines(single_part_reverse)
                # single_part_reverse = []
                # for p in single_part:
                #     u, v = p.split(" ")
                #     single_part_reverse.append(f"{v} {u}")
                # f.writelines(single_part_reverse)
                print(f"Sub Network {path[:-4]}_{i} Splitted Successfully")


def dataset_loader(path):
    """
    load network from dataset file
    :param path:
    :return:
    """
    G = nx.read_edgelist(path, create_using=nx.DiGraph, nodetype=int)
    sub_network_nodes = list(G.nodes)
    sub_network_node_num = len(sub_network_nodes)
    G = nx.to_scipy_sparse_matrix(G)
    return G, sub_network_node_num


def sub_dataset_loader(path):
    """
    load sub network from sub dataset file
    :param path: path of sub network dataset. format: name_party_totalParty_totalNodesOfCompleteNetwork.txt
    :return:
    """
    G = nx.read_edgelist(path, create_using=nx.DiGraph, nodetype=int)
    total_node_num = path[:-4].split("_")[-1]
    sub_network_nodes = list(G.nodes)
    sub_network_node_num = len(sub_network_nodes)
    for node in range(total_node_num):
        if node not in sub_network_nodes:
            G.add_node(node)
    G = nx.to_scipy_sparse_matrix(G)
    return G, sub_network_node_num

def embedding_result_format(embedding_matrix):
    """
    format output embedding result
    :param embedding_matrix:
    :return:
    """
    res = {}
    for i in range(len(embedding_matrix)):
        res[str(i)] = list(embedding_matrix[i])
    return res




if __name__ == '__main__':
    total_node_dict = {
        "blogcatalog":10312,
        "ppi-ne":3890,
        "flickr-ne":80513
    }
    dataset_splitter("datasets/ppi/edges.txt", total_node_dict["ppi-ne"], 2)
    # dataset_splitter("datasets/blogcatalog/edges.txt", total_node_dict["blogcatalog"], 2,False)
    # dataset_splitter("datasets/flickr/edges.txt", total_node_dict["flickr-ne"], 2,False)

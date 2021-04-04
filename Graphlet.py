import pandas as pd
import networkx as nx
from DAO import DAO
import numpy as np
import os
import glob


# func that writes edges and nodes in each model to a txt file. use those files in R
def write_graph_to_txt(graph, model_ID):
    nx.write_edgelist(graph, "GraphletsEdges/edgelist{0}.txt".format(model_ID), delimiter=' ', data=[])
    x = graph.nodes()
    nodes = ""
    for y in x:
        nodes += str(y) + '\n'
    file = open("GraphletsNodes/nodelist{0}.txt".format(model_ID), "w")
    file.write(nodes)
    file.close()


# a func that checks for cycles in a model
def check_cycle(graph):
    import matplotlib.pyplot as plt
    G = nx.DiGraph(graph.edges())
    print(len(list(nx.simple_cycles(G))))
    nx.draw(G, pos=nx.spring_layout(G))
    plt.draw()


# a func that combines all output csv files from R into one csv file
def combine_csv():
    os.chdir("C:/Users/albil/Documents/GitHub/OCLRecommendation/GraphletsCSV")
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # export to csv
    combined_csv.rename(columns={"Unnamed: 0": "ObjectID"}, inplace=True)
    combined_csv.sort_values(by=['ObjectID'], inplace=True)
    combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')







class Graphlet():
    def __init__(self):
        self.MODELS_NUMBER = 319
        self.dao = DAO()
        self.df_objects = self.dao.getObjects()

    # a method that converts model's relations to a graph
    def model2graph(self, graph, model_ID):
        result = self.df_relations.loc[self.df_relations['ModelID'] == model_ID]
        for index, relation in result.iterrows():
            if int(relation[0]) != int(relation[2]):
                graph.add_edge(int(relation[0]), int(relation[2]))
        write_graph_to_txt(graph, model_ID)
        # check_cycle(graph)
        return graph

    # a method that creates a list of all graphs
    def embedd_and_write(self):
        lst = list()
        for i in range(1, self.MODELS_NUMBER):
            graph = nx.Graph()
            lst.append(self.model2graph(graph, i))
        return lst

    def run(self):
        relations_df_cols_to_retain = ['ObjectID1', 'ModelID', 'ObjectID2']
        df_relations = pd.read_csv("relations_final.csv").sort_values(by=['ObjectID1'])
        self.df_relations = df_relations[relations_df_cols_to_retain]
        self.df_relations = self.df_relations.astype(int)
        self.df_relations.reset_index(drop=True, inplace=True)
        lst_graph = self.embedd_and_write()
        return lst_graph

    def csv_to_features(self):
        graphlets = pd.read_csv("combined_csv.csv")
        graphlets['ObjectID'] = graphlets["ObjectID"].str.split(",").str.get(0).str.split("[").str.get(1)
        graphlets['ObjectID'] = graphlets['ObjectID'].astype(int)
        graphlets.sort_values(by=['ObjectID'], inplace=True)
        updated_objs = pd.merge(self.df_objects, graphlets, on='ObjectID', how='left')
        print(updated_objs)
        updated_objs.fillna(0, inplace=True)
        updated_objs = updated_objs.reset_index()
        print(updated_objs)
        features_to_retain = [ "O" + str(i) for i in range(0,73) ]
        # features_to_retain.append("ObjectID")
        updated_objs = updated_objs[features_to_retain]
        updated_objs.to_csv("final_graphlet_features.csv", index=False)
        return updated_objs


g = Graphlet()
# x = g.run()
# combine_csv()
g.csv_to_features()
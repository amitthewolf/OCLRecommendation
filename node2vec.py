import pandas as pd
import networkx as nx
import nodevectors
import math
from DAO import DAO
import umap

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#from karateclub import TENE

class node2vec():
    def __init__(self, features_num, use_attributes_flag, use_inheritance_flag, return_weight,walklen, epochs):
        self.MODELS_NUMBER = 319
        self.dao = DAO()
        self.df_objects = self.dao.getObjects()
        self.features_num = features_num
        self.use_atts = use_attributes_flag
        self.use_inher = use_inheritance_flag
        self.return_weight = return_weight
        self.walklen = walklen
        self.epochs = epochs
        # self.createRelationsDF()

        # function that adds inheritance edges

    def inherit_edges(self, graph):
        y = set(self.df_objects['ObjectID'].values.flatten())
        for index, row in self.df_objects.iterrows():
            if str(row[10]).isdigit() and row[10] in y:
                graph.add_edge(str((row[0])), str((row[10])), edge1=row['ObjectID'], edge2=row['inheriting_from'],
                               inheritance='True')

    # function that adds nodes, edges according to modelID
    def model2graph(self, graph, model_ID):
        result = self.df_relations.loc[self.df_relations['ModelID'] == model_ID]
        for index, relation in result.iterrows():
            if not graph.has_node(relation[0]) and self.use_atts == 'True':
                atts = self.df_objects.loc[self.df_objects['ObjectID'] == relation[0]]['properties_names'].iloc[0]
                atts = atts.split(',')
                atts = {str(v): k for v, k in enumerate(atts)}
                graph.add_node(str(relation[0]), model_ID=model_ID, object_ID=relation[0], **atts)
            if not graph.has_node(relation[2]) and self.use_atts == 'True':
                atts = self.df_objects.loc[self.df_objects['ObjectID'] == relation[2]]['properties_names'].iloc[0]
                atts = atts.split(',')
                atts = {str(v): k for v, k in enumerate(atts)}
                graph.add_node(str(relation[2]), model_ID=model_ID, object_ID=relation[2], **atts)
            graph.add_edge(str(relation[0]), str(relation[2]), edge1=relation[0], edge2=relation[2])

    def createRelationsDF(self):
        df_relations = pd.read_sql("Select ObjectID1,ModelID, ObjectID2 from relations", self.dao.conn)
        df_relations = df_relations.dropna()

        # remove objects that exist in relations but don't exist in object tables
        y = set(self.df_objects['ObjectID'].values.flatten())
        for index, row in df_relations.iterrows():
            if row[0] not in y or row[2] not in y:
                df_relations.drop(index, inplace=True)

        # add objects from objects table that don't exist in relations table
        z1 = set(df_relations['ObjectID1'].values.flatten())
        z2 = set(df_relations['ObjectID2'].values.flatten())
        z = z1 | z2
        df = pd.DataFrame([], columns=['ObjectID1', 'ModelID', 'ObjectID2'])
        for index, row in self.df_objects.iterrows():
            if row[0] not in z:
                df = df.append({'ObjectID1': row[0], 'ModelID': row[1], 'ObjectID2': row[0]}, ignore_index=True)
        df_relations = pd.concat([df, df_relations], axis=0)

        df_relations = df_relations.dropna()
        df_relations = df_relations.drop_duplicates()

        df_relations.to_csv('relations_final.csv')

    def embedd_and_write(self,features_num):
        # init a graph
        graph = nx.Graph()
        # add all nodes and edges to the graph
        for i in range(1, self.MODELS_NUMBER):
            self.model2graph(graph, i)

        if self.use_inher == 'True':
            self.inherit_edges(graph)

        H = nx.Graph()
        H.add_nodes_from(sorted(graph.nodes(data=True)))
        H.add_edges_from(graph.edges(data=True))

        # check if there are any objects that don't overlap.
        # n = set(H.nodes)
        # m = set(df_objects['ObjectID'].values.flatten())
        # print(n ^ m)

        # fit node2vec
        node2vec_model = nodevectors.Node2Vec(n_components=int(features_num), return_weight=float(self.return_weight),
                                              walklen=int(self.walklen), epochs=int(self.epochs))
        # embeddings = node2vec_model.fit_transform(H)
        node2vec_model.fit(H)
        y = H.nodes
        lst = list()
        for x in y:
            lst.append(node2vec_model.predict(str(x)))
        embeddings_col_names = ['N2V_' + str(i) for i in range(1, int(features_num)+1)]
        embeddings_df = pd.DataFrame(data=lst, columns=embeddings_col_names)
        merged_df = pd.concat((self.df_objects, embeddings_df), axis=1)

        # plt.scatter(
        #     lst[:, 0],
        #     lst[:, 1],
        #     c=[sns.color_palette()[x] for x in merged_df.ModelID.map()])
        # plt.gca().set_aspect('equal', 'datalim')
        # plt.title('UMAP projection of the ThreeEyes dataset', fontsize=24)


        return merged_df
        
        # updating DB with new columns
        #self.dao.rewriteObjectTable(merged_df)
        
        # n2v_model = TENE()
        # n2v_model.fit(H,self.df_objects)
        # embddd = n2v_model.get_embedding()


    def run(self):
        relations_df_cols_to_retain = ['ObjectID1', 'ModelID', 'ObjectID2']
        df_relations = pd.read_csv("relations_final.csv").sort_values(by=['ObjectID1'])
        self.df_relations = df_relations[relations_df_cols_to_retain]
        df = self.embedd_and_write(self.features_num)
        return df
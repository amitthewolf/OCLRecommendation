from configparser import ConfigParser

import pandas as pd
import networkx as nx
import nodevectors
import math
from DAO import DAO
import umap

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from karateclub import TENE
from karateclub import FSCNMF
from karateclub import GraphWave


class node2vec():
    def __init__(self, features_num, use_attributes_flag, use_inheritance_flag, return_weight, walklen, epochs,
                 neighbor_weight, use_pca, pca_num):
        self.config = ConfigParser()
        self.config.read('conf.ini')
        self.paths = self.config['paths']
        self.dao = DAO()
        self.MODELS_NUMBER = self.dao.get_models_number()
        self.df_objects = self.dao.getObjects()
        self.features_num = features_num
        self.use_atts = use_attributes_flag
        self.use_inher = use_inheritance_flag
        self.return_weight = return_weight
        self.walklen = walklen
        self.epochs = epochs
        self.neighbor_weight = neighbor_weight

        self.use_pca = use_pca
        self.pca_num = pca_num
        # self.createRelationsDF()

        # function that adds inheritance edges

    def inherit_edges(self, graph):
        y = set(self.df_objects['ObjectID'].values.flatten())
        for index, row in self.df_objects.iterrows():
            if str(row[10]).isdigit() and row[10] in y:
                graph.add_edge(int((row[0])), int((row[10])), edge1=row['ObjectID'], edge2=row['inheriting_from'],
                               inheritance='True')

    # function that adds nodes, edges according to modelID
    def model2graph(self, graph, model_ID):
        result = self.df_relations.loc[self.df_relations['ModelID'] == model_ID]
        for index, relation in result.iterrows():
            if not graph.has_node(relation[0]) and self.use_atts == 'True':
                atts = self.df_objects.loc[self.df_objects['ObjectID'] == relation[0]]['properties_names'].iloc[0]
                atts = atts.split(',')
                atts = {str(v): k for v, k in enumerate(atts)}
                graph.add_node(int(relation[0]), model_ID=model_ID, object_ID=relation[0], **atts)
            if not graph.has_node(relation[2]) and self.use_atts == 'True':
                atts = self.df_objects.loc[self.df_objects['ObjectID'] == relation[2]]['properties_names'].iloc[0]
                atts = atts.split(',')
                atts = {str(v): k for v, k in enumerate(atts)}
                graph.add_node(int(relation[2]), model_ID=model_ID, object_ID=relation[2], **atts)
            graph.add_edge(int(relation[0]), int(relation[2]), edge1=relation[0], edge2=relation[2])

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

        df_relations.to_csv(self.paths['RELATIONS'])

    def embedd_and_write(self, features_num):
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

        # fit node2vec

        # H = nx.convert_node_labels_to_integers(H, first_label=0)
        # n2v_model = TENE(dimensions=features_num, alpha=0.5)
        # df_objects_copy = self.prepareObjectDF()
        # n2v_model.fit(H,df_objects_copy)
        # embeddings = n2v_model.get_embedding()

        node2vec_model = nodevectors.Node2Vec(n_components=int(features_num), return_weight=float(self.return_weight),
                                              walklen=int(self.walklen), epochs=int(self.epochs),
                                              neighbor_weight=float(self.neighbor_weight))
        node2vec_model.fit(H)
        y = H.nodes
        z = H.nodes.data('model_ID')
        embeddings = list()
        for x in y:
            embeddings.append(node2vec_model.predict(str(x)))

        if self.use_pca == 'True':
            pca = PCA(n_components=int(self.pca_num))
            # modi = TSNE(n_components=int(self.pca_num), random_state=42)
            # tsne_data = modi.fit_transform(embeddings)
            print('reducing dimensions')
            pca_data = pca.fit_transform(embeddings)
            embeddings_col_names = ['N2V_' + str(i) for i in range(1, int(self.pca_num) + 1)]
            embeddings_df = pd.DataFrame(data=pca_data, columns=embeddings_col_names)
            merged_df = pd.concat((self.df_objects, embeddings_df), axis=1)
        else:
            embeddings_col_names = ['N2V_' + str(i) for i in range(1, int(features_num) + 1)]
            embeddings_df = pd.DataFrame(data=embeddings, columns=embeddings_col_names)
            merged_df = pd.concat((self.df_objects, embeddings_df), axis=1)
        # self.plot_embedding(embeddings, z)

        return merged_df

        # updating DB with new columns
        # self.dao.rewriteObjectTable(merged_df)

        # check if there are any objects that don't overlap.
        # n = set(H.nodes)
        # m = set(df_objects['ObjectID'].values.flatten())
        # print(n ^ m)

    def run(self):
        relations_df_cols_to_retain = ['ObjectID1', 'ModelID', 'ObjectID2']
        df_relations = pd.read_csv(self.paths['RELATIONS']).sort_values(by=['ObjectID1'])
        self.df_relations = df_relations[relations_df_cols_to_retain]
        self.df_relations = self.df_relations.astype(int)
        df = self.embedd_and_write(self.features_num)
        return df

    def prepareObjectDF(self):
        df_objects_copy = pickle.loads(pickle.dumps(self.df_objects))
        objects_df_cols_to_retain = ['RelationNum', 'AttributeNum', 'is_abstract']
        df_objects_copy = df_objects_copy[objects_df_cols_to_retain]
        df_objects_copy = df_objects_copy.astype(int)
        return df_objects_copy

    def plot_embedding(self, model, labels):
        data_1000 = model
        labels_1000 = [i[1] for i in labels]

        modi = TSNE(n_components=2, random_state=42)
        tsne_data = modi.fit_transform(data_1000)
        # stnadard_embedding = umap.UMAP(random_state=42).fit_transform(model)
        # plt.scatter(stnadard_embedding[:,0], stnadard_embedding[:,1], s=0.1)

        tsne_data = np.vstack((tsne_data.T, labels_1000)).T
        tsne_df = pd.DataFrame(data=tsne_data, columns=('x', 'y', 'label'))

        sns.FacetGrid(tsne_df, hue='label', size=6).map(plt.scatter, 'x', 'y')
        plt.show()
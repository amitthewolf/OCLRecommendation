import pandas as pd
import networkx as nx
import nodevectors
import math
from DAO import DAO


class node2vec():
    def __init__(self, features_num):
        self.MODELS_NUMBER = 319
        self.dao = DAO()
        self.df_objects = self.dao.getObjects()
        self.features_num = features_num
        # self.createRelationsDF()

    # function that adds nodes, edges according to modelID
    def model2graph(self, graph, model_ID):
        result = self.df_relations.loc[self.df_relations['ModelID'] == model_ID]
        for index, relation in result.iterrows():
            if not math.isnan(relation[2]) and not math.isnan(relation[0]):
                graph.add_edge(relation[0], relation[2], edge1=relation[0], edge2=relation[2])
                graph.add_node(relation[0], model_ID=model_ID, object_ID=relation[0])
                graph.add_node(relation[2], model_ID=model_ID, object_ID=relation[2])

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

        H = nx.Graph()
        H.add_nodes_from(sorted(graph.nodes(data=True)))
        H.add_edges_from(graph.edges(data=True))

        # check if there are any objects that don't overlap.
        # n = set(H.nodes)
        # m = set(df_objects['ObjectID'].values.flatten())
        # print(n ^ m)

        # fit node2vec
        ggvec_model = nodevectors.GGVec(n_components=int(features_num))
        embeddings = ggvec_model.fit_transform(H)

        embeddings_col_names = ['N2V_' + str(i) for i in range(1, int(features_num)+1)]
        embeddings_df = pd.DataFrame(data=embeddings, columns=embeddings_col_names)
        merged_df = pd.concat((self.df_objects, embeddings_df), axis=1)

        return merged_df

        # updating DB with new columns
        #self.dao.rewriteObjectTable(merged_df)


    def run(self):
        relations_df_cols_to_retain = ['ObjectID1', 'ModelID', 'ObjectID2']
        df_relations = pd.read_csv("relations_final.csv").sort_values(by=['ObjectID1'])
        self.df_relations = df_relations[relations_df_cols_to_retain]
        df = self.embedd_and_write(self.features_num)
        return df
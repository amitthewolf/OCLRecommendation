
import pandas as pd
import networkx as nx
import nodevectors
import math
from DAO import DAO

# function that adds nodes, edges according to modelID
def model2graph(graph, model_ID):

    result = df_relations.loc[df_relations['ModelID'] == model_ID]
    for index, relation in result.iterrows():
      if not math.isnan(relation[2]) and not math.isnan(relation[0]):
        graph.add_edge(relation[0], relation[2], edge1=relation[0], edge2=relation[2])
        graph.add_node(relation[0], model_ID=model_ID, object_ID=relation[0])
        graph.add_node(relation[2], model_ID=model_ID, object_ID=relation[2])

def createRelationsDF():
  df_relations = pd.read_sql("Select ObjectID1,ModelID, ObjectID2 from relations", dao.conn)
  df_relations = df_relations.dropna()


  # add objects from objects table that dont exist in relations table
  i = 256453
  for index, row in df_objects.iterrows():
    if row[0] not in df_relations.values:
      df_relations.loc[i] = [row[0],row[1],row[0]]
      i+=1

  df_relations = df_relations.dropna()
  df_relations = df_relations.drop_duplicates()

  #remove objects that exist in relations but doesnt exist in object tables

  for index, row in df_relations.iterrows():
    if row[0] not in df_objects.values or row[2] not in df_objects.values:
      df_relations.drop(df_relations.index[index])
  #
  df_relations.to_csv('relations_final.csv')


MODELS_NUMBER = 319
dao = DAO()
df_objects = dao.getObjects()

relations_df_cols_to_retain = ['ObjectID1','ModelID', 'ObjectID2']
df_relations = pd.read_csv("relations_final.csv").sort_values(by=['ObjectID1'])
df_relations = df_relations[relations_df_cols_to_retain]

# init a graph
graph = nx.Graph()
# add all nodes and edges to the graph
for i in range(1, MODELS_NUMBER):
  model2graph(graph, i)
H = nx.Graph()
H.add_nodes_from(sorted(graph.nodes(data=True)))
H.add_edges_from(graph.edges(data=True))

# fit node2vec
ggvec_model = nodevectors.GGVec()
embeddings = ggvec_model.fit_transform(H)

embeddings_col_names = [ 'N2V_' + str(i) for i in range(1,33) ]
embeddings_df = pd.DataFrame(data=embeddings,columns=embeddings_col_names)
merged_df = pd.concat((df_objects,embeddings_df),axis=1)

# updating DB with new columns
# dao.rewriteObjectTable(merged_df)

# merged_df.to_csv('object_final.csv')




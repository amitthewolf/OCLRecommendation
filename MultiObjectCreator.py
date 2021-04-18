from DAO import DAO
import itertools
import pandas as pd
from itertools import product



class MultiObjectCreator:

    def __init__(self):
        self.dao = DAO()

        self.ref_const_df = self.dao.get_const_ref()
        self.my_df = pd.DataFrame()

    def run(self,df,features):

        self.df_objects = df
        all_models_ids = self.df_objects['ModelID'].unique()

        for model_id in all_models_ids:

            all_obj_ids_in_model = self.df_objects.loc[self.df_objects['ModelID'] == model_id]['ObjectID'].tolist()
            obj_ids_in_model_with_consts = self.ref_const_df.loc[self.ref_const_df['ModelID'] == model_id]['ObjectID'].tolist()
            obj_ids_in_model_with_no_const = list(set(all_obj_ids_in_model) ^ set(obj_ids_in_model_with_consts))
            consts_ids_in_model = self.ref_const_df.loc[self.ref_const_df['ModelID'] == model_id]['ConstraintID'].unique()

            positive_pairs_ctr = 0
            negative_pairs_ctr = 0
            for const_id_in_model in consts_ids_in_model:
                objs_in_const_df = self.ref_const_df.loc[self.ref_const_df['ConstraintID'] == const_id_in_model]
                objs_in_const_number = objs_in_const_df.shape[0]

                if objs_in_const_number > 1:
                    objects_ids_in_constraint = objs_in_const_df['ObjectID'].tolist()
                    for positive_subset in itertools.combinations(objects_ids_in_constraint, 2):
                        obj_df_1 = self.df_objects.loc[self.df_objects['ObjectID'] == positive_subset[0]].reset_index(drop=True)
                        obj_df_2 = self.df_objects.loc[self.df_objects['ObjectID'] == positive_subset[1]].reset_index(drop=True)
                        if not obj_df_1.empty and not obj_df_2.empty:
                            new_df = obj_df_1.join(obj_df_2, lsuffix="_1", rsuffix="_2")
                            new_df['ContainsConstraints'] = 1
                            self.my_df = pd.concat([self.my_df,new_df], axis=0)
                            positive_pairs_ctr += 1

            if positive_pairs_ctr > 0:
                combinations = list(product(obj_ids_in_model_with_no_const,obj_ids_in_model_with_consts ,repeat=1))
                if len(combinations) == 0:
                    self.my_df = self.my_df[self.my_df['ModelID_1'] != model_id]
                for negative_subset in combinations:
                    if negative_pairs_ctr == positive_pairs_ctr:
                        break
                    obj_df_1 = self.df_objects.loc[self.df_objects['ObjectID'] == negative_subset[0]].reset_index(drop=True)
                    obj_df_2 = self.df_objects.loc[self.df_objects['ObjectID'] == negative_subset[1]].reset_index(drop=True)
                    new_df = obj_df_1.join(obj_df_2, lsuffix="_1", rsuffix="_2")
                    new_df['ContainsConstraints'] = 0
                    if not obj_df_1.empty and not obj_df_2.empty:
                        self.my_df = pd.concat([self.my_df, new_df], axis=0)
                        negative_pairs_ctr += 1

        self.my_df.to_csv("pairs_df.csv")

        final_features = []
        for feature in features:
            final_features.append(feature + str("_1"))
            final_features.append(feature + str("_2"))

        return self.my_df, final_features


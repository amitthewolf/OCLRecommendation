from DAO import DAO
import itertools
import pandas as pd
from itertools import product
import random
from tqdm import tqdm

class GroupCreator:

    def __init__(self):
        self.dao = DAO()
        self.ref_const_df = self.dao.get_const_ref()
        self.my_df = pd.DataFrame()
        self.max_pos_pairs_ctr = 0
        self.max_neg_pairs_ctr = 20
        self.deleted_models_num = 0
        self.total_pos = 0
        self.total_neg = 0
        self.not_found = 0
        self.GROUP_SIZE = 3
        self.target = ''

    def create_groups_df(self, df, target):
        self.target = target
        self.df_objects = df
        all_models_ids = self.ref_const_df['ModelID'].unique()

        print("Creating pairs dataframe ...")
        for model_id in tqdm(all_models_ids):

            positive_groups_in_model_number = 0
            negative_groups_in_model_number = 0
            positive_combinations = set()

            consts_ids_in_model = self.ref_const_df.loc[self.ref_const_df['ModelID'] == model_id]['ConstraintID'].unique()

            for const_id_in_model in consts_ids_in_model:
                objs_in_const_df = self.ref_const_df.loc[self.ref_const_df['ConstraintID'] == const_id_in_model]
                objs_in_const_number = objs_in_const_df.shape[0]

                # ADDING POSITIVE RECORDS
                if objs_in_const_number > 1:
                    object_ids_in_const_set = set(objs_in_const_df['ObjectID'].tolist())

                    for positive_group in itertools.combinations(sorted(object_ids_in_const_set), self.GROUP_SIZE):
                        positive_combinations.add(positive_group)
                        positive_groups_in_model_number = self.create_group_record(positive_group, 1, model_id, positive_groups_in_model_number)

            # IF POSITIVE RECORDS ADDED, CREATE ALSO NEGATIVE RECORDS
            if positive_groups_in_model_number > 0:
                all_obj_ids_in_model = self.df_objects.loc[self.df_objects['ModelID'] == model_id]['ObjectID'].tolist()
                negative_combs_set = self.create_negative_combinations(all_obj_ids_in_model,positive_combinations)
                # if cant create equal number of negative records, delete positive pairs
                if len(negative_combs_set) < positive_groups_in_model_number:
                    self.my_df = self.my_df[self.my_df['ModelID_1'] != model_id]
                    self.deleted_models_num += 1
                    continue
                for negative_group in negative_combs_set:
                    # STOP WHEN POSITIVE RECORDS NUM = NEGATIVE RECORDS NUM
                    if negative_groups_in_model_number == self.max_neg_pairs_ctr:
                        break
                    negative_groups_in_model_number = self.create_group_record(negative_group, 0, model_id, negative_groups_in_model_number)

        for i in range(1, self.GROUP_SIZE + 1):
            self.my_df = self.my_df.drop("ModelID_{}".format(i),axis=1)

        self.print_pairs_creation_stats()

        return self.my_df

    def print_pairs_creation_stats(self):
        print('-' * 50)
        print("pairs creation stats: \n")
        print("Input : {} Models". format(len(self.ref_const_df['ModelID'].unique())))
        print("Output : {} Models ". format(len(self.my_df['ModelID'].unique())))
        print(" Number of deleted models : {}" .format(self.deleted_models_num))
        print(" Number of positive records found in objects dataframe : {}" .format(self.total_pos))
        print(" Number of negative records found in objects dataframe : {}" .format(self.total_neg))
        print(" Number of negative pairs created : {}" .format(self.my_df.groupby(self.target).size()[0]))
        print(" Number of positive pairs created : {}" .format(self.my_df.groupby(self.target).size()[1]))

    def subtract_sets(self, x, y):
        final_set = set()
        while y:
            sub = y.pop()
            if sub in x:
                pass
            elif (sub[1], sub[0]) in x:
                pass
            else:
                final_set.add(sub)
        return final_set


    def get_features(self,features):
        final_features = []
        for feature in features:
            for i in range(1,self.GROUP_SIZE + 1):
                final_features.append(feature + str("_{}".format(i)))
        return final_features


    def add_pair(self, obj_id_1, obj_id_2, target_var,model_id,ctr):
        obj_1 = self.df_objects.loc[self.df_objects['ObjectID'] == obj_id_1].reset_index(drop=True)
        obj_2 = self.df_objects.loc[self.df_objects['ObjectID'] == obj_id_2].reset_index(drop=True)
        if (not obj_1.empty) and (not obj_2.empty):
            new_df = obj_1.join(obj_2, lsuffix="_1", rsuffix="_2")
            new_df['PairInConstraint'] = target_var
            new_df['ModelID'] = model_id
            self.my_df = pd.concat([self.my_df, new_df], axis=0)
            if target_var == 1:
                self.total_pos += 1
            if target_var == 0:
                self.total_neg += 1
            ctr += 1
        else:
            self.not_found += 1
        return ctr

    def create_group_record(self, obj_ids, target_var, model_id, ctr):

        objects_records = []
        for i,ID in enumerate(obj_ids):
            obj_from_db = self.df_objects.loc[self.df_objects['ObjectID'] == ID].reset_index(drop=True)
            if not obj_from_db.empty:
                objects_records.append(obj_from_db.add_suffix("_{}".format(i + 1)))


        if len(obj_ids) == len(objects_records):

            # objects_with_suffix = self.check_if_objects_exist_in_db(objects_records)

            if objects_records :
                new_df = pd.concat(objects_records, axis=1)
                new_df['PairInConstraint'] = target_var
                new_df['ModelID'] = model_id
                self.my_df = pd.concat([self.my_df, new_df], axis=0)
                if target_var == 1:
                    self.total_pos += 1
                if target_var == 0:
                    self.total_neg += 1
                ctr += 1
            else:
                self.not_found += 1
            return ctr


    def check_if_objects_exist_in_db(self,objs):
        objects_with_suffix = []
        for i,obj in enumerate(objs):
            if not obj.empty:
                objects_with_suffix.append(obj.add_suffix("_{}".format(i + 1)))
            else:
                return None
        return objects_with_suffix

    def create_negative_combinations(self, all_obj_ids_in_model, positive_combinations):
        all_combinations = set(itertools.combinations(sorted(all_obj_ids_in_model),  self.GROUP_SIZE))
        combinations = set(positive_combinations ^ all_combinations)
        combinations = list(combinations)
        random.shuffle(combinations)
        return combinations
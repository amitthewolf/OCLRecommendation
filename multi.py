from DAO import DAO
import itertools
import pandas as pd
from itertools import product


dao = DAO()
df_objects = dao.getObjects()
ref_const_df = dao.get_const_ref()

# constraint_ids = ref_const_df['ConstraintID'].unique()
x = 0
my_df = pd.DataFrame()

# all_models_ids = df_objects['ModelID'].unique()
all_models_ids = [1,2,3,4]


for model_id in all_models_ids:

    all_obj_ids_in_model = df_objects.loc[df_objects['ModelID'] == model_id]['ObjectID'].tolist()
    obj_ids_in_model_with_consts = ref_const_df.loc[ref_const_df['ModelID'] == model_id]['ObjectID'].tolist()
    obj_ids_in_model_with_no_const = list(set(all_obj_ids_in_model) ^ set(obj_ids_in_model_with_consts))
    consts_ids_in_model = ref_const_df.loc[ref_const_df['ModelID'] == model_id]['ConstraintID'].unique()

    positive_pairs_ctr = 0
    negative_pairs_ctr = 0
    for const_id_in_model in consts_ids_in_model:
        objs_in_const_df = ref_const_df.loc[ref_const_df['ConstraintID'] == const_id_in_model]
        objs_in_const_number = objs_in_const_df.shape[0]

        if objs_in_const_number > 1:
            print('ConstraintID ', const_id_in_model)
            objects_ids_in_constraint = objs_in_const_df['ObjectID'].tolist()
            objects_ids_not_in_constraint = list(set(all_obj_ids_in_model) ^ set(objects_ids_in_constraint))
            for positive_subset in itertools.combinations(objects_ids_in_constraint, 2):
                obj_df_1 = df_objects.loc[df_objects['ObjectID'] == positive_subset[0]].reset_index(drop=True)
                obj_df_2 = df_objects.loc[df_objects['ObjectID'] == positive_subset[1]].reset_index(drop=True)
                new_df = obj_df_1.join(obj_df_2, lsuffix="_1", rsuffix="_2")
                my_df = pd.concat([my_df,new_df], axis=0)
                positive_pairs_ctr += 1
                print(my_df)
    if positive_pairs_ctr > 0:
        combinations = list(product(obj_ids_in_model_with_no_const, obj_ids_in_model_with_consts, repeat=1))
        # for negative_subset in itertools.combinations(obj_ids_in_model_with_no_const, 2):
        for negative_subset in combinations:
            if negative_pairs_ctr == positive_pairs_ctr:
                break
            obj_df_1 = df_objects.loc[df_objects['ObjectID'] == negative_subset[0]].reset_index(drop=True)
            obj_df_2 = df_objects.loc[df_objects['ObjectID'] == negative_subset[1]].reset_index(drop=True)
            new_df = obj_df_1.join(obj_df_2, lsuffix="_1", rsuffix="_2")
            my_df = pd.concat([my_df, new_df], axis=0)
            negative_pairs_ctr += 1
            print(my_df)


    print('*' * 50)

print(my_df.shape)
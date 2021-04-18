from DAO import DAO
import itertools
import pandas as pd

dao = DAO()
df_objects = dao.getObjects()
ref_const_df = dao.get_const_ref()

constraint_ids = ref_const_df['ConstraintID'].unique()
x = 0
my_df = pd.DataFrame()
for const in constraint_ids:
    objs_in_const = ref_const_df.loc[ref_const_df['ConstraintID'] == const]

    if objs_in_const.shape[0] > 1:
        print('ConstraintID ', const)

        model_id = objs_in_const['ModelID'].iloc[0]
        objects_in_model = df_objects.loc[df_objects['ModelID'] == model_id]['ObjectID'].tolist()
        objects_in_constraint = objs_in_const['ObjectID'].tolist()
        objects_not_in_constraint = list(set(objects_in_model) ^ set(objects_in_constraint))
        for subset in itertools.combinations(objects_in_constraint, 2):
            obj_df_1 = df_objects.loc[df_objects['ObjectID'] == subset[0]].reset_index(drop=True)
            obj_df_2 = df_objects.loc[df_objects['ObjectID'] == subset[1]].reset_index(drop=True)
            new_df = obj_df_1.join(obj_df_2, lsuffix="_1", rsuffix="_2")
            my_df = pd.concat([my_df,new_df], axis=0)
            # for obj in subset:
            #     obj_df = df_objects.loc[df_objects['ObjectID'] == obj]
            #     print(obj_df)
            #     print(type(obj_df))

                # print(subset)
        print('*' * 50)

print(x)
print(my_df.shape)
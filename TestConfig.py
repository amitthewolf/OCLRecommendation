from configparser import ConfigParser
from itertools import product
import random

class TestConfig:

    def __init__(self,graphlets_flag,models_number,test_method,random=True):
        self.models_number = models_number
        self.config = ConfigParser()
        self.config.read('conf.ini')
        self.classifier_section = self.config['classifier']
        self.n2v_section = self.config['node2vec']

        self.n2v_flag = self.n2v_section['n2v_flag']
        fixed_section = self.config['fixed_params']
        self.random = fixed_section['random']
        self.graphlet_flag = graphlets_flag
        self.target = self.classifier_section['Target'].split(',')
        self.parse_classifier_params(self.classifier_section)
        self.method = test_method
        self.pairs_creation_flag = self.config['pairs']['re_create_df']

        if self.n2v_flag == 'True':
            self.parse_n2v_params(self.n2v_section)
        if self.random == 'False':
            self.create_grid_search_combs()


    def parse_classifier_params(self,classifier_section):
        self.sampling_strategy_list = classifier_section['sampling'].split(',')
        self.target_list = classifier_section['Target'].split(',')

        test_ratio_list = classifier_section['test_ratio'].split(',')
        self.test_ratio_list = [ float(i) for i in test_ratio_list ]
        cross_val_k_list = classifier_section['cross_val_k'].split(',')
        self.cross_val_k_list = [ int(i) for i in cross_val_k_list ]

    def parse_n2v_params(self, n2v_section):
        n2v_features_num_list = n2v_section['n2v_features_num'].split(',')
        self.n2v_features_num_list = [int(i) for i in n2v_features_num_list]

        n2v_return_weight_list = n2v_section['n2v_return_weight'].split(',')
        self.n2v_return_weight_list = [float(i) for i in n2v_return_weight_list]

        n2v_walklen_list = n2v_section['n2v_walklen'].split(',')
        self.n2v_walklen_list = [int(i) for i in n2v_walklen_list]

        n2v_epochs_list = n2v_section['n2v_epochs'].split(',')
        self.n2v_epochs_list = [int(i) for i in n2v_epochs_list]

        n2v_neighbor_weight_list = n2v_section['n2v_neighbor_weight'].split(',')
        self.n2v_neighbor_weight_list = [float(i) for i in n2v_neighbor_weight_list]

        pca_list = n2v_section['pca_num'].split(',')
        self.pca_list = [int(i) for i in pca_list]

    def create_grid_search_combs(self):
        if self.n2v_flag == 'True':
            self.combinations = list(product(self.sampling_strategy_list,self.test_ratio_list,self.cross_val_k_list,self.target_list,
                                         self.n2v_features_num_list,self.n2v_return_weight_list, self.n2v_walklen_list, self.n2v_epochs_list,self.n2v_neighbor_weight_list,self.pca_list))
        else:
            self.combinations = list(product(self.sampling_strategy_list, self.test_ratio_list, self.cross_val_k_list, self.target_list))

    def set_grid_params_for_iter(self,params_list):
        self.sampling_strategy = params_list[0]
        self.test_ratio =  params_list[1]
        self.cross_val_k =  params_list[2]
        self.target =  params_list[3]
        if self.n2v_flag == 'True':
            self.n2v_features_num = params_list[4]
            self.n2v_return_weight = params_list[5]
            self.n2v_walklen = params_list[6]
            self.n2v_epochs = params_list[7]
            self.n2v_neighbor_weight = params_list[8]
            self.pca = params_list[9]

    def set_random_params_for_iter(self):
        self.sampling_strategy = random.choice(self.sampling_strategy_list)
        self.test_ratio = random.choice(self.test_ratio_list)
        self.cross_val_k = random.choice(self.cross_val_k_list)
        self.n2v_features_num = random.choice(self.n2v_features_num_list)
        self.n2v_return_weight = random.choice(self.n2v_return_weight_list)
        self.n2v_walklen = random.choice(self.n2v_walklen_list)
        self.n2v_epochs = random.choice(self.n2v_epochs_list)
        self.n2v_neighbor_weight = random.choice(self.n2v_neighbor_weight_list)
        self.pca = random.choice(self.pca_list)
        self.target = random.choice(self.target_list)

    def update_iteration_params(self,iter_num):
        if self.random == 'True':
            self.set_random_params_for_iter()
        else:
            self.set_grid_params_for_iter(self.combinations[iter_num])



# Acts as an entry point for planning and training using the eRDTransformer framework.  
# Brings together configuation, data processing, model inference, and action planning for the eRDTransformer.

import numpy as np
import argparse
import pickle
import sys
import os
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import copy
import subprocess as subp

sys.path.append(os.getcwd())

from relational_precond.utils.colors import bcolors
from relational_precond.utils.data_utils import str2bool

from relational_precond.utils.data_utils import get_norm, get_activation, rotate_2d, scale_min_max

from relational_precond.model.GNN_pytorch_geometry import PointConv
from relational_precond.model.GNN_pytorch_geometry import GNNModelOptionalEdge, EmbeddingNetTorch, QuickReadoutNet


from relational_precond.utils import math_util
from relational_precond.utils import torch_util
from relational_precond.dataloader.farthest_point_sampling import farthest_point_sampling

from relational_precond.config.base_config import BaseVAEConfig

from itertools import permutations

import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from torch.utils.tensorboard import SummaryWriter

import random




ALL_OBJ_PAIRS_GNN = 'all_object_pairs_gnn'
ALL_OBJ_PAIRS_GNN_NEW = 'all_object_pairs_gnn_new'
ALL_OBJ_PAIRS_GNN_RAW_INFO = 'all_object_pairs_gnn_raw_obj_info'




def create_log_dirs(config):
    args = config.args
    # Create logger directory if required
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(config.get_logger_dir()):
        os.makedirs(config.get_logger_dir())
    if not os.path.exists(config.get_model_checkpoint_dir()):
        os.makedirs(config.get_model_checkpoint_dir())



class MultiObjectVoxelPrecondTrainerE2E(object):
    def __init__(self, config):

        self.config = config

        

        

        
        args = config.args
        self.use_tensorboard = args.use_tensorboard

        self.timestr = time.strftime("%Y-%m-%d-%H-%M-%S")

        self.params = {
            'theta_predicte_lr_fb_ab' : np.pi / args.relation_angle, # 45 degrees
            'occ_IoU_threshold' : 0.5,
        }

        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir = "../runs/"+self.timestr)
            self.loss_iter = 0

        


        
        self.max_evaluation = args.max_evaluation

        
        self.using_multi_step_statistics = args.using_multi_step_statistics

        self.using_multi_step = args.using_multi_step

        self.total_sub_step = args.total_sub_step


       
        
        
        self.remove_env_edges = args.remove_env_edges


        self.learn_update_edge_weights = args.learn_update_edge_weights

        self.use_boundary_relations = args.use_boundary_relations

        self.add_real_env = args.add_real_env

        self.consider_z_offset = args.consider_z_offset

        self.z_range = args.z_range

        self.x_middle_point = args.x_middle_point

        self.y_middle_point = args.y_middle_point

        self.z_middle_point = args.z_middle_point

        self.evaluate_pickplace = args.evaluate_pickplace

        
        self.saved_data_num = 1

        self.seperate_env_id = args.seperate_env_id

        self.max_env_num = args.max_env_num

        self.enable_seperate_pointconv = args.enable_seperate_pointconv

        self.seperate_action_emb = args.seperate_action_emb

        

        self.seperate_discrete_continuous = args.seperate_discrete_continuous

        self.simple_encoding = args.simple_encoding

        self.use_discrete_z = args.use_discrete_z

        

        
        self.complicated_pre_dynamics = args.complicated_pre_dynamics



        self.more_weight_regulirization = args.more_weight_regulirization



        self.assgin_total_step = args.total_sub_step

        self.use_env_for_graph_search = args.use_env_for_graph_search


        self.using_latent_memory = args.using_latent_memory

        self.pe = args.pe

        if args.use_specific_relations:
            self.relation_effective_id = [6]
        else:
            self.relation_effective_id = np.arange(args.z_dim)

        
        
       

        self.stacking = True

        self.pushing = args.pushing

        self.seperate = True
        

        self.use_point_cloud_embedding = True
        self.e2e = True
        
        self.all_classifier = False
        self.all_gt = False

        self.all_gt_sigmoid = True 

        self.new_relational_classifier = True 

        
        self.pick_place = args.pick_place

        if self.pushing:
            self.stacking = True

        
        self.manual_relations = args.manual_relations
        
        

        self.set_max = args.set_max
        if args.seperate_env_id:
            self.max_objects = args.max_objects + args.max_env_num
        else:
            self.max_objects = args.max_objects
        self.execute_planning = args.execute_planning





        self.seperate_range = args.seperate_range

        self.random_sampling_relations = args.random_sampling_relations

        self.using_delta_training = args.using_delta_training

        self.cem_planning = args.cem_planning

        self.graph_search = args.graph_search
        
        self.using_multi_step_latent = args.using_multi_step_latent



        self.using_latent_regularization = args.using_latent_regularization

        self.save_many_data = args.save_many_data

        self.test_next_step = args.test_next_step

        

       
        self.use_shared_latent_embedding = args.use_shared_latent_embedding

        self.use_seperate_latent_embedding = args.use_seperate_latent_embedding

        self.total_iters = args.total_iters
        self.action_selections = args.action_selections
        self.x_range = args.x_range

        self.y_range = args.y_range


        self.combine_push_pickplace = args.combine_push_pickplace

        self.push_3_steps = args.push_3_steps

        self.stack_push = args.stack_push

        self.push_pickplace = args.push_pickplace

        self.manual_specificy_goal_list = args.manual_specificy_goal_list

        self.max_training_step = args.max_training_step

        self.disable_push = args.disable_push

        self.disable_pickplace = args.disable_pickplace

        self.total_skill_num = 2

        self.consider_end_range = args.consider_end_range


        self.fast_training = args.fast_training

        self.fast_training_test = args.fast_training_test

        self.only_high_push = args.only_high_push


        self.get_hidden_label = args.get_hidden_label

        self.use_transformer = args.use_transformer

        self.torch_embedding = args.torch_embedding

        self.train_env_identity = args.train_env_identity

        self.graph_search_time = []

        
        
        self.previous_threshold = -100
        self.node_pose_list = []
        self.action_list = []
        self.goal_relation_list = []
        self.gt_extents_range_list = []
        self.gt_pose_list = []

        self.node_pose_list_planning = []  # only save it if planning success
        self.action_list_planning = []
        self.goal_relation_list_planning = []
        self.gt_extents_range_list_planning = []
        self.gt_pose_list_planning = []

        
                
           
        args = config.args
        
        # TODO: Use the arguments saved in the emb_checkpoint_dir to create 

        
        if args.train_type == 'all_object_pairs_gnn_new':
            
            
            self.emb_model_planar = PointConv(normal_channel=False)
            
            if self.e2e:
                self.emb_model = PointConv(normal_channel=False)
                if self.enable_seperate_pointconv:
                    self.emb_model_env = PointConv(normal_channel=False)
                
                
                
                
                
            
            if self.pick_place:
                self.num_nodes = 3
            elif self.pushing:
                self.num_nodes = 4
            else:
                self.num_nodes = 3
            if self.set_max:
                node_inp_size, edge_inp_size = self.max_objects + 3, args.z_dim
            else:
                node_inp_size, edge_inp_size = self.num_nodes + 3, args.z_dim
            
            self.node_inp_size = node_inp_size
            
            self.edge_inp_size = edge_inp_size

            
            
            edge_classifier = True
            self.edge_classifier = edge_classifier
            node_emb_size, edge_emb_size = 128, 128
            self.node_emb_size = node_emb_size
            self.edge_emb_size = edge_emb_size

            self.node_inp_size = self.node_emb_size
        
            self.edge_inp_size = self.edge_emb_size

            if self.use_transformer:
                self.classif_model = EmbeddingNetTorch(n_objects = self.max_objects,
                                    width = self.node_emb_size*2, 
                                    layers = args.n_layers, 
                                    heads = args.n_heads, 
                                    input_feature_num = self.max_objects + 3,
                                    d_hidden = 64, 
                                    n_unary = self.max_objects + 3, 
                                    n_binary = args.z_dim,
                                    simple_encoding = args.simple_encoding,
                                    seperate_discrete_continuous = args.seperate_discrete_continuous,
                                    torch_embedding = args.torch_embedding,
                                    use_seperate_latent_embedding = args.use_seperate_latent_embedding)
                self.classif_model_decoder = QuickReadoutNet(n_objects = self.max_objects,
                                        width = self.node_emb_size*2, 
                                        layers = args.n_layers, 
                                        heads = args.n_heads, 
                                        input_feature_num = self.max_objects + 3,
                                        d_hidden = 64, 
                                        n_unary = self.max_objects + 3, 
                                        n_binary = args.z_dim,
                                        train_env_identity = args.train_env_identity,
                                        one_bit_env = args.one_bit_env,
                                        pe = self.pe)
            else:
                self.classif_model = GNNModelOptionalEdge(
                            self.node_inp_size, 
                            self.edge_inp_size,
                            relation_output_size = args.z_dim, 
                            node_output_size = node_emb_size, 
                            predict_edge_output = True,
                            edge_output_size = edge_emb_size,
                            graph_output_emb_size=16, 
                            node_emb_size=node_emb_size, 
                            edge_emb_size=edge_emb_size,
                            message_output_hidden_layer_size=128,  
                            message_output_size=128, 
                            node_output_hidden_layer_size=64,
                            all_classifier = self.all_classifier,
                            predict_obj_masks=False,
                            predict_graph_output=False,
                            use_edge_embedding = False,
                            use_edge_input = False, 
                            max_objects = self.max_objects,
                            use_shared_latent_embedding = args.use_shared_latent_embedding,
                            use_seperate_latent_embedding = args.use_seperate_latent_embedding,

                            learn_update_edge_weights = args.learn_update_edge_weights,
                            seperate_env_id = args.seperate_env_id,
                            max_env_num = args.max_env_num,
                            seperate_action_emb = args.seperate_action_emb,
                            seperate_discrete_continuous = args.seperate_discrete_continuous,
                            simple_encoding = args.simple_encoding,
                            one_bit_env = args.one_bit_env,
                            larger_output_sigmoid = args.larger_output_sigmoid
                        )
                self.classif_model_decoder = GNNModelOptionalEdge(
                            self.node_emb_size, 
                            self.edge_emb_size,
                            relation_output_size = args.z_dim, 
                            node_output_size = self.node_inp_size, 
                            predict_edge_output = True,
                            edge_output_size = edge_inp_size,
                            graph_output_emb_size=16, 
                            node_emb_size=node_emb_size, 
                            edge_emb_size=edge_emb_size,
                            message_output_hidden_layer_size=128,  
                            message_output_size=128, 
                            node_output_hidden_layer_size=64,
                            all_classifier = self.all_classifier,
                            predict_obj_masks=False,
                            predict_graph_output=False,
                            use_edge_embedding = False,
                            use_edge_input = True, 
                            max_objects = self.max_objects,
                            use_shared_latent_embedding = args.use_shared_latent_embedding,
                            use_seperate_latent_embedding = args.use_seperate_latent_embedding,
                            
                            learn_update_edge_weights = args.learn_update_edge_weights,
                            seperate_env_id = args.seperate_env_id,
                            max_env_num = args.max_env_num,
                            seperate_action_emb = args.seperate_action_emb,
                            seperate_discrete_continuous = args.seperate_discrete_continuous,
                            simple_encoding = args.simple_encoding,
                            one_bit_env = args.one_bit_env,
                            larger_output_sigmoid = args.larger_output_sigmoid
                        )
            

        else:
            raise ValueError(f"Invalid train type: {args.train_type}")


       

        self.dynamics_loss = nn.MSELoss()
        if args.logit_loss:
            print('enter logit loss!')
            self.bce_loss = nn.BCEWithLogitsLoss()
        else:
            self.bce_loss = nn.BCELoss()
    

    def get_model_list(self):
        return [self.emb_model ,self.classif_model,self.classif_model_decoder]
        

    def set_model_device(self, device=torch.device("cpu")):
        model_list = self.get_model_list()
        for m in model_list:
            m.to(device)



    def model_checkpoint_dir(self):
        '''Return the directory to save models in.'''
        return self.config.get_model_checkpoint_dir()

    def model_checkpoint_filename(self, epoch):
        return os.path.join(self.model_checkpoint_dir(),
                            'cp_{}.pth'.format(epoch))

    
    
    def load_checkpoint(self, checkpoint_path):
        cp_models = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        self.emb_model.load_state_dict(cp_models['emb_model'])
        self.classif_model.load_state_dict(cp_models['classif_model'])
        self.classif_model_decoder.load_state_dict(cp_models['classif_model_decoder'])
        
     
    
    def create_graph(self, num_nodes, node_inp_size, node_pose, edge_size, edge_feature, action):
        if self.remove_env_edges:
            nodes = [0,1,2]
        else:
            nodes = list(range(num_nodes))
        # print(nodes)
        # Create a completely connected graph
        edges = list(permutations(nodes, 2))
        # print(edges)
        edge_index = torch.LongTensor(np.array(edges).T)
        x = node_pose #torch.zeros((num_nodes, node_inp_size))#torch.eye(node_inp_size).float()
        edge_attr = edge_feature #torch.rand(len(edges), edge_size)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, action = action)
        # Recreate x as target
        data.y = x
        return data

    
    def planner(self):
        device = self.device
        action_selections = self.action_selections
        self.min_cost = 1e5
        loss_list = []
        all_action_list = []
        
        if not (self.use_seperate_latent_embedding or self.use_shared_latent_embedding):
            self.total_skill_num = 1
        
        if self.use_discrete_z:
            z_range = [-1, 0, 1]
        else:
            z_range = [0]
        
        if self.train_env_identity:
            obj_select_range = []
            for each_env_id in range(self.env_identity.shape[0]):
                if self.env_identity[each_env_id, 0] > 0.5:
                    obj_select_range.append(each_env_id)
        else:
            obj_select_range = range(self.num_nodes)


        for each_z in z_range:
            for skill_iter in range(self.total_skill_num):
                if self.use_seperate_latent_embedding or self.use_shared_latent_embedding:
                    if self.only_high_push:
                        skill_iter = 2
                    elif self.disable_push:
                        skill_iter = 0
                    elif self.disable_pickplace:
                        skill_iter = 1
                    
                
                if skill_iter == 0:
                    self.consider_z_offset = True
                else:
                    self.consider_z_offset = False

                for obj_mov in obj_select_range:
                    if self.seperate_range:
                        middle_point = [[1,0.3], [0,0.3]]
                    else:
                        middle_point = [[0.5,0.6]]
                    for current_middle_point in middle_point:
                        if skill_iter == 0:
                            print('try pick-and-place object', obj_mov)
                        elif skill_iter == 1:
                            print('try push object', obj_mov)

                        
                        action_mu = np.zeros((action_selections, 1, 3))
                        action_sigma = np.ones((action_selections, 1, 3))
                        
                        
                        i_iter = 0
                        while i_iter < self.total_iters:
                            # print(i_iter)
                            action_noise = np.zeros((action_selections, 1, 3))
                            # if self.consider_z_offset:
                            #     action_noise = np.zeros((action_selections, 1, 3))
                            # else:
                            #     action_noise = np.zeros((action_selections, 1, 2))
                            action_noise[:,:,0] = (np.random.rand(action_selections, 1) - self.x_middle_point) * self.x_range
                            action_noise[:,:,1] = (np.random.rand(action_selections, 1) - self.y_middle_point) * self.y_range
                            if self.consider_z_offset:
                                action_noise[:,:,2] = (np.random.rand(action_selections, 1) - self.z_middle_point) * self.z_range
                            #action_noise = (np.random.rand(action_selections, 1, 2) - 0.5) * 0.4 # change range to (-0.2, 0.2)
                            act = action_mu + action_noise*action_sigma
                            costs = []
                            for j in range(action_selections):
                                action_numpy = np.zeros((self.num_nodes, 3))
                                action_numpy[obj_mov][0] = act[j, 0, 0]
                                action_numpy[obj_mov][1] = act[j, 0, 1]
                                if self.consider_z_offset:
                                    action_numpy[obj_mov][2] = act[j, 0, 2]
                                else:
                                    if self.use_discrete_z:
                                        action_numpy[obj_mov][2] = each_z
                                    else:
                                        action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)
                                
                                if self.use_seperate_latent_embedding or self.use_shared_latent_embedding:
                                    if self.set_max:
                                        action = np.zeros((self.num_nodes, self.max_objects + 3 + 1))
                                    else:
                                        action = np.zeros((self.num_nodes, self.num_nodes + 3))
                                    for i in range(action.shape[0]):
                                        action[i][0] = skill_iter
                                        action[i][obj_mov + 1] = 1
                                        action[i][-3:] = action_numpy[obj_mov]
                                else:
                                    if self.set_max:
                                        action = np.zeros((self.num_nodes, self.max_objects + 3))
                                    else:
                                        action = np.zeros((self.num_nodes, self.num_nodes + 3))
                                    for i in range(action.shape[0]):
                                        action[i][obj_mov] = 1
                                        action[i][-3:] = action_numpy[obj_mov]
                                
                                sample_action = torch.Tensor(action).to(device)
                                
                                
                                this_sequence = []
                                this_sequence.append(sample_action)
                                loss_func = nn.MSELoss()
                                test_loss = 0
                                current_latent = self.this_time_step_embed 
                                if not self.use_transformer:
                                    egde_latent = self.this_time_step_edge_embed 
                                if self.use_seperate_latent_embedding:
                                    for seq in range(len(this_sequence)):
                                        if self.seperate_discrete_continuous:
                                            if self.torch_embedding:
                                                discrete_action = self.classif_model.one_hot_encoding_embed(torch.argmax(this_sequence[seq][:, 1:-3], dim = 1))
                                            else:
                                                discrete_action = self.classif_model.one_hot_encoding_embed(this_sequence[seq][:, 1:-3])
                                            
                                            if self.seperate_action_emb:
                                                if skill_iter == 0:
                                                    continuous_action = self.classif_model.continuous_action_emb(this_sequence[seq][:, -3:])
                                                elif skill_iter == 1:
                                                    continuous_action = self.classif_model.continuous_action_emb_1(this_sequence[seq][:, -3:])
                                            else:
                                                continuous_action = self.classif_model.continuous_action_emb(this_sequence[seq][:, -3:])
                                            current_action = torch.cat((discrete_action, continuous_action), axis = -1)
                                        else:
                                            current_action = self.classif_model.action_emb(this_sequence[seq][:, 1:])
                                        if self.use_transformer:
                                            # print('enter direct transformer')
                                            # print(current_action)
                                            current_action = current_action.view(1, current_action.shape[0], current_action.shape[1])
                                            current_action = current_action[0, 0, :]
                                            current_action = current_action.view(1, 1, current_action.shape[0])
                                            # print('current_action', current_action.shape)
                                            graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                            # print('graph_node_action', graph_node_action.shape)
                                        else:
                                            graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                        
                                        if True:
                                            if skill_iter == 0:
                                                current_latent = self.classif_model.graph_dynamics_0(graph_node_action)
                                            elif skill_iter == 1:
                                                # print(graph_node_action.shape)
                                                current_latent = self.classif_model.graph_dynamics_1(graph_node_action)
                                            elif skill_iter == 2:
                                                # print(graph_node_action.shape)
                                                current_latent = self.classif_model.graph_dynamics_2(graph_node_action)


                                    if not self.use_transformer:
                                        for seq in range(len(this_sequence)):
                                            #print([current_latent, this_sequence[seq]])
                                            #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                            edge_num = egde_latent.shape[0]
                                            edge_action_list = []
                                            if self.seperate_discrete_continuous:
                                                discrete_action = self.classif_model.one_hot_encoding_embed(this_sequence[seq][:, 1:-3])
                                                if self.seperate_action_emb:
                                                    if skill_iter == 0:
                                                        continuous_action = self.classif_model.continuous_action_emb(this_sequence[seq][:, -3:])
                                                    elif skill_iter == 1:
                                                        continuous_action = self.classif_model.continuous_action_emb_1(this_sequence[seq][:, -3:])
                                                else:
                                                    continuous_action = self.classif_model.continuous_action_emb(this_sequence[seq][:, -3:])
                                                current_action = torch.cat((discrete_action, continuous_action), axis = -1)
                                            else:
                                                current_action = self.classif_model.action_emb(this_sequence[seq][:, 1:])
                                            for _ in range(edge_num):
                                                edge_action_list.append(current_action[0])
                                            edge_action = torch.stack(edge_action_list)
                                            graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)

                                            if skill_iter == 0:
                                                egde_latent = self.classif_model.graph_edge_dynamics_0(graph_edge_action)
                                            elif skill_iter == 1:
                                                egde_latent = self.classif_model.graph_edge_dynamics_1(graph_edge_action)
                                            
                                else:
                                    for seq in range(len(this_sequence)):
                                        #print([current_latent, this_sequence[seq]])
                                        current_action = self.classif_model.action_emb(this_sequence[seq])

                                        graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                        current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                    for seq in range(len(this_sequence)):
                                        #print([current_latent, this_sequence[seq]])
                                        #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                        edge_num = egde_latent.shape[0]
                                        edge_action_list = []
                                        current_action = self.classif_model.action_emb(this_sequence[seq])
                                        for _ in range(edge_num):
                                            edge_action_list.append(current_action[0])
                                        edge_action = torch.stack(edge_action_list)
                                        graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                        egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                
                                if self.use_transformer:
                                    data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, 0, None, sample_action)
                                else:
                                    data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
        
                                batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                                outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.action, skill_iter)
                                

                                if self.graph_search:
                                    for each_sub_goal_index in self.subgoal_index_list:
                                        index_i = self.all_index_i_list[each_sub_goal_index]
                                        index_j = self.all_index_j_list[each_sub_goal_index]
                                        if self.pick_place or self.use_boundary_relations:
                                            current_goal = self.hard_coded_relations
                                            self.x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(self.hard_coded_relations).to(device)
                                        else:
                                            current_goal = self.all_goal_list[each_sub_goal_index]
                                            self.x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(self.goal_relations_list[current_goal]).to(device)
                                        test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][index_i, index_j], self.x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                else:
                                    test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][self.index_i, self.index_j], self.x_tensor_dict_next['batch_all_obj_pair_relation'][self.index_i, self.index_j])
                                    
                    
                                costs.append(test_loss.detach().cpu().numpy())
                                

                            index = np.argsort(costs)
                            elite = act[index,:,:]
                            
                            elite = elite[:3, :, :]

                            if np.max(elite.std(axis = 0)[0]) > 0.2:
                                continue

                            
                            
                            action_mu = elite.mean(axis = 0)
                            action_sigma = elite.std(axis = 0)
                            if self.use_discrete_z:
                                print([action_mu, action_sigma, each_z])
                            else:
                                print('Continuous action parameter mean and sigma')
                                print([action_mu[0, :3], action_sigma[0, :3]])
                            i_iter = i_iter + 1
                            
                        
                        
                        chosen_action = action_mu
                        action_numpy = np.zeros((self.num_nodes, 3))
                        action_numpy[obj_mov][0] = chosen_action[0, 0]
                        action_numpy[obj_mov][1] = chosen_action[0, 1]
                        if self.consider_z_offset:
                            action_numpy[obj_mov][2] = chosen_action[0, 2]
                        else:
                            if self.use_discrete_z:
                                action_numpy[obj_mov][2] = each_z
                            else:
                                action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)

                        action_numpy_variance = np.zeros((self.num_nodes, 3))
                        
                        action_numpy_variance[obj_mov][0] = action_sigma[0, 0]
                        action_numpy_variance[obj_mov][1] = action_sigma[0, 1]
                        if self.consider_z_offset:
                            action_numpy_variance[obj_mov][2] = action_sigma[0, 2]
                        if self.use_seperate_latent_embedding or self.use_shared_latent_embedding:
                            if self.set_max:
                                action = np.zeros((self.num_nodes, self.max_objects + 3 + 1))
                            else:
                                action = np.zeros((self.num_nodes, self.num_nodes + 3))
                            for i in range(action.shape[0]):
                                action[i][0] = skill_iter
                                action[i][obj_mov + 1] = 1
                                action[i][-3:] = action_numpy[obj_mov]
                        else:
                            if self.set_max:
                                action = np.zeros((self.num_nodes, self.max_objects + 3))
                            else:
                                action = np.zeros((self.num_nodes, self.num_nodes + 3))
                            for i in range(action.shape[0]):
                                action[i][obj_mov] = 1
                                action[i][-3:] = action_numpy[obj_mov]
                                
                        sample_action = torch.Tensor(action).to(device)
                        
                        this_sequence = []
                        this_sequence.append(sample_action)

                        this_sequence_variance = []
                        this_sequence_variance.append(action_numpy_variance)
                        loss_func = nn.MSELoss()
                        test_loss = 0
                        current_latent = self.this_time_step_embed
                        if not self.use_transformer:
                            egde_latent = self.this_time_step_edge_embed
                        if self.use_seperate_latent_embedding:
                            for seq in range(len(this_sequence)):
                                if self.seperate_discrete_continuous:
                                    if self.torch_embedding:
                                        discrete_action = self.classif_model.one_hot_encoding_embed(torch.argmax(this_sequence[seq][:, 1:-3], dim = 1))
                                    else:
                                        discrete_action = self.classif_model.one_hot_encoding_embed(this_sequence[seq][:, 1:-3])
                                    if self.seperate_action_emb:
                                        if skill_iter == 0:
                                            continuous_action = self.classif_model.continuous_action_emb(this_sequence[seq][:, -3:])
                                        elif skill_iter == 1:
                                            continuous_action = self.classif_model.continuous_action_emb_1(this_sequence[seq][:, -3:])
                                    else:
                                        continuous_action = self.classif_model.continuous_action_emb(this_sequence[seq][:, -3:])
                                    current_action = torch.cat((discrete_action, continuous_action), axis = -1)
                                else:
                                    current_action = self.classif_model.action_emb(this_sequence[seq][:, 1:])

                                if self.use_transformer:
                                    current_action = current_action.view(1, current_action.shape[0], current_action.shape[1])
                                    current_action = current_action[0, 0, :]
                                    current_action = current_action.view(1, 1, current_action.shape[0])
                                    graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                else:
                                    graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                
                                if True:
                                    if skill_iter == 0:
                                        current_latent = self.classif_model.graph_dynamics_0(graph_node_action)
                                    elif skill_iter == 1:
                                        current_latent = self.classif_model.graph_dynamics_1(graph_node_action)
                                    elif skill_iter == 2:
                                        current_latent = self.classif_model.graph_dynamics_2(graph_node_action)


                            if not self.use_transformer:
                                for seq in range(len(this_sequence)):
                                    
                                    edge_num = egde_latent.shape[0]
                                    edge_action_list = []
                                    if self.seperate_discrete_continuous:
                                        discrete_action = self.classif_model.one_hot_encoding_embed(this_sequence[seq][:, 1:-3])
                                        if self.seperate_action_emb:
                                            if skill_iter == 0:
                                                continuous_action = self.classif_model.continuous_action_emb(this_sequence[seq][:, -3:])
                                            elif skill_iter == 1:
                                                continuous_action = self.classif_model.continuous_action_emb_1(this_sequence[seq][:, -3:])
                                        else:
                                            continuous_action = self.classif_model.continuous_action_emb(this_sequence[seq][:, -3:])
                                        current_action = torch.cat((discrete_action, continuous_action), axis = -1)
                                    else:
                                        current_action = self.classif_model.action_emb(this_sequence[seq][:, 1:])
                                
                                    for _ in range(edge_num):
                                        edge_action_list.append(current_action[0])
                                    edge_action = torch.stack(edge_action_list)
                                    graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                    
                                    if skill_iter == 0:
                                        egde_latent = self.classif_model.graph_edge_dynamics_0(graph_edge_action)
                                    elif skill_iter == 1:
                                        egde_latent = self.classif_model.graph_edge_dynamics_1(graph_edge_action)
                                    
                        else:
                            for seq in range(len(this_sequence)):
                                #print([current_latent, this_sequence[seq]])
                                current_action = self.classif_model.action_emb(this_sequence[seq])

                                graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                current_latent = self.classif_model.graph_dynamics(graph_node_action)
                            for seq in range(len(this_sequence)):
                                
                                edge_num = egde_latent.shape[0]
                                edge_action_list = []
                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                for _ in range(edge_num):
                                    edge_action_list.append(current_action[0])
                                edge_action = torch.stack(edge_action_list)
                                graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)

                        if self.use_transformer:
                            data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, 0, None, sample_action)
                        else:
                            data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
    
                        batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                        outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.action, skill_iter)
                        
                        
                        
                        if True:
                            if self.graph_search:
                                for each_sub_goal_index in self.subgoal_index_list:
                                    index_i = self.all_index_i_list[each_sub_goal_index]
                                    index_j = self.all_index_j_list[each_sub_goal_index]
                                    if self.pick_place or self.use_boundary_relations:
                                        current_goal = self.hard_coded_relations
                                        self.x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(self.hard_coded_relations).to(device)
                                    else:
                                        current_goal = self.all_goal_list[each_sub_goal_index]
                                        self.x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(self.goal_relations_list[current_goal]).to(device)
                                    test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][index_i, index_j], self.x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                            else:
                                test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][self.index_i, self.index_j], self.x_tensor_dict_next['batch_all_obj_pair_relation'][self.index_i, self.index_j])
                                
                        #sample_list.append(outs_edge['pred_edge'])
                        print('test_loss', test_loss)

                        if self.graph_search:
                            for each_sub_goal_index in self.subgoal_index_list:
                                index_i = self.all_index_i_list[each_sub_goal_index]
                                index_j = self.all_index_j_list[each_sub_goal_index]
                                print('predicted_relations',outs_decoder_2['pred_sigmoid'][index_i, index_j])
                                print('ground truth relations', self.x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                        else:
                            print('predicted_relations',outs_decoder_2['pred_sigmoid'][self.index_i, self.index_j])
                            print('ground truth relations', self.x_tensor_dict_next['batch_all_obj_pair_relation'][self.index_i, self.index_j])
                
                        loss_list.append(test_loss)
                        if(test_loss.detach().cpu().numpy() < self.min_cost):
                            self.min_prediction = outs_decoder_2['pred_sigmoid'][:, :]
                            self.min_action = this_sequence
                            self.min_action_variance = this_sequence_variance
                            self.min_pose = outs_decoder_2['pred'][:, :]
                            self.min_cost = test_loss.detach().cpu().numpy()
                            min_latent = current_latent
                            if not self.use_transformer:
                                min_latent_edge = egde_latent
            
                if self.use_seperate_latent_embedding or self.use_shared_latent_embedding:
                    if self.disable_push or self.disable_pickplace or self.only_high_push:
                        break
            
        self.this_time_step_embed = min_latent
        if not self.use_transformer:
            self.this_time_step_edge_embed = min_latent_edge                        

    
    def run_model_on_batch_torch_geometry_pick_primitive_new_relational_classifier_point_cloud_e2e_manual_relations(self,
                           batch_size,
                           train=False,
                           threshold = 0):
        batch_result_dict = {}
        device = self.config.get_device()
        self.device = self.config.get_device()


        voxel_data_single = torch.Tensor(np.array(self.total_sample_point_cloud)).to(device)
        self.num_nodes = self.obj_num_for_save
    
        img_emb_single = self.emb_model(voxel_data_single)
        
        
        

        if self.use_point_cloud_embedding:

            A = np.arange(self.max_objects)

            select_obj_num_range = A[:self.obj_num_for_save]

            
            
            if self.set_max:
                one_hot_encoding = np.zeros((self.num_nodes, self.max_objects))
            else:
                one_hot_encoding = np.zeros((self.num_nodes, self.num_nodes))
            
            for one_hot_i in range(len(select_obj_num_range)):
                one_hot_encoding[one_hot_i][(int)(select_obj_num_range[one_hot_i])] = 1
            
            
            one_hot_encoding_tensor = torch.Tensor(one_hot_encoding).to(device)
            
            if self.torch_embedding:
                latent_one_hot_encoding = self.classif_model.one_hot_encoding_embed(torch.argmax(one_hot_encoding_tensor, dim = 1))
            else:
                latent_one_hot_encoding = self.classif_model.one_hot_encoding_embed(one_hot_encoding_tensor)
                
            
            node_pose = torch.cat([img_emb_single, latent_one_hot_encoding], dim = 1)
            

        action_torch = torch.zeros(self.num_nodes, self.max_objects + 3 + 1)  ## place holder for action
        
                   

                            
        if self.manual_relations:
            if self.manual_specificy_goal_list:
                ## left, right, behind, front, below, top, contact, in-boundary in original view
                
                print('Manually specify goal relations')

                current_goal_relations = np.array([[1., 0., 0., 1., 0., 0., 1., 1],  # 1-2 
                                                [1., 0., 0., 1., 1., 1., 0., 0],   # 1-3
                                                [0., 0., 0., 1., 0., 1., 1., 1],   # 1-4
                                                [0., 0., 1., 0., 0., 0., 1., 0],   # 2-1
                                                [0., 0., 0., 1., 1., 1., 0., 0],   # 2-3  
                                                [0., 0., 0., 1., 1., 1., 1., 1],   # 2-4
                                                [0., 0., 1., 0., 1., 0., 0., 0],   # 3-1
                                                [0., 0., 1., 0., 0., 0., 0., 0],  # 3-2 # 
                                                [0., 0., 0., 1., 0., 0., 1., 1],     # 3-4
                                                [0., 0., 1., 0., 0., 0., 0., 0],     # 4-1
                                                [0., 0., 1., 0., 0., 0., 0., 0],     # 4-2
                                                [0., 0., 1., 0., 0., 0., 0., 0]    # 4-3
                                                ])



                # print(self.gt_extents)
                x_tensor_dict_next = dict()
                x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(current_goal_relations).to(device)
                
        


                
                ## here to specify which subgoals you want to achieve
                
                ## goal 1, move the first object to the top layer of the shelf
                # index_i = [1,2,2]
                # index_j = [6,5,6] 

                ## goal 2, move the second object to the top layer of the shelf
                index_i = [4,5,5]
                index_j = [6,5,6] 


                ## For more examples, refer to our paper. 
                


                
                x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(current_goal_relations).to(device) 

            

        num_nodes = self.num_nodes


        data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
        batch = Batch.from_data_list([data_1]).to(device)
        skill_label = action_torch.detach().cpu().numpy()[0][0]
        outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.action, skill_label)



        if self.use_transformer:
            data_1_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs['current_embed'], 0, None, action_torch)
        else:
            data_1_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred'], self.edge_emb_size, outs['pred_edge'], action_torch)                      
        batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
        outs_decoder = self.classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.action, skill_label)

        print('current relations', outs_decoder['pred_sigmoid'][index_i, index_j])
        

        if self.execute_planning:

            
            self.this_time_step_embed = outs['current_embed']
            if not self.use_transformer:
                self.this_time_step_edge_embed = outs['edge_embed']
            self.index_i = index_i 
            self.index_j = index_j
            self.x_tensor_dict_next = x_tensor_dict_next
        
            
            if self.train_env_identity:
                print('predicted env identity', outs_decoder['env_identity'])
                self.env_identity = outs_decoder['env_identity'].cpu().detach().numpy()
                
            self.planner() 
            
            
            
            
            this_mean_action_numpy = self.min_action[0].detach().cpu().numpy()[0]
            print('best_action', this_mean_action_numpy)
            print('best action: ')
            if this_mean_action_numpy[0] == 0:
                print('Skill is pick-and-place')
            elif this_mean_action_numpy[0] == 1:
                print('Skill is push')

            print('manipulated object id', np.argmax(this_mean_action_numpy[1: 1+ self.max_objects]) + 1)

            print('continuous parameters:', this_mean_action_numpy[-3: -1])
            

                    
            
                        
        

        return batch_result_dict



    def run_planning(self, train=True, threshold = 0.8):
        print("Begin planner")
        args = self.config.args
        device = self.config.get_device()

        batch_size = args.batch_size


        self.set_model_device(device)

        self.test_dir_list = self.config.args.test_dir

        files = sorted(os.listdir(self.test_dir_list[0]))
            
            
        self.test_pcd_path = [
            os.path.join(self.test_dir_list[0], p) for p in files if 'demo' in p]

        
        for test_dir in self.test_pcd_path[args.start_test_id: args.start_test_id + args.test_max_size]: 
            with open(test_dir, 'rb') as f:
                data, attrs = pickle.load(f)

            leap = 1
            data_size = 128  ## make sure all points are larger than 128
            self.initial_scene_label = 0
            for k, v in data.items():
                if 'point_cloud' in k and 'last' not in k:
                    if(v.shape[0] == 0): ## make sure the point cloud is not empty
                        leap = 0
                        break
                    if(v[self.initial_scene_label].shape[0] < data_size): ## only check for first point cloud since we only use this one.
                        leap = 0
                        break
            if leap == 0:
                continue

            total_objects = 0
            for k, v in data.items():
                if 'point_cloud' in k and 'sampling' in k and 'last' not in k:
                    total_objects += 1
            
            self.obj_num_for_save = total_objects

            self.total_sample_point_cloud = []
            point_string = 'point_cloud_'
            for j in range(total_objects):
                self.total_sample_point_cloud.append(data[point_string + str(j+1) + 'sampling'][self.initial_scene_label][:data_size, :].T.astype(np.float32)) ## get the point cloud at the initial scene.


            batch_result_dict = self.run_model_on_batch_torch_geometry_pick_primitive_new_relational_classifier_point_cloud_e2e_manual_relations(
                batch_size,
                train=train,
                threshold = threshold)

            if self.graph_search:
                print('self.graph_search_time', self.graph_search_time)
                print('mean of self.graph_search_time', sum(self.graph_search_time)/len(self.graph_search_time))

        return batch_result_dict
       


def main(args):
    dtype = torch.FloatTensor
    if args.cuda:
        dtype = torch.cuda.FloatTensor

    # Load the args from previously saved checkpoint
    if len(args.checkpoint_path) > 0:
        config_pkl_path = os.path.join(args.result_dir, 'config.pkl')
        with open(config_pkl_path, 'rb') as config_f:
            old_args = pickle.load(config_f)
            print(bcolors.c_red("Did load config: {}".format(config_pkl_path)))


    config = BaseVAEConfig(args, dtype=dtype)
    create_log_dirs(config)

    trainer = MultiObjectVoxelPrecondTrainerE2E(config)

    if len(args.checkpoint_path) > 0:
        trainer.load_checkpoint(args.checkpoint_path)
        
        result_dict = trainer.run_planning(train=False,
                                            threshold = 0)
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training/planning for eRDTransformer.')


    parser.add_argument('--cuda', type=int, default=1, help="Use cuda")
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory to save results.')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Checkpoint to test on.')

    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for each step')

    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for encoder/decoder.')

    parser.add_argument('--save_freq_iters', type=int, default=1001,
                        help='Frequency at which to save models.')


    
    parser.add_argument('--train_dir', action='append',
                        help='Path to training directory.')
    parser.add_argument('--test_dir', required=True, action='append',
                        help='Path to test directory.')
    
    parser.add_argument('--train_type', type=str, default='all_object_pairs_gnn_new',
                        choices=[
                            'all_object_pairs_gnn_new',
                            'PointConv'
                            ],
                        help='Training type to follow.')
    parser.add_argument('--emb_lr', type=float, default=0.0001,
                        help='Learning rate to use for pointconvembeddings.')


    parser.add_argument('--z_dim', type=int, default=8,
                        help='number of relations to use.')

    
    parser.add_argument('--save_data_path', type=str, default='', 
                        help='Path to savetxt file to get goal relations.')

    
    parser.add_argument('--evaluate_end_relations', type=str2bool, default=False,
                        help='whether to use the evaluate the end relations especially for the real-world case')
    parser.add_argument('--set_max', type=str2bool, default=True,
                        help='whether to use set_max method')
    parser.add_argument('--max_objects', type=int, default=8,
                        help='max_objects in this experiments')
    parser.add_argument('--total_sub_step', type=int, default=2,
                        help='total sub steps for multi-step test')
    
    parser.add_argument('--online_planning', type=str2bool, default=False,
                        help='whether to use online_planning in real_data')
    parser.add_argument('--online_planning_ros', type=str2bool, default=False,
                        help='whether to use online_planning and get the info from ros')    
    parser.add_argument('--pointconv_baselines', type=str2bool, default=False,
                        help='whether to use pointconv baselines as a comparison')
    parser.add_argument('--mlp', type=str2bool, default=False,
                        help='whether to use mlp baselines as a comparison')

    parser.add_argument('--use_multiple_test_dataset', type=str2bool, default=False,
                        help='whether to use use_multiple_test_dataset')
    parser.add_argument('--manual_relations', type=str2bool, default=True,
                        help='whether to use manual_relations')
    parser.add_argument('--execute_planning', type=str2bool, default=True,
                        help='whether to use execute_planning in the test')
    parser.add_argument('--consider_current_relations', type=str2bool, default=False,
                        help='whether consider current relations')
    parser.add_argument('--consider_end_relations', type=str2bool, default=True,
                        help='whether consider end relations in the evaluate_end_relations')
    parser.add_argument('--evaluate_pickplace', type=str2bool, default=False,
                        help='whether to use evaluate_pickplace')
    parser.add_argument('--updated_behavior_params', type=str2bool, default=True,
                        help='whether to use updated_behavior_params')
    parser.add_argument('--start_id', type=int, default=0,
                        help='start_id in hthe training')
    parser.add_argument('--max_size', type=int, default=0,
                        help='max_size if the training dataset')
    parser.add_argument('--start_test_id', type=int, default=0,
                        help='start_test_id of the test dataset')
    parser.add_argument('--test_max_size', type=int, default=0,
                        help='test_max_size of the test dataset') 
    parser.add_argument('--set_random_seed', type=str2bool, default=False,
                        help='whether to set random seed')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for numpy and torch')
    parser.add_argument('--seperate_range', type=str2bool, default=False,
                        help='whether to seperate_range')
    parser.add_argument('--random_sampling_relations', type=str2bool, default=True,
                        help='whether to random_sampling_relations')
    parser.add_argument('--using_delta_training', type=str2bool, default=False,
                        help='whether to use using_delta_training')
    parser.add_argument('--cem_planning', type=str2bool, default=True,
                        help='whether to use cem planning')
    parser.add_argument('--pick_place', type=str2bool, default=False,
                        help='whether to use pick place skill')        
    parser.add_argument('--pushing', type=str2bool, default=True,
                        help='whether to use pushing skill')  
    parser.add_argument('--rcpe', type=str2bool, default=False,
                        help='whether to use relational classifier and pose estimation baselines')    
    parser.add_argument('--using_multi_step', type=str2bool, default=False,
                        help='whether to use multi step planning as a task and motion planning style') 
    parser.add_argument('--graph_search', type=str2bool, default=False,
                        help='whether to use graph search in the multi-step planning')
    parser.add_argument('--using_multi_step_latent', type=str2bool, default=False,
                        help='whether to use using_multi_step_latent which means some model-based RL sampling methods') 
    parser.add_argument('--test_next_step', type=str2bool, default=False,
                        help='whether to use test_next_step which means plans actions based on first steps. also whether to use test_next_step for pickplace') 
    parser.add_argument('--using_latent_regularization', type=str2bool, default=True,
                        help='whether to use use regularization loss in the latent space.') 
    parser.add_argument('--save_many_data', type=str2bool, default=False,
                        help='whether to use save many data together') 
    parser.add_argument('--using_multi_step_statistics', type=str2bool, default=False,
                        help='whether to use using_multi_step_statistics to get statistics for the multi-step test.')

    parser.add_argument('--use_shared_latent_embedding', type=str2bool, default=False,
                        help='whether to use shared latent embedding.')
    parser.add_argument('--use_seperate_latent_embedding', type=str2bool, default=True,
                        help='whether to use seperate latent embedding, which means that we use one latent dynamics for pickplace and one latent dynamics for push.')
    parser.add_argument('--total_iters', type=int, default=2,
                        help='number of iterations in the CEM planning')
    parser.add_argument('--action_selections', type=int, default=200,
                        help='number of action_selections in the CEM planning')
    parser.add_argument('--x_range', type=float, default=0.1,
                        help='x_range sampling in the CEM planning')
    parser.add_argument('--y_range', type=float, default=0.8,
                        help='y_range sampling in the CEM planning')
    parser.add_argument('--z_range', type=float, default=1.0,
                        help='z_range sampling in the CEM planning')
    parser.add_argument('--x_middle_point', type=float, default=0.5,
                        help='x_middle_point sampling in the CEM planning')
    parser.add_argument('--y_middle_point', type=float, default=0.5,
                        help='y_middle_point sampling in the CEM planning')
    parser.add_argument('--z_middle_point', type=float, default=0.5,
                        help='z_middle_point in the CEM planning')
    parser.add_argument('--first_save_sampling', type=str2bool, default=False,
                        help='whether to first save and sampling points before the full setup.')
    parser.add_argument('--combine_push_pickplace', type=str2bool, default=False,
                        help='whether to use both push and pickplace during the planning.')
    parser.add_argument('--push_3_steps', type=str2bool, default=True,
                        help='whether to use push_3_steps to deal with multi-step problem.')
    parser.add_argument('--stack_push', type=str2bool, default=False,
                        help='whether to use first 3stack then push.')
    parser.add_argument('--push_pickplace', type=str2bool, default=False,
                        help='whether to use first push then pickplace dataset.')
    parser.add_argument('--manual_specificy_goal_list', type=str2bool, default=True,
                        help='whether to use manual specify goal list.')
    
    parser.add_argument('--max_training_step', type=int, default=10,
                        help='the max training forward step in push_3_steps')
    
    parser.add_argument('--obj_num_for_save', type=int, default=3,
                        help='the number of objects for online learning based on ros')
    parser.add_argument('--max_evaluation', type=int, default=20,
                        help='the max evaluation num for each num of objects and each number of relations')
    parser.add_argument('--disable_pickplace', type=str2bool, default=False,
                        help='whether to disable pickplace during sampling.')
    parser.add_argument('--disable_push', type=str2bool, default=False,
                        help='whether to disable push during sampling.')
    
    parser.add_argument('--diverse_planner', type=str2bool, default=False,
                        help='whether to use the diverse_planner.')
    parser.add_argument('--single_test', type=str2bool, default=False,
                        help='whether to use the single step test.')
    parser.add_argument('--single_push', type=str2bool, default=False,
                        help='whether to use the single step test.')
    parser.add_argument('--hidden_leap', type=str2bool, default=False,
                        help='whether to use the hidden leap.')
    
    
    parser.add_argument('--remove_env_edges', type=str2bool, default=False,
                        help='whether to remove environment edges.')

    parser.add_argument('--learn_update_edge_weights', type=str2bool, default=False,
                        help='whether to use dynamic edge weight.')
    parser.add_argument('--use_boundary_relations', type=str2bool, default=True,
                        help='whether to use boundary relations.') 
    parser.add_argument('--add_real_env', type=str2bool, default=False,
                        help='whether to add real env.')     
    parser.add_argument('--consider_z_offset', type=str2bool, default=True,
                        help='whether to consider_z_offset.')
    
    parser.add_argument('--seperate_env_id', type=str2bool, default=False,
                        help='whether to seperate env id and object id.')
    parser.add_argument('--add_memory', type=str2bool, default=False,
                        help='whether to add_memory or not.')
    
    parser.add_argument('--max_env_num', type=int, default=0,
                        help='the max number for the environment nodes.')
    parser.add_argument('--enable_seperate_pointconv', type=str2bool, default=False,
                        help='whether to enable seperate pointconv encoder for object and environment.')
    
    parser.add_argument('--seperate_action_emb', type=str2bool, default=False,
                        help='whether to use different action embed based on the different skills.')

    
    parser.add_argument('--seperate_discrete_continuous', type=str2bool, default=True,
                        help='whether to seperate_discrete_continuous or not.')

    parser.add_argument('--simple_encoding', type=str2bool, default=False,
                        help='whether to simple_encoding or not.')

    parser.add_argument('--all_hidden_data', type=str2bool, default=False,
                        help='whether to all_hidden_data or not.')

    parser.add_argument('--env_first_step', type=str2bool, default=False,
                        help='whether to env_first_step or not.')

    parser.add_argument('--use_discrete_z', type=str2bool, default=False,
                        help='whether to use discrete z value for different action in z dimension.')


    parser.add_argument('--use_tensorboard', type=str2bool, default=True,
                        help='whether to use tensorboard in pytorch or not for visualization purpose.')
    
     
    
    parser.add_argument('--complicated_pre_dynamics', type=str2bool, default=False,
                        help='whether to use complicated_pre_dynamics or not.')

    parser.add_argument('--fast_training', type=str2bool, default=False,
                        help='whether to use fast_training or not.')

    parser.add_argument('--fast_training_test', type=str2bool, default=True,
                        help='whether to use fast_training or not.')     

    parser.add_argument('--add_test_data', type=str2bool, default=False,
                        help='whether to use add test data or not.') 

    parser.add_argument('--get_numer_of_parameters', type=str2bool, default=False,
                        help='whether to get number of parameters or not.')     

    parser.add_argument('--one_bit_env', type=str2bool, default=True,
                        help='whether to use one bit to represent environment identity.') 

    parser.add_argument('--enable_return', type=str2bool, default=False,
                        help='whether to use return for saving and sampling data.')  

    parser.add_argument('--augument_data', type=str2bool, default=False,
                        help='whether to use augument_data or not.') 

    parser.add_argument('--use_specific_relations', type=str2bool, default=False,
                        help='whether to use use_specific_relations.')  

    parser.add_argument('--logit_loss', type=str2bool, default=False,
                        help='whether to use logit_loss or not.')  

    
    parser.add_argument('--double_push', type=str2bool, default=False,
                        help='whether to use double_push or not.') 

    parser.add_argument('--stack_pickplace', type=str2bool, default=False,
                        help='whether to use stack_pickplace or not.') 

    parser.add_argument('--more_weight_regulirization', type=str2bool, default=False,
                        help='whether to use more_weight_regulirization or not.') 
    
    parser.add_argument('--lfd_search', type=str2bool, default=False,
                        help='whether to use lfd for graph search or not.') 

    parser.add_argument('--use_env_for_graph_search', type=str2bool, default=True,
                        help='whether to use_env_for_graph_search or not.') 

   
    
    
    parser.add_argument('--adam_task', type=str2bool, default=False,
                        help='whether to ues adam_task or not.')

    parser.add_argument('--only_high_push', type=str2bool, default=False,
                        help='whether to ues only_high_push or not.')
    
    
    
    parser.add_argument('--larger_output_sigmoid', type=str2bool, default=False,
                        help='whether to ues larger_output_sigmoid or not.')

    parser.add_argument('--pe', type=str2bool, default=False,
                        help='whether to ues pose estimation or not.')


    
    parser.add_argument('--bookshelf_env_shift', type=int, default=0,
                        help='to solve the shift in segmentation id in bookshelf case. if shift use 1, else use 0')
    
    parser.add_argument('--save_sampling_max', type=int, default=10000,
                        help='the max_number for save and sampling')
    parser.add_argument('--consider_end_range', type=int, default=5,
                        help='the consider range for sampling relations. Default is 5, which means behind, front, below, top, contact')

    
    
    parser.add_argument('--view_points', type=str2bool, default=False,
                        help='whether to ues view_points or not.')

    parser.add_argument('--get_hidden_label', type=str2bool, default=False,
                        help='whether to get hidden label from dataloader.')
    
    parser.add_argument('--using_latent_memory', type=str2bool, default=False,
                        help='whether to using_latent_memory.')
    
    
    
    parser.add_argument('--relation_angle', type=float, default=4.0,
                        help='real_relation angle = np.pi/relation_angle')
    parser.add_argument('--graph_search_threshold', type=float, default=0.5,
                        help='threhold for graph search success rate definition')

    parser.add_argument('--use_transformer', type=str2bool, default=True,
                        help='whether to use transformer.')

    parser.add_argument('--torch_embedding', type=str2bool, default=True,
                        help='whether to use torch_embedding or not.')

    parser.add_argument('--train_env_identity', type=str2bool, default=True,
                        help='whether to use train_env_identity or not.')
    


    parser.add_argument('--n_layers', type=int, default=2,
                        help='the nlayers for transformer')

    parser.add_argument('--n_heads', type=int, default=2,
                        help='the nheads for transformer')
    
    
    
    
                        
    args = parser.parse_args()
    # pprint.pprint(args.__dict__)
    np.set_printoptions(precision=4, linewidth=120)

    if args.set_random_seed:
        seed = args.seed  # previous version is all 0
        np.random.seed(seed)
        torch.manual_seed(seed)

    main(args)

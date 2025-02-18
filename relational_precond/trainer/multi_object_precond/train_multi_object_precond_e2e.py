# Contains the full training processes for eRDTransformer.

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


from relational_precond.dataloader.real_robot_dataloader import AllPairVoxelDataloaderPointCloud3stack

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

        self.save_real_data_info = args.save_real_data_info

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

        self.save_all_planning_info = args.save_all_planning_info
        self.manual_relations = args.manual_relations
        

        self.set_max = args.set_max
        if args.seperate_env_id:
            self.max_objects = args.max_objects + args.max_env_num
        else:
            self.max_objects = args.max_objects





        self.seperate_range = args.seperate_range

        self.random_sampling_relations = args.random_sampling_relations

        self.using_delta_training = args.using_delta_training

        self.cem_planning = args.cem_planning

        
        self.using_multi_step_latent = args.using_multi_step_latent

        
        self.using_latent_regularization = args.using_latent_regularization

        self.save_many_data = args.save_many_data

        self.test_next_step = args.test_next_step

        

        self.sampling_once = args.sampling_once
       
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

    

        self.dataloader = AllPairVoxelDataloaderPointCloud3stack(
            config,
            use_multiple_train_dataset = args.use_multiple_train_dataset,
            pick_place = self.pick_place, 
            pushing = self.pushing,
            stacking = self.stacking, 
            set_max = self.set_max, 
            max_objects = self.max_objects,
            start_id = args.start_id, 
            max_size = args.max_size, 
            start_test_id = args.start_test_id, 
            test_max_size = args.test_max_size,
            updated_behavior_params = args.updated_behavior_params,
            save_data_path = args.save_data_path,
            evaluate_pickplace = args.evaluate_pickplace,
            using_multi_step_statistics = args.using_multi_step_statistics,
            total_multi_steps = args.total_sub_step,
            use_shared_latent_embedding = args.use_shared_latent_embedding,
            use_seperate_latent_embedding = args.use_seperate_latent_embedding,
            push_3_steps = args.push_3_steps,
            single_test = args.single_test,
            single_push = args.single_push,
            stack_pickplace = args.stack_pickplace, 
            use_boundary_relations = args.use_boundary_relations,
            consider_z_offset = args.consider_z_offset,
            seperate_env_id = args.seperate_env_id,
            max_env_num = args.max_env_num,
            env_first_step = args.env_first_step,
            use_discrete_z = args.use_discrete_z,
            fast_training = args.fast_training,
            one_bit_env = args.one_bit_env,
            relation_angle = args.relation_angle,
            bookshelf_env_shift = args.bookshelf_env_shift,
            get_hidden_label = args.get_hidden_label
            )
             
           
        args = config.args
        
        # TODO: Use the arguments saved in the emb_checkpoint_dir to create 

        
        if args.train_type == 'all_object_pairs_gnn_new':
            
            
            self.emb_model_planar = PointConv(normal_channel=False)
            
            if self.e2e:
                self.emb_model = PointConv(normal_channel=False)
                if self.enable_seperate_pointconv:
                    self.emb_model_env = PointConv(normal_channel=False)
                
                
                
                
            if self.set_max:
                node_inp_size, edge_inp_size = self.max_objects + 3, args.z_dim
            
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


        
        
        self.opt_emb = optim.Adam(self.emb_model.parameters(), lr=args.emb_lr)
        if self.enable_seperate_pointconv:
            self.opt_emb_env = optim.Adam(self.emb_model_env.parameters(), lr=args.emb_lr)
        self.opt_classif = optim.Adam(self.classif_model.parameters(), lr=args.learning_rate) 
        self.opt_classif_decoder = optim.Adam(self.classif_model_decoder.parameters(), lr=args.learning_rate) 
        
        


       

        self.dynamics_loss = nn.MSELoss()
        if args.logit_loss:
            self.bce_loss = nn.BCEWithLogitsLoss()
        else:
            self.bce_loss = nn.BCELoss()

    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
  

    
    def get_model_list(self):
        return [self.emb_model ,self.classif_model,self.classif_model_decoder]
        
    def get_state_dict(self):
        return {
            'emb_model': self.emb_model.state_dict(),
            'classif_model': self.classif_model.state_dict(),
            'classif_model_decoder': self.classif_model_decoder.state_dict(),
        }

    def set_model_device(self, device=torch.device("cpu")):
        model_list = self.get_model_list()
        for m in model_list:
            m.to(device)
    
    def set_model_to_train(self):
        model_list = self.get_model_list()
        for m in model_list:
            m.train()

    def set_model_to_eval(self):
        model_list = self.get_model_list()
        for m in model_list:
            m.eval()

    

    def model_checkpoint_dir(self):
        '''Return the directory to save models in.'''
        return self.config.get_model_checkpoint_dir()

    def model_checkpoint_filename(self, epoch):
        return os.path.join(self.model_checkpoint_dir(),
                            'cp_{}.pth'.format(epoch))

    
    def save_checkpoint(self, epoch):
        cp_filepath = self.model_checkpoint_filename(epoch)
        torch.save(self.get_state_dict(), cp_filepath)
        print(bcolors.c_red("Save checkpoint: {}".format(cp_filepath)))

    def load_checkpoint(self, checkpoint_path):
        cp_models = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        self.emb_model.load_state_dict(cp_models['emb_model'])
        self.classif_model.load_state_dict(cp_models['classif_model'])
        self.classif_model_decoder.load_state_dict(cp_models['classif_model_decoder'])
        
    def one_step_process_all(self, batch_data, push_3_steps = False): ## assume batch_size = 1
        '''Process raw batch data and collect relevant objects in a dict.'''
        # proc_batch_dict = {
        #     # GPU
        #     'batch_voxel_list_single': [],     # (B, N, D)  
        #     'batch_all_obj_pair_relation': [], # (B, N*(N-1), D)
        #     'batch_action': [],    # (B, N, D)
        #     'batch_one_hot_encoding': [], # (B, N, D)
        #     'batch_edge_attr': [],
        #     # CPU
        #     'batch_num_objects': [],
        #     'batch_skill_label': [],
        # }

        args = self.config.args
        x_dict = dict()

        
        assert(len(batch_data) == 1)

        for b, data in enumerate(batch_data):
            # 'num_objects': total_objects_fast,
            #     'action': all_relation_fast,
            #     'relation': all_relation_fast, 
            #     'all_object_pair_voxels_single': obj_voxels_single_fast,
            #     'one_hot_encoding': one_hot_encoding_tensor_fast,
            #     'edge_attr': edge_attr_fast,
            #     'all_action_label':all_action_label_fast
            
            x_dict['batch_num_objects'] = data['num_objects']
            x_dict['batch_action'] = data['action']
            x_dict['batch_all_obj_pair_relation'] = data['relation']
            x_dict['batch_one_hot_encoding'] = data['one_hot_encoding']
            x_dict['batch_edge_attr'] = data['edge_attr']
            x_dict['batch_skill_label'] = data['all_action_label']
            x_dict['batch_voxel_list_single'] = data['all_object_pair_voxels_single']
            x_dict['batch_env_identity'] = data['env_identity']
            x_dict['batch_6DOF_pose'] = data['all_6DOF_pose_fast']
            
            if self.get_hidden_label:
                x_dict['batch_all_hidden_label'] = data['all_hidden_label']
            
        
        return x_dict
            
    
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

    def run_model_on_batch_torch_geometry_pick_primitive_new_relational_classifier_point_cloud_e2e_fast_training(self,
                           x_tensor_dict,
                           x_tensor_dict_next,
                           batch_size,
                           train=False,
                           threshold = 0):

        batch_result_dict = {}
        device = self.config.get_device()
        args = self.config.args

        total_loss = 0

        
        total_steps = x_tensor_dict['batch_all_obj_pair_relation'].shape[0] # should be 4 for pushing and 3 for stacking
        


        self.num_nodes = x_tensor_dict['batch_num_objects'] 

        data_list = []

        current_latent_list = []
        current_output_classifier_list = []
        current_env_identity_list = []


        predict_latent_list = [[], [], []]  ## last step does not have predicted_latent any more. 
        decoder_classifier_list = [[], [], []]
        decoder_env_identity_list = [[], [], []]


        action_torch_list = []

        action_label_list = []


        
        
        x_tensor_dict['batch_all_obj_pair_relation'] = x_tensor_dict['batch_all_obj_pair_relation'] 

       
        for this_step in range(total_steps):
            
            
            voxel_data_single = x_tensor_dict['batch_voxel_list_single'][this_step]


            if this_step != total_steps - 1:
                action = x_tensor_dict['batch_action'][this_step]
                action_label = x_tensor_dict['batch_skill_label'][this_step]
            else: 
                action = x_tensor_dict['batch_action'][this_step - 1] 
                action_label = x_tensor_dict['batch_skill_label'][this_step - 1]
            
            
            # print('action_label', action_label)
            img_emb_single = self.emb_model(voxel_data_single)

            self.edge_index = x_tensor_dict['batch_edge_attr']
            

            action_torch = action

            action_label_list.append(action_label)

            action_torch_list.append(action_torch)
            
                
            one_hot_encoding_tensor = x_tensor_dict['batch_one_hot_encoding'] 

            
            if self.torch_embedding:
                latent_one_hot_encoding = self.classif_model.one_hot_encoding_embed(torch.argmax(one_hot_encoding_tensor, dim = 1))
            else:
                one_hot_encoding_tensor = one_hot_encoding_tensor.float()
                latent_one_hot_encoding = self.classif_model.one_hot_encoding_embed(one_hot_encoding_tensor)
            
            
            node_pose = torch.cat([img_emb_single, latent_one_hot_encoding], dim = 1)
            

            outs = self.classif_model(node_pose, self.edge_index, None, action_torch, action_label)
            
            
            
            if self.use_transformer:
                outs_decoder = self.classif_model_decoder(outs['current_embed'],  self.edge_index, None, action_torch, action_label)
            else:
                outs_decoder = self.classif_model_decoder(outs['pred'],  self.edge_index, outs['pred_edge'], action_torch, action_label)

            if self.use_transformer:
                current_latent_list.append([outs['current_embed']])
            else:
                current_latent_list.append([outs['pred'], outs['pred_edge']])
            

            current_output_classifier_list.append(outs_decoder['pred_sigmoid'][:])

            if self.train_env_identity:
                current_env_identity_list.append(outs_decoder['env_identity'][:]) 

            
            

            data = []
            data_list.append(data)
        

        
        self.node_effective_id = np.arange(self.num_nodes)
        self.edge_effective_id = np.arange(x_tensor_dict['batch_all_obj_pair_relation'][0].shape[0])

                
        
        
        
        # print(self.node_effective_id)
        # print(self.edge_effective_id)
        current_step_prediction_loss = 0
        regularization_loss = 0
        next_step_prediction_loss = 0
        current_env_identity_loss = 0
        next_env_identity_loss = 0

        for this_step in range(len(data_list) - 1):
            current_latent = current_latent_list[this_step][0]
            if not self.use_transformer:
                egde_latent = current_latent_list[this_step][1]

            
            for seq in range(this_step, len(data_list) - 1):
                
                if self.use_seperate_latent_embedding:
                    
                    skill_label = action_label_list[seq]

                    if self.seperate_discrete_continuous:
                        if self.torch_embedding:
                            discrete_action = self.classif_model.one_hot_encoding_embed(torch.argmax(action_torch_list[seq][:, 1:-3], dim = 1))
                        else:
                            discrete_action = self.classif_model.one_hot_encoding_embed(action_torch_list[seq][:, 1:-3])
                        if self.seperate_action_emb:
                            if skill_label == 0:
                                continuous_action = self.classif_model.continuous_action_emb(action_torch_list[seq][:, -3:])
                            elif skill_label == 1:
                                continuous_action = self.classif_model.continuous_action_emb_1(action_torch_list[seq][:, -3:])
                        else:
                            continuous_action = self.classif_model.continuous_action_emb(action_torch_list[seq][:, -3:])
                        current_action = torch.cat((discrete_action, continuous_action), axis = -1)
                    else:
                        current_action = self.classif_model.action_emb(action_torch_list[seq][:, 1:])
                
                    if self.use_transformer:
                        current_action = current_action.view(1, current_action.shape[0], current_action.shape[1])
                        current_action = current_action[0, 0, :]
                        current_action = current_action.view(1, 1, current_action.shape[0])
                        # print('current_action', current_action.shape)
                        graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                        # print('graph_node_action', graph_node_action.shape)
                    else:
                        graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                    
                    if True:
                        if skill_label == 0:
                            current_latent = self.classif_model.graph_dynamics_0(graph_node_action)
                        elif skill_label == 1:
                            # print(graph_node_action.shape)
                            current_latent = self.classif_model.graph_dynamics_1(graph_node_action)
                        elif skill_label == 2:
                            # print(graph_node_action.shape)
                            current_latent = self.classif_model.graph_dynamics_2(graph_node_action)


                    if not self.use_transformer:
                        edge_num = egde_latent.shape[0]
                        edge_action_list = []

                        if self.seperate_discrete_continuous:
                            discrete_action = self.classif_model.one_hot_encoding_embed(action_torch_list[seq][:, 1:-3])
                            if self.seperate_action_emb:
                                if skill_label == 0:
                                    continuous_action = self.classif_model.continuous_action_emb(action_torch_list[seq][:, -3:])
                                elif skill_label == 1:
                                    continuous_action = self.classif_model.continuous_action_emb_1(action_torch_list[seq][:, -3:])
                            else:
                                continuous_action = self.classif_model.continuous_action_emb(action_torch_list[seq][:, -3:])
                            current_action = torch.cat((discrete_action, continuous_action), axis = -1)
                        else:
                            current_action = self.classif_model.action_emb(action_torch_list[seq][:, 1:])
                    
                        for _ in range(edge_num):
                            edge_action_list.append(current_action[0])
                        edge_action = torch.stack(edge_action_list)
                        graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)

                        if skill_label == 0:
                            egde_latent = self.classif_model.graph_edge_dynamics_0(graph_edge_action)
                        elif skill_label == 1:
                            egde_latent = self.classif_model.graph_edge_dynamics_1(graph_edge_action)
                        
                    if self.use_transformer:
                        predict_latent_list[this_step].append([current_latent])
                    else:
                        predict_latent_list[this_step].append([current_latent, egde_latent])
                
                
                if self.use_transformer:
                    outs_decoder_2_edge = self.classif_model_decoder(current_latent, self.edge_index, None, action_torch_list[0], action_label_list[0]) # action here is not important since we only use decoder here.  
                else:
                    outs_decoder_2_edge = self.classif_model_decoder(current_latent, self.edge_index, egde_latent, action_torch_list[0], action_label_list[0]) # action here is not important since we only use decoder here.  


                
                decoder_classifier_list[this_step].append(outs_decoder_2_edge['pred_sigmoid'][:])

                if self.train_env_identity:
                    decoder_env_identity_list[this_step].append(outs_decoder_2_edge['env_identity'][:]) 

                
                

        for i in range(len(current_output_classifier_list)):
            # print('1')
            # print([current_output_classifier_list[i].shape, x_tensor_dict['batch_all_obj_pair_relation'][i][:, :].shape])
            # print(current_output_classifier_list.shape)
            # print(x_tensor_dict['batch_all_obj_pair_relation'].shape)
            total_loss += self.bce_loss(current_output_classifier_list[i][self.edge_effective_id], x_tensor_dict['batch_all_obj_pair_relation'][i][self.edge_effective_id, :])
            current_step_prediction_loss += self.bce_loss(current_output_classifier_list[i][self.edge_effective_id], x_tensor_dict['batch_all_obj_pair_relation'][i][self.edge_effective_id, :])

            if self.train_env_identity:
                current_env_identity_loss += self.bce_loss(current_env_identity_list[i][self.node_effective_id], x_tensor_dict['batch_env_identity'][i][self.node_effective_id])
                total_loss += self.bce_loss(current_env_identity_list[i][self.node_effective_id], x_tensor_dict['batch_env_identity'][i][self.node_effective_id])

            
           

        if self.using_latent_regularization:
            if self.use_transformer:
                for i in range(len(predict_latent_list)): ## make this code more clear and easy to understand
                    this_step_all_predict_latent = predict_latent_list[i]
                    # print(len(this_step_all_predict_latent))
                    for j in range(len(this_step_all_predict_latent)):
                        # print(this_step_all_predict_latent[j])
                        if j < self.max_training_step:
                            if self.more_weight_regulirization:
                                print('enter more regulirization weight')
                                total_loss += (10 * self.dynamics_loss(this_step_all_predict_latent[j][0][:, self.node_effective_id], current_latent_list[i+j+1][0][:, self.node_effective_id]))
                                regularization_loss += (10 * self.dynamics_loss(this_step_all_predict_latent[j][0][:, self.node_effective_id], current_latent_list[i+j+1][0][:, self.node_effective_id]))
                            else:
                                total_loss += self.dynamics_loss(this_step_all_predict_latent[j][0][:, self.node_effective_id], current_latent_list[i+j+1][0][:, self.node_effective_id])
                                regularization_loss += self.dynamics_loss(this_step_all_predict_latent[j][0][:, self.node_effective_id], current_latent_list[i+j+1][0][:, self.node_effective_id])
            else:
                for i in range(len(predict_latent_list)): ## make this code more clear and easy to understand
                    this_step_all_predict_latent = predict_latent_list[i]
                    for j in range(len(this_step_all_predict_latent)):
                        if j < self.max_training_step:
                            # print([i,j])
                            for node_or_edge_id in range(2):
                                # print('2')
                                # print([this_step_all_predict_latent[j][node_or_edge_id].shape, current_latent_list[i+j+1][node_or_edge_id].shape])
                                if node_or_edge_id == 0:
                                    if self.more_weight_regulirization:
                                        # print('enter more regulirization weight')
                                        total_loss += (10 * self.dynamics_loss(this_step_all_predict_latent[j][node_or_edge_id][self.node_effective_id], current_latent_list[i+j+1][node_or_edge_id][self.node_effective_id]))
                                        regularization_loss += (10 * self.dynamics_loss(this_step_all_predict_latent[j][node_or_edge_id][self.node_effective_id], current_latent_list[i+j+1][node_or_edge_id][self.node_effective_id]))
                                    else:
                                        total_loss += self.dynamics_loss(this_step_all_predict_latent[j][node_or_edge_id][self.node_effective_id], current_latent_list[i+j+1][node_or_edge_id][self.node_effective_id])
                                        regularization_loss += self.dynamics_loss(this_step_all_predict_latent[j][node_or_edge_id][self.node_effective_id], current_latent_list[i+j+1][node_or_edge_id][self.node_effective_id])
                                else:
                                    if self.more_weight_regulirization:
                                        # print('enter more regulirization weight')
                                        total_loss += (10 * self.dynamics_loss(this_step_all_predict_latent[j][node_or_edge_id][self.edge_effective_id], current_latent_list[i+j+1][node_or_edge_id][self.edge_effective_id]))
                                        regularization_loss += (10 * self.dynamics_loss(this_step_all_predict_latent[j][node_or_edge_id][self.edge_effective_id], current_latent_list[i+j+1][node_or_edge_id][self.edge_effective_id]))
                                    else:
                                        total_loss += self.dynamics_loss(this_step_all_predict_latent[j][node_or_edge_id][self.edge_effective_id], current_latent_list[i+j+1][node_or_edge_id][self.edge_effective_id])
                                        regularization_loss += self.dynamics_loss(this_step_all_predict_latent[j][node_or_edge_id][self.edge_effective_id], current_latent_list[i+j+1][node_or_edge_id][self.edge_effective_id])

        for i in range(len(decoder_classifier_list)):
            this_step_all_decoder_classifier = decoder_classifier_list[i]
            for j in range(len(this_step_all_decoder_classifier)):
                # print('3')
                # print(this_step_all_decoder_classifier[j].shape, x_tensor_dict['batch_all_obj_pair_relation'][i + j + 1][:, :].shape)
                if j < self.max_training_step:
                    # print([i,j])
                    total_loss += self.bce_loss(this_step_all_decoder_classifier[j][self.edge_effective_id], x_tensor_dict['batch_all_obj_pair_relation'][i + j + 1][self.edge_effective_id, :])
                    next_step_prediction_loss += self.bce_loss(this_step_all_decoder_classifier[j][self.edge_effective_id], x_tensor_dict['batch_all_obj_pair_relation'][i + j + 1][self.edge_effective_id, :])

        if self.train_env_identity:
            for i in range(len(decoder_env_identity_list)):
                this_step_decoder_env_identity_list = decoder_env_identity_list[i]
                for j in range(len(this_step_decoder_env_identity_list)):
                    if j < self.max_training_step:
                        next_env_identity_loss += self.bce_loss(this_step_decoder_env_identity_list[j][self.node_effective_id], x_tensor_dict['batch_env_identity'][i + j + 1][self.node_effective_id])
                        total_loss += self.bce_loss(this_step_decoder_env_identity_list[j][self.node_effective_id], x_tensor_dict['batch_env_identity'][i + j + 1][self.node_effective_id])


        
        

                    
                    
                
        if self.use_tensorboard:
            if train:
                self.writer.add_scalar("Loss/train", total_loss, self.loss_iter)
                self.writer.add_scalar("Loss/current_step", current_step_prediction_loss, self.loss_iter)
                self.writer.add_scalar("Loss/next_step", next_step_prediction_loss, self.loss_iter)
                self.writer.add_scalar("Loss/regularization", regularization_loss, self.loss_iter)
                if self.train_env_identity:
                    self.writer.add_scalar("Loss/current_step_env", current_env_identity_loss, self.loss_iter)
                    self.writer.add_scalar("Loss/next_step_env", next_env_identity_loss, self.loss_iter)
                self.loss_iter += 1
                if self.loss_iter % 100 == 1:
                    self.writer.flush()
            else:
                self.writer.add_scalar("Loss/test", total_loss, self.loss_iter)
                self.writer.add_scalar("Loss/test_current_step", current_step_prediction_loss, self.loss_iter)
                self.writer.add_scalar("Loss/test_next_step", next_step_prediction_loss, self.loss_iter)
                self.writer.add_scalar("Loss/test_regularization", regularization_loss, self.loss_iter)
                if self.train_env_identity:
                    self.writer.add_scalar("Loss/test_current_step_env", current_env_identity_loss, self.loss_iter)
                    self.writer.add_scalar("Loss/test_next_step_env", next_env_identity_loss, self.loss_iter)
                self.loss_iter += 1

        # print(total_loss)
        if train:
            self.opt_emb.zero_grad()

            self.opt_classif.zero_grad()
            self.opt_classif_decoder.zero_grad()
        

            total_loss.backward()
            self.opt_emb.step()
            
            self.opt_classif.step()
            self.opt_classif_decoder.step()
            



        
        
        
                              
        return batch_result_dict

    

    
    def get_next_data_from_dataloader(self, dataloader, train):
        args = self.config.args
        data = None
        if args.train_type == ALL_OBJ_PAIRS_GNN_NEW:
            data, data_next = dataloader.get_next_all_object_pairs_for_scene(train)
        else:
            raise ValueError(f"Invalid train type: {args.train_type}")
        return data, data_next
   
    def train_next(self, train=True, threshold = 0.8):
        print("Begin training")
        args = self.config.args
        dataloader = self.dataloader
        device = self.config.get_device()

        train_data_size = dataloader.number_of_scene_data(train)



        self.set_model_device(device)

        
        
        num_epochs = args.num_epochs if train else 1
        



        
        
        if train_data_size == 0:
            raise ValueError("Training total size == 0")
        else:
            if self.fast_training:
                dataloader.put_all_data_device(device)
            for e in range(num_epochs):
                dataloader.reset_scene_batch_sampler(train=train, shuffle=train)

                batch_size = args.batch_size 
                num_batches = train_data_size // batch_size
                if train_data_size % batch_size != 0:
                    num_batches += 1

                
                data_idx = 0


                for batch_idx in range(num_batches):

                    batch_data = []
                    batch_data_next = []

                    while len(batch_data) < batch_size and data_idx < train_data_size:  # in current version, we totally ignore batch size
                        data, data_next = self.get_next_data_from_dataloader(dataloader, train)
                        batch_data.append(data)
                        batch_data_next.append(data_next)
                        data_idx = data_idx + 1


                    x_tensor_dict = self.one_step_process_all(batch_data)
                    x_tensor_dict_next = self.one_step_process_all(batch_data_next)
                
                    
                    batch_result_dict = self.run_model_on_batch_torch_geometry_pick_primitive_new_relational_classifier_point_cloud_e2e_fast_training(
                        x_tensor_dict,
                        x_tensor_dict_next,
                        batch_size,
                        train=train,
                        threshold = threshold)
                    
                    

                
                

                if train:
                    self.save_checkpoint(e) # save checkpoint after every epoch too. 
        
        
        return batch_result_dict
            
          


def main(args):
    dtype = torch.FloatTensor
    if args.cuda:
        dtype = torch.cuda.FloatTensor

    config = BaseVAEConfig(args, dtype=dtype)
    create_log_dirs(config)

    trainer = MultiObjectVoxelPrecondTrainerE2E(config)


    config_pkl_path = os.path.join(args.result_dir, 'config.pkl')
    config_json_path = os.path.join(args.result_dir, 'config.json')
    with open(config_pkl_path, 'wb') as config_f:
        pickle.dump((args), config_f, protocol=2)
        print(bcolors.c_red("Did save config: {}".format(config_pkl_path)))
    with open(config_json_path, 'w') as config_json_f:
        config_json_f.write(json.dumps(args.__dict__))

    result_dict = trainer.train_next()
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training/planning for eRDTransformer.')


    parser.add_argument('--cuda', type=int, default=1, help="Use cuda")
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory to save results.')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Checkpoint to test on.')

    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for each step')

    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for encoder/decoder.')

    parser.add_argument('--save_freq_iters', type=int, default=1001,
                        help='Frequency at which to save models.')


    
    parser.add_argument('--train_dir', required=True, action='append',
                        help='Path to training directory.')
    parser.add_argument('--test_dir', action='append',
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

    
    
    parser.add_argument('--set_max', type=str2bool, default=True,
                        help='whether to use set_max method')
    parser.add_argument('--max_objects', type=int, default=8,
                        help='max_objects in this experiments')
    parser.add_argument('--total_sub_step', type=int, default=2,
                        help='total sub steps for multi-step test')
    
    parser.add_argument('--save_all_planning_info', type=str2bool, default=False,
                        help='whether to save all planning info')

   
    
    parser.add_argument('--mlp', type=str2bool, default=False,
                        help='whether to use mlp baselines as a comparison')
    parser.add_argument('--use_multiple_train_dataset', type=str2bool, default=False,
                        help='whether to use use_multiple_train_dataset')

    parser.add_argument('--manual_relations', type=str2bool, default=True,
                        help='whether to use manual_relations')


    parser.add_argument('--evaluate_pickplace', type=str2bool, default=False,
                        help='whether to use evaluate_pickplace')
    parser.add_argument('--updated_behavior_params', type=str2bool, default=True,
                        help='whether to use updated_behavior_params')
    parser.add_argument('--start_id', type=int, default=5,
                        help='start_id in hthe training')
    parser.add_argument('--max_size', type=int, default=10,
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
    
    parser.add_argument('--using_multi_step', type=str2bool, default=False,
                        help='whether to use multi step planning as a task and motion planning style') 
    
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
    parser.add_argument('--sampling_once', type=str2bool, default=False,
                        help='whether to sampling_once for random sample goal.')
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
    parser.add_argument('--z_range', type=float, default=0,
                        help='z_range sampling in the CEM planning')
    parser.add_argument('--x_middle_point', type=float, default=0.5,
                        help='x_middle_point sampling in the CEM planning')
    parser.add_argument('--y_middle_point', type=float, default=0.5,
                        help='y_middle_point sampling in the CEM planning')
    parser.add_argument('--z_middle_point', type=float, default=0,
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
    parser.add_argument('--save_real_data_info', type=str2bool, default=False,
                        help='whether to disable push during sampling.')
   
    parser.add_argument('--single_test', type=str2bool, default=False,
                        help='whether to use the single step test.')
    parser.add_argument('--single_push', type=str2bool, default=True,
                        help='whether to use the single step test.')
    parser.add_argument('--hidden_leap', type=str2bool, default=False,
                        help='whether to use the hidden leap.')
    
    parser.add_argument('--remove_env_edges', type=str2bool, default=False,
                        help='whether to remove environment edges.')
    parser.add_argument('--use_part_graph', type=str2bool, default=False,
                        help='whether to use part graph.')
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

    parser.add_argument('--fast_training', type=str2bool, default=True,
                        help='whether to use fast_training or not.')

    parser.add_argument('--fast_training_test', type=str2bool, default=False,
                        help='whether to use fast_training or not.')     

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

    parser.add_argument('--use_transformer', type=str2bool, default=True,
                        help='whether to use transformer.')

    parser.add_argument('--torch_embedding', type=str2bool, default=True,
                        help='whether to use torch_embedding or not.')

    parser.add_argument('--train_env_identity', type=str2bool, default=True,
                        help='whether to use train_env_identity or not.')
    
    
    parser.add_argument('--relation_angle', type=float, default=4.0,
                        help='real_relation angle = np.pi/relation_angle')

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

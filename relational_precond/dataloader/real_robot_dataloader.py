# Contains classes and functions for loading real robot dataset(s).
# Manages and preprocesses the data for training and testing.

import numpy as np

import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import time

import copy
import math

from itertools import permutations

from relational_precond.dataloader.farthest_point_sampling import farthest_point_sampling
from relational_precond.utils import math_util

from relational_precond.utils.data_utils import get_norm, get_activation, scale_min_max

import torch
from torchvision.transforms import functional as F

import torch.nn as nn 
import torch.nn.functional as F

from relational_precond.utils import torch_util

from typing import List, Dict



def rotate_2d(x, theta):
    """ Rotate x by theta degrees (counter clockwise)

        @param x: a 2D vector (numpy array with shape 2)
        @param theta: a scalar representing radians
    """

    rot_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ], dtype=np.float32)

    return rot_mat.dot(x)


class RobotAllPairSceneObjectPointCloudVariablestack(object):
    def __init__(self,  
                 scene_path, 
                 scene_type, 
                 relation_angle,
                 pick_place = False, 
                 push = False,
                 set_max = False,
                 train = True,
                 scene_pos_info=None,
                 max_objects = 5,
                 test_dir_1 = None,
                 updated_behavior_params = False, 
                 evaluate_pickplace = False,
                 this_one_hot_encoding = None,
                 use_shared_latent_embedding = False, 
                 use_seperate_latent_embedding = False,
                 push_3_steps = False,
                 stack_push = False,
                 push_pickplace = False, 
                 single_test = False,
                 single_push = False,
                 double_pickplace = False,
                 use_boundary_relations = False,
                 consider_z_offset = False,
                 seperate_env_id = False,
                 max_env_num = 0,
                 env_first_step = False,
                 use_discrete_z = False,
                 fast_training = False,
                 one_bit_env = False,
                 bookshelf_env_shift = 0,
                 double_push = False,
                 enable_return = False,
                 add_memory = False,
                 get_hidden_label = False,
                 stack_pickplace = False):


        
        

        
        self.double_push = double_push

        self.evaluate_pickplace = evaluate_pickplace
        self.updated_behavior_params = updated_behavior_params
        self.train = train
        self.scene_path = scene_path
        self.scene_type = scene_type
        self.set_max = set_max
        self.pushing = push
        self.use_shared_latent_embedding = use_shared_latent_embedding
        self.use_seperate_latent_embedding = use_seperate_latent_embedding
        self.push_3_steps = push_3_steps
        self.pick_place = pick_place
        self.stack_push = stack_push
        self.push_pickplace = push_pickplace
        self.single_test = single_test
        self.single_push = single_push
        self.use_boundary_relations = use_boundary_relations
        self.consider_z_offset = consider_z_offset
        self.seperate_env_id = seperate_env_id
        self.max_env_num = max_env_num
        self.env_first_step = env_first_step
        self.use_discrete_z = use_discrete_z
        self.double_pickplace = double_pickplace
        self.enable_return = enable_return
        self.stack_pickplace = stack_pickplace
        self.get_hidden_label = get_hidden_label

        self.fast_training = fast_training

        self.one_bit_env = one_bit_env

        self.bookshelf_env_shift = bookshelf_env_shift

        self.add_memory = add_memory
        

        if self.push_3_steps:
            if self.stack_pickplace:
                self.sample_time_step = [0, 11, -1]
            elif self.double_push:
                self.sample_time_step = [0, 6, -1]
            elif self.single_test or self.single_push:
                self.sample_time_step = [0, -1]
            elif self.push_pickplace:
                self.sample_time_step = [0, 4, -1]
            elif self.stack_push:
                self.sample_time_step = [0, 10, 19, -1]
            elif self.double_pickplace:
                if self.enable_return:
                    print('eneter enable_return')
                    self.sample_time_step = [0, 11, -1]
                else:
                    self.sample_time_step = [0, 10, -1]
            elif self.pushing:
                if self.enable_return:
                    print('eneter enable_return')
                    self.sample_time_step = [0, 6, 13, -1]
                else:
                    self.sample_time_step = [0, 3, 7, -1]
            else:
                self.sample_time_step = [0, 10, -1]
        
       

        

        self.params = {
            'theta_predicte_lr_fb_ab' : np.pi / relation_angle, # keep consistent for 45 degrees since there are some problems about 90 degrees. 
            'occ_IoU_threshold' : 0.5,
        }

        
        with open(self.scene_path, 'rb') as f:
            data, attrs = pickle.load(f)

        
        
        #print('enter')
        self.all_point_cloud_1 = []
        self.all_point_cloud_2 = []
        self.all_point_cloud_3 = []
            
        self.all_point_cloud_last = []
        data_size = 128
        self.scale = 1
        self.all_pos_list_1 = []
        self.all_pos_list_2 = []
        self.all_pos_list_3 = []
        self.all_pos_list_last = []
        self.all_orient_list = []
        self.all_orient_list_last = []
        self.all_point_cloud = []
        self.all_pos_list = []
        self.all_hidden_label_list = []
        self.all_pos_list_p = []
        self.all_gt_pose_list = []
        self.all_gt_orientation_list = []
        self.all_gt_max_pose_list = []
        self.all_gt_min_pose_list = []
        self.all_gt_extents_range_list = []
        self.all_gt_extents_list = []
        self.all_relation_list = []
        self.all_initial_bounding_box = []
        self.all_bounding_box = []
        self.all_axis_bounding_box = []
        self.all_rotated_bounding_box = []
        self.all_gt_identity_list = []
        self.all_rgb_identity_list = []
        self.all_gt_env_identity_list = []
        
        
        total_objects = 0
        if total_objects == 0:
            for k, v in data['objects'].items():
                if 'block' in k:
                    total_objects += 1

        
        
        if self.seperate_env_id:
            self.total_pure_obj_num = 0
            self.total_pure_env_num = 0
            for k, v in attrs['objects'].items():
                if 'block' in k and v['fix_base_link'] == False:
                    self.total_pure_obj_num += 1
                if 'block' in k and v['fix_base_link'] == True:
                    self.total_pure_env_num += 1
          



        
        self.total_objects = total_objects

        

        self.obj_pair_list = list(permutations(range(total_objects), 2))
        
        for i in range(data['point_cloud_1'].shape[0]):
            self.all_point_cloud.append([])
            self.all_pos_list.append([])
            self.all_gt_pose_list.append([])
            self.all_gt_identity_list.append([])
            self.all_gt_env_identity_list.append([])
            self.all_rgb_identity_list.append([])
            self.all_gt_orientation_list.append([])
            self.all_gt_max_pose_list.append([])
            self.all_gt_min_pose_list.append([])
            self.all_pos_list_p.append([])
            self.all_relation_list.append([])
            #self.all_initial_bounding_box.append([])
            self.all_bounding_box.append([])
            self.all_axis_bounding_box.append([])
            self.all_rotated_bounding_box.append([])
        

        #print(data)
        if self.set_max:
            # print(self.seperate_env_id)
            if self.seperate_env_id:
                

                self.max_objects_obj = max_objects - self.max_env_num ## max object number
                self.max_objects_env = self.max_env_num

                self.max_objects = self.max_objects_obj + self.max_objects_env

                A = np.arange(self.max_objects_obj)
                
                if train:
                    np.random.shuffle(A)
                                #print(A)
                select_obj_num_range_obj = A[:self.total_pure_obj_num]
                one_hot_encoding_obj = np.zeros((self.total_pure_obj_num, self.max_objects))
                for i in range(len(select_obj_num_range_obj)):
                    one_hot_encoding_obj[i][select_obj_num_range_obj[i]] = 1
                
                ## env part
                
                A = np.arange(self.max_objects_obj, self.max_objects)
                
                if train:
                    np.random.shuffle(A)
                                #print(A)
                select_obj_num_range_env = A[:self.total_pure_env_num]
                one_hot_encoding_env = np.zeros((self.total_pure_env_num, self.max_objects))
                for i in range(len(select_obj_num_range_env)):
                    one_hot_encoding_env[i][select_obj_num_range_env[i]] = 1

                ## combination part
                
                select_obj_num_range_obj = select_obj_num_range_obj.reshape(select_obj_num_range_obj.shape[0], 1)
                select_obj_num_range_env = select_obj_num_range_env.reshape(select_obj_num_range_env.shape[0], 1)
                # print(select_obj_num_range_obj.shape)
                # print(select_obj_num_range_env.shape)
                select_obj_num_range = np.concatenate((select_obj_num_range_obj, select_obj_num_range_env), axis = 0)
                select_obj_num_range = select_obj_num_range.reshape(select_obj_num_range.shape[0], )
                # print(select_obj_num_range.shape)
                self.select_obj_num_range = select_obj_num_range
                

                

                one_hot_encoding = np.zeros((self.total_pure_env_num + self.total_pure_obj_num, self.max_objects))

                # print(one_hot_encoding_obj.shape)
                one_hot_encoding[:self.total_pure_obj_num,:] = one_hot_encoding_obj
                
                one_hot_encoding[-self.total_pure_env_num:, :] = one_hot_encoding_env

                # print(one_hot_encoding)
                
            else:
                self.max_objects = max_objects
                A = np.arange(self.max_objects)
                #print('max objects',self.max_objects)
                
                if train:
                    np.random.shuffle(A)
                                #print(A)
                select_obj_num_range = A[:total_objects]
                self.select_obj_num_range = select_obj_num_range
                one_hot_encoding = np.zeros((total_objects, self.max_objects))
                for i in range(len(select_obj_num_range)):
                    one_hot_encoding[i][select_obj_num_range[i]] = 1
            if self.fast_training:
                self.one_hot_encoding_tensor_fast = torch.tensor(one_hot_encoding)
                self.total_objects_fast = total_objects
        else:
            A = np.arange(total_objects)
            np.random.shuffle(A)
            select_obj_num_range = A[:total_objects]
            self.select_obj_num_range = select_obj_num_range
        block_string = 'block_'
        for j in range(total_objects):
            if 'extents' in attrs['objects'][block_string + str(j+1)]: 
                if attrs['objects'][block_string + str(j+1)]['extents_ranges'] == None:
                    attrs['objects'][block_string + str(j+1)]['extents_ranges'] = [[attrs['objects'][block_string + str(j+1)]['extents'][0], attrs['objects'][block_string + str(j+1)]['extents'][0]], [attrs['objects'][block_string + str(j+1)]['extents'][1], attrs['objects'][block_string + str(j+1)]['extents'][1]], [attrs['objects'][block_string + str(j+1)]['extents'][2], attrs['objects'][block_string + str(j+1)]['extents'][2]]]
                self.all_gt_extents_range_list.append(attrs['objects'][block_string + str(j+1)]['extents_ranges'])
                self.all_gt_extents_list.append(attrs['objects'][block_string + str(j+1)]['extents'])
            else:
                self.all_gt_extents_list.append([attrs['objects'][block_string + str(j+1)]['x_extent'], attrs['objects'][block_string + str(j+1)]['y_extent'], attrs['objects'][block_string + str(j+1)]['z_extent']])


        for i in range(data['point_cloud_1'].shape[0]):
            point_string = 'point_cloud_'
            block_string = 'block_'

            

            if True:
                if 'contact' in data:  ## don't encode the big object in terms of the contact data
                    contact_array = np.zeros((total_objects, total_objects))

                    #print('contact length', len(data['contact']))
                    time_step = self.sample_time_step[i]
                    #print('contact [0] length', len(data['contact'][time_step]))
                    for contact_i in range(len(data['contact'][time_step])):
                        if self.bookshelf_env_shift > 0:
                            # print('env shift!')
                            if data['contact'][time_step][contact_i]['body0'] > 0 and data['contact'][time_step][contact_i]['body0'] < total_objects + 1 and data['contact'][time_step][contact_i]['body1'] > 0 and data['contact'][time_step][contact_i]['body1'] < total_objects + 1:
                                contact_array[data['contact'][time_step][contact_i]['body0'] - 1, data['contact'][time_step][contact_i]['body1'] - 1] = 1
                                contact_array[data['contact'][time_step][contact_i]['body1'] - 1, data['contact'][time_step][contact_i]['body0'] - 1] = 1
                        else:
                            if data['contact'][time_step][contact_i]['body0'] > -1 and data['contact'][time_step][contact_i]['body0'] < total_objects and data['contact'][time_step][contact_i]['body1'] > -1 and data['contact'][time_step][contact_i]['body1'] < total_objects:
                                contact_array[data['contact'][time_step][contact_i]['body0'], data['contact'][time_step][contact_i]['body1']] = 1
                                contact_array[data['contact'][time_step][contact_i]['body1'], data['contact'][time_step][contact_i]['body0']] = 1


                    #print(contact_array)
                else:
                    contact_array = - np.ones((total_objects, total_objects))
                for j in range(total_objects):

                    each_obj = j
                    current_block = "block_" + str(each_obj + 1)
                    initial_bounding_box = []
                    TF_matrix = []
                    for inner_i in range(2):
                        for inner_j in range(2):
                            for inner_k in range(2):
                                if True:
                                    step = 0
                                    if 'extents' in attrs['objects'][current_block]: #self.pushing and not self.pick_place: self.pushing:
                                        initial_bounding_box.append(math_util.pose_to_homogeneous(np.array([data['objects'][current_block]['position'][step][0] + ((inner_i*2) - 1)*attrs['objects'][current_block]['extents'][0]/2, 
                                        data['objects'][current_block]['position'][step][1] + ((inner_j*2) - 1)*attrs['objects'][current_block]['extents'][1]/2, 
                                        data['objects'][current_block]['position'][step][2] + ((inner_k*2) - 1)*attrs['objects'][current_block]['extents'][2]/2]), np.array([0,0,0,1])))
                                        TF_matrix.append(np.matmul(math_util.homogeneous_transpose(math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], np.array([0,0,0,1]))), initial_bounding_box[-1]))
                                    else:
                                        #self.all_gt_extents_list.append([attrs['objects'][block_string + str(j+1)]['x_extent'], attrs['objects'][block_string + str(j+1)]['y_extent'], attrs['objects'][block_string + str(j+1)]['z_extent']])
                                        
                                        initial_bounding_box.append(math_util.pose_to_homogeneous(np.array([data['objects'][current_block]['position'][step][0] + ((inner_i*2) - 1)*attrs['objects'][current_block]['x_extent']/2, 
                                        data['objects'][current_block]['position'][step][1] + ((inner_j*2) - 1)*attrs['objects'][current_block]['y_extent']/2, 
                                        data['objects'][current_block]['position'][step][2] + ((inner_k*2) - 1)*attrs['objects'][current_block]['z_extent']/2]), np.array([0,0,0,1])))
                                        TF_matrix.append(np.matmul(math_util.homogeneous_transpose(math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], np.array([0,0,0,1]))), initial_bounding_box[-1]))
                                    
                    
                    initial_bounding_box = np.array(initial_bounding_box)
                    #print(initial_bounding_box.shape)

                    rotated_bounding_box = np.zeros((initial_bounding_box.shape[0], initial_bounding_box.shape[1], initial_bounding_box.shape[2]))
                    TF_rotated_bounding_matrix = []
                    
                    for inner_i in range(initial_bounding_box.shape[0]):
                        rotated_bounding_box[inner_i, :, :] = math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], data['objects'][current_block]['orientation'][0])@TF_matrix[inner_i]
                        TF_rotated_bounding_matrix.append(np.matmul(math_util.homogeneous_transpose(math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], data['objects'][current_block]['orientation'][0])), rotated_bounding_box[inner_i, :, :]))
                        #print(math_util.homogeneous_to_position(rotated_bounding_box[inner_i, :, :]))

                    
                    #print('final bounding box')
                    self.all_rotated_bounding_box[i].append(np.array(TF_rotated_bounding_matrix))

                    final_bounding_box = np.zeros((initial_bounding_box.shape[0], initial_bounding_box.shape[1], initial_bounding_box.shape[2]))
                    final_array = np.zeros((initial_bounding_box.shape[0], 3))
                    if True:
                        for inner_i in range(rotated_bounding_box.shape[0]): 
                            final_bounding_box[inner_i,:, :] = math_util.pose_to_homogeneous(data['objects'][current_block]['position'][self.sample_time_step[i]], data['objects'][current_block]['orientation'][self.sample_time_step[i]])@TF_rotated_bounding_matrix[inner_i]
                            final_array[inner_i, :] = math_util.homogeneous_to_position(final_bounding_box[inner_i, :, :])
                    
                        
                    

                    self.all_bounding_box[i].append(final_array)
                    
                    max_current_pose = np.max(final_array, axis = 0)[:3]
                    min_current_pose = np.min(final_array, axis = 0)[:3]

                    max_min_extents = max_current_pose - min_current_pose
                    axis_final_array = [] #np.zeros((final_array.shape[0], final_array.shape[1]))

                    for max_min_i in range(2):
                        for max_min_j in range(2):
                            for max_min_z in range(2):
                                axis_final_array.append([min_current_pose[0] + max_min_i * max_min_extents[0], min_current_pose[1] + max_min_j * max_min_extents[1], min_current_pose[2] + max_min_z * max_min_extents[2]])
                    axis_final_array = np.array(axis_final_array)

                    self.all_axis_bounding_box[i].append(axis_final_array)
                    
                    
                    
                    self.all_gt_max_pose_list[i].append(max_current_pose)
                    self.all_gt_min_pose_list[i].append(min_current_pose)


                    
                    if self.env_first_step and attrs['objects']['block_'+str(j+1)]['fix_base_link']:
                        
                        self.all_point_cloud[i].append(data[point_string + str(j+1) + 'sampling'][0][:data_size, :])
                    else:
                        self.all_point_cloud[i].append(data[point_string + str(j+1) + 'sampling'][i][:data_size, :])

                    self.all_pos_list[i].append(self.get_point_cloud_center(data[point_string + str(j+1) + 'sampling'][i])) ## consider the case with memory
                    

                    if True:
                        if True:
                            identity_max_objects = 10
                            rgb_identity_max = 10
                            env_identity_max = 2
                            current_identity_list = []
                            rgb_identity_list = []
                            # env_identity_list = []
                            if each_obj >= 3:
                                current_identity_list = [1,0,0,0,0,0,0,0,0,0]
                            else:
                                for object_identity_id in range(identity_max_objects):
                                    if object_identity_id == each_obj:
                                        current_identity_list.append(1)
                                    else:
                                        current_identity_list.append(0)
                                
                            
                            for rgb_identity_id in range(rgb_identity_max):
                                if rgb_identity_id == each_obj:
                                    rgb_identity_list.append(1)
                                else:
                                    rgb_identity_list.append(0)

                            if self.one_bit_env:
                                if self.seperate_env_id:
                                    if self.select_obj_num_range[each_obj] < (max_objects - self.max_env_num):
                                        env_identity_list = [0]
                                    else:
                                        env_identity_list = [1]
                                else:
                                    if attrs['objects']['block_'+str(each_obj+1)]['fix_base_link']:
                                        env_identity_list = [0]
                                    else:
                                        env_identity_list = [1]
                            else:
                                if self.seperate_env_id:
                                    if self.select_obj_num_range[each_obj] < (max_objects - self.max_env_num):
                                        env_identity_list = [0,1]
                                    else:
                                        env_identity_list = [1,0]
                                else:
                                    if attrs['objects']['block_'+str(each_obj+1)]['fix_base_link']:
                                        env_identity_list = [0,1]
                                    else:
                                        env_identity_list = [1,0]
                            
                            # print(env_identity_list)
                            self.all_rgb_identity_list[i].append(rgb_identity_list)
                            self.all_gt_identity_list[i].append(current_identity_list)
                            self.all_gt_env_identity_list[i].append(env_identity_list)
                            self.all_gt_pose_list[i].append(data['objects'][block_string + str(j+1)]['position'][self.sample_time_step[i]])
                            self.all_gt_orientation_list[i].append(data['objects'][block_string + str(j+1)]['orientation'][self.sample_time_step[i]])
                        

                if self.get_hidden_label:
                    self.all_hidden_label_list.append(data['hidden_label'][i])
                
                for obj_pair in self.obj_pair_list:
                    (anchor_idx, other_idx) = obj_pair
                    if True:
                        self.all_relation_list[i].append(self.get_ground_truth_tf_relations_general_chris_1(contact_array, anchor_idx, other_idx ,self.all_axis_bounding_box[i][anchor_idx], self.all_gt_pose_list[i][anchor_idx], self.all_gt_max_pose_list[i][anchor_idx], self.all_gt_min_pose_list[i][anchor_idx], self.all_gt_extents_list[anchor_idx], self.all_axis_bounding_box[i][other_idx], self.all_gt_pose_list[i][other_idx], self.all_gt_max_pose_list[i][other_idx],self.all_gt_min_pose_list[i][other_idx][:], self.all_gt_extents_list[other_idx])[:])
                        

            
            

        self.random_push = False
        if push and not self.pick_place:
            self.random_push = True
        
        
        
        self.action_1 = []

        
        
        if self.push_3_steps:
            if self.stack_pickplace:
                self.all_action_list = []
                for each_action_step in range(data['point_cloud_1'].shape[0] - 1): ## should be 4 - 1
                    if each_action_step == data['point_cloud_1'].shape[0] - 2:
                        self.all_action_list.append([])
                        this_pickplace_string = 'pick_place_'
                        move_obj = -1
                        
                        for i in range(total_objects):
                            if str(i+1) in attrs['behavior_params'][this_pickplace_string + str(each_action_step + 1)]['target_object']:
                                move_obj = i

                        for i in range(self.max_objects):
                            self.all_action_list[each_action_step].append(0)
                        if self.train:
                            self.all_action_list[each_action_step][select_obj_num_range[move_obj]] = 1
                        else:
                            self.all_action_list[each_action_step][move_obj] = 1

                        for i in range(3):
                            # yixuan note: use + 1 for moving block 1 and block 2, use + 2 for moving block 2 and block 3
                            self.all_action_list[each_action_step].append(attrs['behavior_params'][this_pickplace_string + str(each_action_step + 1)]['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params'][this_pickplace_string + str(each_action_step + 1)]['behaviors']['pick']['init_object_pose'][i])
                            # self.all_action_list[each_action_step].append(attrs['behavior_params']['stack_objects']['behaviors'][this_pickplace_string + str(each_action_step + 1)]['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['stack_objects']['behaviors'][this_pickplace_string + str(each_action_step + 1)]['behaviors']['pick']['init_object_pose'][i])
                        if self.use_discrete_z:
                            self.discrete_z_buffer = 0.05
                            if self.all_action_list[each_action_step][-1] > 0 and (data['objects']['block_' + str(move_obj + 1)]['position'][-1][2] - self.discrete_z_buffer > data['objects']['block_' + str(move_obj + 1)]['position'][0][2]):
                                self.all_action_list[each_action_step][-1] = 1
                            elif self.all_action_list[each_action_step][-1] < 0 and (data['objects']['block_' + str(move_obj + 1)]['position'][-1][2] + self.discrete_z_buffer < data['objects']['block_' + str(move_obj + 1)]['position'][0][2]):
                                self.all_action_list[each_action_step][-1] = -1
                            else:
                                self.all_action_list[each_action_step][-1] = 0
                        elif not self.consider_z_offset:
                            self.all_action_list[each_action_step][-1] = 0 
                        if self.use_shared_latent_embedding or self.use_seperate_latent_embedding:
                            self.all_action_list[each_action_step].insert(0, 0)
                    else:
                        self.all_action_list.append([])
                        this_pickplace_string = 'pick_place_block_'
                        move_obj = each_action_step + 1
                        for i in range(self.max_objects):
                            self.all_action_list[each_action_step].append(0)
                        if self.train:
                            self.all_action_list[each_action_step][select_obj_num_range[move_obj]] = 1
                        else:
                            self.all_action_list[each_action_step][move_obj] = 1

                        for i in range(3):
                            self.all_action_list[each_action_step].append(attrs['behavior_params']['stack_objects']['behaviors'][this_pickplace_string + str(each_action_step + 2)]['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['stack_objects']['behaviors'][this_pickplace_string + str(each_action_step + 2)]['behaviors']['pick']['init_object_pose'][i])
                        if self.use_discrete_z:
                            self.discrete_z_buffer = 0.05
                            if self.all_action_list[each_action_step][-1] > 0 and (data['objects']['block_' + str(move_obj + 1)]['position'][-1][2] - self.discrete_z_buffer > data['objects']['block_' + str(move_obj + 1)]['position'][0][2]):
                                self.all_action_list[each_action_step][-1] = 1
                            elif self.all_action_list[each_action_step][-1] < 0 and (data['objects']['block_' + str(move_obj + 1)]['position'][-1][2] + self.discrete_z_buffer < data['objects']['block_' + str(move_obj + 1)]['position'][0][2]):
                                self.all_action_list[each_action_step][-1] = -1
                            else:
                                self.all_action_list[each_action_step][-1] = 0
                        elif not self.consider_z_offset:
                            self.all_action_list[each_action_step][-1] = 0 
                        if self.use_shared_latent_embedding or self.use_seperate_latent_embedding:
                            self.all_action_list[each_action_step].insert(0, 0)
            
                print('action', self.all_action_list)
            elif self.double_push:
                self.all_action_list = []
                for each_action_step in range(data['point_cloud_1'].shape[0] - 1): ## total three steps for push_3_steps args
                    self.all_action_list.append([])
                    this_push_string = 'push_step_'
                    move_obj = -1
                    
                    for i in range(total_objects):
                        if str(i+1) in attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['target_object']:
                            move_obj = i
                    print('move obj', move_obj)
                    for i in range(self.max_objects):
                        self.all_action_list[each_action_step].append(0)
                    if self.train:
                        self.all_action_list[each_action_step][select_obj_num_range[move_obj]] = 1
                    else:
                        self.all_action_list[each_action_step][move_obj] = 1
                    
                    if not self.updated_behavior_params:
                        for i in range(3):
                            self.all_action_list[each_action_step].append(attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['behaviors']['push']['target_pose'][i] - attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['behaviors']['approach']['target_pose'][i])
                    else:
                        for i in range(3):
                            self.all_action_list[each_action_step].append(attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['target_object_pose'][i] - attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['init_object_pose'][i])
                    # print(attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['push_height_offset'])
                    self.all_action_list[each_action_step][-1] = 0 
                    if self.use_shared_latent_embedding or self.use_seperate_latent_embedding:
                        if attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['push_height_offset'] > 0.08:
                            self.all_action_list[each_action_step].insert(0, 2)  
                        else:
                            self.all_action_list[each_action_step].insert(0, 1)
                
            
            elif self.single_push:
                self.all_action_list = []
                for each_action_step in range(data['point_cloud_1'].shape[0] - 1):
                    # print(attrs['behavior_params'])
                    self.all_action_list.append([])
                    move_obj = -1
                    
                    for i in range(total_objects):
                        if str(i+1) in attrs['behavior_params']['']['target_object']:
                            move_obj = i
                    for i in range(self.max_objects):
                        self.all_action_list[each_action_step].append(0)
                    if self.train:
                        self.all_action_list[each_action_step][select_obj_num_range[move_obj]] = 1
                    else:
                        self.all_action_list[each_action_step][move_obj] = 1
                    
                    # for i in range(3):
                    #     self.all_action_list[each_action_step].append(attrs['behavior_params']['pick_place_1']['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['pick_place_1']['behaviors']['pick']['init_object_pose'][i])
                    
                    if not self.updated_behavior_params:
                        for i in range(3):
                            self.all_action_list[each_action_step].append(attrs['behavior_params']['']['behaviors']['push']['target_pose'][i] - attrs['behavior_params']['']['behaviors']['approach']['target_pose'][i])
                    else:
                        for i in range(3):
                            self.all_action_list[each_action_step].append(attrs['behavior_params']['']['target_object_pose'][i] - attrs['behavior_params']['']['init_object_pose'][i])
                    
                    self.all_action_list[each_action_step][-1] = 0 
                    
                    if self.use_shared_latent_embedding or self.use_seperate_latent_embedding:
                        self.all_action_list[each_action_step].insert(0, 1)  
                
            elif self.single_test:
                self.all_action_list = []
                for each_action_step in range(data['point_cloud_1'].shape[0] - 1):
                    self.all_action_list.append([])
                    move_obj = -1
                    
                    for i in range(total_objects):
                        if str(i+1) in attrs['behavior_params']['pick_place_1']['target_object']:
                            move_obj = i
                    for i in range(self.max_objects):
                        self.all_action_list[each_action_step].append(0)
                    if self.train:
                        # print(select_obj_num_range)
                        self.all_action_list[each_action_step][select_obj_num_range[move_obj]] = 1
                    else:
                        self.all_action_list[each_action_step][move_obj] = 1
                    for i in range(3):
                        self.all_action_list[each_action_step].append(attrs['behavior_params']['pick_place_1']['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['pick_place_1']['behaviors']['pick']['init_object_pose'][i])
                    if self.use_discrete_z:
                        self.discrete_z_buffer = 0.05
                        if self.all_action_list[each_action_step][-1] > 0 and (data['objects']['block_' + str(move_obj + 1)]['position'][-1][2] - self.discrete_z_buffer > data['objects']['block_' + str(move_obj + 1)]['position'][0][2]):
                            self.all_action_list[each_action_step][-1] = 1
                        elif self.all_action_list[each_action_step][-1] < 0 and (data['objects']['block_' + str(move_obj + 1)]['position'][-1][2] + self.discrete_z_buffer < data['objects']['block_' + str(move_obj + 1)]['position'][0][2]):
                            self.all_action_list[each_action_step][-1] = -1
                        else:
                            self.all_action_list[each_action_step][-1] = 0
                    elif not self.consider_z_offset:
                        self.all_action_list[each_action_step][-1] = 0 
                    if self.use_shared_latent_embedding or self.use_seperate_latent_embedding:
                        self.all_action_list[each_action_step].insert(0, 0)  
                
            elif self.push_pickplace:
                self.all_action_list = []
                for each_action_step in range(data['point_cloud_1'].shape[0] - 1): ## total three steps for push_3_steps args
                    if each_action_step == data['point_cloud_1'].shape[0] - 1 - 1:
                        self.all_action_list.append([])
                        move_obj = -1
                        
                        for i in range(total_objects):
                            if str(i+1) in attrs['behavior_params']['pick_place_2']['target_object']:
                                move_obj = i
                        for i in range(self.max_objects):
                            self.all_action_list[each_action_step].append(0)
                        if self.train:
                            self.all_action_list[each_action_step][select_obj_num_range[move_obj]] = 1
                        else:
                            self.all_action_list[each_action_step][move_obj] = 1
                        for i in range(3):
                            self.all_action_list[each_action_step].append(attrs['behavior_params']['pick_place_2']['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['pick_place_2']['behaviors']['pick']['init_object_pose'][i])
                        if self.use_discrete_z:
                            self.discrete_z_buffer = 0.05
                            if self.all_action_list[each_action_step][-1] > 0 and (data['objects']['block_' + str(move_obj + 1)]['position'][-1][2] - self.discrete_z_buffer > data['objects']['block_' + str(move_obj + 1)]['position'][0][2]):
                                self.all_action_list[each_action_step][-1] = 1
                            elif self.all_action_list[each_action_step][-1] < 0 and (data['objects']['block_' + str(move_obj + 1)]['position'][-1][2] + self.discrete_z_buffer < data['objects']['block_' + str(move_obj + 1)]['position'][0][2]):
                                self.all_action_list[each_action_step][-1] = -1
                            else:
                                self.all_action_list[each_action_step][-1] = 0
                        elif not self.consider_z_offset:
                            self.all_action_list[each_action_step][-1] = 0 
                        if self.use_shared_latent_embedding or self.use_seperate_latent_embedding:
                            self.all_action_list[each_action_step].insert(0, 0)  

                    else:
                        self.all_action_list.append([])
                        this_push_string = 'push_step_'
                        move_obj = -1
                        
                        for i in range(total_objects):
                            if str(i+1) in attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['target_object']:
                                move_obj = i
                        print('move obj', move_obj)
                        for i in range(self.max_objects):
                            self.all_action_list[each_action_step].append(0)
                        if self.train:
                            self.all_action_list[each_action_step][select_obj_num_range[move_obj]] = 1
                        else:
                            self.all_action_list[each_action_step][move_obj] = 1
                        
                        if not self.updated_behavior_params:
                            for i in range(3):
                                self.all_action_list[each_action_step].append(attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['behaviors']['push']['target_pose'][i] - attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['behaviors']['approach']['target_pose'][i])
                        else:
                            for i in range(3):
                                self.all_action_list[each_action_step].append(attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['target_object_pose'][i] - attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['init_object_pose'][i])
                        self.all_action_list[each_action_step][-1] = 0 
                        
                        if self.use_shared_latent_embedding or self.use_seperate_latent_embedding:
                            self.all_action_list[each_action_step].insert(0, 1)
            elif self.stack_push:
                self.all_action_list = []
                for each_action_step in range(data['point_cloud_1'].shape[0] - 1): ## should be 4 - 1
                    if each_action_step == data['point_cloud_1'].shape[0] - 2:
                        self.all_action_list.append([])
                        this_push_string = 'push_step_'
                        move_obj = -1

                        for i in range(total_objects):
                            if str(i+1) in attrs['behavior_params'][this_push_string + str(1)]['target_object']:
                                move_obj = i
                        print('move obj', move_obj)
                        for i in range(self.max_objects):
                            self.all_action_list[each_action_step].append(0)
                        if self.train:
                            self.all_action_list[each_action_step][select_obj_num_range[move_obj]] = 1
                        else:
                            self.all_action_list[each_action_step][move_obj] = 1
                        
                        if not self.updated_behavior_params:
                            for i in range(3):
                                self.all_action_list[each_action_step].append(attrs['behavior_params'][this_push_string + str(1)]['behaviors']['push']['target_pose'][i] - attrs['behavior_params'][this_push_string + str(1)]['behaviors']['approach']['target_pose'][i])
                        else:
                            for i in range(3):
                                self.all_action_list[each_action_step].append(attrs['behavior_params'][this_push_string + str(1)]['target_object_pose'][i] - attrs['behavior_params'][this_push_string + str(1)]['init_object_pose'][i])
                        self.all_action_list[each_action_step][-1] = 0 
                        if self.use_shared_latent_embedding or self.use_seperate_latent_embedding:
                            self.all_action_list[each_action_step].insert(0, 1)
                    else:
                        self.all_action_list.append([])
                        this_pickplace_string = 'pick_place_block_'
                        move_obj = each_action_step + 1
                        for i in range(self.max_objects):
                            self.all_action_list[each_action_step].append(0)
                        if self.train:
                            self.all_action_list[each_action_step][select_obj_num_range[move_obj]] = 1
                        else:
                            self.all_action_list[each_action_step][move_obj] = 1

                        for i in range(3):
                            # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
                            # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                            self.all_action_list[each_action_step].append(attrs['behavior_params']['stack_objects']['behaviors'][this_pickplace_string + str(each_action_step + 2)]['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['stack_objects']['behaviors'][this_pickplace_string + str(each_action_step + 2)]['behaviors']['pick']['init_object_pose'][i])
                        if self.use_discrete_z:
                            self.discrete_z_buffer = 0.05
                            if self.all_action_list[each_action_step][-1] > 0 and (data['objects']['block_' + str(move_obj + 1)]['position'][-1][2] - self.discrete_z_buffer > data['objects']['block_' + str(move_obj + 1)]['position'][0][2]):
                                self.all_action_list[each_action_step][-1] = 1
                            elif self.all_action_list[each_action_step][-1] < 0 and (data['objects']['block_' + str(move_obj + 1)]['position'][-1][2] + self.discrete_z_buffer < data['objects']['block_' + str(move_obj + 1)]['position'][0][2]):
                                self.all_action_list[each_action_step][-1] = -1
                            else:
                                self.all_action_list[each_action_step][-1] = 0
                        elif not self.consider_z_offset:
                            self.all_action_list[each_action_step][-1] = 0 
                        if self.use_shared_latent_embedding or self.use_seperate_latent_embedding:
                            self.all_action_list[each_action_step].insert(0, 0)
            elif self.double_pickplace:
                # print(attrs['behavior_params'])
                self.all_action_list = []
                for each_action_step in range(data['point_cloud_1'].shape[0] - 1): ## should be 3 - 1
                    self.all_action_list.append([])
                    this_pickplace_string = 'pick_place_'
                    move_obj = -1
                    
                    for i in range(total_objects):
                        if str(i+1) in attrs['behavior_params'][this_pickplace_string + str(each_action_step + 1)]['target_object']:
                            move_obj = i

                    for i in range(self.max_objects):
                        self.all_action_list[each_action_step].append(0)
                    if self.train:
                        self.all_action_list[each_action_step][select_obj_num_range[move_obj]] = 1
                    else:
                        self.all_action_list[each_action_step][move_obj] = 1

                    
                    for i in range(3):
                        # yixuan note: use + 1 for moving block 1 and block 2, use + 2 for moving block 2 and block 3
                        self.all_action_list[each_action_step].append(attrs['behavior_params'][this_pickplace_string + str(each_action_step + 1)]['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params'][this_pickplace_string + str(each_action_step + 1)]['behaviors']['pick']['init_object_pose'][i])
                        # self.all_action_list[each_action_step].append(attrs['behavior_params']['stack_objects']['behaviors'][this_pickplace_string + str(each_action_step + 1)]['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['stack_objects']['behaviors'][this_pickplace_string + str(each_action_step + 1)]['behaviors']['pick']['init_object_pose'][i])
                    if self.use_discrete_z:
                        self.discrete_z_buffer = 0.05
                        if self.all_action_list[each_action_step][-1] > 0 and (data['objects']['block_' + str(move_obj + 1)]['position'][-1][2] - self.discrete_z_buffer > data['objects']['block_' + str(move_obj + 1)]['position'][0][2]):
                            self.all_action_list[each_action_step][-1] = 1
                        elif self.all_action_list[each_action_step][-1] < 0 and (data['objects']['block_' + str(move_obj + 1)]['position'][-1][2] + self.discrete_z_buffer < data['objects']['block_' + str(move_obj + 1)]['position'][0][2]):
                            self.all_action_list[each_action_step][-1] = -1
                        else:
                            self.all_action_list[each_action_step][-1] = 0
                    elif not self.consider_z_offset:
                        self.all_action_list[each_action_step][-1] = 0 
                    if self.use_shared_latent_embedding or self.use_seperate_latent_embedding:
                        self.all_action_list[each_action_step].insert(0, 0)
                # print(self.all_action_list)
            elif self.pushing:
                #print('enter random push')
                self.all_action_list = []
                for each_action_step in range(data['point_cloud_1'].shape[0] - 1): ## total three steps for push_3_steps args
                    self.all_action_list.append([])
                    this_push_string = 'push_step_'
                    move_obj = -1
                    
                    for i in range(total_objects):
                        if str(i+1) in attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['target_object']:
                            move_obj = i
                    print('move obj', move_obj)
                    for i in range(self.max_objects):
                        self.all_action_list[each_action_step].append(0)
                    if self.train:
                        self.all_action_list[each_action_step][select_obj_num_range[move_obj]] = 1
                    else:
                        self.all_action_list[each_action_step][move_obj] = 1
                    
                    if not self.updated_behavior_params:
                        for i in range(3):
                            self.all_action_list[each_action_step].append(attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['behaviors']['push']['target_pose'][i] - attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['behaviors']['approach']['target_pose'][i])
                    else:
                        for i in range(3):
                            self.all_action_list[each_action_step].append(attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['target_object_pose'][i] - attrs['behavior_params'][this_push_string + str(each_action_step + 1)]['init_object_pose'][i])
                    self.all_action_list[each_action_step][-1] = 0 
                    if self.use_shared_latent_embedding or self.use_seperate_latent_embedding:
                        self.all_action_list[each_action_step].insert(0, 1)
            
            else:
                print(attrs['behavior_params'])
                self.all_action_list = []
                for each_action_step in range(data['point_cloud_1'].shape[0] - 1): ## should be 3 - 1
                    self.all_action_list.append([])
                    this_pickplace_string = 'pick_place_block_'
                    move_obj = each_action_step # use + 0 for moving block 1 and block 2, use + 1 for moving block 2 and block 3
                    for i in range(self.max_objects):
                        self.all_action_list[each_action_step].append(0)
                    if self.train:
                        self.all_action_list[each_action_step][select_obj_num_range[move_obj]] = 1
                    else:
                        self.all_action_list[each_action_step][move_obj] = 1

                    
                    for i in range(3):
                        # yixuan note: use + 1 for moving block 1 and block 2, use + 2 for moving block 2 and block 3
                        self.all_action_list[each_action_step].append(attrs['behavior_params']['stack_objects']['behaviors'][this_pickplace_string + str(each_action_step + 1)]['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['stack_objects']['behaviors'][this_pickplace_string + str(each_action_step + 1)]['behaviors']['pick']['init_object_pose'][i])
                    if self.use_discrete_z:
                        self.discrete_z_buffer = 0.05
                        if self.all_action_list[each_action_step][-1] > 0 and (data['objects']['block_' + str(move_obj + 1)]['position'][-1][2] - self.discrete_z_buffer > data['objects']['block_' + str(move_obj + 1)]['position'][0][2]):
                            self.all_action_list[each_action_step][-1] = 1
                        elif self.all_action_list[each_action_step][-1] < 0 and (data['objects']['block_' + str(move_obj + 1)]['position'][-1][2] + self.discrete_z_buffer < data['objects']['block_' + str(move_obj + 1)]['position'][0][2]):
                            self.all_action_list[each_action_step][-1] = -1
                        else:
                            self.all_action_list[each_action_step][-1] = 0
                    elif not self.consider_z_offset:
                        self.all_action_list[each_action_step][-1] = 0 
                    if self.use_shared_latent_embedding or self.use_seperate_latent_embedding:
                        self.all_action_list[each_action_step].insert(0, 0)
                         
                
        
        self.obj_voxels_single = []
        self.obj_voxels_by_obj_pair_dict = []
        self.obj_voxels_by_obj_pair_dict_anchor = []
        self.obj_voxels_by_obj_pair_dict_other = []
        for i in range(data['point_cloud_1'].shape[0]):
            if self.fast_training:
                self.obj_voxels_single.append([])
            else:
                self.obj_voxels_single.append(dict())
            self.obj_voxels_by_obj_pair_dict.append(dict())
            self.obj_voxels_by_obj_pair_dict_anchor.append(dict())
            self.obj_voxels_by_obj_pair_dict_other.append(dict())


        self.obj_voxels_by_obj_pair_dict_last = dict()
        self.obj_voxels_by_obj_pair_dict_anchor_last = dict()
        self.obj_voxels_by_obj_pair_dict_other_last = dict()

        self.obj_voxels_status_by_obj_pair_dict = dict()
        self.obj_pcd_path_by_obj_pair_dict = dict()

        

        for obj_id in range(self.total_objects):
            for i in range(data['point_cloud_1'].shape[0]):
                total_point_cloud = self.all_point_cloud[i][obj_id]
                #print(total_point_cloud.shape)
                if self.fast_training:
                    self.obj_voxels_single[i].append(total_point_cloud.T)
                else:
                    self.obj_voxels_single[i][obj_id] = total_point_cloud.T
                

            
        
        for obj_pair in self.obj_pair_list:
            (anchor_idx, other_idx) = obj_pair

            for i in range(data['point_cloud_1'].shape[0]):
                total_point_cloud = np.concatenate((self.all_point_cloud[i][anchor_idx], self.all_point_cloud[i][other_idx]), axis = 0)
                #print(total_point_cloud.shape)
                self.obj_voxels_by_obj_pair_dict[i][obj_pair] = total_point_cloud.T
                self.obj_voxels_by_obj_pair_dict_anchor[i][obj_pair] = self.all_point_cloud[i][anchor_idx].T
                self.obj_voxels_by_obj_pair_dict_other[i][obj_pair] = self.all_point_cloud[i][other_idx].T

            
            self.obj_pcd_path_by_obj_pair_dict[obj_pair] = (
                scene_path, 
                scene_path # yixuan test
            )
        
        if self.fast_training:
            self.all_relation_fast = torch.Tensor(self.all_relation_list)
            self.obj_voxels_single_fast = torch.Tensor(self.obj_voxels_single)
            self.env_identity_list_fast = torch.Tensor(self.all_gt_env_identity_list)
            nodes = list(range(self.total_objects_fast))
            # Create a completely connected graph
            edges = list(permutations(nodes, 2))
            edge_index = torch.LongTensor(np.array(edges).T)
            self.edge_attr_fast = edge_index
            
            self.all_action_label_fast = []
            for i in range(len(self.all_action_list)):
                self.all_action_label_fast.append((int)(self.all_action_list[i][0]))
            if self.add_memory:
                self.all_action_label_fast = torch.Tensor(self.all_action_label_fast).int()

            self.all_action_fast_temp = torch.Tensor(self.all_action_list)
            self.all_action_fast = torch.zeros((self.all_action_fast_temp.shape[0], self.total_objects_fast, self.all_action_fast_temp.shape[1]))
            for i in range(self.total_objects_fast):
                self.all_action_fast[:, i, :] = self.all_action_fast_temp
            

            
            self.current_pose_fast = torch.zeros((2,6,9)) ## placeholder
            
            self.debug = False
            if self.debug:
                for debug_i in range(self.obj_voxels_single_fast.shape[0]):
                    for debug_j in range(self.obj_voxels_single_fast.shape[1]):
                        print(self.obj_voxels_single_fast[debug_i][debug_j].T.shape)
                        print(self.get_point_cloud_center(self.obj_voxels_single_fast[debug_i][debug_j].numpy().T))
        
        
    def get_ground_truth_tf_relations_general_chris_1(self, contact_arr, anchor_id, other_id, anchor_bounding_box ,anchor_pose, anchor_pose_max, anchor_pose_min, anchor_extents, other_bounding_box, other_pose, other_pose_max, other_pose_min, other_extents): # to start, assume no orientation
        action = []

        # print(anchor_bounding_box.shape)
        # print(anchor_pose.shape)
        cf_o1_bbox_corners = np.concatenate([anchor_bounding_box, np.expand_dims(anchor_pose, axis=0)], axis=0).T

        cf_o2_bbox_corners = np.concatenate([other_bounding_box, np.expand_dims(other_pose, axis=0)], axis=0).T

        # print(cf_o1_bbox_corners.shape)
        
        o1_predicate_args = {
            'obj_id' : anchor_id,
            'cf_bbox_corners' : cf_o1_bbox_corners, 
        }
        o2_predicate_args = {
            'obj_id' : other_id,
            'cf_bbox_corners' : cf_o2_bbox_corners,
        }

        self.compute_left_right_predicates(o1_predicate_args, o2_predicate_args, action)

        
        self.compute_above_below_predicates(o1_predicate_args, o2_predicate_args, action)

        self.compute_front_behind_predicates(o1_predicate_args, o2_predicate_args, action)



        if((other_pose[2] - anchor_pose[2]) > 0):
            current_extents = np.array(anchor_pose_max) - np.array(anchor_pose_min)
        else:
            current_extents = np.array(other_pose_max) - np.array(other_pose_min)
        
        

        sudo_contact = 0
        if np.abs(other_pose[2] - anchor_pose[2]) > 0.04 and np.abs(other_pose[2] - anchor_pose[2]) < 0.12:
            if np.abs(other_pose[0] - anchor_pose[0]) < current_extents[0]/2 and np.abs(other_pose[1] - anchor_pose[1]) < current_extents[1]/2:
                sudo_contact = 1
        
        # print(contact_arr)        
        if contact_arr[0][0] == -1: # simple trick to deal with unsaved contact relations
            action.append(sudo_contact)
        else:
            action.append(contact_arr[anchor_id][other_id])

        # print(self.use_boundary_relations)
        
        if self.use_boundary_relations:
            # print('enter computation')
            # print(other_bounding_box)
            
            
            pair_corner = [[[other_bounding_box[0][0], other_bounding_box[0][1]], [other_bounding_box[3][0], other_bounding_box[3][1]]], 
            [[other_bounding_box[0][0], other_bounding_box[0][1]], [other_bounding_box[5][0], other_bounding_box[5][1]]], 
            [[other_bounding_box[7][0], other_bounding_box[7][1]], [other_bounding_box[3][0], other_bounding_box[3][1]]], 
            [[other_bounding_box[7][0], other_bounding_box[7][1]], [other_bounding_box[5][0], other_bounding_box[5][1]]]]
            # print(pair_corner)
            self.boundary_length = 0.10
            # print(action)
            all_dist_list = []
            if action[5] == 1 and current_extents[0] > 0.2 and current_extents[1] > 0.2:
                # print(anchor_pose)
                # print(other_pose)
                for each_pair_corner in pair_corner:
                    # print(each_pair_corner)
                    all_dist_list.append(self.get_distance_from_point_to_line(anchor_pose[:2], each_pair_corner[0], each_pair_corner[1]))
                # print(all_dist_list)
                if min(all_dist_list) < self.boundary_length:
                    action.append(1)
                else:
                    action.append(0)
            else:
                action.append(0)

            # if action[5] == 1:
            #     if np.abs(anchor_pose[0] - other_pose[0]) > (current_extents[0]/2 - self.boundary_length) or np.abs(anchor_pose[1] - other_pose[1]) > (current_extents[1]/2 - self.boundary_length):
            #         action.append(1)
            #     else:
            #         action.append(0)
            # else:
            #     action.append(0)

        return action

    

    def get_distance_from_point_to_line(self, point, line_point1, line_point2):
        #对于两点坐标为同一点时,返回点与点的距离
        if line_point1 == line_point2:
            point_array = np.array(point )
            point1_array = np.array(line_point1)
            return np.linalg.norm(point_array -point1_array )
        #计算直线的三个参数
        A = line_point2[1] - line_point1[1]
        B = line_point1[0] - line_point2[0]
        C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
            (line_point2[0] - line_point1[0]) * line_point1[1]
        #根据点到直线的距离公式计算距离
        distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
        return distance

    
    def compute_left_right_predicates(self, o1_predicate_args, o2_predicate_args, predicates):
        """ Compute left-right predicates.
            Use camera frame coordinates.
            Relation rules:
                1) o1 center MUST be in half-space defined by o2 UPPER corner and theta (xz plane)
                2) o1 center MUST be in half-space defined by o2 LOWER corner and theta (xz plane)
                3) do same as 1) for xy
                4) do same as 2) for xy
                5) o1 center MUST be to left of all o2 corners
                6) All o1 corners MUST be to the left of o2 center
        """

        def left_of(cf_o1_bbox_corners, cf_o2_bbox_corners):

            # Check xz
            o1_xz_center = cf_o1_bbox_corners[[0,2], 8] # [x,z]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-z plane, and two left-most corners. 
            o2_upper_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[2,:8].max()]) # [x,z]
            o2_lower_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[2,:8].min()]) # [x,z]

            # Upper half-space defined by p'n + d = 0
            upper_normal = rotate_2d(np.array([-1,0]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            upper_d = -1 * o2_upper_corner.dot(upper_normal)
            first_rule = o1_xz_center.dot(upper_normal) + upper_d >= 0

            # Lower half-space defined by p'n + d = 0
            lower_normal = rotate_2d(np.array([0,1]), self.params['theta_predicte_lr_fb_ab'])
            lower_d = -1 * o2_lower_corner.dot(lower_normal)
            second_rule = o1_xz_center.dot(lower_normal) + lower_d >= 0

            xz_works = first_rule and second_rule

            # Check xy
            o1_xy_center = cf_o1_bbox_corners[[0,1], 8] # [x,y]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-y plane, and two left-most corners. 
            o2_upper_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[1,:8].max()]) # [x,y]
            o2_lower_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[1,:8].min()]) # [x,y]

            # Upper half-space defined by p'n + d = 0
            upper_normal = rotate_2d(np.array([-1,0]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            upper_d = -1 * o2_upper_corner.dot(upper_normal)
            third_rule = o1_xy_center.dot(upper_normal) + upper_d >= 0

            # Lower half-space defined by p'n + d = 0
            lower_normal = rotate_2d(np.array([0,1]), self.params['theta_predicte_lr_fb_ab'])
            lower_d = -1 * o2_lower_corner.dot(lower_normal)
            fourth_rule = o1_xy_center.dot(lower_normal) + lower_d >= 0

            xy_works = third_rule and fourth_rule

            # All corners check
            fifth_rule = np.all(o1_xz_center[0] <= cf_o2_bbox_corners[0,:8].min())

            # o1 right corners check
            sixth_rule = np.all(cf_o1_bbox_corners[0, :8].max() <= cf_o2_bbox_corners[0,8])

            return xz_works and xy_works and fifth_rule and sixth_rule

        obj1_id = o1_predicate_args['obj_id']
        obj2_id = o2_predicate_args['obj_id']

        # For symmetry, check if o1 is left of o2, and if o2 is right of o1
        cf_o1_bbox_corners = o1_predicate_args['cf_bbox_corners'].copy()
        cf_o2_bbox_corners = o2_predicate_args['cf_bbox_corners'].copy()
        o1_left_of_o2 = left_of(cf_o1_bbox_corners, cf_o2_bbox_corners)

        o2_left_of_o1 = left_of(cf_o2_bbox_corners, cf_o1_bbox_corners)

        cf_o1_bbox_corners[0,:] = cf_o1_bbox_corners[0,:] * -1
        cf_o2_bbox_corners[0,:] = cf_o2_bbox_corners[0,:] * -1
        o2_right_of_o1 = left_of(cf_o2_bbox_corners, cf_o1_bbox_corners)

        o1_right_of_o2 = left_of(cf_o1_bbox_corners, cf_o2_bbox_corners)

        if o1_left_of_o2 or o2_right_of_o1:
            predicates.append(1)
            predicates.append(0)  
        elif o2_left_of_o1 or o1_right_of_o2:
            predicates.append(0)
            predicates.append(1)  
        else:
            predicates.append(0)
            predicates.append(0)                


    def compute_front_behind_predicates(self, o1_predicate_args, o2_predicate_args, predicates):
        """ Compute front-behind predicates.
            Use camera frame coordinates.
            Relation rules:
                1) o1 center MUST be in half-space defined by o2 LEFT corner and theta (xz plane)
                2) o1 center MUST be in half-space defined by o2 RIGHT corner and theta (xz plane)
                3) do same as 1) for yz
                4) do same as 2) for yz
                5) o1 center MUST be behind all o2 corners
                6) All o1 corners MUST be behind o2 center
        """

        def behind(cf_o1_bbox_corners, cf_o2_bbox_corners):

            # Check xz
            o1_xz_center = cf_o1_bbox_corners[[0,2], 8] # [x,z]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-z plane, and two left-most corners. 
            o2_left_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[2,:8].max()]) # [x,z]
            o2_right_corner = np.array([cf_o2_bbox_corners[0,:8].max(), cf_o2_bbox_corners[2,:8].max()]) # [x,z]

            # Left half-space defined by p'n + d = 0
            left_normal = rotate_2d(np.array([1,0]), self.params['theta_predicte_lr_fb_ab'])
            left_d = -1 * o2_left_corner.dot(left_normal)
            first_rule = o1_xz_center.dot(left_normal) + left_d >= 0

            # Right half-space defined by p'n + d = 0
            right_normal = rotate_2d(np.array([0,1]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            right_d = -1 * o2_right_corner.dot(right_normal)
            second_rule = o1_xz_center.dot(right_normal) + right_d >= 0

            xz_works = first_rule and second_rule

            # Check yz
            o1_yz_center = cf_o1_bbox_corners[[1,2], 8] # [y,z]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-z plane, and two left-most corners. 
            o2_left_corner = np.array([cf_o2_bbox_corners[1,:8].min(), cf_o2_bbox_corners[2,:8].max()]) # [y,z]
            o2_right_corner = np.array([cf_o2_bbox_corners[1,:8].max(), cf_o2_bbox_corners[2,:8].max()]) # [y,z]

            # Left half-space defined by p'n + d = 0
            left_normal = rotate_2d(np.array([1,0]), self.params['theta_predicte_lr_fb_ab'])
            left_d = -1 * o2_left_corner.dot(left_normal)
            third_rule = o1_yz_center.dot(left_normal) + left_d >= 0

            # Right half-space defined by p'n + d = 0
            right_normal = rotate_2d(np.array([0,1]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            right_d = -1 * o2_right_corner.dot(right_normal)
            fourth_rule = o1_yz_center.dot(right_normal) + right_d >= 0

            yz_works = third_rule and fourth_rule

            # All corners check
            fifth_rule = np.all(o1_xz_center[1] >= cf_o2_bbox_corners[2,:8].max())

            # o1 near corners check
            sixth_rule = np.all(cf_o1_bbox_corners[2, :8].min() >= cf_o2_bbox_corners[2,8])

            return xz_works and yz_works and fifth_rule and sixth_rule

        obj1_id = o1_predicate_args['obj_id']
        obj2_id = o2_predicate_args['obj_id']

        # For symmetry, check if o1 is behind of o2, and if o2 is in front of o1
        cf_o1_bbox_corners = o1_predicate_args['cf_bbox_corners'].copy()
        cf_o2_bbox_corners = o2_predicate_args['cf_bbox_corners'].copy()
        o1_behind_o2 = behind(cf_o1_bbox_corners, cf_o2_bbox_corners)

        o2_behind_o1 = behind(cf_o2_bbox_corners, cf_o1_bbox_corners)

        cf_o1_bbox_corners[2,:] = cf_o1_bbox_corners[2,:] * -1
        cf_o2_bbox_corners[2,:] = cf_o2_bbox_corners[2,:] * -1
        o2_in_front_of_o1 = behind(cf_o2_bbox_corners, cf_o1_bbox_corners) 

        o1_in_front_of_o2 = behind(cf_o1_bbox_corners, cf_o2_bbox_corners)               

        if o1_behind_o2 or o2_in_front_of_o1:
            predicates.append(0)
            predicates.append(1)
        elif o2_behind_o1 or o1_in_front_of_o2:
            predicates.append(1)
            predicates.append(0)
        else:
            predicates.append(0)
            predicates.append(0)

    def compute_above_below_predicates(self, o1_predicate_args, o2_predicate_args, predicates):
        """ Compute above-below predicates.
            Use camera frame coordinates.
            Relation rules:
                1) o1 center MUST be in half-space defined by o2 LEFT corner and theta (xy plane)
                2) o1 center MUST be in half-space defined by o2 RIGHT corner and theta (xy plane)
                3) do same as 1) for zy
                4) do same as 2) for zy
                6) o1 center MUST be above all o2 corners
                7) All o1 corners MUST be above o2 center
                rule = ((1 & 2 & 3 & 4)) & 6 & 7
        """

        def above(cf_o1_bbox_corners, cf_o2_bbox_corners):

            # Check xy
            o1_xy_center = cf_o1_bbox_corners[[0,1], 8] # [x,y]

            # Get camera-frame axis-aligned bbox corners for o2
            o2_left_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[1,:8].max()]) # [x,y]
            o2_right_corner = np.array([cf_o2_bbox_corners[0,:8].max(), cf_o2_bbox_corners[1,:8].max()]) # [x,y]

            # Left half-space defined by p'n + d = 0
            left_normal = rotate_2d(np.array([1,0]), self.params['theta_predicte_lr_fb_ab'])
            left_d = -1 * o2_left_corner.dot(left_normal)
            first_rule = o1_xy_center.dot(left_normal) + left_d >= 0

            # Right half-space defined by p'n + d = 0
            right_normal = rotate_2d(np.array([0,1]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            right_d = -1 * o2_right_corner.dot(right_normal)
            second_rule = o1_xy_center.dot(right_normal) + right_d >= 0

            xy_works = first_rule and second_rule

            # Check zy
            o1_zy_center = cf_o1_bbox_corners[[2,1], 8] # [z,y]

            # Get camera-frame axis-aligned bbox corners for o2 
            o2_left_corner = np.array([cf_o2_bbox_corners[2,:8].min(), cf_o2_bbox_corners[1,:8].max()]) # [z,y]
            o2_right_corner = np.array([cf_o2_bbox_corners[2,:8].max(), cf_o2_bbox_corners[1,:8].max()]) # [z,y]

            # Left half-space defined by p'n + d = 0
            left_normal = rotate_2d(np.array([1,0]), self.params['theta_predicte_lr_fb_ab'])
            left_d = -1 * o2_left_corner.dot(left_normal)
            third_rule = o1_zy_center.dot(left_normal) + left_d >= 0

            # Right half-space defined by p'n + d = 0
            right_normal = rotate_2d(np.array([0,1]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            right_d = -1 * o2_right_corner.dot(right_normal)
            fourth_rule = o1_zy_center.dot(right_normal) + right_d >= 0

            zy_works = third_rule and fourth_rule

            # All corners check
            fifth_rule = np.all(o1_xy_center[1] >= cf_o2_bbox_corners[1,:8].max())

            # o1 bottom corners check
            sixth_rule = np.all(cf_o1_bbox_corners[1, :8].min() >= cf_o2_bbox_corners[1,8])

            return xy_works and zy_works and fifth_rule and sixth_rule

        obj1_id = o1_predicate_args['obj_id']
        obj2_id = o2_predicate_args['obj_id']

        # For symmetry, check if o1 is above o2, and if o2 is below o1
        cf_o1_bbox_corners = o1_predicate_args['cf_bbox_corners'].copy()
        cf_o2_bbox_corners = o2_predicate_args['cf_bbox_corners'].copy()
        o1_above_o2 = above(cf_o1_bbox_corners, cf_o2_bbox_corners)
        o2_above_o1 = above(cf_o2_bbox_corners, cf_o1_bbox_corners)

        cf_o1_bbox_corners[1,:] = cf_o1_bbox_corners[1,:] * -1
        cf_o2_bbox_corners[1,:] = cf_o2_bbox_corners[1,:] * -1
        o2_below_o1 = above(cf_o2_bbox_corners, cf_o1_bbox_corners)    
        o1_below_o2 = above(cf_o1_bbox_corners, cf_o2_bbox_corners)        

        if o1_above_o2 or o2_below_o1:
            # predicates.append((obj1_id, obj2_id, 'above'))
            # predicates.append((obj2_id, obj1_id, 'below'))
            predicates.append(0)
            predicates.append(1)
        elif o2_above_o1 or o1_below_o2:
            predicates.append(1)
            predicates.append(0)
        else:
            predicates.append(0)
            predicates.append(0)

    
    def get_point_cloud_center(self, v):
        # print(v.shape)
        # print(np.min(v[:, :], axis = 0))
        # print(np.max(v[:, :], axis = 0))

        A = np.min(v[:, :], axis = 0) + (np.max(v[:, :], axis = 0) - np.min(v[:, :], axis = 0))/2
        A_1 = [A[1], A[2], A[0]]
        # print(A_1)
        # time.sleep(10)
        return np.array(A_1)
    
    def get_point_cloud_max(self, v):
        A = (np.max(v[:, :], axis = 0)) #np.min(v[:, :], axis = 0) + (np.max(v[:, :], axis = 0) - np.min(v[:, :], axis = 0))/2
        A_1 = [A[1], A[2], A[0]]
        return np.array(A_1)

    def get_point_cloud_min(self, v):
        A = (np.min(v[:, :], axis = 0)) #np.min(v[:, :], axis = 0) + (np.max(v[:, :], axis = 0) - np.min(v[:, :], axis = 0))/2
        A_1 = [A[1], A[2], A[0]]
        return np.array(A_1)
    
    
    def __len__(self):
        return len(self.obj_voxels_by_obj_pair_dict)
    
    
    
    def get_all_object_pair_voxels_fast_3steps(self):

        return self.all_relation_fast, self.obj_voxels_single_fast, self.one_hot_encoding_tensor_fast, self.total_objects_fast, self.edge_attr_fast, self.all_action_fast, self.all_action_label_fast, self.env_identity_list_fast, self.current_pose_fast, self.all_hidden_label_list

    def put_all_things_on_device(self, device):
        self.all_relation_fast = self.all_relation_fast.to(device)
        self.obj_voxels_single_fast = self.obj_voxels_single_fast.to(device)
        self.one_hot_encoding_tensor_fast = self.one_hot_encoding_tensor_fast.to(device)
        self.edge_attr_fast = self.edge_attr_fast.to(device)
        self.all_action_fast = self.all_action_fast.to(device)
        self.env_identity_list_fast = self.env_identity_list_fast.to(device)
        self.current_pose_fast = self.current_pose_fast.to(device)
        if self.add_memory:
            self.all_action_label_fast = self.all_action_label_fast.to(device)

    def get_obj_num(self):
        return self.total_objects
    

class AllPairVoxelDataloaderPointCloud3stack(object):
    def __init__(self, 
                 config,
                 relation_angle, 
                 max_objects = 5, 
                 use_multiple_train_dataset = False,
                 pick_place = False,
                 pushing = False,
                 stacking = False, 
                 set_max = False,
                 train_dir_list=None,
                 test_dir_list=None,
                 load_contact_data=False,
                 start_id = 0, 
                 max_size = 0,   # begin on max_size = 8000 for all disturbance data
                 start_test_id = 0, 
                 test_max_size = 2,
                 updated_behavior_params = False,
                 save_data_path = None, 
                 using_multi_step_statistics = False,
                 total_multi_steps = 0,
                 use_shared_latent_embedding = False,
                 use_seperate_latent_embedding = False,
                 push_3_steps = False,
                 stack_push = False,
                 push_pickplace = False,
                 single_test = False,
                 single_push = False, ## for single push in push_3_steps
                 double_pickplace = False, 
                 use_boundary_relations = False,
                 consider_z_offset = False,
                 seperate_env_id = False,
                 max_env_num = False,
                 env_first_step = False,
                 use_discrete_z = False,
                 fast_training = False,
                 evaluate_pickplace = False,
                 one_bit_env = False,
                 bookshelf_env_shift = 0,
                 double_push = False,
                 stack_pickplace = False,
                 enable_return = False,
                 test_data_loader = False,
                 add_memory = False,
                 get_hidden_label = False
                 ):
        #self.train = train
        
        self.test_data_loader = test_data_loader
        self.enable_return = enable_return
        self.double_push = double_push
        self.total_multi_steps = total_multi_steps
        self.using_multi_step_statistics = using_multi_step_statistics
        self.evaluate_pickplace = evaluate_pickplace
        self.updated_behavior_params = updated_behavior_params
        stacking = stacking
        self.set_max = set_max
        self._config = config
        self.transforms = None
        self.pos_grid = None
        self.pick_place = pick_place
        self.pushing = pushing
        self.stacking = stacking
        self.load_contact_data = load_contact_data
        self.use_shared_latent_embedding = use_shared_latent_embedding
        self.use_seperate_latent_embedding = use_seperate_latent_embedding
        self.push_3_steps = push_3_steps
        self.stack_push = stack_push
        self.push_pickplace = push_pickplace
        self.single_test = single_test
        self.single_push = single_push
        self.double_pickplace = double_pickplace
        self.use_boundary_relations = use_boundary_relations
        self.consider_z_offset = consider_z_offset
        self.seperate_env_id = seperate_env_id
        self.env_first_step = env_first_step
        self.max_env_num = max_env_num
        self.use_discrete_z = use_discrete_z
        self.fast_training = fast_training
        self.relation_angle = relation_angle
        self.add_memory = add_memory
        self.get_hidden_label = get_hidden_label
        self.stack_pickplace = stack_pickplace

        self.one_bit_env = one_bit_env
        self.bookshelf_env_shift = bookshelf_env_shift

        self.fail_reasoning_num = 0

        self.scene_type = "data_in_line"

        

        demo_idx = 0
        self.train_idx_to_data_dict = {}
        idx_to_data_dict = {}


        data_size = 128
        self.max_size = max_size
        self.test_max_size = test_max_size

        
        if self.push_3_steps:
            if self.single_test or self.single_push:
                total_steps = 2
            elif self.push_pickplace:
                total_steps = 3
            elif self.stack_push:
                total_steps = 4
            elif self.double_pickplace or self.double_push or self.stack_pickplace:
                total_steps = 3
            elif self.pushing:
                total_steps = 4
            else:
                total_steps = 3
        elif self.pick_place:
            total_steps = 2
        elif self.pushing:
            total_steps = 2
        elif self.stacking:
            total_steps = 3
        
        self.motion_planner_fail_num = 0
        self.train_id = 0
        self.test_id = 0    




        self.use_multiple_train_dataset = use_multiple_train_dataset
        if not self.use_multiple_train_dataset:
            self.train_dir_list = train_dir_list \
                if train_dir_list is not None else config.args.train_dir
        

        self.all_goal_relations = np.ones((50000,5,1))
        self.all_predicted_relations = np.ones((50000,5,1))
        self.all_index_i_list = np.ones((50000,5,1))
        self.all_index_j_list = np.ones((50000,5,1))
    
        files = sorted(os.listdir(self.train_dir_list[0]))       
        self.train_pcd_path = [
            os.path.join(self.train_dir_list[0], p) for p in files if 'demo' in p]
        for train_dir in self.train_pcd_path[start_id:start_id+max_size]:
            self.current_goal_relations = self.all_goal_relations[self.train_id] # simplify for without test end_relations
            self.current_predicted_relations = self.all_predicted_relations[self.train_id]
            self.current_index_i = self.all_index_i_list[self.train_id]  ## test_id?  I change it to train_id
            self.current_index_j = self.all_index_j_list[self.train_id]
            self.train_id += 1
            
            print('loaded the data at path: ', train_dir)            
            with open(train_dir, 'rb') as f:
                data, attrs = pickle.load(f)
            total_objects = 0
            for k, v in data.items():
                if 'point_cloud' in k and 'sampling' in k and 'last' not in k:
                    total_objects += 1
            this_one_hot_encoding = np.zeros((1, total_objects))
            
            leap = 1


            for k, v in data.items():
                if 'point_cloud' in k and 'last' not in k and 'leap' not in k:
                    # print(k)
                    # print(v.shape[0])
                    if(v.shape[0] == 0):
                        leap = 0
                        break
                    if(v.shape[0] != total_steps):
                        leap = 0
                        break
                    for i in range((v.shape[0])):
                        # print([k, v[i].shape])
                        if(v[i].shape[0] < data_size):
                            leap = 0
                            break
                
            
            eps = 1e-3
            
            if leap == 0:
                continue
            
            idx_to_data_dict[demo_idx] = {}
            
            idx_to_data_dict[demo_idx]['objects'] = data['objects']

            idx_to_data_dict[demo_idx]['att_objects'] = attrs['objects']

            idx_to_data_dict[demo_idx]['goal_relations'] = self.current_goal_relations
            idx_to_data_dict[demo_idx]['predicted_relations'] = self.current_predicted_relations
            idx_to_data_dict[demo_idx]['index_i'] = self.current_index_i
            idx_to_data_dict[demo_idx]['index_j'] = self.current_index_j
            idx_to_data_dict[demo_idx]['this_one_hot_encoding'] = this_one_hot_encoding

            total_point_cloud = np.concatenate((data['point_cloud_1'][0][:data_size][:], data['point_cloud_2'][0][:data_size][:]), axis = 0)
            idx_to_data_dict[demo_idx]['point_cloud'] = total_point_cloud.T
            idx_to_data_dict[demo_idx]['point_cloud_1'] = data['point_cloud_1'][0][:data_size][:].T
            idx_to_data_dict[demo_idx]['point_cloud_2'] = data['point_cloud_2'][0][:data_size][:].T
            idx_to_data_dict[demo_idx]['point_cloud_3'] = data['point_cloud_3'][0][:data_size][:].T
            self.train = True
            
            all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(train_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects ,train = self.train, updated_behavior_params = self.updated_behavior_params, use_shared_latent_embedding = self.use_shared_latent_embedding, push_3_steps = self.push_3_steps, use_seperate_latent_embedding = self.use_seperate_latent_embedding, stack_push = self.stack_push, push_pickplace = self.push_pickplace, single_test = self.single_test, single_push = self.single_push, use_boundary_relations = self.use_boundary_relations, consider_z_offset = self.consider_z_offset, seperate_env_id = self.seperate_env_id, max_env_num = self.max_env_num, env_first_step = self.env_first_step, use_discrete_z = self.use_discrete_z, double_pickplace = self.double_pickplace, fast_training = self.fast_training, one_bit_env = self.one_bit_env, relation_angle = self.relation_angle, bookshelf_env_shift = self.bookshelf_env_shift, double_push = self.double_push, enable_return = self.enable_return, add_memory = self.add_memory, stack_pickplace = self.stack_pickplace, get_hidden_label = self.get_hidden_label)
            
            
            idx_to_data_dict[demo_idx]['path'] = train_dir
            idx_to_data_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
            demo_idx += 1           
        self.train_idx_to_data_dict.update(idx_to_data_dict)
                


        self.train_scene_sample_order = {}
        self.test_scene_sample_order = {}

    def get_fail_reasoning_num(self):
        return self.fail_reasoning_num
    

    
    def get_demo_data_dict(self, train=True):
        data_dict = self.train_idx_to_data_dict if train else self.test_idx_to_data_dict
        return data_dict
    
    
    def number_of_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        return len(data_dict)

    def reset_scene_batch_sampler(self, train=True, shuffle=True) -> None:
        if train:
            sampling_dict = self.train_scene_sample_order
        elif not train:
            sampling_dict = self.test_scene_sample_order
        else:
            raise ValueError("Invalid value")

        data_dict = self.get_demo_data_dict(train)
        order = sorted(data_dict.keys())
        #print(order)
        if shuffle:
            np.random.shuffle(order)
        #print(order)

        sampling_dict['order'] = order
        sampling_dict['idx'] = 0
        


    def number_of_scene_data(self, train=True):
        return self.number_of_scenes(train)

    


    def get_all_object_pairs_for_scene_index(self, scene_idx, train=True):
        data_dict = self.get_demo_data_dict(train)[scene_idx]
 
        #print(data_dict)
        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        
        # we need to make sure all the following return arguments have the same value and structure. 
        if self.fast_training:
            all_relation_fast, obj_voxels_single_fast, one_hot_encoding_tensor_fast, total_objects_fast, edge_attr_fast, all_action_fast, all_action_label_fast, all_env_identity_fast, all_6DOF_pose_fast, all_hidden_label_list = scene_voxel_obj.get_all_object_pair_voxels_fast_3steps()
        

        data = {
            'num_objects': total_objects_fast,
            'action': all_action_fast,
            'relation': all_relation_fast, 
            'all_object_pair_voxels_single': obj_voxels_single_fast,
            'one_hot_encoding': one_hot_encoding_tensor_fast,
            'edge_attr': edge_attr_fast,
            'all_action_label':all_action_label_fast,
            'env_identity': all_env_identity_fast,
            'all_6DOF_pose_fast': all_6DOF_pose_fast,
            'all_hidden_label': all_hidden_label_list
        }
        data_last = {
            'num_objects': total_objects_fast,
            'action': all_action_fast,
            'relation': all_relation_fast, 
            'all_object_pair_voxels_single': obj_voxels_single_fast,
            'one_hot_encoding': one_hot_encoding_tensor_fast,
            'edge_attr': edge_attr_fast,
            'all_action_label':all_action_label_fast,
            'env_identity': all_env_identity_fast,
            'all_6DOF_pose_fast': all_6DOF_pose_fast,
            'all_hidden_label': all_hidden_label_list
        }
        
        return data, data_last
    
    def get_next_all_object_pairs_for_scene(self, train=True):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        

        data, data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        sample_order_dict['idx'] += 1
        
        return data, data_next
       

    def put_all_data_device(self, device):
        for i in range(len(self.train_idx_to_data_dict)):
            data_dict = self.train_idx_to_data_dict[i]['scene_voxel_obj']
            data_dict.put_all_things_on_device(device)


    
   

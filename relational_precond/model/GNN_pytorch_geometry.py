# Implements graph neural network components using PyTorch.
# GNN Embeder, PointConv.

import numpy as np
from collections import OrderedDict
from itertools import permutations

import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Data, Batch
from torch.nn import TransformerEncoder, TransformerEncoderLayer


from relational_precond.model.pointconv_util_groupnorm import PointConvDensitySetAbstraction


class EmbeddingNetTorch(nn.Module):
    def __init__(
            self, n_objects: int,
            width: int, layers: int, heads: int, input_feature_num: int,
            d_hidden: int, n_unary: int, n_binary: int,
            seperate_discrete_continuous = False,
            simple_encoding = False,
            torch_embedding = False,
            complicated_pre_dynamics = True,
            train_env_identity = False, 
            total_env_identity = 2,
            one_bit_env = False,
            direct_transformer = True,
            enable_high_push = False,
            use_seperate_latent_embedding = True
        ): 
        super().__init__()
        d_input = width

        self.use_seperate_latent_embedding = use_seperate_latent_embedding

        self.train_env_identity = train_env_identity

        self.direct_transformer = direct_transformer

        self.enable_high_push = enable_high_push

        if one_bit_env:
            total_env_identity = 1

        self.seperate_discrete_continuous = seperate_discrete_continuous

        self.simple_encoding = simple_encoding

        
        self.torch_embedding = torch_embedding

        self.complicated_pre_dynamics = complicated_pre_dynamics
        
        encoder_layers = TransformerEncoderLayer(width, heads, batch_first = True)

        self.transformer = TransformerEncoder(encoder_layers, layers)

        if self.train_env_identity:
            # self.env_output_identity = nn.Sequential(
            #     nn.Linear(d_input, d_hidden),
            #     nn.ReLU(),
            #     nn.Linear(d_hidden, d_hidden),
            #     nn.ReLU(),
            #     nn.Linear(d_hidden, total_env_identity),
            #     nn.Sigmoid()
            # )

            self.env_output_identity = nn.Sequential(
                nn.Linear(d_input, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, total_env_identity),
                nn.Sigmoid()
            ) # small one


        self.one_hot_encoding_dim = 128 

        if self.torch_embedding:
            self.one_hot_encoding_embed = nn.Sequential(
                nn.Embedding(n_objects, self.one_hot_encoding_dim)
            )
        elif self.simple_encoding:
            self.one_hot_encoding_embed = nn.Sequential(
                nn.Linear(n_objects, self.one_hot_encoding_dim)
            )
        else:


            self.one_hot_encoding_embed = nn.Sequential(
                    nn.Linear(n_objects, self.one_hot_encoding_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.one_hot_encoding_dim, self.one_hot_encoding_dim)
                )
        
        # print(self.one_hot_encoding_embed)
        if self.seperate_discrete_continuous:
            self.continuous_action_emb = nn.Sequential(
                nn.Linear(3, self.one_hot_encoding_dim),
                nn.ReLU(),
                nn.Linear(self.one_hot_encoding_dim, self.one_hot_encoding_dim)
            )
           
        elif self.simple_encoding:
            self.action_emb = nn.Sequential(
                nn.Linear(input_feature_num, width)
            )
        else:
            self.action_emb = nn.Sequential(
                nn.Linear(input_feature_num, width),
                nn.ReLU(),
                nn.Linear(width, width)
            )
        

        if not self.direct_transformer:
            if self.complicated_pre_dynamics:
                self.pre_dynamics_0 = nn.Sequential(
                    nn.Linear(width+self.one_hot_encoding_dim+self.one_hot_encoding_dim, width),
                    nn.ReLU(),
                    nn.Linear(width, width)
                )
                self.pre_dynamics_1 = nn.Sequential(
                    nn.Linear(width+self.one_hot_encoding_dim+self.one_hot_encoding_dim, width),
                    nn.ReLU(),
                    nn.Linear(width, width)
                )
            else:
                self.pre_dynamics = nn.Sequential(
                    nn.Linear(width+self.one_hot_encoding_dim+self.one_hot_encoding_dim, width)
                ) ## I may need to set this pre_dynamics function to be more complicated MLP for example. 
            
        if self.use_seperate_latent_embedding:
            encoder_layers_1 = TransformerEncoderLayer(width, heads, batch_first = True)
            self.graph_dynamics_0 = TransformerEncoder(encoder_layers_1, layers)
            encoder_layers_2 = TransformerEncoderLayer(width, heads, batch_first = True)
            self.graph_dynamics_1 = TransformerEncoder(encoder_layers_2, layers)
            if self.enable_high_push:
                encoder_layers_3 = TransformerEncoderLayer(width, heads, batch_first = True)
                self.graph_dynamics_2 = TransformerEncoder(encoder_layers_3, layers)
        else:
            encoder_layers_1 = TransformerEncoderLayer(width, heads, batch_first = True)
            self.graph_dynamics_0 = TransformerEncoder(encoder_layers_1, layers)
            self.action_type_embed = nn.Sequential(
                nn.Embedding(2, self.one_hot_encoding_dim*2)
            )

        

        self.n_unary = n_unary
        self.n_binary = n_binary
        self.d_hidden = d_hidden

        self.f_unary = self.get_head_unary(d_input, d_hidden, 1, n_unary)
        self.f_binary = self.get_head(d_input, d_hidden, 2, n_binary)
        

        self.ln_post = nn.LayerNorm(width)

    def get_head(self, d_input, d_hidden, n_args, n_binary):
        if d_hidden > 1:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, n_binary),
                nn.Sigmoid()
            )
        else:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden),
                nn.Sigmoid()
            )
        return head

    def get_head_unary(self, d_input, d_hidden, n_args, n_unary):
        if d_hidden > 1:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, n_unary)
            )
        else:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden)
            )
        return head

    
    def forward(self, objs, edge_index, edge_attr, actions, skill_label_placeholder): #(self, objs: torch.Tensor, actions: torch.Tensor):
        
        if objs.shape[0] != 1: ## hack for only batch size == 1
            objs = objs.view(1, objs.shape[0], objs.shape[1])
        
        batch_size, n_obj = objs.shape[:2] # # shape = [*, n_obj, width]

        x_current = objs # x = (pointconv feature + one hot encoding(128)) * 
        

        ## READOUT for current step
        # print(actions)
        

        n_obj = x_current.shape[1]
        z = self.f_unary(x_current)

        if self.train_env_identity:
            self.env_identity = self.env_output_identity(x_current)

            # print('env_identity', self.env_identity.shape)
            self.env_identity = self.env_identity.view(self.env_identity.shape[1], self.env_identity.shape[2])
        
        z = z.view(z.shape[1], z.shape[2])
        

        x1 = x_current[:, edge_index[0, :], :]


        x2 = x_current[:, edge_index[1, :], :]
        

        concat_x = torch.cat([x1, x2], dim=-1)
        y = self.f_binary(concat_x)
        
        y = y.view(y.shape[1], y.shape[2])
        
        
        ## dynamics part for next step.
        x = objs # x = (pointconv feature + one hot encoding(128)) * 
        x = self.transformer(x) 

        

        
        skill_label = skill_label_placeholder

        
        if self.direct_transformer:
            actions = actions[0, :]
            # print(actions.shape)
            actions = actions.view(1, actions.shape[0])
        # print(actions)

        if self.seperate_discrete_continuous:
            if self.torch_embedding:
                # print(actions[:, 1:-3].shape)
                max_index_tensor = torch.argmax(actions[:, 1:-3], dim=1)
                # print(max_index_tensor.shape)
                discrete_action = self.one_hot_encoding_embed(max_index_tensor)
            else:
                discrete_action = self.one_hot_encoding_embed(actions[:, 1:-3])
            continuous_action = self.continuous_action_emb(actions[:, -3:])
            action_embedding = torch.cat((discrete_action, continuous_action), axis = -1)
        else:
            action_embedding = self.action_emb(actions[:, 1:])
        action_embedding = action_embedding.view(batch_size, action_embedding.shape[0], action_embedding.shape[1])
        
        # print(action_embedding.shape)
        if not self.direct_transformer:
            action_x = torch.cat((action_embedding, x), axis = 2)
            
            if self.transformer_dynamics:
                if self.complicated_pre_dynamics:
                    # print('enter complicated')
                    if skill_label == 0:
                        action_x = self.pre_dynamics_0(action_x)
                    elif skill_label == 1:
                        action_x = self.pre_dynamics_1(action_x)
                    elif skill_label == 2:
                        action_x = self.pre_dynamics_2(action_x)

                else:
                    action_x = self.pre_dynamics(action_x)

        
        # print(action_embedding.shape)
        # print(x.shape)
        if self.direct_transformer:
            action_x = torch.cat((x, action_embedding), axis = 1)
        # print('action_x', action_x.shape)
        if skill_label == 0:
            next_predict = self.graph_dynamics_0(action_x)
        elif skill_label == 1:
            next_predict = self.graph_dynamics_1(action_x)
        elif skill_label == 2:
            next_predict = self.graph_dynamics_2(action_x)
        
        if self.direct_transformer:
            next_predict = next_predict[:, :-1, :]

        

        return_dict = {'pred': z,  ## z single object semantics like color, object pose etc
        'current_embed': x, 'pred_embedding':next_predict}

        return_dict['pred_sigmoid'] = y ## relations 

        if self.train_env_identity:
            return_dict['env_identity'] = self.env_identity

        return return_dict


class QuickReadoutNet(nn.Module):
    def __init__(
            self, n_objects: int,
            width: int, layers: int, heads: int, input_feature_num: int,
            d_hidden: int, n_unary: int, n_binary: int,
            train_env_identity = False, total_env_identity = 2,
            one_bit_env = False, 
            pe = False,
            transformer_decoder = False
        ): 

        super().__init__()
        d_input = width

        self.train_env_identity = train_env_identity
        self.pe = pe
        self.transformer_decoder = transformer_decoder

        if one_bit_env:
            total_env_identity = 1

        if self.transformer_decoder:
            encoder_layers = TransformerEncoderLayer(width, heads, batch_first = True)
            self.transformer = TransformerEncoder(encoder_layers, layers)


        
        self.one_hot_encoding_dim = 128 

        self.one_hot_encoding_embed = nn.Sequential(
                nn.Linear(n_objects, self.one_hot_encoding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.one_hot_encoding_dim, self.one_hot_encoding_dim)
            )
        
        self.action_emb = nn.Sequential(
                nn.Linear(input_feature_num, width),
                nn.ReLU(),
                nn.Linear(width, width)
            )

        if self.train_env_identity:
            # self.env_output_identity = nn.Sequential(
            #     nn.Linear(d_input, d_hidden),
            #     nn.ReLU(),
            #     nn.Linear(d_hidden, d_hidden),
            #     nn.ReLU(),
            #     nn.Linear(d_hidden, total_env_identity),
            #     nn.Sigmoid()
            # )

            self.env_output_identity = nn.Sequential(
                nn.Linear(d_input, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, total_env_identity),
                nn.Sigmoid()
            ) # small one

        if self.pe:
            self.pose_estimation = nn.Sequential(
                nn.Linear(d_input, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, 9)
            )



        self.n_unary = n_unary
        self.n_binary = n_binary
        self.d_hidden = d_hidden

        self.f_unary = self.get_head_unary(d_input, d_hidden, 1, n_unary)
        self.f_binary = self.get_head(d_input, d_hidden, 2, n_binary)
        


        self.ln_post = nn.LayerNorm(width)

    def get_head(self, d_input, d_hidden, n_args, n_binary):
        if d_hidden > 1:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, n_binary),
                nn.Sigmoid()
            )
        else:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden),
                nn.Sigmoid()
            )
        return head

    def get_head_unary(self, d_input, d_hidden, n_args, n_unary):
        if d_hidden > 1:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, n_unary)
            )
        else:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden)
            )
        return head

    
    def forward(self, objs, edge_index, edge_attr, actions, skill_label): #(self, objs: torch.Tensor, actions: torch.Tensor):
        # print(objs.shape)
        # objs = self.embed_func(objs)
        
        # print(objs.shape)
        
        if objs.shape[0] != 1: ## hack for only batch size == 1
            objs = objs.view(1, objs.shape[0], objs.shape[1])
        batch_size, n_obj = objs.shape[:2] # # shape = [*, n_obj, width]

        x = objs # x = (pointconv feature + one hot encoding(128)) * 

        if self.transformer_decoder:
            # print('enter transformer decoder')
            x = self.transformer(x)
        

        n_obj = x.shape[1]
        # print('x', x[0, :, 0])
        z = self.f_unary(x)

        if self.train_env_identity:
            self.env_identity = self.env_output_identity(x)

            # print('env_identity', self.env_identity.shape)
            self.env_identity = self.env_identity.view(self.env_identity.shape[1], self.env_identity.shape[2])
        
        if self.pe:
            self.predicted_pose = self.pose_estimation(x)
            self.predicted_pose = self.predicted_pose.view(self.predicted_pose.shape[1], self.predicted_pose.shape[2])
        
        z = z.view(z.shape[1], z.shape[2])
        

        x1 = x[:, edge_index[0, :], :]

        x2 = x[:, edge_index[1, :], :]


        concat_x = torch.cat([x1, x2], dim=-1)
        y = self.f_binary(concat_x)
        
        y = y.view(y.shape[1], y.shape[2])
        

        return_dict = {'pred': z,
        'current_embed': x}

        return_dict['pred_sigmoid'] = y
        if self.train_env_identity:
            return_dict['env_identity'] = self.env_identity
        if self.pe:
            return_dict['predicted_pose'] = self.predicted_pose

        return return_dict
        

class GNNModelOptionalEdge(MessagePassing):
    def __init__(self, 
                 in_channels, 
                 edge_inp_size,
                 node_output_size, 
                 relation_output_size, 
                 max_objects = 5, 
                 graph_output_emb_size=16, 
                 node_emb_size=32, 
                 edge_emb_size=32,
                 message_output_hidden_layer_size=128,  
                 message_output_size=128, 
                 node_output_hidden_layer_size=64,
                 edge_output_size=16,
                 use_latent_action = True,
                 latent_action_dim = 128, 
                 all_classifier = False,
                 predict_obj_masks=False,
                 predict_graph_output=False,
                 use_edge_embedding=False,
                 predict_edge_output=False,
                 use_edge_input=False,
                 node_embedding = False,
                 use_shared_latent_embedding = False,
                 use_seperate_latent_embedding = False,
                 total_object_identity = 10,
                 total_rgb_identity = 10,
                 total_env_identity = 2,
                 learn_update_edge_weights = False,
                 seperate_env_id = False,
                 max_env_num = 0,
                 seperate_action_emb = False,
                 seperate_discrete_continuous = False,
                 simple_encoding = False,
                 one_bit_env = False,
                 larger_output_sigmoid = False):
        self.larger_output_sigmoid = larger_output_sigmoid
        self.one_bit_env = one_bit_env
        self.relation_output_size = relation_output_size

        self.seperate_action_emb = seperate_action_emb
        # define the relation_output_size by hand for all baselines. 
        # Make sure all the planning stuff keeps the same for all our comparison approaches. 
        super(GNNModelOptionalEdge, self).__init__(aggr='mean')
        # all edge output will be classifier
        self.all_classifier = all_classifier


        self.learn_update_edge_weights = learn_update_edge_weights

        if self.one_bit_env:
            total_env_identity = 1

        
        self.node_inp_size = in_channels
        # Predict if an object moved or not
        self._predict_obj_masks = predict_obj_masks
        # predict any graph level output
        self._predict_graph_output = predict_graph_output

        self.latent_action_dim = latent_action_dim
        self.use_latent_action = use_latent_action

        self.use_seperate_latent_embedding = use_seperate_latent_embedding

        
        
        self.use_one_hot_embedding = True
        if self.use_one_hot_embedding: 
            self.one_hot_encoding_dim = 128

        
        total_objects = max_objects

        self.total_objects = total_objects

        action_dim = total_objects + 3
        if use_shared_latent_embedding:
            action_dim = action_dim + 1
        self.simple_encoding = simple_encoding
        self.seperate_discrete_continuous = seperate_discrete_continuous
        if self.use_latent_action:
            if self.seperate_discrete_continuous:
                self._in_channels = self.latent_action_dim + self.one_hot_encoding_dim
            else:
                self._in_channels = self.latent_action_dim
            if self.seperate_discrete_continuous:
                self.continuous_action_emb = nn.Sequential(
                    nn.Linear(3, self.latent_action_dim),
                    nn.ReLU(),
                    nn.Linear(self.latent_action_dim, self.latent_action_dim)
                )
                # print('enter the new continuous action embed')
                # self.continuous_action_emb = nn.Sequential(
                #     nn.Linear(3, self.latent_action_dim)
                # )
                if self.seperate_action_emb:
                    self.continuous_action_emb_1 = nn.Sequential(
                        nn.Linear(3, self.latent_action_dim)
                    )
            elif self.simple_encoding:
                self.action_emb = nn.Sequential(
                    nn.Linear(action_dim, self.latent_action_dim)
                )
            else:
                self.action_emb = nn.Sequential(
                    nn.Linear(action_dim, self.latent_action_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.latent_action_dim, self.latent_action_dim)
                )
            # if self.seperate_action_emb:
            #     self.action_emb_1 = nn.Sequential(
            #         nn.Linear(action_dim, self.latent_action_dim),
            #         nn.ReLU(inplace=True),
            #         nn.Linear(self.latent_action_dim, self.latent_action_dim)
            #     )
        else:
            self._in_channels = action_dim

        if self.use_one_hot_embedding: 
            if self.simple_encoding:
                self.one_hot_encoding_embed = nn.Sequential(
                    nn.Linear(total_objects, self.one_hot_encoding_dim)
                )
                # print('enter simple version')
            else:
                self.one_hot_encoding_embed = nn.Sequential(
                        nn.Linear(total_objects, self.one_hot_encoding_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.one_hot_encoding_dim, self.one_hot_encoding_dim)
                    )

        
        

        self._use_edge_dynamics = True

        self.use_edge_input = use_edge_input
        if self.use_edge_input:
            self.use_one_hot_embedding = False
        
        if use_edge_input == False:
            edge_inp_size = 0
            use_edge_embedding = False
            self._use_edge_dynamics = False
        self._edge_inp_size = edge_inp_size

        self._node_emb_size = node_emb_size
        self.node_embedding = node_embedding
        if self.node_embedding:
            self.node_emb = nn.Sequential(
                nn.Linear(in_channels, self._node_emb_size),
                nn.ReLU(inplace=True),
                nn.Linear(self._node_emb_size, self._node_emb_size)
            )
        if not self.node_embedding:
            if self.use_one_hot_embedding:
                self.node_inp_size += self.one_hot_encoding_dim
                self._node_emb_size = self.node_inp_size
                
            else:
                self._node_emb_size = self.node_inp_size

        self.edge_emb_size = edge_emb_size
        self._use_edge_embedding = use_edge_embedding
        self._test_edge_embedding = False
        if use_edge_embedding:
            self.edge_emb = nn.Sequential(
                nn.Linear(edge_inp_size, edge_emb_size),
                nn.ReLU(inplace=True),
                nn.Linear(edge_emb_size, edge_emb_size)
            )

        self._message_layer_size = message_output_hidden_layer_size
        self._message_output_size = message_output_size
        #print('node input size', self.node_inp_size)
        if self.node_embedding:
            message_inp_size = 2*self._node_emb_size + edge_emb_size if use_edge_embedding else \
                2 * self._node_emb_size + edge_inp_size
        else:
            message_inp_size = 2*self.node_inp_size + edge_emb_size if use_edge_embedding else \
                2 * self.node_inp_size + edge_inp_size
        # if use_edge_input == False:
        #     message_inp_size = 2 * self._node_emb_size
        self.message_info_mlp = nn.Sequential(
            nn.Linear(message_inp_size, self._message_layer_size),
            nn.ReLU(),
            # nn.Linear(self._message_layer_size, self._message_layer_size),
            # nn.ReLU(),
            nn.Linear(self._message_layer_size, self._message_output_size)
            )
        if self.learn_update_edge_weights:
            self.message_weight_predictor = nn.Sequential(
                nn.Linear(message_inp_size, self._message_layer_size),
                nn.ReLU(),
                # nn.Linear(self._message_layer_size, self._message_layer_size),
                # nn.ReLU(),
                nn.Linear(self._message_layer_size, 1),
                nn.Sigmoid()
                )

        self._node_output_layer_size = node_output_hidden_layer_size
        self._per_node_output_size = node_output_size
        graph_output_emb_size = 0
        self._per_node_graph_output_size = graph_output_emb_size
        self.node_output_mlp = nn.Sequential(
            nn.Linear(self._node_emb_size + self._message_output_size, self._node_output_layer_size),
            nn.ReLU(),
            nn.Linear(self._node_output_layer_size, node_output_size + graph_output_emb_size)
        )

        

        action_dim = self._in_channels
        self.action_dim = action_dim
        self.dynamics =  nn.Sequential(
            nn.Linear(self._in_channels+action_dim, 128),  # larger value
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self._in_channels)
        )

        if self._use_edge_dynamics:
            self.edge_dynamics =  nn.Sequential(
                nn.Linear(self._edge_inp_size+action_dim, 128),  # larger value
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self._edge_inp_size)
            )

        
        self.add_dropout = False
        if self.use_seperate_latent_embedding:
            if self.add_dropout:
                self.graph_dynamics_0 = nn.Sequential(
                    nn.Linear(node_output_size+action_dim, 512),  # larger value
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, node_output_size)
                )

                self.graph_edge_dynamics_0 = nn.Sequential(
                    nn.Linear(edge_output_size+action_dim, 512),  # larger value
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, edge_output_size)
                )

                self.graph_dynamics_1 = nn.Sequential(
                    nn.Linear(node_output_size+action_dim, 512),  # larger value
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, node_output_size)
                )

                self.graph_edge_dynamics_1 = nn.Sequential(
                    nn.Linear(edge_output_size+action_dim, 512),  # larger value
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, edge_output_size)
                )
            else:
                self.graph_dynamics_0 = nn.Sequential(
                    nn.Linear(node_output_size+action_dim, 512),  # larger value
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, node_output_size)
                )

                self.graph_edge_dynamics_0 = nn.Sequential(
                    nn.Linear(edge_output_size+action_dim, 512),  # larger value
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, edge_output_size)
                )

                self.graph_dynamics_1 = nn.Sequential(
                    nn.Linear(node_output_size+action_dim, 512),  # larger value
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, node_output_size)
                )

                self.graph_edge_dynamics_1 = nn.Sequential(
                    nn.Linear(edge_output_size+action_dim, 512),  # larger value
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, edge_output_size)
                )
               
        else:
            self.graph_dynamics = nn.Sequential(
                nn.Linear(node_output_size+action_dim, 512),  # larger value
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, node_output_size)
            )

            self.graph_edge_dynamics = nn.Sequential(
                nn.Linear(edge_output_size+action_dim, 512),  # larger value
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, edge_output_size)
            )

         
        if self._predict_graph_output:
            self._graph_pred_mlp = nn.Sequential(
                nn.Linear(graph_output_emb_size, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            )
        
        self._should_predict_edge_output = predict_edge_output
        if predict_edge_output:
            self._edge_output_size = edge_output_size
            # TODO: Add edge attributes as well, should be easy
            if True:
                self._edge_output_mlp = nn.Sequential(
                    nn.Linear(edge_inp_size + 2 * self._node_emb_size + 2 * self._message_output_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, edge_output_size)
                )
                self._edge_output_sigmoid = nn.Sequential(
                    nn.Linear(edge_inp_size + 2 * self._node_emb_size + 2 * self._message_output_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.relation_output_size),
                    nn.Sigmoid()
                )
                
                if self.larger_output_sigmoid:
                    print('larger_output_sigmoid ')
                    self._edge_output_sigmoid = nn.Sequential(
                        nn.Linear(edge_inp_size + 2 * self._node_emb_size + 2 * self._message_output_size, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, self.relation_output_size),
                        nn.Sigmoid()
                    )
                else:
                    self._edge_output_sigmoid = nn.Sequential(
                        nn.Linear(edge_inp_size + 2 * self._node_emb_size + 2 * self._message_output_size, 64),
                        nn.ReLU(),
                        nn.Linear(64, self.relation_output_size),
                        nn.Sigmoid()
                    )
            self._pred_edge_output = None


    def forward(self, x, edge_index, edge_attr, action, skill_label):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_x has shape [E, edge_features]

        # Get node embeddings for input features
        #print(x.shape)
        # print(self.node_emb)
        self._test_edge_embedding = False
        
        # if self.use_seperate_latent_embedding:
        #     skill_label = (int)(action.cpu().detach().numpy()[0][0])
            # print('skill_label', skill_label)
        if self.use_latent_action:
            # print(action.shape)
            # print(self.action_emb)
            if self.seperate_discrete_continuous:
                discrete_action = self.one_hot_encoding_embed(action[:, 1:-3])
                
                if self.seperate_action_emb:
                    if skill_label == 0:
                        continuous_action = self.continuous_action_emb(action[:, -3:])
                    elif skill_label == 1:
                        continuous_action = self.continuous_action_emb_1(action[:, -3:])
                else:
                    continuous_action = self.continuous_action_emb(action[:, -3:])
                action = torch.cat((discrete_action, continuous_action), axis = -1)
            elif self.use_seperate_latent_embedding:
                action = self.action_emb(action[:, 1:])
            else:
                action = self.action_emb(action)
        if self.node_embedding:
            x = self.node_emb(x)

        # Begin the message passing scheme
        total_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        #print(total_out)

        # Get outputs for every ndoe vs overall graph
        node_out_index = torch.arange(self._per_node_output_size).to(x.device)
        graph_out_index = torch.arange(
            self._per_node_output_size, 
            self._per_node_output_size+self._per_node_graph_output_size).to(x.device)

        # Get node level outputs, that is [0..node_out_index-1] values from total_out
        out = torch.index_select(total_out, dim=1, index=node_out_index)
        #import pdb; pdb.set_trace()
        if self._predict_obj_masks:
            mask_index = [out.size(1) - 1]
            state_pred_index = [i for i in range(out.size(1)-1)]

            state_pred_out = torch.index_select(out, 1, torch.LongTensor(state_pred_index).to(x.device))
            mask_out = torch.index_select(out, 1, torch.LongTensor(mask_index).to(x.device))[:, 0]
        else:
            state_pred_out = out
            mask_out = None

        # Get graph level outputs, i.e., [node_out_index, end] values from total_out
        if self._predict_graph_output:
            graph_out = torch.index_select(total_out, dim=1, index=graph_out_index)
            graph_out = global_add_pool(graph_out, batch)
            graph_preds = self._graph_pred_mlp(graph_out)
        else:
            graph_preds = None

        
        # print(self._per_node_output_size)
        if self.use_seperate_latent_embedding:
            graph_node_action = torch.cat((state_pred_out, action), axis = 1)
            if skill_label == 0:
                pred_node_embedding = self.graph_dynamics_0(graph_node_action)
            elif skill_label == 1:
                pred_node_embedding = self.graph_dynamics_1(graph_node_action)
            elif skill_label == 2:
                pred_node_embedding = self.graph_dynamics_2(graph_node_action)
            # pred_node_embedding = self.graph_dynamics[skill_label](graph_node_action)

            
            edge_num = self._pred_edge_output.shape[0]
            edge_action_list = []
            for _ in range(edge_num):
                edge_action_list.append(action[0][:])
            edge_action = torch.stack(edge_action_list)
            # print('self._pred_edge_output shape', self._pred_edge_output.shape)
            # print('edge action shape', edge_action.shape)
            graph_edge_node_action = torch.cat((self._pred_edge_output, edge_action), axis = 1)
            if skill_label == 0:
                pred_graph_edge_embedding = self.graph_edge_dynamics_0(graph_edge_node_action)
            elif skill_label == 1:
                pred_graph_edge_embedding = self.graph_edge_dynamics_1(graph_edge_node_action)
            
        else:
            graph_node_action = torch.cat((state_pred_out, action), axis = 1)
            pred_node_embedding = self.graph_dynamics(graph_node_action)

            
            edge_num = self._pred_edge_output.shape[0]
            edge_action_list = []
            for _ in range(edge_num):
                edge_action_list.append(action[0][:])
            edge_action = torch.stack(edge_action_list)
            graph_edge_node_action = torch.cat((self._pred_edge_output, edge_action), axis = 1)
            pred_graph_edge_embedding = self.graph_edge_dynamics(graph_edge_node_action)
        
        return_dict = {'pred': state_pred_out,
        'current_embed': state_pred_out, 'pred_embedding':pred_node_embedding, 'edge_embed': self._pred_edge_output, 'pred_edge_embed': pred_graph_edge_embedding}
        
        if self._should_predict_edge_output:
            return_dict['pred_edge'] = self._pred_edge_output
            #print(self._pred_edge_output_sigmoid)
            return_dict['pred_sigmoid'] = self._pred_edge_output_sigmoid
        if self.all_classifier:
            return_dict['pred_edge_classifier'] = self._pred_edge_classifier
        

        return return_dict

    
    
    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # edge_attr is the edge attribute between x_i and x_j

        # x_i is the central node that aggregates information
        # x_j is the neighboring node that passes on information.

        # Concatenate features for sender node (x_j) and receiver x_i and get the message from them
        # Maybe there is a better way to get this message information?

        if self._test_edge_embedding:
            edge_inp = edge_attr
        else:
            if self._use_edge_embedding:
                assert self.edge_emb is not None, "Edge embedding model cannot be none"
                # print(edge_attr.shape)
                # print(self.edge_emb)
                edge_inp = self.edge_emb(edge_attr)
            else:
                edge_inp = edge_attr
        self._edge_inp = edge_inp
        #print('edge in GNN', self._edge_inp)

        #print(edge_inp.shape)
        if self.use_edge_input:
            # print(x_i.shape)
            # print(x_j.shape)
            # print(edge_inp)
            # print(edge_inp.shape)
            

            x_ij = torch.cat([x_i, x_j, edge_inp], dim=1)
            # print(x_ij.shape)
            # print(self.message_info_mlp)
            out = self.message_info_mlp(x_ij)
        else:
            x_ij = torch.cat([x_i, x_j], dim=1)
            # print(x_ij.shape)
            # print(self.message_info_mlp)
            out = self.message_info_mlp(x_ij)
        #print('out', out.shape)
        # print(out)
        if self.learn_update_edge_weights:
            self.edge_weights = self.message_weight_predictor(x_ij)
            out = torch.mul(out, self.edge_weights)
        return out

    def update(self, x_ij_aggr, x, edge_index, edge_attr):
        # We can transform the node embedding, or use the transformed embedding directly as well.
        inp = torch.cat([x, x_ij_aggr], dim=1)
        # print('edge_index',edge_index)
        if self._should_predict_edge_output:
            source_node_idxs, target_node_idxs = edge_index[0, :], edge_index[1, :]
            if self.use_edge_input:
                edge_inp = torch.cat([
                    self._edge_inp,
                    x[source_node_idxs], x[target_node_idxs],
                    x_ij_aggr[source_node_idxs], x_ij_aggr[target_node_idxs]], dim=1)
            else:
                edge_inp = torch.cat([
                    x[source_node_idxs], x[target_node_idxs],
                    x_ij_aggr[source_node_idxs], x_ij_aggr[target_node_idxs]], dim=1)
            # print(edge_inp.shape)
            # print(self._edge_output_sigmoid)
            # print(self._edge_output_mlp)
            self._pred_edge_output = self._edge_output_mlp(edge_inp)
            self._pred_edge_output_sigmoid = self._edge_output_sigmoid(edge_inp)
            #print(self._pred_edge_output_sigmoid)
            if self.all_classifier:
                self._pred_edge_classifier = []
                for pred_classifier in self.all_classifier_list:
                    pred_classifier = pred_classifier.to(x.device)
                    self._pred_edge_classifier.append(F.softmax(pred_classifier(edge_inp), dim = 1))
        
        
        return self.node_output_mlp(inp)

    def edge_decoder_result(self):
        if self._should_predict_edge_output:
            return self._pred_edge_output
        else:
            return None

    def forward_decoder(self, x, edge_index, edge_attr, batch, action):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_x has shape [E, edge_features]

        # Get node embeddings for input features
        # print(x.shape)
        # print(self.node_emb)
        #x = self.node_emb(x)
        

        # Begin the message passing scheme
        self._test_edge_embedding = True
        total_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        #print(total_out)

        # Get outputs for every ndoe vs overall graph
        node_out_index = torch.arange(self._per_node_output_size).to(x.device)
        graph_out_index = torch.arange(
            self._per_node_output_size, 
            self._per_node_output_size+self._per_node_graph_output_size).to(x.device)

        # Get node level outputs, that is [0..node_out_index-1] values from total_out
        out = torch.index_select(total_out, dim=1, index=node_out_index)
        #import pdb; pdb.set_trace()
        if self._predict_obj_masks:
            mask_index = [out.size(1) - 1]
            state_pred_index = [i for i in range(out.size(1)-1)]

            state_pred_out = torch.index_select(out, 1, torch.LongTensor(state_pred_index).to(x.device))
            mask_out = torch.index_select(out, 1, torch.LongTensor(mask_index).to(x.device))[:, 0]
        else:
            state_pred_out = out
            mask_out = None

        # Get graph level outputs, i.e., [node_out_index, end] values from total_out
        if self._predict_graph_output:
            graph_out = torch.index_select(total_out, dim=1, index=graph_out_index)
            graph_out = global_add_pool(graph_out, batch)
            graph_preds = self._graph_pred_mlp(graph_out)
        else:
            graph_preds = None

        #print(state_pred_out.shape)
        # print(state_pred_out.shape)
        # print(action.shape)
        state_action = torch.cat((state_pred_out, action), axis = 1)
        #print(state_action.shape)
        pred_state = self.dynamics(state_action)

        # print(self._pred_edge_output.shape)
        # print(action.shape)
        edge_action = torch.zeros((self._pred_edge_output.shape[0], self._pred_edge_output.shape[1] + self.action_dim))
        edge_action[:,:self._pred_edge_output.shape[1]] = self._pred_edge_output
        edge_action[:,self._pred_edge_output.shape[1]:] = action[0]
        edge_action = edge_action.to(x.device)
        #print(edge_action)

        #edge_action = torch.cat((self._pred_edge_output, action), axis = 1)
        #print(state_action.shape)
        
        if self._use_edge_dynamics:
            dynamics_edge = self.edge_dynamics(edge_action)

        graph_node_action = torch.cat((x, action), axis = 1)
        pred_node_embedding = self.graph_dynamics(graph_node_action)

        #edge_action = torch.stack([action[0][:], action[0][:], action[0][:], action[0][:], action[0][:], action[0][:]])
        edge_num = self._edge_inp.shape[0]
        edge_action_list = []
        for _ in range(edge_num):
            edge_action_list.append(action[0][:])
        edge_action = torch.stack(edge_action_list)
        
        graph_edge_node_action = torch.cat((self._edge_inp, edge_action), axis = 1)
        pred_graph_edge_embedding = self.graph_edge_dynamics(graph_edge_node_action)
        return_dict = {'pred': state_pred_out, 'object_mask': mask_out, 'graph_pred': graph_preds, 'pred_state': pred_state, 
        'current_embed': x, 'pred_embedding':pred_node_embedding, 'edge_embed': self._edge_inp, 'pred_edge_embed': pred_graph_edge_embedding}
        if self._should_predict_edge_output:
            return_dict['pred_edge'] = self._pred_edge_output
            return_dict['pred_sigmoid'] = self._pred_edge_output_sigmoid
        if self.all_classifier:
            return_dict['pred_edge_classifier'] = self._pred_edge_classifier
        if self._use_edge_dynamics:
            return_dict['dynamics_edge'] = dynamics_edge

        return return_dict


class PointConv(nn.Module):
    def __init__(self, normal_channel=False):
        super(PointConv, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        
        rgb_channel = 0
        self.normal_channel = normal_channel
        # self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        # self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        # self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[512], bandwidth = 0.4, group_all=True)
        
        
        self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel+rgb_channel, mlp=[32], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=64, nsample=16, in_channel= 32 + 3, mlp=[64], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel= 64 + 3, mlp=[128], bandwidth = 0.4, group_all=True) # version 3-5

        # self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        # self.sa2 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel= 64 + 3, mlp=[16], bandwidth = 0.2, group_all=True) # version feb
        
        
        #self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[128], bandwidth = 0.4, group_all=True)  

        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.GroupNorm(1, 128)
        self.drop1 = nn.Dropout(0.5)



        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.GroupNorm(1, 64)
        self.drop3 = nn.Dropout(0.5)    

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.GroupNorm(1, 32)
        self.drop4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(32, 3)


    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            # print('xyz.shape',xyz.shape)
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        
        x = l3_points.view(B, 128)
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))



        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        # x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        # x = self.fc5(x)

        return x


import tensorflow as tf
import numpy as np
import json
import logging, logging.config
import os,sys,time
import random
import scipy.sparse as sp

from tqdm import tqdm
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from metrics import bleu
from collections import defaultdict

def get_logger(log_dir, config_id):
    """
    Initialize Logger
    """
    if not os.path.exists(log_dir+config_id):
        os.makedirs(log_dir+config_id)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
     
    handler1 = logging.StreamHandler(sys.stdout)
    handler1.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)
 
    handler2 = logging.FileHandler(os.path.join(log_dir+config_id, "logs.log"),"w", encoding=None, delay="true")
    handler2.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    handler2.setFormatter(formatter)
    logger.addHandler(handler2)
    return logger

def reduce_data_size(data):
    """
    Reduce data size for debugging
    """
    dtype = ['train','test','dev']
    size = 237
    for d in dtype:
        dataset = data[d]
        for k in list(dataset.keys()):
            dataset[k] = dataset[k][0:size]
    
    return data

class GCN_SeAN(object):
    
    
    def load_data(self):
        self.logger.info("Reading Data....")
        self.data = json.load(open(self.params.data_dir+self.params.data_name,'r'))
        if self.params.debug==True:
            self.data = reduce_data_size(self.data)
            self.params.epochs = 2
        self.vocab = self.data['vocab']
        self.words = list(self.vocab.keys())
        self.invert_vocab = {v:k for k,v in self.vocab.items()}
        self.dep2id = self.data['dep2id']
        self.rel2id = self.data['rel2id']
        self.total_deps = len(self.dep2id)
        self.total_rels = len(self.rel2id)
        self.rel2id = self.data['rel2id']
        self.logger.info("Finished Reading Data!!")
        self.max_decoder_len = max(self.data['train']['dec_lens']+self.data['test']['dec_lens']+self.data['dev']['dec_lens'])
        if self.params.l2 == 0.0:
            self.regularizer = None
        else:
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.params.l2)
        
        if self.params.init=='xavier':
            self.initializer = tf.contrib.layers.xavier_initializer(seed = self.params.seed)
        elif self.params.init=='unit_variance':
            self.initializer = tf.contrib.layers.variance_scaling_initializer()
        elif self.params.init=='trunc':
            self.initializer = tf.truncated_normal_initializer(stddev=0.1,seed = self.params.seed)
        elif self.params.init=='none':
            self.initializer = None
        
        if self.params.use_dep_labels == False:
            for dtype in ['train', 'test', 'dev']:
                    for i, edges in enumerate(self.data[dtype]['context_deEdges']):
                        for j, edge in enumerate(edges):
                            self.data[dtype]['context_deEdges'][i][j] = (edge[0], edge[1], 0)
                    
                    for k, edges in enumerate(self.data[dtype]['q_edges']):
                        for l, edge in enumerate(edges):
                            self.data[dtype]['q_edges'][k][l] = (edge[0], edge[1], 0)
            self.total_deps = 1
        
        self.data_list={}
        for dtype in ['train', 'test', 'dev']:
            self.data_list[dtype] = []
            for i in range(len(self.data[dtype]['context_deEdges'])):
                self.data_list[dtype].append([self.data[dtype][key][i] for key in sorted(list(self.data['train'].keys()))])
        
    def getBatches(self, data):
        num_batches = len(data) // self.params.batch_size
        dummy = [[],0,[],[],0,[],[],[],0,[],0,[]]
        for i in range(num_batches+1):
            start_idx = i * self.params.batch_size
            if len(data[start_idx:])<self.params.batch_size:
                new = data[start_idx:]
                for j in range(self.params.batch_size - len(data[start_idx:])):
                    new.append(dummy)
                yield new
            else:
                yield data[start_idx : start_idx + self.params.batch_size]

    def get_words_from_ids(self,ids):
        ids_list=ids.tolist()
         
        r=[]
        for i in ids_list:
            c=''
            for j in i:
                c=c+' '+self.invert_vocab[j]
                if self.invert_vocab[j]=='<EOS>':
                    break
            r.append(c.strip())
        return r

    
    def get_adj(self, edgeList, batch_size, max_nodes, max_labels):

        adj_main_in, adj_main_out = [], []

        for edges in edgeList:
            adj_in, adj_out = {}, {}

            in_ind, in_data   =  defaultdict(list),  defaultdict(list)
            out_ind, out_data = defaultdict(list), defaultdict(list)

            for src, dest, lbl in edges:
                out_ind [lbl].append((src, dest))
                out_data[lbl].append(1.0)

                in_ind  [lbl].append((dest, src))
                in_data [lbl].append(1.0)

            for lbl in range(max_labels):
                if lbl not in out_ind and lbl not in in_ind:
                    adj_in [lbl] = sp.coo_matrix((max_nodes, max_nodes))
                    adj_out[lbl] = sp.coo_matrix((max_nodes, max_nodes))
                else:
                    adj_in [lbl] = sp.coo_matrix((in_data[lbl],  zip(*in_ind[lbl])),  shape=(max_nodes, max_nodes))
                    adj_out[lbl] = sp.coo_matrix((out_data[lbl], zip(*out_ind[lbl])), shape=(max_nodes, max_nodes))

            adj_main_in.append(adj_in)
            adj_main_out.append(adj_out)

        return adj_main_in, adj_main_out
    
    def get_targs_list(self,ids):
        r=[]
        for i in ids:
            c=''
            for j in i:
                c=c+' '+self.invert_vocab[j]
                if self.invert_vocab[j]=='<EOS>':
                    break
            r.append(c.strip())
        return r
    
    def pad_batch_data(self, data, seq_len):
            temp = np.zeros((len(data), seq_len), np.int32)
            mask = np.zeros((len(data), seq_len), np.float32)
            
            for i, ele in enumerate(data):
                temp[i, :len(ele)] = ele[:seq_len]
                mask[i, :len(ele)] = np.ones(len(ele[:seq_len]), np.float32)
    
            return temp, mask

    '''
    ####################################################################################################
    #########################            CREATE FEED DICT FROM A BATCH        ##########################
    ####################################################################################################
    '''            
    def create_feed_dict(self,batch,drop,forward_only):
        context_deEdges,context_lens,context_tokens,dec_ip,dec_lens,dec_targ,kbEdges,kbEnts,kbEnts_len,q_edges,q_lens,q_tokens = zip(*batch)
        
        context_tokens_padded, context_seq_len_mask = self.pad_batch_data(context_tokens,max(1,max(context_lens)))
        kbEnts_padded, kb_seq_len_mask = self.pad_batch_data(kbEnts,max(1,max(kbEnts_len)))
        q_tokens_padded,q_tokens_mask = self.pad_batch_data(q_tokens,max(1,max(q_lens)))
        
        dec_ip_padded,_ = self.pad_batch_data(dec_ip,self.max_decoder_len)
        dec_targ_padded, decoder_targ_mask = self.pad_batch_data(dec_targ,self.max_decoder_len)
        
        
        context_adj_in, context_adj_out = self.get_adj(context_deEdges, self.params.batch_size,max(1,max(context_lens)), self.total_deps)
        query_adj_in, query_adj_out = self.get_adj(q_edges, self.params.batch_size, max(1,max(q_lens)), self.total_deps)
        kb_adj_in, kb_adj_out = self.get_adj(kbEdges, self.params.batch_size, max(1,max(kbEnts_len)), self.total_rels)
        
        feed_dict = {}
        feed_dict[self.context] = context_tokens_padded
        feed_dict[self.context_lens] = context_lens
        feed_dict[self.max_c_seq_len] = max(1,max(context_lens))
        feed_dict[self.context_seq_len_mask] = context_seq_len_mask
        
        feed_dict[self.query] = q_tokens_padded
        feed_dict[self.query_lens] = q_lens
        feed_dict[self.max_q_seq_len] = max(1,max(q_lens))
        feed_dict[self.query_seq_len_mask] = q_tokens_mask
        
        feed_dict[self.kb_ents] = kbEnts_padded
        feed_dict[self.kb_ent_lens] = kbEnts_len
        feed_dict[self.max_kb_ents] = max(1,max(kbEnts_len))
        feed_dict[self.kb_seq_len_mask] = kb_seq_len_mask
        
        feed_dict[self.decoder_inps] = dec_ip_padded
        feed_dict[self.decoder_targets] = dec_targ_padded
        feed_dict[self.target_weights] = decoder_targ_mask
        feed_dict[self.decoder_lens] = dec_lens
        feed_dict[self.forward_only] = forward_only
        feed_dict[self.dropout] = drop
        
        for i in range(self.params.batch_size):
                for lbl in range(self.total_deps):
                    feed_dict[self.context_de_adj_mat_in[i][lbl]] = tf.SparseTensorValue(indices = np.array([context_adj_in[i][lbl].row, context_adj_in[i][lbl].col]).T,
                                                          values      = context_adj_in[i][lbl].data,
                                                    dense_shape    = context_adj_in[i][lbl].shape)
                    feed_dict[self.context_de_adj_mat_out[i][lbl]] = tf.SparseTensorValue(indices = np.array([context_adj_out[i][lbl].row, context_adj_out[i][lbl].col]).T,
                                                          values      = context_adj_out[i][lbl].data,
                                                    dense_shape    = context_adj_out[i][lbl].shape)
                    
                    
                    feed_dict[self.query_de_adj_mat_in[i][lbl]] = tf.SparseTensorValue(indices = np.array([query_adj_in[i][lbl].row, query_adj_in[i][lbl].col]).T,
                                                          values      = query_adj_in[i][lbl].data,
                                                    dense_shape    = query_adj_in[i][lbl].shape)
                    feed_dict[self.query_de_adj_mat_out[i][lbl]] = tf.SparseTensorValue(indices = np.array([query_adj_out[i][lbl].row, query_adj_out[i][lbl].col]).T,
                                                          values      = query_adj_out[i][lbl].data,
                                                    dense_shape    = query_adj_out[i][lbl].shape)
                for lbl_r in range(self.total_rels):
                    feed_dict[self.kb_adj_mat_in[i][lbl_r]] = tf.SparseTensorValue(indices = np.array([kb_adj_in[i][lbl_r].row, kb_adj_in[i][lbl_r].col]).T,
                                                          values      = kb_adj_in[i][lbl_r].data,
                                                    dense_shape    = kb_adj_in[i][lbl_r].shape)
                    feed_dict[self.kb_adj_mat_out[i][lbl_r]] = tf.SparseTensorValue(indices = np.array([kb_adj_out[i][lbl_r].row, kb_adj_out[i][lbl_r].col]).T,
                                                          values      = kb_adj_out[i][lbl_r].data,
                                                    dense_shape    = kb_adj_out[i][lbl_r].shape)
        
        return feed_dict

    '''
    ####################################################################################################
    #########################            ADD PLACEHOLDERS TO THE GRAPH        ##########################
    ####################################################################################################
    '''            
    
    def add_placeholders(self):
        self.context = tf.placeholder(tf.int32,[None,None], name="context_input")
        self.context_de_adj_mat_in = [{lbl: tf.sparse_placeholder(tf.float32,  shape=[None, None],  name='context_de_adj_mat_in_{}'.  format(lbl)) for lbl in range(self.total_deps)} for _ in range(self.params.batch_size)]
        self.context_de_adj_mat_out = [{lbl: tf.sparse_placeholder(tf.float32,  shape=[None, None],  name='context_de_adj_mat_out_{}'. format(lbl)) for lbl in range(self.total_deps)} for _ in range(self.params.batch_size)]
        self.context_lens = tf.placeholder(tf.int32,[None], name="context_input_lengths")
        self.max_c_seq_len = tf.placeholder(tf.int32, shape=(), name='max_context_seq_len')
        self.context_seq_len_mask = tf.placeholder(tf.float32,shape=[None,None],name="context_seq_len_mask")
        
        self.query = tf.placeholder(tf.int32,[None,None],name="query_input")
        self.query_de_adj_mat_in = [{lbl: tf.sparse_placeholder(tf.float32,  shape=[None, None],  name='query_de_adj_mat_in_{}'.  format(lbl)) for lbl in range(self.total_deps)} for _ in range(self.params.batch_size)]
        self.query_de_adj_mat_out = [{lbl: tf.sparse_placeholder(tf.float32,  shape=[None, None],  name='query_de_adj_mat_out_{}'. format(lbl)) for lbl in range(self.total_deps)} for _ in range(self.params.batch_size)]
        self.query_lens = tf.placeholder(tf.int32,[None], name="query_input_lengths")
        self.max_q_seq_len = tf.placeholder(tf.int32, shape=(), name='max_query_seq_len')
        self.query_seq_len_mask = tf.placeholder(tf.float32,shape=[None,None],name="query_seq_len_mask")

        self.kb_ents = tf.placeholder(tf.int32,[None,None],name="KB_entities")
        self.kb_ent_lens = tf.placeholder(tf.int32,[None],name="KB_entity_lengths")
        self.kb_adj_mat_in = [{lbl: tf.sparse_placeholder(tf.float32,  shape=[None, None],  name='kb_adj_mat_in_{}'.  format(lbl)) for lbl in range(self.total_rels)} for _ in range(self.params.batch_size)]
        self.kb_adj_mat_out = [{lbl: tf.sparse_placeholder(tf.float32,  shape=[None, None],  name='kb_adj_mat_out_{}'. format(lbl)) for lbl in range(self.total_rels)} for _ in range(self.params.batch_size)]
        self.max_kb_ents = tf.placeholder(tf.int32,shape=(),name="max_kb_entities")
        self.kb_seq_len_mask = tf.placeholder(tf.float32,shape=[None,None],name="kb_seq_len_mask")
        
        self.decoder_inps = tf.placeholder(tf.int32,[None,self.max_decoder_len],name = "decoder_inputs")
        self.decoder_targets = tf.placeholder(tf.int32,[None,self.max_decoder_len],name = "decoder_targets")
        self.target_weights = tf.placeholder(tf.int32,[None,self.max_decoder_len],name="decoder_target_weights")
        self.decoder_lens = tf.placeholder(tf.int32,[None],name = "decoder_target_lengths")
        self.forward_only = tf.placeholder(tf.bool,name="foward_only")
        self.dropout =tf.placeholder(tf.float32,name="dropout")

    '''
    ####################################################################################################
    #########################            GRAPH CONVOLUTION LAYER        ################################
    ####################################################################################################
    '''
         
    def GCNLayer(self, gcn_in,         
               in_dim,         
               gcn_dim,         
               batch_size,         
               max_nodes,        
               max_labels,         
               adj_in,         
               adj_out,        
               num_layers=1,    
               name="GCN"):
        out = []
        out.append(gcn_in)

        for layer in range(num_layers):
            gcn_in    = out[-1]                        
            if len(out) > 1: in_dim = gcn_dim                 

            with tf.name_scope('%s-%d' % (name,layer)):

                act_sum = tf.zeros([batch_size, max_nodes, gcn_dim])
                
                for lbl in range(max_labels):

                    with tf.variable_scope('label-%d_name-%s_layer-%d' % (lbl, name, layer),reuse=tf.AUTO_REUSE):

                        w_in   = tf.get_variable('w_in',   [in_dim, gcn_dim],initializer=self.initializer,regularizer=self.regularizer)
                        b_in   = tf.get_variable('b_in',   [1, gcn_dim],initializer=self.initializer,regularizer=self.regularizer)

                        w_out  = tf.get_variable('w_out',  [in_dim, gcn_dim],initializer=self.initializer,regularizer=self.regularizer)
                        b_out  = tf.get_variable('b_out',  [1, gcn_dim],initializer=self.initializer,regularizer=self.regularizer)

                        w_loop = tf.get_variable('w_loop', [in_dim, gcn_dim],initializer=self.initializer,regularizer=self.regularizer)

                        if self.params.edge_gate:
                            w_gin  = tf.get_variable('w_gin',  [in_dim, 1],initializer=self.initializer,regularizer=self.regularizer)
                            b_gin  = tf.get_variable('b_gin',  [1],initializer=self.initializer,regularizer=self.regularizer)

                            w_gout = tf.get_variable('w_gout', [in_dim, 1],initializer=self.initializer,regularizer=self.regularizer)
                            b_gout = tf.get_variable('b_gout', [1],initializer=self.initializer,regularizer=self.regularizer)

                            w_gloop = tf.get_variable('w_gloop',[in_dim, 1],initializer=self.initializer,regularizer=self.regularizer)

                    with tf.name_scope('in_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
                        inp_in  = tf.tensordot(gcn_in, w_in, axes=[2,0]) + tf.expand_dims(b_in, axis=0)
                        in_t    = tf.stack([tf.sparse_tensor_dense_matmul(adj_in[i][lbl], inp_in[i]) for i in range(batch_size)])
                        if self.params.dropout != 1.0:
                            in_t= tf.nn.dropout(in_t, keep_prob=self.dropout)

                        if self.params.edge_gate:
                            inp_gin = tf.tensordot(gcn_in, w_gin, axes=[2,0]) + tf.expand_dims(b_gin, axis=0)
                            in_gate = tf.stack([tf.sparse_tensor_dense_matmul(adj_in[i][lbl], inp_gin[i]) for i in range(batch_size)])
                            in_gsig = tf.sigmoid(in_gate)
                            in_act   = in_t * in_gsig
                        else:
                            in_act   = in_t

                    with tf.name_scope('out_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
                        inp_out  = tf.tensordot(gcn_in, w_out, axes=[2,0]) + tf.expand_dims(b_out, axis=0)
                        out_t    = tf.stack([tf.sparse_tensor_dense_matmul(adj_out[i][lbl], inp_out[i]) for i in range(batch_size)])
                        if self.params.dropout != 1.0:
                            out_t = tf.nn.dropout(out_t, keep_prob=self.dropout)

                        if self.params.edge_gate:
                            inp_gout = tf.tensordot(gcn_in, w_gout, axes=[2,0]) + tf.expand_dims(b_gout, axis=0)
                            out_gate = tf.stack([tf.sparse_tensor_dense_matmul(adj_out[i][lbl], inp_gout[i]) for i in range(batch_size)])
                            out_gsig = tf.sigmoid(out_gate)
                            out_act  = out_t * out_gsig
                        else:
                            out_act = out_t

                    with tf.name_scope('self_loop'):
                        inp_loop  = tf.tensordot(gcn_in, w_loop,  axes=[2,0])
                        if self.params.dropout != 1.0:
                            inp_loop  = tf.nn.dropout(inp_loop, keep_prob=self.dropout)

                        if self.params.edge_gate:
                            inp_gloop = tf.tensordot(gcn_in, w_gloop, axes=[2,0])
                            loop_gsig = tf.sigmoid(inp_gloop)
                            loop_act  = inp_loop * loop_gsig
                        else:
                            loop_act = inp_loop


                    act_sum += in_act + out_act + loop_act
                gcn_out = tf.nn.relu(act_sum)
                out.append(gcn_out)

        return out
    
    
    '''
    ####################################################################################################
    #########################            REDUCE BILSTM STATES TO A SINGLE STATE        #################
    ####################################################################################################
    '''
    
    def _reduce_states(self, fw_st, bw_st):
        
        hidden_dim = self.params.rnn_hidden_units
        with tf.variable_scope('reduce_final_state'):
    
          w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.initializer,regularizer=self.regularizer)
          w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32,initializer=self.initializer,regularizer=self.regularizer)
          bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.initializer,regularizer=self.regularizer)
          bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.initializer,regularizer=self.regularizer)
    
          old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) 
          old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) 
          new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) 
          new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) 
          return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) 


    '''
    ####################################################################################################
    #########################            BIDIRECTIONAL RNN ENCODER        ##############################
    ####################################################################################################
    '''

    def birnn_encode(self,scope,embeddings,seq_lens):
            with tf.variable_scope(scope):
                if self.params.rnn_unit=='lstm':
                        fw_cell    = tf.nn.rnn_cell.LSTMCell(self.params.rnn_hidden_units)
                        bk_cell    = tf.nn.rnn_cell.LSTMCell(self.params.rnn_hidden_units)
                elif self.params.rnn_unit=='gru':
                        fw_cell    = tf.nn.rnn_cell.GRUCell(self.params.rnn_hidden_units)
                        bk_cell    = tf.nn.rnn_cell.GRUCell(self.params.rnn_hidden_units)
                    
                hidden,state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bk_cell,embeddings,
                                                                    sequence_length=seq_lens, dtype=tf.float32)
                if self.params.rnn_unit=='lstm':
                    final_state = self._reduce_states(state[0],state[1])
                elif self.params.rnn_unit=='gru':
                    final_state = state
                return tf.concat((hidden[0],hidden[1]), axis=2),final_state
    
    '''
    ####################################################################################################
    #########################            ATTENTION DECODER        ######################################
    ####################################################################################################
    '''
    def attention_decoder(self,decoder_inputs,initial_state,
                context_attention_states,kb_attention_states,q_attention_states,
                cell,output_projection,output_size,kb_mask,q_mask,context_mask,
                loop_function,initial_state_attention,dtype=None,scope=None):

        with tf.variable_scope(scope or 'attention_decoder', dtype=dtype) as scope:
    
            dtype = scope.dtype
    
            batch_size = tf.shape(decoder_inputs[0])[0] 
    
            context_attn_length = context_attention_states.get_shape()[1].value 
            if context_attn_length == None: 
                context_attn_length = tf.shape(context_attention_states)[1]
            context_attn_size = context_attention_states.get_shape()[2].value
    
            kb_attn_length = kb_attention_states.get_shape()[1].value 
            if kb_attn_length == None: 
                kb_attn_length = tf.shape(kb_attention_states)[1]
            kb_attn_size = kb_attention_states.get_shape()[2].value
    
            q_attn_length = q_attention_states.get_shape()[1].value 
            if q_attn_length == None: 
                q_attn_length = tf.shape(q_attention_states)[1]
            q_attn_size = q_attention_states.get_shape()[2].value
    
            state = initial_state  
            outputs = []
            prev = None
            
            def masked_attention(e,enc_padding_mask):
              attn_dist = nn_ops.softmax(e) 
              attn_dist *= enc_padding_mask 
              attn_dist +=1e-10
              masked_sums = tf.reduce_sum(attn_dist, axis=1) 
              return attn_dist / tf.reshape(masked_sums, [-1, 1])
            
            def attention(query,attn_size,V,hidden_features,
                          attn_length,attn_states,name,mask):
                
                cs = [] 
                if nest.is_sequence(query):
                    query_list = nest.flatten(query)
                query = tf.concat(query_list,1) 
    
                with tf.variable_scope("Attention"+name) as scope:
                    y = _linear(
                        args=query, output_size=attn_size, bias=True,
                        bias_initializer = self.initializer, kernel_initializer = self.initializer)
    
                    
                    y = tf.reshape(y, [-1, 1, 1, attn_size]) 
                    
                    s = tf.reduce_sum(V * tf.nn.tanh(hidden_features + y), [2, 3])
                    a_masked = masked_attention(s,mask)
                    c = tf.reduce_sum(tf.reshape(
                        a_masked, [-1, attn_length, 1, 1])*attn_states, [1,2])
                    cs=tf.reshape(c, [-1, attn_size])
    
                return cs,a_masked
   
            hidden_context = tf.reshape(context_attention_states,
                [-1, context_attn_length, 1, context_attn_size]) 
            
            k1= tf.get_variable("AttnW1",[1, 1, context_attn_size, context_attn_size],initializer = self.initializer) 
            hidden_features_context=tf.nn.conv2d(hidden_context, k1, [1,1,1,1], "SAME") 
            V1=tf.get_variable("V1_attention_softmax", [context_attn_size],initializer = self.initializer) 
            
            hidden_kb = tf.reshape(kb_attention_states,
                [-1, kb_attn_length, 1, kb_attn_size]) 
            k2= tf.get_variable("AttnW2",[1, 1, kb_attn_size, kb_attn_size],initializer=self.initializer) 
            hidden_features_kb=tf.nn.conv2d(hidden_kb, k2, [1,1,1,1], "SAME")
            V2=tf.get_variable("V2_attention_softmax", [kb_attn_size],initializer = self.initializer)     
            
            hidden_q = tf.reshape(q_attention_states,
                [-1, q_attn_length, 1, q_attn_size]) 
            k3= tf.get_variable("AttnW3",[1, 1, q_attn_size, q_attn_size],initializer=self.initializer) 
            hidden_features_q=tf.nn.conv2d(hidden_q, k3, [1,1,1,1], "SAME")
            V3=tf.get_variable("V3_attention_softmax", [q_attn_size],initializer = self.initializer)    
            
            h_q = tf.zeros([batch_size,q_attn_size],dtype=dtype)
            h_q.set_shape([None,q_attn_size])        
            
            h_context = tf.zeros([batch_size,context_attn_size],dtype=dtype)
            h_context.set_shape([None,context_attn_size])       
            
            h_kb = tf.zeros([batch_size,kb_attn_size],dtype=dtype)
            h_kb.set_shape([None,kb_attn_size])        
            
            h_final=[h_context] 
            
            context_wts_l=[]
            kb_wts_l=[]
            q_wts_l=[]
            theta1_l=[]
            theta2_l=[]
            V_theta1=tf.get_variable("V_theta1_attention_softmax", [context_attn_size,1],initializer=self.initializer)
            V_theta2=tf.get_variable("V_theta2_attention_softmax", [kb_attn_size,1],initializer=self.initializer)
            for i, inp in enumerate(decoder_inputs):
    
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
    
                if loop_function is not None and prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        inp = loop_function(prev, i)  
    
                h_q,q_wts = attention([state],q_attn_size,V3,hidden_features_q,
                                    q_attn_length,hidden_q,'_q',q_mask)
                
                
                h_context,context_wts = attention([state]+[h_q],context_attn_size,V1,hidden_features_context,
                                  context_attn_length,hidden_context,'_context',context_mask) 
                
                
                h_kb,kb_wts = attention([state]+[h_q]+[h_context],kb_attn_size,V2,hidden_features_kb,
                                    kb_attn_length,hidden_kb,'_kb',kb_mask)

                state_projected =_linear(state,context_attn_size,False,
                                         bias_initializer = self.initializer, kernel_initializer = self.initializer)
                
                with tf.variable_scope("linear_theta1"):
                    theta1=tf.matmul(tf.nn.tanh(state_projected+_linear(h_context,context_attn_size,bias=False,
                                                                        bias_initializer = self.initializer, kernel_initializer = self.initializer)),V_theta1)
    
                with tf.variable_scope("linear_theta2"):
                    theta2=tf.matmul(tf.nn.tanh(state_projected+_linear(h_kb,kb_attn_size,bias=False,
                                                                        bias_initializer = self.initializer, kernel_initializer = self.initializer)),V_theta2)
    
                
                h_final=[theta1*h_context+theta2*h_kb]
                
                input_size = inp.get_shape().with_rank(2)[1] 
                
                with tf.variable_scope("linear1"):
                    x = _linear(
                    args=[inp]+h_final+[h_q], output_size=input_size, bias=True,
                    bias_initializer = self.initializer, kernel_initializer = self.initializer)  
    
                
                cell_outputs, state = cell(x, state) 
    
                with tf.variable_scope('attention_output_projection'):
                    output = _linear(
                        args=[cell_outputs]+h_final, output_size=output_size,
                        bias=True,
                        bias_initializer = self.initializer, kernel_initializer = self.initializer)
                if loop_function is not None:
                    prev = output
                outputs.append(output)
                context_wts_l.append(context_wts)
                kb_wts_l.append(kb_wts)
                q_wts_l.append(q_wts)
                theta1_l.append(theta1)
                theta2_l.append(theta2)

            return outputs, state, [context_wts_l,kb_wts_l,q_wts,theta1_l,theta2_l]

    
    '''
    ####################################################################################################
    #########################            FEED PREVIOUS MECHANISM        ################################
    ####################################################################################################
    '''
    
    def _extract_argmax_and_embed(self,W_embedding, output_projection,
    update_embedding=False):
    
        def loop_function(prev, _):
            prev = tf.matmul(prev, output_projection[0]) + output_projection[1]
            prev_symbol = tf.argmax(prev, axis=1) 
            embedded_prev_symbol = tf.nn.embedding_lookup(W_embedding, prev_symbol) 
     
            if not update_embedding:
                embedded_prev_symbol = tf.stop_gradient(embedded_prev_symbol)
    
            return embedded_prev_symbol
    
        return loop_function
    
    '''
    ####################################################################################################
    #########################           EMBEDDING ATTENTION DECODER        #############################
    ####################################################################################################
    '''
    
    def embedding_attention_decoder(self,decoder_inputs,initial_state,context_attention_states,kb_attention_states,
                    q_attention_states,cell,num_symbols,kb_mask,q_mask,context_mask,output_projection,feed_previous,
                    update_embedding_for_previous=True,scope=None, dtype=None, initial_state_attention=False):

        output_size = cell.output_size
    
        if feed_previous==False:
             tf.get_variable_scope().reuse_variables()
        with tf.variable_scope(scope or "embedding_attention_decoder",
            dtype=dtype) as scope:
    
            loop_function = self._extract_argmax_and_embed(
                self.embedding_mat, output_projection,
                update_embedding_for_previous) if feed_previous else None

            return self.attention_decoder(
                decoder_inputs,
                initial_state,
                context_attention_states,kb_attention_states,q_attention_states,
                cell,output_projection,
                output_size=output_size,
                kb_mask=kb_mask,
                q_mask = q_mask,
                context_mask=context_mask,
                loop_function=loop_function,
                initial_state_attention=initial_state_attention)
    
    '''
    ####################################################################################################
    #########################            THE WHOLE SEQ2SEQ GRAPH        ################################
    ####################################################################################################
    '''

    def add_seq2seq(self):
        
        with tf.variable_scope('Word_Embeddings') as scope:
            self.embedding_mat = tf.get_variable('word_embeddings',shape=[len(self.vocab),self.params.word_emb_dim], initializer=self.initializer, trainable=True, regularizer=self.regularizer)
            self.context_embeddings = tf.nn.embedding_lookup(self.embedding_mat, self.context)
            self.query_embeddings = tf.nn.embedding_lookup(self.embedding_mat, self.query)
            self.kb_ents_embeddings = tf.nn.embedding_lookup(self.embedding_mat,self.kb_ents)
            self.decoder_inputs_emb = tf.nn.embedding_lookup(self.embedding_mat,self.decoder_inps)
        
        with tf.variable_scope('Context_Embedding') as scope:
            if self.params.use_rnn:
                self.context_rnn_out, self.context_rnn_state = self.birnn_encode(scope,self.context_embeddings,self.context_lens)
                c_in_dims = self.params.rnn_hidden_units*2
            else:
                self.context_rnn_out = self.context_embeddings
                c_in_dims = self.params.word_emb_dim 
            
        with tf.variable_scope('Query_Embedding') as scope:
            if self.params.use_rnn:
                self.query_rnn_out, self.query_rnn_state = self.birnn_encode(scope,self.query_embeddings,self.query_lens)
                q_in_dims = self.params.rnn_hidden_units*2
            else:
                self.query_rnn_out = self.query_embeddings
                q_in_dims = self.params.word_emb_dim
            
        self.context_gcn_out = self.GCNLayer(self.context_rnn_out,c_in_dims,self.params.gcn_hidden_units,
                                            self.params.batch_size,self.max_c_seq_len,self.total_deps,
                                            self.context_de_adj_mat_in,self.context_de_adj_mat_out,self.params.num_hops,name="GCN")[-1]
        
        self.query_gcn_out = self.GCNLayer(self.query_rnn_out,q_in_dims,self.params.gcn_hidden_units,
                                            self.params.batch_size,self.max_q_seq_len,self.total_deps,
                                            self.query_de_adj_mat_in,self.query_de_adj_mat_out,self.params.num_hops,name="GCN")[-1]
        
        self.kb_gcn_out = self.GCNLayer(self.kb_ents_embeddings,self.params.word_emb_dim,self.params.gcn_hidden_units,
                                        self.params.batch_size,self.max_kb_ents,self.total_rels,
                                        self.kb_adj_mat_in,self.kb_adj_mat_out,self.params.num_hops,name = "KBGCN")[-1]
        
        with tf.variable_scope('Decoder_initial_state') as scope:
            self.decoder_init = tf.get_variable(name="decoder_init_state",shape=[self.params.batch_size,self.params.gcn_hidden_units],
                                                      dtype=tf.float32,initializer = self.initializer,regularizer = self.regularizer) 
            if self.params.rnn_unit =='gru':
                self.decoder_init_state = self.decoder_init
            elif self.params.rnn_unit =='lstm':
                self.decoder_init_state = tf.contrib.rnn.LSTMStateTuple(self.decoder_init,self.decoder_init)
        
        with tf.variable_scope('Decoder') as scope:             
            if self.params.rnn_unit=='lstm':
                self.decoder_cell = tf.nn.rnn_cell.LSTMCell(self.params.gcn_hidden_units)
            elif self.params.rnn_unit=='gru':
                self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.params.gcn_hidden_units)
            
            
            self.list_decoder_inputs = tf.unstack(self.decoder_inputs_emb, axis=1)    
    
            W_softmax = tf.get_variable("W_softmax",
    
                    shape=[self.params.gcn_hidden_units, len(self.vocab)],
                    dtype=tf.float32,initializer=self.initializer,regularizer = self.regularizer)       
                
            b_softmax = tf.get_variable("b_softmax",
                    shape=[len(self.vocab)],
                    dtype=tf.float32,initializer = self.initializer,regularizer = self.regularizer)
                
            output_projection = (W_softmax, b_softmax)
            self.all_decoder_outputs, self.decoder_state,self.attn_wts = tf.cond(self.forward_only,
                        lambda: self.embedding_attention_decoder(
                        decoder_inputs=self.list_decoder_inputs,
                        initial_state=self.decoder_init_state,
                        context_attention_states=self.context_gcn_out,
                        kb_attention_states=self.kb_gcn_out,
                        q_attention_states=self.query_gcn_out,
                        cell=self.decoder_cell,
                        num_symbols=len(self.vocab),
                        kb_mask=self.kb_seq_len_mask,
                        q_mask = self.query_seq_len_mask,
                        context_mask=self.context_seq_len_mask,
                        output_projection=output_projection,
                        feed_previous=True),
                        lambda: self.embedding_attention_decoder(
                        decoder_inputs=self.list_decoder_inputs,
                        initial_state=self.decoder_init_state,
                        context_attention_states=self.context_gcn_out,
                        kb_attention_states=self.kb_gcn_out,
                        q_attention_states=self.query_gcn_out,
                        cell=self.decoder_cell,
                        num_symbols=len(self.vocab),
                        kb_mask=self.kb_seq_len_mask,
                        q_mask = self.query_seq_len_mask,
                        context_mask=self.context_seq_len_mask,
                        output_projection=output_projection,
                        feed_previous=False))           
            l=[]
            for i in self.all_decoder_outputs:             
                logits = tf.matmul(i, W_softmax) + b_softmax
                l.append(logits)
            
            self.decoder_outputs = tf.stack(l,axis=1)
            
    '''
    ####################################################################################################
    #########################            LOSS, OPTIMIZER, PREDICTOR        #############################
    ####################################################################################################
    '''
    def add_loss(self, decoder_ops):
        with tf.name_scope('Loss_op'):
            loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_ops,
                                                targets=self.decoder_targets,
                                                weights=tf.cast(
                                                        tf.sequence_mask(self.decoder_lens,maxlen = self.max_decoder_len),
                                                        tf.float32))
            if self.regularizer != None: 
                loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss

    def add_optimizer(self, loss):
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
            train_op  = optimizer.minimize(loss)
        return train_op
    
    def add_predictions(self,decoder_ops):
        with tf.name_scope('Predictions'):
            output_probs = tf.nn.softmax(decoder_ops,axis=2)
            return tf.argmax(output_probs,axis=2)
    '''
    ####################################################################################################
    #########################            INITIALIZE        #############################################
    ####################################################################################################
    '''
    def __init__(self,args):
        self.params = args
        self.logger = get_logger(self.params.logs_dir, self.params.config_id)
        self.logger.info(args.flag_values_dict())
        self.load_data()
        self.logger.info("Building Graph....")
        t0 = time.time()
        self.add_placeholders()
        self.add_seq2seq()
        self.loss_op = self.add_loss(self.decoder_outputs)
        self.train_op = self.add_optimizer(self.loss_op)
        self.preds = self.add_predictions(self.decoder_outputs)
        self.saver = tf.train.Saver()
        t1 = time.time()
        self.logger.info('Time to build graph: %i seconds', t1 - t0)
        
        
    '''
    ####################################################################################################
    #########################            RUN INFERENCE        ##########################################
    ####################################################################################################
    '''
    def predict(self,sess,data_name):
        pred_losses = []
        preds_all=np.zeros(self.max_decoder_len)
        for step, batch in enumerate(self.getBatches(self.data_list[data_name])):
            feed = self.create_feed_dict(batch,drop = 1.0,forward_only=True)
            loss, preds,= sess.run([self.loss_op, self.preds], feed_dict=feed)
            pred_losses.append(loss)
            preds_all = np.row_stack((preds_all,preds))

        preds_ids=np.delete(preds_all,0,0)
        preds_list=self.get_words_from_ids(preds_ids)[0:len(self.data_list[data_name])]
        labels_list=self.get_targs_list(self.data[data_name]['dec_targ'])
        bl=bleu.moses_multi_bleu(preds_list,labels_list)
        
        return np.mean(pred_losses),bl,preds_list,labels_list


    '''
    ####################################################################################################
    #########################            RUN EPOCH         #############################################
    ####################################################################################################
    '''

    def run_epoch(self,sess,epoch):
        t0 = time.time()
        losses = []
        preds_all=np.zeros(self.max_decoder_len)
        for step, batch in enumerate(self.getBatches(self.data_list['train'])):
            feed = self.create_feed_dict(batch,drop = self.params.dropout,forward_only=False)
            loss, preds, _= sess.run([self.loss_op, self.preds, self.train_op], feed_dict=feed)
            losses.append(loss)
            preds_all = np.row_stack((preds_all,preds))
            self.logger.info('Epoch:{} \t Step:{} \t Batch Loss: {} \t Avg loss: {}'.format(epoch+1, step,loss, np.mean(losses)))

        preds_ids=np.delete(preds_all,0,0)
        preds_list=self.get_words_from_ids(preds_ids)[0:len(self.data_list['train'])]
        labels_list=self.get_targs_list(self.data['train']['dec_targ'])
        bl=bleu.moses_multi_bleu(preds_list,labels_list)

        self.logger.info('Train Loss:{}, Train BLEU: {}'.format(np.mean(losses), bl))
        t1 = time.time()
        self.logger.info('Time to run an epoch: %i seconds', t1 - t0)
        return np.mean(losses), bl,preds_list,labels_list

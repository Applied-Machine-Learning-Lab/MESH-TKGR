import math
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder_emb import ConvTransE, ConvTransR


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False,weight_loss=1):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.loss_weight=weight_loss

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w11 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w11)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.w21 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w21)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)
        self.emb_rel1 = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel1)
        self.entity_decoder = nn.Sequential(
            nn.Linear(4096, 1024),  
            nn.ReLU(),  
            nn.Linear(1024, 100)  
        )
        self.relation_decoder = nn.Sequential(
            nn.Linear(4096, 1024),  
            nn.ReLU(),  
            nn.Linear(1024, 100)  
        )
        self.entity_decoder_nhis = nn.Sequential(
            nn.Linear(4096, 1024),  
            nn.ReLU(),  
            nn.Linear(1024, 100)  
        )
        self.relation_decoder_nhis = nn.Sequential(
            nn.Linear(4096, 1024),  
            nn.ReLU(),  
            nn.Linear(1024, 100)  
        )
        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)
        self.dynamic_emb1 = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb1)
        #self.dynamic_emb = self.make_embedding(h_dim)
        #torch.save(self.dynamic_emb, '../models/words_emb.pth')
        self.llm_entity_emb = torch.load('../models/entity_emb.pth')
        llm_relation_emb = torch.load('../models/relation_emb.pth')
        self.llm_relation_emb=torch.cat((llm_relation_emb,llm_relation_emb.clone()),dim=0).to(self.gpu)
 
        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.words_emb1 = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb1)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.statci_rgcn_layer1 = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)
        self.rgcn1 = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)
        self.time_gate_weight1 = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight1, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias1 = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias1)                                                                  

        self.decoder_gate_weight = nn.Parameter(torch.Tensor(2*100, 1))    
        nn.init.xavier_uniform_(self.decoder_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.decoder_gate_bias = nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.decoder_gate_bias)

        self.decoder_gate_weight_nhis = nn.Parameter(torch.Tensor(2*100, 1))    
        nn.init.xavier_uniform_(self.decoder_gate_weight_nhis, gain=nn.init.calculate_gain('relu'))
        self.decoder_gate_bias_nhis = nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.decoder_gate_bias_nhis)

        self.decoder_gate_weight_all = nn.Parameter(torch.Tensor(2*100, 1))    
        nn.init.xavier_uniform_(self.decoder_gate_weight_all, gain=nn.init.calculate_gain('relu'))
        self.decoder_gate_bias_all = nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.decoder_gate_bias_all)

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)
        self.relation_cell_11= nn.GRUCell(self.h_dim*2, self.h_dim)

       
       
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, 100, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, 100, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob1 = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder1 = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob_nhis = ConvTransE(num_ents, 100, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob1_nhis = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError 

    def forward(self, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)   
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)  
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            self.h = time_weight * current_h + (1-time_weight) * self.h
            #print(self.h.shape)
            history_embs.append(self.h)
        return history_embs, static_emb, self.h_0, gate_list, degree_list
    def forward1(self, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb1, self.words_emb1), dim=0)  
            self.statci_rgcn_layer1(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb1) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel1, x_input), dim=1)
                self.h_0 = self.relation_cell_11(x_input, self.emb_rel)    
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel1, x_input), dim=1)
                self.h_0 = self.relation_cell_11(x_input, self.h_0)  
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            current_h = self.rgcn1.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight1) + self.time_gate_bias1)
            self.h = time_weight * current_h + (1-time_weight) * self.h
            #print(self.h.shape)
            history_embs.append(self.h)
        return history_embs, static_emb, self.h_0, gate_list, degree_list

    def make_embedding(self,h_dim):

        
        word_id_to_word = {}
        with open('../data/ICEWS14s/entity2id.txt', 'r') as file:
            for line in file:
                word, word_id = line.strip().split('\t')
                word_id_to_word[int(word_id)] = word

       
        model_path = "../../GenTKG-main/Llama-2-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)

      
        words_emb = torch.nn.Parameter(torch.zeros(self.num_ents, h_dim), requires_grad=True).float()
        linear_layer = nn.Linear(4096, h_dim)
       
        
        with torch.no_grad():
            for word_id, word in word_id_to_word.items():
                print(word_id)
                inputs = tokenizer(word, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
                outputs = model(**inputs)
                word_embedding = outputs.last_hidden_state.mean(dim=1)  
                words_emb[word_id] = linear_layer(word_embedding)
                
        return words_emb

        

    def predict(self, test_graph,num_rels, static_graph, test_triplets,mask, t, use_cuda,print_score=False):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            all_triples = test_triplets
            #print(all_triples)
            inverse_mask=mask.clone()
            all_masks=torch.cat([mask,inverse_mask],dim=0)
            all_masks=all_masks.to(self.gpu)
            evolve_embs, _, r_emb1, _, _ = self.forward(test_graph, static_graph, use_cuda)
            evolve_embs1, _, r_emb1_nhis, _, _ = self.forward(test_graph, static_graph, use_cuda)
            embedding1 = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1] 
            embedding1_nhis = F.normalize(evolve_embs1[-1]) if self.layer_norm else evolve_embs1[-1] 
           
            llm_entity_emb=self.entity_decoder(self.llm_entity_emb)
            llm_relation_emb=self.relation_decoder(self.llm_relation_emb)
            tensor_t = torch.full((llm_entity_emb.size(0), 1), t).to(self.gpu)
            embedding = F.normalize(llm_entity_emb)
            tensor_t = torch.full((llm_relation_emb.size(0), 1), t).to(self.gpu)
            r_emb = F.normalize(llm_relation_emb)
            e1_embedded_all = F.tanh(embedding)
            e1_embedded = e1_embedded_all[all_triples[:, 0]]
            rel_embedded = r_emb[all_triples[:, 1]]
            stacked_inputs = torch.cat((e1_embedded, rel_embedded), dim=1)
            score_weight = F.sigmoid(torch.mm(stacked_inputs, self.decoder_gate_weight) + self.decoder_gate_bias)

            llm_entity_emb_nhis=self.entity_decoder_nhis(self.llm_entity_emb)
            llm_relation_emb_nhis=self.relation_decoder_nhis(self.llm_relation_emb)
            tensor_t = torch.full((llm_entity_emb_nhis.size(0), 1), t).to(self.gpu)
            embedding_nhis = F.normalize(llm_entity_emb_nhis)
            tensor_t = torch.full((llm_relation_emb_nhis.size(0), 1), t).to(self.gpu)
            r_emb_nhis = F.normalize(llm_relation_emb_nhis)
            e1_embedded_all = F.tanh(embedding_nhis)
            e1_embedded_nhis = e1_embedded_all[all_triples[:, 0]]
            rel_embedded_nhis = r_emb_nhis[all_triples[:, 1]]
            stacked_inputs_nhis = torch.cat((e1_embedded_nhis, rel_embedded_nhis), dim=1)
            score_weight_nhis = F.sigmoid(torch.mm(stacked_inputs_nhis, self.decoder_gate_weight_nhis) + self.decoder_gate_bias_nhis)

            score_weight_all = F.sigmoid(torch.mm(stacked_inputs, self.decoder_gate_weight_all) + self.decoder_gate_bias_all)
            #embedding=torch.add(embedding, llm_entity_emb)
            #r_emb=torch.add(r_emb, llm_relation_emb)
            #score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            #scores_ob_his = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            #scores_ob_nhis=self.decoder_ob_nhis.forward(embedding, r_emb, all_triples, mode="test")
            #print(scores_ob_his.shape)
            #core=torch.where(mask.unsqueeze(1) == 1, scores_ob_his, scores_ob_nhis)
            score_his_emb= score_weight*self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")+(1-score_weight)*self.decoder_ob1.forward(embedding1, r_emb1, all_triples, mode="test")
            score_nhis_emb = score_weight_nhis*self.decoder_ob_nhis.forward(embedding, r_emb, all_triples, mode="test")+(1-score_weight_nhis)*self.decoder_ob1_nhis.forward(embedding1, r_emb1, all_triples, mode="test")
            
            score_emb=(score_weight_all)*score_his_emb+(1-score_weight_all)*score_nhis_emb
            candidate_emb=F.tanh(embedding1)
            candidate_emb_nhis=F.tanh(embedding1_nhis)
            #score=(torch.mm(score_emb, candidate_emb.transpose(1, 0))+torch.mm(score_emb, candidate_emb_nhis.transpose(1, 0)))/2
            score=torch.mm(score_emb, candidate_emb.transpose(1, 0))
            #print(score_weight_all)
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            #print(mask)
            #print(torch.mean(torch.norm(score_weight_all,dim=1)))
            #print(torch.mean(torch.norm(score_weight,dim=1)))
            #print(torch.mean(torch.norm(score_weight_nhis,dim=1)))
            #print(torch.mean(torch.norm(embedding1,dim=1)))
            #print(torch.mean(torch.norm(embedding_nhis,dim=1)))
            #print(torch.mean(torch.norm(embedding1_nhis,dim=1)))
            #print(torch.mean(score_weight_all[mask==1]))
            #print(torch.var(score_weight_all[mask==1]))
            #print(torch.mean(score_weight_all[mask==0]))
            #print(torch.var(score_weight_all[mask==0]))
            #print(torch.cat((score_weight_all,score_weight,score_weight_nhis,mask.unsqueeze(1)),dim=1))
            if print_score:
                print(score_weight)
                print(score_weight_nhis)
                print(score_weight_all)
            return all_triples, score, score_rel,score_weight_all,score_weight_all[mask==1],score_weight_all[mask==0],score_weight,score_weight_nhis


    def get_loss(self, glist, triples,mask, static_graph, t,use_cuda):
        """
        :param glist:
        :param triplets:
        :param static_graph: 
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0]]
        #mask = [int(value) for value in mask]
        #print(mask)
        #mask=mask[0]
        #mask = torch.tensor(mask)
        inverse_mask=mask.clone()
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)
        all_triples=triples#!!!
        all_masks=torch.cat([mask,inverse_mask],dim=0)
        #print("all_triples.shape",all_triples.shape)
        #print("all_masks.shape",all_masks.shape)
        all_masks=all_masks.to(self.gpu)
        all_masks=mask#!!!
        #print(all_triples.shape)
        #print(all_masks.shape)
        #print(all_triples.shape)
        evolve_embs, static_emb, r_emb1, _, _ = self.forward(glist, static_graph, use_cuda)
        evolve_embs_nhis, static_emb, r_emb1_nhis, _, _ = self.forward(glist, static_graph, use_cuda)
        #print(len(evolve_embs))
        llm_entity_emb=self.entity_decoder(self.llm_entity_emb)
        llm_relation_emb=self.relation_decoder(self.llm_relation_emb)
        llm_entity_emb_nhis=self.entity_decoder_nhis(self.llm_entity_emb)
        llm_relation_emb_nhis=self.relation_decoder_nhis(self.llm_relation_emb)
        
        pre_emb1 = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
        pre_emb1_nhis = F.normalize(evolve_embs_nhis[-1]) if self.layer_norm else evolve_embs_nhis[-1]
        
        tensor_t = torch.full((llm_entity_emb.size(0), 1), t).to(self.gpu)
        pre_emb = F.normalize(torch.cat((llm_entity_emb, tensor_t), dim=1))
        pre_emb = F.normalize(llm_entity_emb)
        #r_emb=llm_relation_emb
        tensor_t = torch.full((llm_relation_emb.size(0), 1), t).to(self.gpu)
        r_emb = F.normalize(llm_relation_emb)
        e1_embedded_all = F.tanh(pre_emb)
        e1_embedded = e1_embedded_all[all_triples[:, 0]]
        rel_embedded = r_emb[all_triples[:, 1]]
        stacked_inputs = torch.cat((e1_embedded, rel_embedded), dim=1)
        #print(stacked_inputs.shape)
        #print(self.decoder_gate_weight.shape)
        score_weight = F.sigmoid(torch.mm(stacked_inputs, self.decoder_gate_weight) + self.decoder_gate_bias)

        tensor_t = torch.full((llm_entity_emb_nhis.size(0), 1), t).to(self.gpu)
        pre_emb_nhis = F.normalize(llm_entity_emb_nhis)
        #r_emb=llm_relation_emb
        tensor_t = torch.full((llm_relation_emb_nhis.size(0), 1), t).to(self.gpu)
        r_emb_nhis = F.normalize(llm_relation_emb_nhis)
        
        e1_embedded_all = F.tanh(pre_emb_nhis)
        e1_embedded = e1_embedded_all[all_triples[:, 0]]
        rel_embedded = r_emb[all_triples[:, 1]]
        stacked_inputs_nhis = torch.cat((e1_embedded, rel_embedded), dim=1)
        #print(stacked_inputs.shape)
        #print(self.decoder_gate_weight.shape)
        score_weight_nhis = F.sigmoid(torch.mm(stacked_inputs_nhis, self.decoder_gate_weight_nhis) + self.decoder_gate_bias_nhis)
        score_weight_all = F.sigmoid(torch.mm(stacked_inputs, self.decoder_gate_weight_all) + self.decoder_gate_bias_all)
        if self.entity_prediction:
            scores_ob_his_emb = (score_weight*self.decoder_ob.forward(pre_emb, r_emb, all_triples))+((1-score_weight)*self.decoder_ob1.forward(pre_emb1, r_emb1, all_triples))
           
            scores_ob_nhis_emb=(score_weight_nhis*self.decoder_ob_nhis.forward(pre_emb, r_emb, all_triples))+(1-score_weight_nhis)*self.decoder_ob1_nhis.forward(pre_emb1, r_emb1, all_triples)
           #print(scores_ob_his.shape)
            #scores_ob=torch.where(all_masks.unsqueeze(1) == 1, scores_ob_his, scores_ob_nhis)
            #scores_ob=mask * scores_ob_his + (1 - mask) * scores_ob_nhis
            
            score_emb=(score_weight_all)*scores_ob_his_emb+(1-score_weight_all)*scores_ob_nhis_emb
            score_emb_graph=self.decoder_ob1.forward(pre_emb1, r_emb1, all_triples)
            score_emb_llm=self.decoder_ob.forward(pre_emb, r_emb, all_triples)
            candidate_emb=F.tanh(pre_emb1)
            candidate_emb_llm=F.tanh(pre_emb)
            candidate_emb_nhis=F.tanh(pre_emb1_nhis)
            #scores_ob=((torch.mm(score_emb, candidate_emb.transpose(1, 0))+torch.mm(score_emb, candidate_emb_nhis.transpose(1, 0)))/2).view(-1, self.num_ents)
            scores_ob=torch.mm(score_emb, candidate_emb.transpose(1, 0)).view(-1, self.num_ents)
            scores_ob_graph=torch.mm(score_emb_graph,candidate_emb.transpose(1, 0)).view(-1, self.num_ents)
            scores_ob_llm=torch.mm(score_emb_llm,candidate_emb_llm.transpose(1, 0)).view(-1, self.num_ents)
            scores_ob_his=torch.mm(scores_ob_his_emb, candidate_emb.transpose(1, 0)).view(-1, self.num_ents)
            #candidate_emb_nhis=F.tanh(pre_emb1_nhis)
            scores_ob_nhis=torch.mm(scores_ob_nhis_emb, candidate_emb_nhis.transpose(1, 0)).view(-1, self.num_ents)
            scores_ob_side=torch.where(all_masks.unsqueeze(1) == 1, scores_ob_his, scores_ob_nhis)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])+self.loss_weight*self.loss_e(scores_ob_side, all_triples[:, 2])+self.loss_e(scores_ob_llm, all_triples[:, 2])
     
        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static
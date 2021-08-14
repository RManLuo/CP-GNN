#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/2 21:33
# @Author  : Raymound luo
# @Mail    : luolinhao1998@gmail.com
# @File    : CGNN.py
# @Software: PyCharm
# @Describe:

import torch
import torch.nn as nn
import dgl.function as fn
import json
import torch.nn.functional as F


class ContextGNN(nn.Module):
    def __init__(self, g, model_config):
        super(ContextGNN, self).__init__()
        self.g = g
        self.primary_type = model_config['primary_type']
        self.auxiliary_embedding = model_config['auxiliary_embedding']
        self.K_length = model_config['K_length']
        self.primary_emb = nn.Embedding(g.number_of_nodes(
            self.primary_type), model_config['embedding_dim'])
        self.enable_gru = model_config['gru']
        self.enable_add_init = model_config['add_init']
        self.ntypes = g.ntypes
        self.etypes = g.etypes
        if self.auxiliary_embedding == "emb":
            self.auxiliary_emb = nn.ModuleDict(
                {ntype: nn.Embedding(g.number_of_nodes(ntype), model_config['embedding_dim']) for ntype in g.ntypes if
                 ntype != self.primary_type})
        else:
            self.auxiliary_tans_fuc = nn.ModuleDict({
                etype: EmbTransformer(model_config['in_dim'], model_config['out_dim'], self.auxiliary_embedding) for
                etype in
                g.etypes
            })

        self.multihead_cgnn = nn.ModuleList(
            [MultiHeadCGNN(model_config['in_dim'], model_config['out_dim'],
                           model_config['num_heads'],
                           merge=model_config['merge'], g_agg_type=model_config['g_agg_type'],
                           drop_out=model_config['drop_out'],
                           cgnn_non_linear=model_config['cgnn_non_linear'],
                           multi_attn_linear=model_config['multi_attn_linear'],
                           ntypes=g.ntypes,
                           etypes=g.etypes,
                           graph_attention=model_config['graph_attention'],
                           kq_linear_out_dim=model_config['kq_linear_out_dim'],
                           path_attention=model_config['path_attention'],
                           c_linear_out_dim=model_config['c_linear_out_dim'],
                           enable_bilinear=model_config['enable_bilinear']) for hop in
             range(self.K_length + 1)  # self.K_length + 1 (self.K_length + 1) // 2 + 1
             ])
        self.loss_weight = nn.Parameter(torch.zeros(self.K_length + 1))

        if self.enable_gru:
            self.gru_gate = nn.GRUCell(
                model_config['out_dim'], model_config['out_dim'])
        self.reset_parameter()

    def reset_parameter(self):
        if self.auxiliary_embedding == "emb":
            [nn.init.normal_(emb.weight)
             for emb in self.auxiliary_emb.values()]
        else:
            [nn.init.normal_(func.linear.weight)
             for func in self.auxiliary_tans_fuc.values()]

    def embedding_transformer(self, primary_feature):
        '''
        Get HIN feature dict for each node type
        :param primary_feature: Primary Graph feature
        :return: h_dict, {ntype, feature}
        '''
        if self.auxiliary_embedding == "emb":
            h_dict = {ntype: emb.weight for ntype, emb in self.auxiliary_emb.items()}
            h_dict[self.primary_type] = primary_feature
        else:
            h_dict = {}
            non_init_graph = self.g.ntypes[:]  # Copy list
            h_dict[self.primary_type] = primary_feature
            non_init_graph.remove(self.primary_type)
            while non_init_graph:
                for srctype, etype, dsttype in self.g.canonical_etypes:
                    if srctype not in non_init_graph and dsttype in non_init_graph:
                        h_dict[dsttype] = self.auxiliary_tans_fuc[etype](self.g, h_dict[srctype], srctype, etype,
                                                                         dsttype)
                        non_init_graph.remove(dsttype)
        return h_dict

    def dump_cgnn_attention_matrix(self, path):
        '''
        Dump multi head attention matrix to json
        :param path: path to json file
        :return:
        '''
        with open(path, 'w') as f:
            multi_head_attention_matrix = {}
            for k, layer in enumerate(self.multihead_cgnn):
                hop_multi_head_matrix = layer.dump_multi_head_attention_matrix()
                multi_head_attention_matrix["length_{}".format(
                    k)] = hop_multi_head_matrix

            json.dump(multi_head_attention_matrix, f)

    def forward(self, k):
        '''

        :param k: k-hop
        :return: K_length feature emb h_feature of primary type
        '''
        h_dict = self.embedding_transformer(
            self.primary_emb.weight)  # Generate emb for auxiliary graph
        # KCI
        # k = k // 2 + 1
        for hop in range(k):
            new_h_dict = self.multihead_cgnn[hop](
                self.g, h_dict)  # multi head CGNN

            if self.enable_gru:  # Enable Gru
                new_primary_feature = self.gru_gate(new_h_dict[self.primary_type],
                                                    h_dict[self.primary_type])  # GRU gate for primary graph
                new_h_dict[self.primary_type] = new_primary_feature
            if self.enable_add_init:
                new_h_dict[self.primary_type] += self.primary_emb.weight
            h_dict = new_h_dict
        return h_dict[self.primary_type]

    def _context_score(self, src, dst, p_emb, context_emb):
        logits = torch.sum((p_emb[src] * context_emb[src])
                           * (p_emb[dst] * context_emb[dst]), dim=1)
        return logits

    def get_loss(self, k_hop, pos_src, pos_dst, neg_src, neg_dst, p_emb, p_context_emb):
        # k_length = k_hop // 2
        k_length = k_hop - 1
        pos_score = self._context_score(pos_src, pos_dst, p_emb, p_context_emb)
        neg_score = self._context_score(neg_src, neg_dst, p_emb, p_context_emb)
        # pos_score = torch.clamp(pos_score, max=10, min=-10)
        pos_loss = -F.logsigmoid(pos_score).view(-1).mean()
        # neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_loss = -F.logsigmoid(-neg_score).view(-1).mean()
        # Multi loss
        weight = torch.exp(-self.loss_weight[k_length])
        return (pos_loss + neg_loss) * weight + self.loss_weight[
            k_length]  # torch.mean(torch.cat((pos_score, neg_score)))


class EmbTransformer(nn.Module):
    def __init__(self, in_feat, out_feat, apply_linear='linear'):
        super(EmbTransformer, self).__init__()
        self.linear = nn.Linear(in_feat, out_feat)
        self.apply_linear = apply_linear

    def forward(self, g, src_h, srctype, etype, dsttype):
        g.nodes[srctype].data['h'] = src_h
        g[etype].update_all(fn.copy_src('h', 'm'), fn.sum(msg='m', out='h'))
        if self.apply_linear == 'non_linear':
            return torch.relu(self.linear(g.nodes[dsttype].data['h']))
        else:
            return self.linear(g.nodes[dsttype].data['h'])


class MultiHeadCGNN(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='linear', g_agg_type="mean", drop_out=0.,
                 cgnn_non_linear=False,
                 multi_attn_linear=False, ntypes=None, etypes=None, graph_attention=True, kq_linear_out_dim=None,
                 path_attention=True,
                 c_linear_out_dim=None, enable_bilinear=False):
        super(MultiHeadCGNN, self).__init__()
        self.heads = nn.ModuleList()
        cgnn_out_dim = out_dim // num_heads if merge == 'stack' or merge == 'linear' else out_dim
        for i in range(num_heads):
            self.heads.append(
                CGNNLayer(in_dim, cgnn_out_dim, g_agg_type, drop_out, cgnn_non_linear,
                          multi_attn_linear, ntypes, etypes, graph_attention, kq_linear_out_dim, path_attention,
                          c_linear_out_dim, enable_bilinear))
        self.merge = merge
        # W_2
        self.cgnn_non_linear = cgnn_non_linear
        if merge == 'linear':
            self.fc = nn.Linear(out_dim, out_dim)
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight)

    def dump_multi_head_attention_matrix(self):
        '''
        Reutrn attention matrix of each head
        :return: multi_head_matrix_dict {h_head, attention matrix of head h}
        '''
        multi_head_matrix_dict = {}
        for h, head in enumerate(self.heads):
            head_matrix = head.dump_attention_matrix()
            multi_head_matrix_dict['head_{}'.format(h)] = head_matrix
        return multi_head_matrix_dict

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        h_dict = {}
        if self.merge == 'linear':
            # concat on the output feature dimension (dim=1)
            for ntype in g.ntypes:
                n_feature = torch.cat([feature[ntype]
                                       for feature in head_outs], dim=1)
                h_dict[ntype] = self.fc(n_feature)
        elif self.merge == 'mean':
            # merge using average
            for ntype in g.ntypes:
                n_feature = torch.stack([feature[ntype]
                                         for feature in head_outs], dim=1)
                h_dict[ntype] = torch.mean(n_feature, dim=1)
        elif self.merge == 'stack':
            for ntype in g.ntypes:
                n_feature = torch.cat([feature[ntype]
                                       for feature in head_outs], dim=1)
                h_dict[ntype] = n_feature
        return h_dict


class CGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, g_agg_type="mean", drop_out=0., cgnn_non_linear=False,
                 multi_attn_linear=False,
                 ntypes=None, etypes=None, graph_attention=True, kq_linear_out_dim=None, path_attention=True,
                 c_linear_out_dim=None, enable_bilinear=False):
        super(CGNNLayer, self).__init__()
        self.graph_attention_matrix = {}
        self.g_agg_type = g_agg_type  # Graph aggregation type
        self.multi_attn_linear = multi_attn_linear
        self.cgnn_non_linear = cgnn_non_linear
        self.graph_attention = graph_attention
        self.path_attention = path_attention
        self.enable_bilinear = enable_bilinear
        if self.multi_attn_linear:
            self.k_linear = nn.ModuleDict()
            self.q_linear = nn.ModuleDict()
            self.c_linear = nn.ModuleDict()
            for ntype in ntypes:
                self.k_linear[ntype] = nn.Linear(
                    in_dim, kq_linear_out_dim, bias=False)
                self.q_linear[ntype] = nn.Linear(
                    in_dim, kq_linear_out_dim, bias=False)
                self.c_linear[ntype] = nn.Linear(
                    in_dim, c_linear_out_dim, bias=False)
        else:
            # K-linear
            self.k_linear = nn.Linear(in_dim, kq_linear_out_dim, bias=False)
            # Q-linear
            self.q_linear = nn.Linear(in_dim, kq_linear_out_dim, bias=False)
            self.c_linear = nn.Linear(in_dim, c_linear_out_dim, bias=False)
        # W_1
        if self.cgnn_non_linear:
            self.fc = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU()
            )
        else:
            self.fc = nn.Linear(in_dim, out_dim)
        self.node_dropout = nn.Dropout2d(drop_out)
        self.attn_fc = nn.Linear(2 * c_linear_out_dim, 1, bias=False)
        if enable_bilinear:
            self.bilinear = nn.ModuleDict()
            for etype in etypes:
                self.bilinear[etype] = nn.Bilinear(kq_linear_out_dim, kq_linear_out_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.multi_attn_linear:
            for linear in self.k_linear.values():
                nn.init.xavier_normal_(linear.weight)
            for linear in self.q_linear.values():
                nn.init.xavier_normal_(linear.weight)
        else:
            nn.init.xavier_normal_(self.k_linear.weight)
            nn.init.xavier_normal_(self.q_linear.weight)
        if self.cgnn_non_linear:
            nn.init.xavier_normal_(self.fc[0].weight)
        else:
            nn.init.xavier_normal_(self.fc.weight)

    def graph_aggregation(self, h):
        '''
        How to aggregate graph information
        :param h: Node embedding (N, D)
        :return: Graph representation (D)
        '''
        if self.g_agg_type == "mean":
            return torch.mean(h, dim=0)
        elif self.g_agg_type == "sum":
            return torch.sum(h, dim=0)
        return

    def dump_attention_matrix(self):
        '''
        Return the attention matrix of each hop
        :return: graph_attention_matrix attention_matrix {etype, attention score}
        '''
        return self.graph_attention_matrix

    def context_edge_attention(self, edges):
        # s_ij = W(h_i||h_j)
        z = torch.cat((edges.src['C_h'], edges.dst['C_h']), dim=1)
        c_A = self.attn_fc(z)
        # c_A = torch.sum(edges.src['Q_h'] * edges.dst['K_h'], dim=1) / (edges.dst['K_h'].shape[-1] ** (1 / 2))
        return {'c_A': c_A.squeeze(-1)}

    def context_attention(self, g):
        '''
        Context edges attention score
        :param g:
        :return: g with k,q feature for each node
        '''
        for ntype in g.ntypes:
            if self.multi_attn_linear:
                g.nodes[ntype].data['C_h'] = self.c_linear[ntype](
                    g.nodes[ntype].data['h'])
            else:
                g.nodes[ntype].data['C_h'] = self.c_linear(
                    g.nodes[ntype].data['h'])
        for etype in g.etypes:
            g.apply_edges(self.context_edge_attention, etype=etype)
        return g

    def msg_func(self, edges):
        return {'h': edges.src['h'], 'C_h': edges.src['C_h'], 'g_A': edges.data['g_A'], 'c_A': edges.data['c_A']}

    def reduce_func(self, nodes):
        # S_j
        # sim_matrix = torch.tanh(torch.bmm(nodes.mailbox['C_h'], nodes.mailbox['C_h'].transpose(1, 2)) / (
        #         nodes.mailbox['C_h'].shape[-1] ** (1 / 2)))
        # sim_J = torch.mean(sim_matrix, dim=2)
        # \sigma(s_ij-S_j)
        e = F.leaky_relu(nodes.mailbox['c_A'])  # nodes.mailbox['c_A'] - sim_J
        alpha = F.softmax(e, dim=1).unsqueeze(-1)
        h = nodes.mailbox['g_A'][:,
            0].unsqueeze(-1) * torch.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        '''
        CGNN
        :param g: graph
        :param h: feature dict {ntype, feature}
        :return: h: feature dict {ntype, feature}
        '''
        # The input is a dictionary of node features for each type
        funcs = {}
        graph_feature = {}
        for ntype in g.ntypes:
            h_feat = h[ntype].unsqueeze(0)  # (1, N , D)
            feat = self.node_dropout(h_feat)
            feat = feat.squeeze(0)  # (N, D)
            graph_feature[ntype] = self.graph_aggregation(
                feat)  # Graph Aggregation
            g.nodes[ntype].data['h'] = feat

        # Context Edges attention:
        if self.path_attention:
            g = self.context_attention(g)

        # Graph Attention
        if self.graph_attention:
            l_hop_graph_attention = {}
            for q_dsttype in g.ntypes:
                context_feature = []
                if self.multi_attn_linear:
                    q_h = self.q_linear[q_dsttype](graph_feature[q_dsttype])
                else:
                    q_h = self.q_linear(graph_feature[q_dsttype])  # k linear

                # Compute attention score for each other graph
                for k_srctype, k_etype, k_dsttype in g.canonical_etypes:  # Score only exists when two graphs are connected
                    if q_dsttype != k_dsttype:
                        continue
                    if self.multi_attn_linear:
                        k_h = self.k_linear[k_srctype](
                            graph_feature[k_srctype])
                    else:
                        k_h = self.k_linear(
                            graph_feature[k_srctype])  # q linear
                    # A = g.number_of_edges(q_etype)  # V
                    if self.enable_bilinear:
                        z = self.bilinear[k_etype](k_h, q_h)
                    else:
                        z = torch.sum(k_h * q_h) / \
                            (k_h.shape[-1] ** (1 / 2))  # ATThead
                    context_feature.append((k_etype, z))
                c_feature = torch.stack(
                    [score for _, score in context_feature])
                c_attention_matrix = torch.softmax(c_feature, dim=0)
                for i, (etype, score) in enumerate(context_feature):
                    att_score = c_attention_matrix[i]
                    g.edges[etype].data['g_A'] = att_score.expand(
                        g.number_of_edges(etype))  # Bind attention score to edge feature
                    l_hop_graph_attention[etype] = float(
                        c_attention_matrix[i].detach().cpu().numpy())
            self.graph_attention_matrix = l_hop_graph_attention

        # h: source node feature, g_A: Graph attention score between two graphs, c_A: Attentnion score between two nodes, m: result
        for etype in g.etypes:
            if self.path_attention:
                funcs[etype] = (self.msg_func, self.reduce_func)
            elif self.graph_attention:
                funcs[etype] = (fn.u_mul_e('h', 'g_A', 'm'), fn.sum('m', 'h'))
            else:
                funcs[etype] = (fn.copy_src('h', 'm'), fn.sum('m', 'h'))
        g.multi_update_all(funcs, 'sum')
        return {ntype: self.fc(g.nodes[ntype].data['h']) for ntype in g.ntypes}

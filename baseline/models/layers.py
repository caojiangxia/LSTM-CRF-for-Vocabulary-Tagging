import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

from utils import torch_utils




class SubjBaseModel(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """
    
    def __init__(self, opt, input_size, filters_num, filter=3):
        super(SubjBaseModel, self).__init__()
        self.input_size = input_size
        self.conv_subj = nn.Conv2d(1, filters_num, (filter, input_size), padding=(int(filter/2), 0))
        self.linear_subj_start = nn.Linear(filters_num, 1)
        self.linear_subj_end = nn.Linear(filters_num, 1)
        self.init_weights()


    def init_weights(self):
        nn.init.xavier_uniform(self.conv_subj.weight)
        nn.init.uniform(self.conv_subj.bias)
        self.linear_subj_start.bias.data.fill_(0)
        init.xavier_uniform(self.linear_subj_start.weight, gain=1) # initialize linear layer

        self.linear_subj_end.bias.data.fill_(0)
        init.xavier_uniform(self.linear_subj_end.weight, gain=1) # initialize linear layer



    def forward(self, hidden, sentence_rep, masks):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """
        batch_size, seq_len, input_size = hidden.shape
        subj_inputs = torch.cat([hidden, seq_and_vec(seq_len,sentence_rep)], dim=2)

        batch_size, seq_len, input_size = subj_inputs.size()
        x = subj_inputs.unsqueeze(1)
        subj_outputs = torch.transpose(F.relu(self.conv_subj(x)).squeeze(3), 1, 2)
        # print(subj_outputs.shape)
        subj_start_logits = F.sigmoid(self.linear_subj_start(subj_outputs))
        subj_end_logits = F.sigmoid(self.linear_subj_end(subj_outputs))
        return subj_start_logits.squeeze(-1), subj_end_logits.squeeze(-1)


class ObjBaseModel(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """
    
    def __init__(self, opt, input_size, filters_num, filter=3):
        super(ObjBaseModel, self).__init__()
        self.input_size = input_size
        self.conv_obj = nn.Conv2d(1, filters_num, (filter, input_size), padding=(int(filter/2), 0))
        self.linear_obj_start = nn.Linear(filters_num, opt['num_class']+1)
        self.linear_obj_end = nn.Linear(filters_num, opt['num_class']+1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.conv_obj.weight)
        nn.init.uniform(self.conv_obj.bias)
        self.linear_obj_start.bias.data.fill_(0)
        init.xavier_uniform(self.linear_obj_start.weight, gain=1) # initialize linear layer

        self.linear_obj_end.bias.data.fill_(0)
        init.xavier_uniform(self.linear_obj_end.weight, gain=1) # initialize linear layer





    def forward(self, hidden, sentence_rep, subj_start_label, subj_end_label, masks):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """
        batch_size, seq_len, input_size = hidden.shape
        # print(subj_end_label)
        # print(hidden.shape)
        subj_start_hidden = torch.gather(hidden, dim=1, index=subj_start_label.unsqueeze(2).repeat(1,1,input_size)).squeeze(1)
        
        subj_end_hidden = torch.gather(hidden, dim=1, index=subj_end_label.unsqueeze(2).repeat(1,1,input_size)).squeeze(1)
        obj_inputs = torch.cat([hidden, seq_and_vec(seq_len,sentence_rep), seq_and_vec(seq_len,subj_start_hidden), seq_and_vec(seq_len,subj_end_hidden)], dim=2)

        batch_size, seq_len, input_size = obj_inputs.size()
        x = obj_inputs.unsqueeze(1)
        obj_outputs = torch.transpose(F.relu(self.conv_obj(x)).squeeze(3), 1, 2)
        obj_start_logits = self.linear_obj_start(obj_outputs)
        obj_end_logits = self.linear_obj_end(obj_outputs)
        return obj_start_logits, obj_end_logits

def seq_and_vec(seq_len, vec):
    return vec.unsqueeze(1).repeat(1,seq_len,1)

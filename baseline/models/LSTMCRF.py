"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
from utils import torch_utils
from models import layers
from models.crf import CRF

class LSTMCRF(object):
    """ A LSTMCRF model for vocabulary tagging """

    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.model = BiLSTM(opt, emb_matrix)
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = Variable(torch.LongTensor(batch[0]).cuda())
            label = Variable(torch.LongTensor(batch[1]).cuda())
        else:
            inputs = Variable(torch.LongTensor(batch[0]))
            label = Variable(torch.LongTensor(batch[1]))

        mask = (inputs.data > 0).long() # padding !!!!!!
        # step forward
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.model(inputs, label, mask)


        #loss = self.criterion(model_predict_logits.view(-1,self.opt["num_class"]), label.view(-1)).view_as(mask)

        loss = torch.sum(loss.mul(mask.float())) / torch.sum(mask.float())

        # backward
        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict_per_instance(self, words):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        if self.opt['cuda']:
            words = Variable(torch.LongTensor(words).cuda())
        else:
            words = Variable(torch.LongTensor(words))

        mask = (words.data > 0).long()
        # forward
        self.model.eval()
        hidden = self.model.based_encoder(words, mask)
        _, tag_seq = self.model.CRF(hidden, mask)
        return list(tag_seq[0].data.cpu().numpy())

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']


class BiLSTM(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(BiLSTM, self).__init__()
        self.drop = nn.Dropout(opt['dropout'])
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=0)

        self.input_size = opt['emb_dim']
        self.rnn = nn.LSTM(self.input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True, \
                           dropout=opt['dropout'], bidirectional=True)
        self.CRF=SubjCRFModel(opt,2*opt["hidden_dim"],opt["num_class"])
        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()

    def init_weights(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)  # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)

        # decide finetuning
        if self.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.topn < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.topn))
            self.emb.weight.register_hook(lambda x: \
                                              torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Finetune all embeddings.")

    def zero_state(self, batch_size):
        state_shape = (2 * self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0

    def zero_state_last(self, batch_size):
        state_shape = (2 * self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0

    def based_encoder(self, words, mask):
        if mask.shape[0] == 1:
            seq_lens = [mask.sum(1).squeeze()]
        else:

            seq_lens = list(mask.sum(1).squeeze())

        batch_size = words.size()[0]

        # embedding lookup
        word_inputs = self.emb(words)
        inputs = [word_inputs]
        inputs = self.drop(torch.cat(inputs, dim=2))  # add dropout to input
        input_size = inputs.size(2)

        try:
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
            h0, c0 = self.zero_state(batch_size)
            hidden, (ht, ct) = self.rnn(inputs, (h0, c0))
            hidden, output_lens = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)
        except Exception as e:
            print(seq_lens)

        #hidden_masked = hidden - ((1 - mask) * 1e10).unsqueeze(2).repeat(1, 1, hidden.shape[2]).float()
        #sentence_rep = F.max_pool1d(torch.transpose(hidden_masked, 1, 2), hidden_masked.size(1)).squeeze(2)
        return hidden

    def forward(self, inputs, label , mask):
        words = inputs  # unpack
        hidden = self.based_encoder(words, mask)
        predict_logits=self.CRF.neg_log_likelihood_loss(hidden,label,mask.byte())
        return predict_logits


class SubjCRFModel(nn.Module):
    def __init__(self, opt, input_size, label_size):
        super(SubjCRFModel, self).__init__()
        self.crf = CRF(label_size, True, average_batch=True)
        self.hidden2tag = nn.Linear(input_size, label_size + 2)
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.init_weights()

    def init_weights(self):
        self.hidden2tag.bias.data.fill_(0)
        init.xavier_uniform_(self.hidden2tag.weight, gain=1)  # initialize linear layer

    def neg_log_likelihood_loss(self, hidden, batch_label, mask):
        batch_size = hidden.size(0)
        seq_len = hidden.size(1)
        output_score = self.hidden2tag(hidden)
        total_loss = self.crf.neg_log_likelihood_loss(output_score, mask, batch_label)
        # scores, tag_seq = self.crf._viterbi_decode(hidden, mask)
        return total_loss

    def forward(self, hidden , mask):
        batch_size = hidden.size(0)
        seq_len = hidden.size(1)
        output_score = self.hidden2tag(hidden)
        scores, tag_seq = self.crf._viterbi_decode(output_score, mask)
        return tag_seq

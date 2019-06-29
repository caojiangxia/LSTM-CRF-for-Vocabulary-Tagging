"""
Train a model on TACRED.
"""

import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import json
from utils.loader import DataLoader
from models.LSTMCRF import LSTMCRF
from utils import helper, score
# from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/dataset')
parser.add_argument('--vocab_dir', type=str, default='data/vocab')
parser.add_argument('--emb_dim', type=int, default=128, help='Word embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=64, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
parser.add_argument('--dropout', type=float, default=0.25, help='Input and RNN dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
parser.add_argument('--subj_loss_weight', type=float, default=2.5, help='Only finetune top N embeddings.')
parser.add_argument('--lr', type=float, default=1.0, help='Applies to SGD and Adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=5, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
parser.add_argument('--seed', type=int, default=43)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()



torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)


# make opt
opt = vars(args)

# load data
train_data = json.load(open(opt['data_dir'] + '/train_me.json'))
dev_data = json.load(open(opt['data_dir'] + '/verify_me.json'))
id2char, char2id = json.load(open(opt['data_dir'] + '/all_chars_me.json'))
id2tag, tag2id = json.load(open(opt['data_dir'] + '/all_tags_me.json'))
id2char = {int(i):j for i,j in id2char.items()}
id2tag = {int(i):j for i,j in id2tag.items()}
opt['num_class'] = len(id2tag)
opt['vocab_size'] = len(char2id)+2
# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(train_data, tag2id,char2id, opt['batch_size'])
dev_batch = DataLoader(dev_data, tag2id, char2id, opt['batch_size'])

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)


# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\dev_p\tdev_r\tdev_f1")
# print model info
helper.print_config(opt)

# model
model = LSTMCRF(opt, emb_matrix=None)

if 0:
    model_file = 'saved_models/00/best_model.pt'
    model.load(model_file)

dev_f1_history = []
current_lr = opt['lr']


global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']


# start training
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss = model.update(batch)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))



    # eval on dev
    print("Evaluating on dev set...")
    dev_f1, dev_p, dev_r = score.evaluate(dev_data, char2id, id2predicate, model)
    
    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    print("epoch {}: train_loss = {:.6f}, dev_p = {:.6f}, dev_r = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
            train_loss, dev_p, dev_r, dev_f1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_p, dev_r, dev_f1))

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    model.save(model_file, epoch)
    if epoch == 1 or dev_f1 > max(dev_f1_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)
    
    # lr schedule
    if len(dev_f1_history) > 10 and dev_f1 <= dev_f1_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad']:
        current_lr *= opt['lr_decay']
        model.update_lr(current_lr)

    dev_f1_history += [dev_f1]
    print("")

print("Training ended with {} epochs.".format(epoch))


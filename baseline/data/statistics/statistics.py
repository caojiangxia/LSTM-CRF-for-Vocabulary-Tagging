import json
import numpy as np
np.set_printoptions(4)
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import  MultipleLocator

from matplotlib.ticker import  FormatStrFormatter

def list_to_distribution(lis, sort=True):
    dis = defaultdict(int)
    for item in lis:
        dis[str(item)] += 1
    if sort:
        return sorted(dis.items(),key=lambda x:int(x[0]))
    else:
        return dis   

def plot(distribution, xlabel, ylabel, title, xmajor, ymajor):
    plt.figure()
    xmajorLocator = MultipleLocator(xmajor)
    ymajorLocator = MultipleLocator(ymajor)
    ax = plt.subplot(111)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    plt.plot([int(item[0]) for item in distribution],[item[1] for item in distribution],'b',lw = 1.5) # 蓝色的线
#     plt.plot([item[1] for item in distribution],'ro') #离散的点
    plt.grid(True)
    plt.axis('tight')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(title+'.png', format='png')



def analysis_of_dataset(dataset_path):
    lens_list = []
    spo_num_list = []
    entitis_num_list = []
    relations_list = []
    entities_list = []
    max_entity = 0
    num = 0
    with open(dataset_path, 'r') as f:
        for line in f.readlines():
            instance = json.loads(line)
            postag = instance['postag']
            text = instance['text']
            spo_list = instance['spo_list']
            lens_list.append(len(postag))
            spo_num_list.append(len(spo_list))
            entitis_num_list.append(len(set([spo['object'] for spo in spo_list]+[spo['subject'] for spo in spo_list])))
            relations_list.extend([spo['predicate'] for spo in spo_list])
            entities_list.extend([spo['object_type'] for spo in spo_list]+[spo['subject_type'] for spo in spo_list])
            curr_word_list = []
            for i in postag:
                curr_word_list.append(i['word'])
            
            
            curr_entity_list = [spo['object'] for spo in spo_list]+[spo['subject'] for spo in spo_list]
            for i in curr_entity_list:
                if i not in curr_word_list:
                    num += 1
        
            
        print('{0}中，句子的平均长度是{1:.2f}词，平均实体数是{3:.2f}，平均关系数是{2:.2f}，最大长度是{4}词，最大实体数是{5}，最大关系数是{6}'.format(
                dataset_path, np.mean(lens_list), np.mean(spo_num_list), np.mean(entitis_num_list),
                max(lens_list), max(spo_num_list), max(entitis_num_list),
                ))
        print('{0}中，句子的长度分布是{1}'.format(
                dataset_path, list_to_distribution(lens_list)
                ))
        plot(list_to_distribution(lens_list),'sentence length', 'number', 'lenghth distribution of sentence',10,100)
        print('{0}中，句子的实体数分布是{1}'.format(
                dataset_path, list_to_distribution(entitis_num_list)
                ))
        plot(list_to_distribution(entitis_num_list),'entities exist in this sentence', 'number', 'entities number distribution of sentence',1,500)

        print('{0}中，句子的关系数分布是{1}'.format(
                dataset_path, list_to_distribution(spo_num_list)
                ))
        plot(list_to_distribution(spo_num_list),'relations exist in this sentence', 'number', 'relations number distribution of sentence',1,500)

        print('{0}中，关系的分布是{1}'.format(
                dataset_path, list_to_distribution(relations_list, sort=False)
                ))
        print('{0}中，实体的分布是{1}'.format(
                dataset_path, list_to_distribution(entities_list, sort=False)
                ))
        # print(len(dataset))
        print(num)

if __name__ == '__main__':
    dev_path = 'dev_data.json'
    train_path = 'train_data.json'
    analysis_of_dataset(dev_path)
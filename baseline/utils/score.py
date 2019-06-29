import numpy as np
from tqdm import tqdm




def evaluate(data, char2id, id2predicate, model):
    acc=0
    all=0
    for d in tqdm(iter(data)):
        pre=model.predict_per_instance(d[0])
        ans=np.array(d[1])
        acc=np.in1d(pre, ans).sum()
        all+=len(ans.reshape(-1))
    return
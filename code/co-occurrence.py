import argparse
import os
import re
import numpy as np
import statistics
import json

from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
from nltk.corpus import stopwords


def normalize(st):
    
    st = st.lower()
    st = re.sub('([.,!?()])', r' \1 ', st)
    st = re.sub('  ',' ', st)
    return st

def add_stop(utt):
    
    if utt[-1] not in '.,!:;?':
        utt = utt + ' .'
    return utt

def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "dialog_dstc_2") #when pushing to git we need to add a download script here
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default=source_dir)
    parser.add_argument("--mat_type", default='freq')

    args = parser.parse_args()
    return args

def get_lines(fname):
    with open(fname,'r') as fp:
        d=fp.readlines()

    lines=[]
    for i in d:
        lines.append(normalize(i))    

    return lines

def remove_num(line):
    l = re.sub('\d+','',line,1).strip()
    ut_resp = l.split('\t')
    
    resp=""
    ut = add_stop(ut_resp[0])
    if len(ut_resp)>1:
        resp = add_stop(ut_resp[1])
    
    return ut+' '+resp 

def remove_triples(lines):
    
    rels=['R_post_code','R_cuisine','R_location','R_phone','R_address','R_price','R_rating']
    
    new_lines=[]
    for line in lines:
        if '\t' in line:
            new_lines.append(remove_num(line))
        elif 'api_call' in line:
            new_lines.append(remove_num(line))
    
    return new_lines

def stringify(lines):
    
    s=""
    for line in lines:
        s = s+" "+line.strip()
    return s


def get_utterance_data(datasets):
    
    all_lines=[]
    for dataset in datasets:
        lines = get_lines(dataset)
        lines = remove_triples(lines)
        all_lines.extend(lines)
    
    all_data = stringify(all_lines)
    all_data = re.sub('  ',' ', all_data)
    return all_data.strip()

def get_freq_mat(corpus):
    
    sparse_matrix = defaultdict(lambda: defaultdict(lambda: 0))
    for sent in sent_tokenize(corpus):
        words = sent.split()
        for word1 in words:
            for word2 in words:
                sparse_matrix[word1][word2]+=1
    
    return sparse_matrix

    
def dense_mat(sparse_matrix):
    
    v_size=len(sparse_matrix)
    
    w2idx={}
    for i,k in enumerate(sparse_matrix.keys()):
        w2idx[k]=i
    
    dense_matrix = np.zeros((v_size, v_size))
    for k in sparse_matrix.keys():
        for k2 in sparse_matrix[k].keys():
            dense_matrix[w2idx[k]][w2idx[k2]] = sparse_matrix[k][k2]
    
    return dense_matrix,w2idx

def Sparsity(mat):
    
    size = mat.shape[0]*mat.shape[0]
    nonzeros = np.count_nonzero(mat)
    zeros = size-nonzeros
    return float(zeros/size)

def remove_stopwords_freq(mat,w2idx):
    
    stops = sorted(set(stopwords.words('english')))
    common_stop=[]
    
    for w in list(w2idx.keys()):
        if w in stops:
            common_stop.append(w)
    
#    common_stop.extend(['.',',',';','?','restaurant','food','town']) #en
    common_stop.extend(['.',',','?','restaurant','food','town'])
    
    ids=[]
    for w in common_stop:
        ids.append(w2idx[w])
    
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            if i in ids or j in ids:
                mat[i][j]=0
    
    return mat

def find_median(mat):
    
    nz_inds = np.nonzero(mat)
     
    freqs=[]
     
    for i in range(len(nz_inds[0])):
         freqs.append(mat[nz_inds[0][i]][nz_inds[1][i]])
     
        
    return statistics.median(freqs)


def wconnection_list(mat,w2idx):
    
    idx2w={}
    
    for k,v in w2idx.items():
        idx2w[v]=k
    
    edges=defaultdict(list)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i!=j:
                if mat[i][j]!=0:
                    edges[idx2w[i]].append(idx2w[j])
    
    return edges

def convert_ppmi(mat):
    
    row_sum = np.sum(mat,axis=1)
    col_sum = np.sum(mat,axis=0)
    total = np.sum(np.sum(mat,axis=0),axis=0)
    
    pij = np.divide(mat,total)
    pi = np.divide(col_sum,total)
    pj = np.divide(row_sum,total)
    
    pipj = np.multiply(pi,pj)
    pipj += 1e-10
    ar = np.divide(pij,pipj)
    ar += 1e-10
    logar = np.log2(ar)
    logar[logar<0]=0
    
    return logar

def main():
    #print(a)
    args = get_args()
    source_dir = args.source_dir 
    target_dir = args.target_dir

    source_fname = source_dir+ '/dialog-dstc2-'    
    target_fname = target_dir+ '/edge-dict-'+str(args.mat_type)+'-dstc2.json'
    
    train_input = source_fname+ 'trn.txt'
    test_input = source_fname+ 'tst.txt'
    dev_input = source_fname+ 'dev.txt'
    
    corpus = get_utterance_data([train_input,test_input,dev_input])
    co_occurrence_dic = get_freq_mat(corpus)
    co_occurrence_mat,w2idx = dense_mat(co_occurrence_dic)
    co_occurrence_mat = remove_stopwords_freq(co_occurrence_mat,w2idx)
    co_occurrence_mat = convert_ppmi(co_occurrence_mat)
    freq_median = find_median(co_occurrence_mat)
    co_occurrence_mat[co_occurrence_mat<freq_median]=0
    edge_dict = wconnection_list(co_occurrence_mat,w2idx)
    
    with open(target_fname,'w+') as fp:
        json.dump(edge_dict,fp)
    
    print("Sparsity : ",Sparsity(co_occurrence_mat))
    
if __name__ == "__main__":
    main()
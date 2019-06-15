import pickle
import re
import os
import argparse
import json
import pandas as pd
import csv
import numpy as np
import spacy
import random
from tqdm import tqdm
from spacy.tokens import Doc
random.seed(1234)

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en',disable=['tagger','ner'],vectors=False)
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def get_args():
    """
    get arguments for preprocessing:
        source_dir: The directory where raw dialogue files are present
        target_dir: The directory where the preprocessed files are gonna be dumped
    """
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "dialog_dstc_2") #when pushing to git we need to add a download script here
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default=source_dir)
    args = parser.parse_args()
    return args

eos = '<EOS>'
beg = '<beg>'
eok = '<EOK>'
def count_kb(content):
    total_api_calls = len(re.findall('\tapi_call',content))
    failed_api_calls = len(re.findall('api_call no result',content))
    return total_api_calls - failed_api_calls

def locate_kb(content):
    
    kb_start_found = False
    start_index = []
    end_index = []
    #kb_counter = count_kb
    
    for turn_no, current_turn in enumerate(content):
        if " r_post_code " in current_turn and not kb_start_found:
            kb_start_found = True
            start_index.append(turn_no)
        if kb_start_found:
            if "<silence>" in current_turn:
                end_index.append(turn_no) 
                kb_start_found = False

    start_index.append(len(content)) 
    return start_index,end_index

def normalize(st):
    """
    Normalize the utterance strings
    """
    st = st.lower()
    st = re.sub('([.,!?()])', r' \1 ', st)
    st = re.sub('  ',' ', st)
    return st

def process_kb(given_kb):
    """
    Return KB Triples
    """
    processed_kb = []
    for i in given_kb:
        processed_kb = processed_kb + [re.sub('\d+','',i,1).strip().split(' ')]
    if processed_kb:
        return processed_kb # only restaurant name is being returned if we are using processed_kb[0]
    return None

def get_kbedgelist(kb,rel2id):
    """
    Create a knowledge graph from the KB triples:
        kb: KB triples
        rel2id: A dictionary which has KB entity relations as keys and unique IDs for relations as values
    """
    
    local_ents=[]
    for triple in kb:
        local_ents.append(triple[0])
        local_ents.append(triple[2])
    
    local_ents_set = sorted(set(local_ents))
    
    
    ent2id={}
    for i,ent in enumerate(local_ents_set):
        ent2id[ent]=i
    
    kb_edges=[]
    for triple in kb:
        kb_edges.append([ent2id[triple[0]],ent2id[triple[2]],rel2id[triple[1]]])
    
    return local_ents_set,kb_edges

def get_graph(data_dics):
    """
    Tokenize, create graphs and vocabularies:
        data_dics: The train, test and valid data dictionaries obtained from append_context() function
    """
    all_deps=[]
    all_dics=[]
    all_tokens=[]
    
    rel2id={'address':0,
            'phone':1,
            'pricerange':2,
            'area':3,
            'food':4,
            'postcode':5,
            'name':6
            }
    
    for data_dic in data_dics: 
        deEdges=[]
        kbEdges=[]
        kbEnts=[]
        kb_ent_lens=[]
        context_tokens=[]
        query_edges=[]
        query_tokens=[]
        context_lens=[]
        q_lens=[]
        dec_lens=[]
        dec_ips=[]
        dec_targs=[]
        for i in tqdm(range(len(data_dic['context_string']))):
            
            dec_ip = data_dic['dec_ip'][i].strip().split()
            dec_ips.append(dec_ip)
            
            dec_targ = data_dic['dec_targ'][i].strip().split()
            dec_targs.append(dec_targ)
            all_tokens.extend(dec_targ)

            doc = nlp(data_dic['context_string'][i].strip())
            context_lens.append(len(doc))
            edges=[]
            c_tokens=[]
            for token in doc:
                edges.append([token.i,token.head.i,token.dep_])
                c_tokens.append(token.text)
                all_deps.append(token.dep_)
                all_tokens.append(token.text)
                
            q_tokens=[]
            q_edges=[]
            q_doc = nlp(data_dic['query_string'][i].strip())
            q_lens.append(len(q_doc))
            for q_token in q_doc:
                q_edges.append([q_token.i,q_token.head.i,q_token.dep_])
                q_tokens.append(q_token.text)
                all_deps.append(q_token.dep_)
                all_tokens.append(q_token.text)
            
            for triple in data_dic['kb'][i]:
                all_tokens.extend([triple[0],triple[1],triple[2]])
            
            deEdges.append(edges)
            context_tokens.append(c_tokens)
            query_edges.append(q_edges)
            query_tokens.append(q_tokens)
            kbent, kbedge = get_kbedgelist(data_dic['kb'][i],rel2id)
            kbEdges.append(kbedge)
            kbEnts.append(kbent)
            kb_ent_lens.append(len(kbent))
            dec_lens.append(len(data_dic['dec_ip'][i].strip().split()))
        final_dic={
            'context_deEdges':deEdges,
            'context_tokens':context_tokens,
            'context_lens':context_lens,
            'kbEdges':kbEdges,
            'kbEnts':kbEnts,
            'kbEnts_len':kb_ent_lens,
            'q_edges':query_edges,
            'q_tokens':query_tokens,
            'q_lens':q_lens,
            'dec_ip':dec_ips,
            'dec_targ':dec_targs,
            'dec_lens':dec_lens
            }
        all_dics.append(final_dic)
        
    all_deps_set = sorted(set(all_deps))
    
    dep2id={}
    for i,d in enumerate(all_deps_set):
        dep2id[d]=i
    
    all_tokens.append('<EOS>')
    all_tokens.append('<GO>')
    
    all_tokens_set = sorted(set(all_tokens))
    
    vocab={}
    for i,t in enumerate(all_tokens_set):
        vocab[t]=i+1
    vocab['<PAD>']=0
    
    return {'train':all_dics[0],
            'test':all_dics[1],
            'dev':all_dics[2],
            'dep2id':dep2id,
            'rel2id':rel2id,
            'vocab':vocab}
    
def replace_deps(data):
    """
    Replace dependency labels in the edges with their IDs
    """
    datasets=['train','test','dev']

    for dataset in datasets:
        d = data[dataset]
        for i in range(len(d['context_deEdges'])):
            edges = data[dataset]['context_deEdges'][i]
            for j,edge in enumerate(edges):
                data[dataset]['context_deEdges'][i][j][2] = data['dep2id'][data[dataset]['context_deEdges'][i][j][2]]
            q_edges = data[dataset]['q_edges'][i]
            for k,q_edge in enumerate(q_edges):
                data[dataset]['q_edges'][i][k][2] = data['dep2id'][data[dataset]['q_edges'][i][k][2]]
    
    return data


def replace_tokens(data):
    """
    Replace tokens with their IDs from the vocabulary
    """
    datasets=['train','test','dev']
    
    for dataset in datasets:
        d = data[dataset]
        for i in range(len(d['context_tokens'])):
            context = data[dataset]['context_tokens'][i]
            for j,token in enumerate(context):
                data[dataset]['context_tokens'][i][j] = data['vocab'][data[dataset]['context_tokens'][i][j]]
            
            kbents = data[dataset]['kbEnts'][i]
            for k,ent in enumerate(kbents):
                data[dataset]['kbEnts'][i][k] = data['vocab'][data[dataset]['kbEnts'][i][k]]
            
            query = data[dataset]['q_tokens'][i]
            for l,q_tok in enumerate(query):
                data[dataset]['q_tokens'][i][l] = data['vocab'][data[dataset]['q_tokens'][i][l]]
            
            for m,dec_i in enumerate(data[dataset]['dec_ip'][i]):
                
                data[dataset]['dec_ip'][i][m] = data['vocab'][data[dataset]['dec_ip'][i][m]]
            
            for n,dec_t in enumerate(data[dataset]['dec_targ'][i]):
                data[dataset]['dec_targ'][i][n] = data['vocab'][data[dataset]['dec_targ'][i][n]]

    return data            
            

    
def prepro(args):
    source_dir = args.source_dir 
    target_dir = args.target_dir
    source_fname = source_dir+ '/'    
    target_fname = target_dir+ '/preprocessed_'
    
    train_input = source_fname + 'cam676_train.json'
    test_input = source_fname+ 'cam676_test.json'
    dev_input = source_fname+ 'cam676_valid.json'
    
    train_data = json.load(open(train_input,'r'))
    test_data = json.load(open(test_input,'r'))
    dev_data = json.load(open(dev_input,'r'))    
    
    final_data = get_graph([train_data,test_data,dev_data])
    final_data = replace_deps(final_data)
    final_data = replace_tokens(final_data)
   
    with open(target_fname+'CamRest676_final_data_dlex.json','w+') as fp1:
        json.dump(final_data,fp1)


def main():
    args = get_args()
    prepro(args)

if __name__ == "__main__":
    main()
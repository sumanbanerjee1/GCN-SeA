import re
import os
import argparse
import json
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
        rand_edges: if True then randomly paired words are joined with edges instead of dependency graph
        use_associations: if True then builds a contextual graph from the co-occurrence matrix instead of dependency graph
        dict_name: The dictionary which contains word associations obtained from the contextual gaph (output of co-occurrence.py)
    """
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "dialog_dstc_2") 
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default=source_dir)
    parser.add_argument("--rand_edges", default=False)
    parser.add_argument("--use_associations", default=False)
    parser.add_argument("--dict_name", default="")
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
        return processed_kb 
    return None

def get_all_dialogs(filename):
    """
    Read all dialogues from the DSTC2 files
    """
    fname=open(filename,'r')
    s = ''
    for i in fname.readlines():
        s = s + normalize(i)
    all_=s.split('\n\n')
    fname.close()
    return all_[0:-1]

def add_stop(utt):
    """
    Add full stops at the end of the utterances
    """
    if utt[-1] not in '.,!:;?':
        utt = utt + ' .'
    return utt

def get_data(fname):
    """
    Get a structured form of the dialogues
    """
    all_dialogues = get_all_dialogs(fname)

    pre_kb = []
    post_kb = []
    kb = []
    utterance = []
    response = []
    count=0
    for dialog_num , single_dialogue in enumerate(all_dialogues):
        history = [beg]
        content = single_dialogue.split('\n')
        len_of_dialogue = len(content)
        kb_start_index, kb_end_index = locate_kb(content) 
        kb_occurences = len(kb_start_index) - 1
        
        for i in range(0,kb_start_index[0]):
            utterance_response = content[i].split('\t')
            utterance_response[0]=add_stop(re.sub('\d+','',utterance_response[0],1).strip())
            
            if len(utterance_response) < 2: #handles api call no result
                history = history + add_stop(re.sub('\d+','',content[i],1).strip()).split(' ')
                continue
            pre_kb.append(history)
            current_utterance = utterance_response[0].split(' ')
            current_response = add_stop(utterance_response[1].strip()).split(' ')
            kb.append([])
            post_kb.append([])
            utterance.append(current_utterance[0:])
            response.append(current_response[0:])
            history = history + current_utterance + current_response

        current_pre = history
        
        for m in range(0,kb_occurences):
        
            current_kb = process_kb(content[kb_start_index[m]:kb_end_index[m]])
            
            
            if kb_occurences > 1: 
                utterance_response = content[kb_start_index[m]-1].split('\t')
                utterance_response[0]=add_stop(re.sub('\d+','',utterance_response[0],1).strip())
                current_utterance = utterance_response[0].split(' ')
                if len(utterance_response)>1:
                    current_response = add_stop(utterance_response[1].strip()).split(' ')
                current_pre = current_pre + current_utterance + current_response
            
            history = []

            for i in range(kb_end_index[m],kb_start_index[m+1]):
                utterance_response = content[i].split('\t')
                utterance_response[0]=add_stop(re.sub('\d+','',utterance_response[0],1).strip())
                pre_kb.append(current_pre) 
                current_utterance = utterance_response[0].split(' ')
                if len(utterance_response)>1:
                    current_response = add_stop(utterance_response[1].strip()).split(' ')
                kb.append(current_kb)
                post_kb.append(history)
                utterance.append(current_utterance[0:])
                response.append(current_response[0:])
                
                history = history + current_utterance + current_response

    data = [pre_kb,kb,post_kb,utterance,response]
    return data


def append_GO(data):
    """
    Append <GO> symbol in decoder inputs
    """
    for i,d in enumerate(data[4]):
        data[4][i]=['<GO>']+d
    
def get_dec_outputs(data):
    """
    Append <EOS> symbol in decoder outputs
    """
    dec_op=[]
    for i in data[4]:
        temp=i+['<EOS>']
        temp=temp[1:]
        dec_op.append(temp)
    data.append(dec_op)

def append_context(data):
    """
    Combine previous utterances into a single dialogue history
    """
    context_string=[]
    q_string=[]
    for i in range(len(data[0])):
        c=[]
        c.extend(data[0][i])
        if len(data[2][i])>0:
            c.extend(data[2][i])
        
        context_string.append(' '.join(c).strip())
        q_string.append(' '.join(data[3][i]).strip())
    return {'context_string': context_string,
            'kb': data[1],
            'query_string':q_string,
            'dec_ip': data[4],
            'dec_targ':data[5]}

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
    

def return_rand_edges(st):
    """
    Function to join random pairs of words in a string with edges
    """
    doc =nlp(st)
    sentences = list(doc.sents)
    sents = []
    
    for i in sentences:
        sents.append(str(i).split())
        
    ranges=[]
    l=0
    for s in sents:
        for i in range(len(s)):
            ranges.append([l,l+len(s)])
        l=l+len(s)
        
    eds=[]
    for i in range(len(st.split())):
        eds.append([i])
        
    for i in range(len(st.split())):
        rando = random.randint(0, len(st.split())-1) # Edges across sentences in a doc
        eds[i].append(rando)
        eds[i].append('Dummy')
        
    return eds



def return_assoc_edges(st,dict_name):
    """
    Function to join co-occurring words with edges to create the contextual graph
    """
    with open(dict_name+'.json','r+') as fp:
        edge_dict = json.load(fp)
        
        
    stlist = st.split(' ')
    
    
    local_ids={}
    for ind,w in enumerate(stlist):
        local_ids[w]=ind
    
    eds=[]
    
    for i,w1 in enumerate(stlist):
        if w1 in list(edge_dict.keys()):
            for j,w2 in enumerate(edge_dict[w1]):    
                if w2 in stlist:
                    eds.append([i,local_ids[w2],'Dummy'])

    for e in eds:
        if [e[1],e[0],'Dummy'] in eds:
            eds.remove([e[1],e[0],'Dummy'])
            
    return eds
    
def get_graph(data_dics,rand_edges,use_associations,dict_name):
    """
    Tokenize, create graphs and vocabularies:
        data_dics: The train, test and valid data dictionaries obtained from append_context() function
        rand_edges: if True create random graphs instead of dependency or contextual graphs
        use_associations: if True creates contextual graphs
        dict_name: The dictionary having word associations extracted from the co-occurrence matrix
    """
    all_deps=[]
    all_dics=[]
    all_tokens=[]
    rel2id={
                'r_post_code':0,
                'r_cuisine':1,
                'r_location':2,
                'r_phone':3,
                'r_address':4,
                'r_price':5,
                'r_rating':6
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
        for i in tqdm(range(len(data_dic['context_string']))):
            
            if use_associations=='True':
                doc = data_dic['context_string'][i].strip().split(' ')
            else:
                doc = nlp(data_dic['context_string'][i])
                
            context_lens.append(len(doc))
            edges=[]
            c_tokens=[]
            for token in doc:
                if use_associations=='True':
                    c_tokens.append(token)
                    all_tokens.append(token)
                else:
                    edges.append([token.i,token.head.i,token.dep_])
                    c_tokens.append(token.text)
                    all_deps.append(token.dep_)
                    all_tokens.append(token.text)
                
            
            if rand_edges=='True':
                    edges = return_rand_edges(data_dic['context_string'][i])
            
            if use_associations=='True':
                    edges = return_assoc_edges(data_dic['context_string'][i],dict_name)

                
            q_tokens=[]
            q_edges=[]
            if use_associations=='True':
                q_doc = data_dic['query_string'][i].strip().split(' ')    
            else:
                q_doc = nlp(data_dic['query_string'][i])

            q_lens.append(len(q_doc))
            for q_token in q_doc:
                if use_associations=='True':
                    q_tokens.append(q_token)
                    all_tokens.append(q_token)
                else:
                    q_edges.append([q_token.i,q_token.head.i,q_token.dep_])
                    q_tokens.append(q_token.text)
                    all_deps.append(q_token.dep_)
                    all_tokens.append(q_token.text)
                
            if rand_edges=='True':
                    q_edges = return_rand_edges(data_dic['query_string'][i])
                    all_deps =['Dummy']
            
            if use_associations=='True':
                    q_edges = return_assoc_edges(data_dic['query_string'][i],dict_name)
                    all_deps =['Dummy']
            
            
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
            dec_lens.append(len(data_dic['dec_ip'][i]))
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
            'dec_ip':data_dic['dec_ip'],
            'dec_targ':data_dic['dec_targ'],
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
    rand_edges = args.rand_edges
    use_associations = args.use_associations
    dict_name = source_dir+'/'+args.dict_name
    source_fname = source_dir+ '/dialog-dstc2-'    
    target_fname = target_dir+ '/preprocessed-dialog-dstc2-'
    
    train_input = source_fname+ 'trn.txt'
    test_input = source_fname+ 'tst.txt'
    dev_input = source_fname+ 'dev.txt'
     
    train_output = get_data(train_input)
    test_output = get_data(test_input)
    dev_output = get_data(dev_input)

    append_GO(train_output)
    append_GO(test_output)
    append_GO(dev_output)

    get_dec_outputs(train_output)
    get_dec_outputs(test_output)
    get_dec_outputs(dev_output)
    
    train_combined = append_context(train_output)
    test_combined = append_context(test_output)
    dev_combined = append_context(dev_output)
    
    final_data = get_graph([train_combined,test_combined,dev_combined],rand_edges,use_associations,dict_name)
    
    final_data = replace_deps(final_data)
    
    final_data = replace_tokens(final_data)
   
    with open(target_fname+'final_data.json','w+') as fp1:
        json.dump(final_data,fp1)


def main():
    args = get_args()
    prepro(args)

if __name__ == "__main__":
    main()
import json
import re
from tqdm import tqdm
import random
from copy import deepcopy

timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat = re.compile("\d{1,3}[.]\d{1,2}")

kb = json.load(open('../data/CamRest676/CamRestDB.json','r'))
val_dic = json.load(open('../data/CamRest676/val_replc_dic.json','r'))

def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def normalize(text):
    '''
    Normalize utterance strings
    '''
    text = text.lower()
    text = re.sub(r'^\s*|\s*$', '', text)
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    #text = re.sub(u"(\u2018|\u2019)", "'", text)

    #text = re.sub(timepat, ' [value_time] ', text)
    #text = re.sub(pricepat, ' [value_price] ', text)
    #text = re.sub(pricepat2, '[value_price]', text)

    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    text = text.replace('-', ' ')
    text = re.sub('[\":\<>@\(\)]', '', text)

    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    text = insertSpace('\'s', text)

    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    
    text = re.sub(' +', ' ', text)
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text

def triplify(rest):
    '''
    Create Triples from a Restaurant's dictionary
    '''
    attrs = list(rest.keys())
    attrs.remove('name')
    attrs.remove('id')
    attrs.remove('type')
    attrs.remove('location')
    k=[]
    entlist=[]
    
    for i in attrs:
        e1= normalize(rest['name'])
        r = normalize(i)
        e2 = normalize(rest[i])
        k.append([e1,r,e2])
        entlist.append(e2)
        
    k.append([e1,'name',e1])
    entlist.append(e1)
    
    return k,list(set(entlist))

def get_triples(goal):
    '''
    Get relevant KB triples according to dialogue goal
    '''
    constraints={} 
    for c in goal['constraints']:
        constraints[c[0]]=c[1]
      
    exception = {'pricerange': 'cheap', 'food': 'european'}
    if(constraints==exception):
        constraints['food']='modern european'
    ents=[]
    triples=[]
    for rest in kb:
        f=0
        for c,v in constraints.items():
          if c in list(rest.keys()):  
            if(rest[c]==v) or v=='dontcare' or(v=='european' and rest[c]=='modern european') or (v=='modern european' and rest[c]=='european'):
                f=1
            else:
                f=0
                break
        if(f==1):
            triple,entlist = triplify(rest)    
            triples.extend(triple)
            ents.extend(entlist)
    
    return triples,list(set(ents))
    
def twospace(text):
    return re.sub(' +', ' ', text)


def turnwise(refined_data):
    '''
    Split dialogue data turnwise to create data instances
    '''
    dic={}
    dic['kb']=[]
    dic['goal']=[]
    dic['context_string']=[]
    dic['query_string']=[]
    dic['dec_targ']=[]
    dic['dec_ip']=[]

    for dialogue in refined_data:
        
        hist='<beg>'
        dic['context_string'].append(twospace(hist))
        for i,t in enumerate(dialogue['turns']):
            
            if((i+1)%2==0):
                
                dic['kb'].append(dialogue['kb'])
                dic['goal'].append(dialogue['goal'])
                dic['dec_targ'].append(twospace(t))
                dic['dec_ip'].append(twospace('<GO> ' + t)[0:-6])
                hist+=' ' + t.strip() + ' '
                if(i!=len(dialogue['turns'])-1):
                    dic['context_string'].append(twospace(hist))
            else:
                dic['query_string'].append(twospace(t)[0:-6])
                hist+=' ' + t.strip() + ' '
    
    return dic        


def turnwise_meta(refined_data):
    
    dic={}
    dic['kb']=[]
    dic['fname']=[]
    dic['goal']=[]
    dic['meta']=[]
    dic['context_string']=[]
    dic['query_string']=[]
    dic['dec_targ']=[]
    dic['dec_ip']=[]

    for dialogue in refined_data:
        
        hist=[]
        h = deepcopy(hist)
        dic['context_string'].append(h)
        for i,t in enumerate(dialogue['turns']):
            if(i%2==0):
                    m = dialogue['meta'][i+1]
                    
            if((i+1)%2==0):
                
                dic['kb'].append(dialogue['kb'])
                dic['goal'].append(dialogue['goal'])
                
                dic['fname'].append(dialogue['fname'])        
                dic['meta'].append(m)
                dic['dec_targ'].append(twospace(t))
                dic['dec_ip'].append(twospace('<GO> ' + t)[0:-6])
                hist.append(t.strip()[0:-6])
                if(i!=len(dialogue['turns'])-1):
                    h = deepcopy(hist)
                    dic['context_string'].append(h)
            else:
                dic['query_string'].append(twospace(t)[0:-6])
                hist.append(t.strip()[0:-6])
    
    return dic       


def ent_normalize(resps):
    '''
    Normalize entities in the KB triples
    '''
    kbents=[]
    for k in kb:
        kv = list(k.values())
        kv = [normalize(v) for v in kv]
        kbents.extend(kv)
    kbents = list(set(kbents))
    
    for i,r in enumerate(resps):
        for k in kbents:
            resps[i] = re.sub(k,'_'.join(k.strip().split()),resps[i])
    
    return resps
        
def dlex(resps,dic):
    '''
    Delexicalize utterances
    '''
    for i,r in enumerate(resps):
        for ent in sorted(list(dic.keys()),key=len,reverse=True):
            if(ent in r):
                r = re.sub(ent,dic[ent],r)
                resps[i] = r
    
    return resps

def main():
    
    data = json.load(open('../data/CamRest676/CamRest676.json','r'))
    refined_data=[]
    cnt=1
    for dialog in tqdm(data):
        d={}
        resps=[]
        meta=[]
        dial = dialog['dial']
        goal = dialog['goal']
        kbtriples,ents = get_triples(goal)
        for turn in dial:
            
            slu = turn['usr']['slu']
            b={}
            for act in slu:
                if(act['act']=='inform'):
                    b[act['slots'][0][0]] = act['slots'][0][1]
            
            
            meta.append({})
            meta.append(b)
            u = normalize(turn['usr']['transcript'])
            s = normalize(turn['sys']['sent'])
            
            for e in sorted(ents,reverse=True,key=len):
                u = re.sub(e,'_'.join(e.strip().split()),u)
                s = re.sub(e,'_'.join(e.strip().split()),s)
            
            resps.append(u + ' <EOS> ')
            resps.append(s + ' <EOS> ')
            
            
        resps = dlex(resps,val_dic) 
        for i,trip in enumerate(kbtriples):
            kbtriples[i][0] = '_'.join(kbtriples[i][0].strip().split())
            kbtriples[i][2] = '_'.join(kbtriples[i][2].strip().split())
            
        d['turns']=ent_normalize(resps)
        d['goal']=goal
        d['kb'] = kbtriples
        d['meta'] = meta
        d['fname'] = 'CAM'+str(cnt)
        cnt+=1
        refined_data.append(d)
    
    random.seed(123)
    random.shuffle(refined_data)
    
    train = refined_data[0:406]
    valid = refined_data[406:541]
    test = refined_data[540:]
    
    train_data = turnwise(train)
    valid_data = turnwise(valid)
    test_data = turnwise(test)
    
    with open('../data/CamRest676/cam676_train.json','w+') as fp:
        json.dump(train_data,fp)
    
    with open('../data/CamRest676/cam676_valid.json','w+') as fp:
        json.dump(valid_data,fp)
    
    with open('../data/CamRest676/cam676_test.json','w+') as fp:
        json.dump(test_data,fp)
        

if __name__ == '__main__':
    main()
from metrics import bleu, rouge
import argparse

def get_args():
    '''
    Parse input arguments:
        preds_path: The directory in which labels and predictions files are dumped after inference
        config_id: The config id mentioned in the labels and predictions filenames
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_path")
    parser.add_argument("--kb_path")
    parser.add_argument("--config_id")
    args = parser.parse_args()
    return args

def read_results(path,num):
    with open(path+"/labels"+str(num)+".txt","r") as fp:
        l=fp.readlines()
    with open(path+"/predictions"+str(num)+".txt","r") as fp:
        p=fp.readlines()
    
    return p,l

def exact_match(p,l):
    c=0
    for i1,i in enumerate(l):
        if p[i1]==l[i1]:
            c+=1
    print("Per-Resp Acc: ",c/len(l))


def moses_bl_rouge(p,l):
    bl = bleu.moses_multi_bleu(p,l)
    x = rouge.rouge(p,l)
    print('BLEU: %f\nROUGE1-F: %f\nROUGE1-P: %f\nROUGE1-R: %f\nROUGE2-F: %f\nROUGE2-P: %f\nROUGE2-R: %f\nROUGEL-F: %f\nROUGEL-P: %f\nROUGEL-R: %f'%(bl,x['rouge_1/f_score'],x['rouge_1/p_score'],x['rouge_1/r_score'],x['rouge_2/f_score'],
                                                    x['rouge_2/p_score'],x['rouge_2/r_score'],x['rouge_l/f_score'],x['rouge_l/p_score'],x['rouge_l/r_score']))


def micro_compute_prf(gold, pred, global_entity_list):
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list:
                    if p not in gold:
                        FP += 1
        else:
            count = 0
        return TP,FP,FN,count

def ent_f1(preds,labels,kb_path):
    with open(kb_path,'r') as fp:
        kb=fp.readlines()
    
    ent=[]
    for i in kb:
        triples = i.split(' ')
        ent.append(triples[1].strip())
        ent.append(triples[3].strip())
    
    ent = set(ent)
    ent_list = sorted(ent)

    mic_pred=0
    les=[]
    all_TP=0
    all_FP=0
    all_FN=0
    for i in range(len(labels)):
        l = labels[i].strip().split()
        le=[]
        for j in l:
            if j in ent_list:
                le.append(j)
        les.append(le)
        p = preds[i].strip().split()
        tp,fp,fn,c = micro_compute_prf(le,p,ent_list)
        all_TP+=tp
        all_FP+=fp
        all_FN+=fn
        mic_pred+=c
    
    mic_prec = all_TP/float(all_TP+all_FP)
    mic_rec = all_TP/float(all_TP + all_FN)
    
    mic_f1=2 * mic_prec * mic_rec / float(mic_prec + mic_rec)
    print("Entity-F1:",mic_f1)


if __name__=='__main__':
    args = get_args()
    result_path = args.preds_path
    kb_path = args.kb_path
    config_id = args.config_id
    print(config_id,"\n")
    preds,labels = read_results(result_path,config_id)
    exact_match(preds,labels)
    moses_bl_rouge(preds,labels)
    ent_f1(preds,labels,kb_path)
    
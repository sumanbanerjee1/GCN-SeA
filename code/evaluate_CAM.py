import json
import argparse
from metrics import bleu,rouge
from evaluator_CAM import evaluateModel

def get_args():
    '''
    Parse input arguments:
        preds_path: The directory in which labels and predictions files are dumped after inference
        config_id: The config id mentioned in the labels and predictions filenames
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_path")
    parser.add_argument("--config_id")
    args = parser.parse_args()
    return args


def moses_bl_rouge(p,l):
    bl = bleu.moses_multi_bleu(p,l)
    x = rouge.rouge(p,l)
    print('Moses BLEU: %f\nROUGE1-F: %f\nROUGE1-P: %f\nROUGE1-R: %f\nROUGE2-F: %f\nROUGE2-P: %f\nROUGE2-R: %f\nROUGEL-F: %f\nROUGEL-P: %f\nROUGEL-R: %f'%(bl,x['rouge_1/f_score'],x['rouge_1/p_score'],x['rouge_1/r_score'],x['rouge_2/f_score'],
                                                    x['rouge_2/p_score'],x['rouge_2/r_score'],x['rouge_l/f_score'],x['rouge_l/p_score'],x['rouge_l/r_score']))

def correct_fnames(pred_sents,label_sents,og_dial):
    
    d = {}
    real ={}
    ind=0
    for i in range(len(og_dial)):
        f = og_dial[i]['fname']
        turn_length = len(og_dial[i]['turns'])
        turn_length = int(turn_length/2)
        dial = pred_sents[ind:ind+turn_length]
        real_dial = label_sents[ind:ind+turn_length]
        ind=ind+turn_length
        d[f] = dial
        real[f] = real_dial
        
    return d,real

if __name__=='__main__':
    args = get_args()
    dir_ = args.preds_path
    config = args.config_id
    
    print('CONFIG:',config)
    with open(dir_+'/predictions'+config+'.txt') as f:
        preds = f.readlines()
        
    with open(dir_+'/labels'+config+'.txt') as f:
        labels = f.readlines()  
    
    preds = [k.strip() for k in preds]
    l,fnames=[],[]
    l=[k.strip() for k in labels]
    
    #dlex_labels = dlex(l,val_dic)
    #dlex_preds = dlex(preds,val_dic)
    
    dlex_labels = l
    dlex_preds = preds
    
    moses_bl_rouge(dlex_preds,dlex_labels)
    
    with open('../data/CamRest676/CAM_dlex_meta_test.json','r') as f:
        og_dial = json.load(f)

    pred_dials_gen,labels_dials_gen = correct_fnames(dlex_preds,dlex_labels,og_dial)
    
    og_dials={}
    for i in range(len(og_dial)):
        f = og_dial[i]['fname']
        og_dials[f] = og_dial[i]
    
    evaluateModel(pred_dials_gen,og_dials,'test')

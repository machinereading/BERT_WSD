import json
import torch

def lines2tsv(lines):
    tsv = []
    sent = []
    for line in lines:
        line = line.strip()
        
        if line != '':
            token = line.split('\t')
            sent.append(token)
        else:
            tsv.append(sent)
            sent = []
    return tsv

def conll2data(conll):
    tsv = lines2tsv(conll)    
    data = []
    for sent in tsv:     
        tok_str, tok_lu, tok_frame= [],[],[]
        for token in sent:
            tok_str.append(token[1])
            tok_lu.append(token[12])
            tok_frame.append(token[13])
        sent_list = []
        sent_list.append(tok_str)
        sent_list.append(tok_lu)
        sent_list.append(tok_frame)
        data.append(sent_list)
    return data     

def load_framenet_data(language):
    if language == 'en':
        with open('./data/fn1.7/fn1.7.fulltext.train.syntaxnet.conll','r') as f:
            trn_conll = f.readlines()            
        with open('./data/fn1.7/fn1.7.dev.syntaxnet.conll','r') as f:
            dev_conll = f.readlines()        
        with open('./data/fn1.7/fn1.7.test.syntaxnet.conll','r') as f:
            tst_conll = f.readlines()
            
        trn_data = conll2data(trn_conll)
        dev_data = conll2data(dev_conll)
        tst_data = conll2data(tst_conll)
    
    print('\n### loading',language,'FrameNet data...')
    print('\t# of sentence in training data:', len(trn_data))
    print('\t# of sentence in dev data:', len(dev_data))
    print('\t# of sentence in test data:', len(tst_data))
    
    return tst_data, dev_data, tst_data

def get_masks(datas, mapdata, num_label=2):
    masks = []
    for idx in datas:
        mask = torch.zeros(num_label)
        try:
            candis = mapdata[str(int(idx[0]))]
        except KeyboardInterrupt:
            raise
        except:
            candis = mapdata[int(idx[0])]
        for candi_idx in candis:
            mask[candi_idx] = 1
        masks.append(mask)
    masks = torch.stack(masks)
    return masks
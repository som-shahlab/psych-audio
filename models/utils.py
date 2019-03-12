import os
import re
import torch
import itertools
import random
import lxml.etree as et
from collections import defaultdict, Counter
from torch.utils.data import DataLoader,Dataset
import pandas as pd

class I2b2Document(object):
    """
    i2b2 Challenge note
    """
    label_map = {
        'NULL': -1,
        'UNKNOWN':1,
        'CURRENT SMOKER':2,
        'SMOKER':2,
        'NON-SMOKER':3,
        'PAST SMOKER':4
        }

    def __init__(self, record, ground_truth=False):

        self.ground_truth = ground_truth
        doc_id, text, sents, y = self._parse(record)

        self.doc_id    = doc_id
        self.text      = text
        self.sentences = sents
        self.tokens    = list(itertools.chain.from_iterable(sents))
        self.label     = I2b2Document.label_map[y]

    def _parse(self, xml):
        doc_id = xml.xpath("@ID")[0].strip()
        text = xml.xpath("TEXT/text()")[0].strip()
        y = xml.xpath("SMOKING/@STATUS")[0] if self.ground_truth else 'NULL'
        # corpus is already split into sentences and tokens
        sents = list(map(lambda s:s.split(), text.split("\n")))
        return doc_id, text, sents, y

    def __repr__(self):
        return f"""I2b2Document({self.doc_id}, "{self.text[:130]}...")"""

class Mimic3Document(object):
    """
    MIMIC-III note
    """
#    label_map = {
#        'NULL': -1,
#        'UNKNOWN':1,
#        'SMOKER':2,
#        'NON-SMOKER':3,
#        'EX SMOKER':4
#        }

    def __init__(self, doc_id, text, y, ground_truth=False):

        self.ground_truth = ground_truth
        sents = self._parse(doc_id, text)

        self.doc_id    = doc_id
        self.text      = text
        self.sentences = sents
        self.tokens    = list(itertools.chain.from_iterable(sents))
        #self.label     = Mimic3Document.label_map[y]
        self.label     = y

    def _parse(self, doc_id, text):
        # corpus is already split into sentences and tokens
        sents = list(map(lambda s:s.split(), text.split("\n")))
        return sents

    def __repr__(self):
        return f"""Mimic3Document({self.doc_id}, "{self.text[:130]}...")"""

class ShcDocument(object):
    """
    SHC note
    """
#    label_map = {
#        'NULL': -1,
#        'UNKNOWN':1,
#        'CURRENT SMOKER':2,
#        'SMOKER':2,
#        'NON-SMOKER':3,
#        'EX SMOKER':4
#        }

    def __init__(self, doc_id, text, y, ground_truth=False):
        self.ground_truth = ground_truth
        sents= self._parse(doc_id, text)

        self.doc_id    = doc_id
        self.text      = text
        self.sentences = sents
        self.tokens    = list(itertools.chain.from_iterable(sents))
        self.label     = y

    def _parse(self, doc_id, text):
        # corpus is already split into sentences and tokens
        sents = list(map(lambda s:s.split(), text.split("\n")))
        return sents

    def __repr__(self):
        return f"""ShcDocument({self.doc_id}, "{self.text[:130]}...")"""

def load_i2b2_dataset(data_root):
    """
    i2b2 2006 Smoking Cessation Challenge dataset
    """
    assert os.path.exists(data_root)

    def parse_xml(fpath, ground_truth):
        docs = [I2b2Document(record, ground_truth) for record in et.parse(fpath).xpath(".//RECORD")]
        return {doc.doc_id:doc for doc in docs}

    fpath = {
     "train":"{}/unannotated_records_deid_smoking.xml".format(data_root),
     "dev":"{}/smokers_surrogate_train_all_version2.xml".format(data_root),
     "test":"{}/smokers_surrogate_test_all_groundtruth_version2.xml".format(data_root)
    }

    #train = parse_xml(fpath["train1"], ground_truth=False)
    #train.update(parse_xml(fpath["train2"], ground_truth=False))
    train  = parse_xml(fpath["train"], ground_truth=False)
    dev    = parse_xml(fpath["dev"], ground_truth=True)
    test   = parse_xml(fpath["test"], ground_truth=True)

    # make certain sets are disjoint
    for doc_id in dev:
        if doc_id in train:
            del train[doc_id]
        if doc_id in test:
            del test[doc_id]

    for doc_id in test:
        if doc_id in train:
            del train[doc_id]
        if doc_id in dev:
            del dev[doc_id]

    print('[TRAIN]', len(train))
    print('[DEV]  ', len(dev))
    print('[TEST] ', len(test))

    return {
        'train':list(train.values()),
        'dev': list(dev.values()),
        'test': list(test.values())
    }

##########################################################
##########################################################
def load_mimic3_dataset(dev_test_ratio, train_size):
    """
    mimic3 dataset
    """

    def get_text(annotated, gt):
        records=dict()
        id=None
        text=None
        with open(annotated,'r') as f:
            for line in f:
                if line.startswith("[[ID="):
                    if id and text:
                        ## add the record if there is another one afterwards
                        if int(id) in gt.index.tolist():
                            records[id]=text.strip()
                        #print(id, text)
                    id=line[line.find("[[ID=")+5:line.find("]]")]
                    text=line[line.find("]]")+2:]
                    continue
                if not line.startswith("[[ID="):
                    text+=line
            if id in gt.index.tolist():
                records[id]=text.strip()
        print(len(records))
        return records

    def get_records(note_dir, train_size):
        small=["notes-1-500000.txt","notes-500001-1000000.txt",
        "notes-1000001-1500000.txt", "notes-1500001-2000000.txt", "notes-2000001-2078705.txt"]
        records=dict()
        i=0
        id=None
        text=None
        rgx = r"""(cigar|smok|tobacc)"""
        for note_file in os.listdir(note_dir):
            if note_file in small:
                with open(note_dir+note_file,'r') as f:
                        for line in f:
                            if line.startswith("[[ID="):
                                if id and text:
                                    ## add the record if there is another one afterwards
                                    if re.search(rgx, text, re.I):
                                        records[id]=text.strip()
                                        #print(id, text)
                                        i+=1
                                        if i>=train_size:
                                            break
                                id=line[line.find("[[ID=")+5:line.find("]]")]
                                text=line[line.find("]]")+2:]
                                continue
                            if not line.startswith("[[ID="):
                                text+=line
                        if i>=train_size:
                            # we stop and do not look at next file
                            break
                        else:
                            # add the record if it is the last one of the file
                            if re.search(rgx, text, re.I):
                                records[id]=text.strip()
                                i+=1
        #print(len(records))
        return records

    def parse_train(fpath, train_size, ground_truth):
        y='NULL' # ground_truth, NO!
        docs = [Mimic3Document(doc_id, text, y, ground_truth) for doc_id, text in get_records(fpath, train_size).items()]
        return {doc.doc_id:doc for doc in docs}

    def parse_test_dev(annotated, gt_files, dev_test_ratio, ground_truth):
        """
        get 2 dictionaries(id, text) for the mimic3 dev / test set
        of relative size defined by ratio
        if ratio = 0.2 >> dev=20%/test=80%
        if ratio = 0.7 >> dev=70%/test=30%
        """
        docs=dict()
        # y # ground_truth, YES!
        gt=pd.read_csv(gt_files, index_col=0)
        print(gt.shape[0])
        docs = [Mimic3Document(doc_id, text, gt.loc[int(doc_id)][0], ground_truth) for doc_id, text in get_text(annotated, gt).items()]
        # random fuzz
        seed=2321
        random.seed(seed)
        docs=random.sample(docs, len(docs))
        # split in dev/test following the ratio
        dev_end_index=len(docs)*float(dev_test_ratio)
        devs=dict();tests=dict()
        devs=docs[:int(dev_end_index)]
        tests=docs[int(dev_end_index):]

        return {doc.doc_id:doc for doc in devs}, {doc.doc_id:doc for doc in tests}

    fpath = {
     "train":"/data2/coulet/ss/mimic3/notes/",
     "dev":"/data2/coulet/ss/mimic3/notes/annotated_notes.txt",
     "test":"/data2/coulet/ss/mimic3/notes/annotated_notes.txt", #useless
     "gt":"/data2/coulet/ss/mimic3/manual_annot/gt.csv" #ground_truth
    }

    train = parse_train(fpath["train"], train_size, ground_truth=False)
    dev, test = parse_test_dev(fpath["dev"], fpath["gt"], dev_test_ratio, ground_truth=True)

    # make certain sets are disjoint
    for doc_id in dev:
        if doc_id in train:
            del train[doc_id]
        if doc_id in test:
            del test[doc_id]

    for doc_id in test:
        if doc_id in train:
            del train[doc_id]
        if doc_id in dev:
            del dev[doc_id]

    print('[TRAIN]', len(train))
    print('[DEV]  ', len(dev))
    print('[TEST] ', len(test))

    return {
        'train':list(train.values()),
        'dev': list(dev.values()),
        'test': list(test.values())
    }

##########################################################
##########################################################
def load_shc_dataset(dev_test_ratio,train_size):
    """
    Shc
    """
    def get_text(annotated, gt):
        records=dict()
        id=None
        text=None
        with open(annotated,'r',encoding="latin-1") as f:
            for line in f:
                if line.startswith("[[ID="):
                    if id and text:
                        ## add the record if there is another one afterwards
                        if int(id) in gt.index.tolist():
                            records[id]=text.strip()
                        #print(id, text)
                    id=line[line.find("[[ID=")+5:line.find("]]")]
                    text=line[line.find("]]")+2:]
                    continue
                if not line.startswith("[[ID="):
                    text+=line
            if id in gt.index.tolist():
                records[id]=text.strip()
        return records

    def get_records(note_dir, train_size):
        small=["notes-clinical-54000001-56000000.txt","notes-clinical-56000001-58000000.txt",
        "notes-clinical-58000001-60000000.txt", "notes-clinical-60000001-74010435.txt"]
        records=dict()
        i=0
        id=None
        text=None
        rgx = r"""(cigar|smok|tobacc)"""
        for note_file in os.listdir(note_dir):
            if note_file in small:
                with open(note_dir+note_file,'r') as f:
                        for line in f:
                            if line.startswith("[[ID="):
                                if id and text:
                                    ## add the record if there is another one afterwards
                                    if re.search(rgx, text, re.I):
                                        records[id]=text.strip()
                                        #print(id, text)
                                        i+=1
                                        if i>=train_size:
                                            break
                                id=line[line.find("[[ID=")+5:line.find("]]")]
                                text=line[line.find("]]")+2:]
                                continue
                            if not line.startswith("[[ID="):
                                text+=line
                        if i>=train_size:
                            # we stop and do not look at next file
                            break
                        else:
                            # add the record if it is the last one of the file
                            if re.search(rgx, text, re.I):
                                records[id]=text.strip()
                                i+=1
        #print(len(records))
        return records

    def parse_train(fpath, train_size, ground_truth):
        y='NULL' # ground_truth, NO!
        docs = [ShcDocument(doc_id, text, y, ground_truth) for doc_id, text in get_records(fpath, train_size).items()]
        return {doc.doc_id:doc for doc in docs}

    def parse_test_dev(annotated, gt_files, dev_test_ratio, ground_truth):
        """
        get 2 dictionaries(id, text) for the mimic3 dev / test set
        of relative size defined by ratio
        if ratio = 0.2 >> dev=20%/test=80%
        if ratio = 0.7 >> dev=70%/test=30%
        """
        docs=dict()
        # y # ground_truth, YES!
        gt=pd.read_csv(gt_files, index_col=0)
        docs = [ShcDocument(doc_id, text, gt.loc[int(doc_id)][0], ground_truth) for doc_id, text in get_text(annotated, gt).items()]
        # random fuzz
        seed=2321
        random.seed(seed)
        docs=random.sample(docs, len(docs))
        # split in dev/test following the ratio
        dev_end_index=len(docs)*float(dev_test_ratio)
        devs=dict();tests=dict()
        devs=docs[:int(dev_end_index)]
        tests=docs[int(dev_end_index):]

        return {doc.doc_id:doc for doc in devs}, {doc.doc_id:doc for doc in tests}

    fpath = {
     "train":"/data2/coulet/ss/shc_sample/notes/",
     "dev":"/data2/coulet/ss/shc_sample/notes/annotated_notes.txt",
     "test":"/data2/coulet/ss/shc_sample/notes/annotated_notes.txt", #useless
     "gt":"/data2/coulet/ss/shc_sample/manual_annot/gt.csv" #ground_truth
    }

    train = parse_train(fpath["train"], train_size, ground_truth=False)
    dev, test = parse_test_dev(fpath["dev"], fpath["gt"], dev_test_ratio, ground_truth=True)

    # make certain sets are disjoint
    for doc_id in dev:
        if doc_id in train:
            del train[doc_id]
        if doc_id in test:
            del test[doc_id]

    for doc_id in test:
        if doc_id in train:
            del train[doc_id]
        if doc_id in dev:
            del dev[doc_id]

    print('[TRAIN]', len(train))
    print('[DEV]  ', len(dev))
    print('[TEST] ', len(test))

    return {
        'train':list(train.values()),
        'dev': list(dev.values()),
        'test': list(test.values())
    }

#
# Pytorch Data Loader Tools
#

class SymbolTable(object):
    """Wrapper for dict to encode unknown symbols"""
    def __init__(self, starting_symbol=2, unknown_symbol=1):
        self.s       = starting_symbol
        self.unknown = unknown_symbol
        self.d       = dict()

    def get(self, w):
        if w not in self.d:
            self.d[w] = self.s
            self.s += 1
        return self.d[w]

    def lookup(self, w):
        return self.d.get(w, self.unknown)

    def lookup_strict(self, w):
        return self.d.get(w)

    def len(self):
        return self.s

    def reverse(self):
        return {v: k for k, v in iteritems(self.d)}


def build_vocab(seqs):

    vocab = Counter()
    for sent in seqs:
        for w in sent:
            vocab[w] += 1
    word_dict = SymbolTable()
    list(map(word_dict.get, vocab))
    return word_dict


def get_padded_seqs(seqs, word_dict, max_seq_len=100):
    X = []
    for seq in seqs:
        x = torch.tensor([word_dict.lookup(w) for w in seq])
        if x.size(0) > max_seq_len:
            x = x[..., 0 : min(x.size(0), max_seq_len)]
        else:
            k = max_seq_len - x.size(0)
            x = torch.cat((x, torch.zeros(k, dtype=torch.long)))
            ## AC to JF: I tried the following line, but it just turn the error the other way!
            #x = torch.cat((x, torch.zeros(k, dtype=torch.float)))
        X.append(x)
    X = torch.stack(X)
    return X


class DocumentDataset(object):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

import numpy as np
import torch

## 20211002
import logging
import random
from sklearn.metrics import f1_score, recall_score, precision_score

import tarfile


class KlueDpInputExample:
    """
    A single training/test example for Dependency Parsing in .conllu format

    Args:
        guid : Unique id for the example
        text : string. the original form of sentence
        token_id : token id
        token : 어절
        pos : POS tag(s)
        head : dependency head
        dep : dependency relation
    """

    def __init__(
        self,
        guid: str,
        text: str,
        sent_id: int,
        token_id: int,
        token: str,
        pos: str,
        head: int,
        dep: str,
    ):
        self.guid = guid
        self.text = text
        self.sent_id = sent_id
        self.token_id = token_id
        self.token = token
        self.pos = pos
        self.head = head
        self.dep = dep


class KlueDpInputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.
    
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        bpe_head_mask : Mask to mark the head token of bpe in aejeol
        head_ids : head ids for each aejeols on head token index
        dep_ids : dependecy relations for each aejeols on head token index
        pos_ids : pos tag for each aejeols on head token index
    """

    def __init__(
        self, guid, ids, mask, bpe_head_mask, bpe_tail_mask, head_ids, dep_ids, pos_ids
    ):
        self.guid = guid
        self.input_ids = ids#       len([i for i in ids if i not in [1]])
        self.attention_mask = mask#  == torch.tensor(mask).sum() 
        self.bpe_head_mask = bpe_head_mask# len([i for i in bpe_head_mask if i not in [0]])
        self.bpe_tail_mask = bpe_tail_mask# == len([i for i in bpe_tail_mask if i not in [0]])
        self.head_ids = head_ids#           == len([i for i in head_ids if i not in [-1]])
        self.dep_ids = dep_ids#             == len([i for i in dep_ids if i not in [-1]])
        self.pos_ids = pos_ids#             == len([i for i in pos_ids if i not in [-1]])


def create_examples(file_path):
    sent_id = -1
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "" or line == "\n" or line == "\t":
                continue

            if line.startswith("#"):
                parsed = line.strip().split("\t")
                if len(parsed) != 2:  # metadata line about dataset
                    continue
                else:
                    sent_id += 1
                    text = parsed[1].strip()
                    guid = parsed[0].replace("##", "").strip()
            else:
                token_list = []
                token_list = (
                    [sent_id]
                    + [token.replace("\n", "") for token in line.split("\t")]
                    + ["-", "-"]
                )
                examples.append(
                    KlueDpInputExample(
                        guid=guid,
                        text=text,
                        sent_id=sent_id,
                        token_id=int(token_list[1]),
                        token=token_list[2],
                        pos=token_list[4],
                        head=token_list[5],
                        dep=token_list[6],
                    )
                )
    return examples


def get_dp_labels():
    """
    label for dependency relations format:

    {structure}_(optional){function}

    """
    dp_labels = [
        "NP",
        "NP_AJT",
        "VP",
        "NP_SBJ",
        "VP_MOD",
        "NP_OBJ",
        "AP",
        "NP_CNJ",
        "NP_MOD",
        "VNP",
        "DP",
        "VP_AJT",
        "VNP_MOD",
        "NP_CMP",
        "VP_SBJ",
        "VP_CMP",
        "VP_OBJ",
        "VNP_CMP",
        "AP_MOD",
        "X_AJT",
        "VP_CNJ",
        "VNP_AJT",
        "IP",
        "X",
        "X_SBJ",
        "VNP_OBJ",
        "VNP_SBJ",
        "X_OBJ",
        "AP_AJT",
        "L",
        "X_MOD",
        "X_CNJ",
        "VNP_CNJ",
        "X_CMP",
        "AP_CMP",
        "AP_SBJ",
        "R",
        "NP_SVJ",
    ]
    return dp_labels


def get_pos_labels():
    """label for part-of-speech tags"""

    return [
        "NNG",
        "NNP",
        "NNB",
        "NP",
        "NR",
        "VV",
        "VA",
        "VX",
        "VCP",
        "VCN",
        "MMA",
        "MMD",
        "MMN",
        "MAG",
        "MAJ",
        "JC",
        "IC",
        "JKS",
        "JKC",
        "JKG",
        "JKO",
        "JKB",
        "JKV",
        "JKQ",
        "JX",
        "EP",
        "EF",
        "EC",
        "ETN",
        "ETM",
        "XPN",
        "XSN",
        "XSV",
        "XSA",
        "XR",
        "SF",
        "SP",
        "SS",
        "SE",
        "SO",
        "SL",
        "SH",
        "SW",
        "SN",
        "NA",
    ]


def flatten_prediction_and_labels(preds, labels, tokens):
    head_preds = list()
    head_labels = list()
    type_preds = list()
    type_labels = list()
    for pred, label in zip(preds, labels):
        head_preds += pred[0].cpu().flatten().tolist()
        head_labels += label[0].cpu().flatten().tolist()
        type_preds += pred[1].cpu().flatten().tolist()
        type_labels += label[1].cpu().flatten().tolist()
    head_preds = np.array(head_preds)
    head_labels = np.array(head_labels)
    type_preds = np.array(type_preds)
    type_labels = np.array(type_labels)

    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_preds = np.delete(head_preds, index)
    head_labels = np.delete(head_labels, index)
    index = [i for i, label in enumerate(type_labels) if label == -1]
    type_preds = np.delete(type_preds, index)
    type_labels = np.delete(type_labels, index)

    return (
        head_preds.tolist(),
        type_preds.tolist(),
        head_labels.tolist(),
        type_labels.tolist(),
    )


def flatten_labels(labels):
    head_labels = list()
    type_labels = list()
    for label in labels:
        head_labels += label[0].cpu().flatten().tolist()
        type_labels += label[1].cpu().flatten().tolist()
    head_labels = np.array(head_labels)
    type_labels = np.array(type_labels)

    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_labels = np.delete(head_labels, index)
    index = [i for i, label in enumerate(type_labels) if label == -1]
    type_labels = np.delete(type_labels, index)

    # classify others label as -3
    others_idx = 15
    for i, label in enumerate(type_labels):
        if label >= others_idx:
            type_labels[i] = -3

    return head_labels.tolist(), type_labels.tolist()


def resize_outputs(outputs, bpe_head_mask, bpe_tail_mask, max_word_length):
    batch_size, input_size, hidden_size = outputs.size()#torch.Size([8, 128, 1024])
    word_outputs = torch.zeros(batch_size, max_word_length + 1, hidden_size * 2).to(
        outputs.device
    )#torch.Size([8, max_word_length, 2048])
    sent_len = list()

    for batch_id in range(batch_size):
        head_ids = [i for i, token in enumerate(bpe_head_mask[batch_id]) if token == 1]# head 위치 정보: subword 기준
        tail_ids = [i for i, token in enumerate(bpe_tail_mask[batch_id]) if token == 1]# tail 위치 정보: subword 기준
        assert len(head_ids) == len(tail_ids)
        # outputs[batch_id].shape == torch.Size([128, 1024])
        word_outputs[batch_id][0] = torch.cat(# outputs[batch_id][0]은 첫 token인 [CLS]에 대응
            (outputs[batch_id][0], outputs[batch_id][0])# [torch.Size([1024]), torch.Size([1024])]
        )  # replace root with [CLS]
        for i, (head, tail) in enumerate(zip(head_ids, tail_ids)):
            word_outputs[batch_id][i + 1] = torch.cat(# concatenate the first and last subword token representations of each word, to form word vector representations.
                (outputs[batch_id][head], outputs[batch_id][tail])# 
            )
        sent_len.append(i + 2)# word 기준 길이

    return word_outputs, sent_len




##############20211002
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_tokenizer(args):
    # return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)
    return AutoTokenizer.from_pretrained("klue/roberta-large")


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        # level=logging.INFO)
                        level=logging.DEBUG)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(data):
    scores = {}
    for k, (preds, labels) in data.items():
        assert len(preds) == len(labels)
        scores.update({
            k : {
                "precision": precision_score(labels, preds, average='macro'),
                "recall": recall_score(labels, preds, average='macro'),
                "f1": f1_score(labels, preds, average='macro')
            }            
        })
    return scores


# def tardir(path, tar_name):
#     with tarfile.open(tar_name, "w") as tar_handle:
#         for root, dirs, files in os.walk(path):
#             for file in files:
#                 tar_handle.add(os.path.join(root, file))
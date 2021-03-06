""" Usage
$ python inference.py --data_dir /data \
                      --model_dir /model \
                      --output_dir /output \
                      [args ...]
"""
import argparse
import os
import tarfile

import torch
from tqdm import utils
from dataloader import KlueDpDataLoader
from model import AutoModelforKlueDp
from transformers import AutoConfig, AutoTokenizer
from utils import flatten_prediction_and_labels
from utils import get_dp_labels, get_pos_labels


# 20211008
import re
from konlpy.tag import Mecab
m = Mecab()
import numpy as np

KLUE_DP_OUTPUT = "output.csv"  # the name of output file should be output.csv


def load_model(model_dir, args):
    # extract tar.gz
    # model_name = args.model_tar_file
    # tarpath = os.path.join(model_dir, model_name)
    # tar = tarfile.open(tarpath, "r:gz")
    # tar.extractall(path=model_dir)

    config = AutoConfig.from_pretrained(os.path.join(model_dir, "config.json"))
    model = AutoModelforKlueDp(config, args)
    model.load_state_dict(torch.load(os.path.join(model_dir, "dp-model.bin"), map_location='cpu'))
    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    # device setup
    num_gpus = torch.cuda.device_count()
    # use_cuda = num_gpus > 0
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    # load model
    model = load_model(model_dir, args)
    model.to(device)
    model.eval()

    # load KLUE-DP-test
    kwargs = {"num_workers": num_gpus, "pin_memory": True} if use_cuda else {}
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    klue_dp_dataset = KlueDpDataLoader(args, tokenizer, data_dir)
    klue_dp_test_loader = klue_dp_dataset.get_test_dataloader(
        args.test_filename, **kwargs
    )

    # inference
    predictions = []
    labels = []
    tokens = []
    for i, batch in enumerate(klue_dp_test_loader):
        input_ids, masks, ids, max_word_length = batch
        input_ids = input_ids.to(device)
        attention_mask, bpe_head_mask, bpe_tail_mask, mask_e, mask_d = (
            mask.to(device) for mask in masks
        )
        head_ids, type_ids, pos_ids = (id.to(device) for id in ids)
        #type id??? model??? ????????????
        #head id??? ??????????????? ?????? ???????????? output??? ????????? ????????? ?????????
        #bilinear ?????? ????????? ?????? training?????? ????????? ???????????? 
        # attention output??? ???????????? head_id?????? ??????
        batch_size, _ = head_ids.size()
        batch_index = torch.arange(0, batch_size).long()

        out_arc, out_type = model(
            bpe_head_mask,
            bpe_tail_mask,
            pos_ids,
            # head_ids,
            None,
            max_word_length,
            mask_e,
            mask_d,
            batch_index,
            input_ids,
            attention_mask,
        )

        heads = torch.argmax(out_arc, dim=2)
        types = torch.argmax(out_type, dim=2)
        
        heads_t = heads.detach().cpu().numpy()
        types_t = types.detach().cpu().numpy()

        for i, each_input in enumerate(input_ids.detach().cpu().numpy()):
            print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(each_input, skip_special_tokens=True)))
            index = [i for i, label in enumerate(head_ids[i]) if label == -1]
            print('HEAD: ',np.delete(np.array(heads_t[i]),index))
            print('DEPREL: ', [get_dp_labels()[k] for k in np.delete(np.array(types_t[i]),index)])
            print('DEPREL Index: ', np.delete(np.array(types_t[i]),index))


        prediction = (heads, types)
        predictions.append(prediction)
        # predictions are valid where labels exist
        label = (head_ids, type_ids)
        labels.append(label)
        

    head_preds, type_preds, _, _ = flatten_prediction_and_labels(predictions, labels)

    # write results to output_dir
    with open(os.path.join(output_dir, KLUE_DP_OUTPUT), "w", encoding="utf8") as f:
        for h, t in zip(head_preds, type_preds):
            f.write(" ".join([str(h), str(t)]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Container environment
    parser.add_argument(
        "--data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/home/tutor/DependencyParsingPractice/klue-dp_code/data")
    )
    parser.add_argument(
        "--model_dir", type=str, default="/home/tutor/DependencyParsingPractice/klue-dp_code/model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/home/tutor/DependencyParsingPractice/klue-dp_code/output"),
    )

    # inference arguments
    parser.add_argument(
        "--model_tar_file",
        type=str,
        default="klue_dp_model.tar.gz",
        help="it needs to include all things for loading baseline model & tokenizer, \
             only supporting transformers.AutoModelForSequenceClassification as a model \
             transformers.XLMRobertaTokenizer or transformers.BertTokenizer as a tokenizer",
    )
    parser.add_argument(
        "--test_filename",
        default="result-head_final.txt",
        # default="klue-dp-v1.1_test.tsv",
        type=str,
        help="Name of the test file (default: klue-dp-v1.1_test.tsv)",
    )
    parser.add_argument("--eval_batch_size", default=64, type=int)

    # model-specific arguments
    parser = AutoModelforKlueDp.add_arguments(parser)

    # parse args
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    inference(data_dir, model_dir, output_dir, args)

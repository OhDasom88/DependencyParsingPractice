import os
from typing import List, Optional

import torch
from torch.utils.data import Dataset, TensorDataset
from transformers import PreTrainedTokenizer
from utils import (KlueDpInputExample, KlueDpInputFeatures, get_dp_labels,
                   get_pos_labels)

# 20211002
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
class KlueDpDataset:
    def __init__(self, args, tokenizer):
        self.hparams = args
        self.tokenizer = tokenizer

    def _create_examples(self, file_path: str) -> List[KlueDpInputExample]:
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
                        + [token.replace("\n", "") for token in line.split("\t")]# INDEX	WORD_FORM	LEMMA	POS	HEAD	DEPREL
                        + ["-", "-"]#?
                    )# data column 
                    examples.append(
                        KlueDpInputExample(
                            guid=guid,
                            text=text,
                            sent_id=sent_id,
                            token_id=int(token_list[1]),
                            token=token_list[2],# the annotation is done at the word level.
                            pos=token_list[4],
                            head=token_list[5],
                            dep=token_list[6],
                        )
                    )
        return examples

    def _convert_features(
        self, examples: List[KlueDpInputExample]
    ) -> List[KlueDpInputFeatures]:
        return self.convert_examples_to_features(
            examples,
            self.tokenizer,
            max_length=self.hparams.max_seq_length,
            dep_label_list=get_dp_labels(),
            pos_label_list=get_pos_labels(),
        )

    def convert_examples_to_features(
        self,
        examples: List[KlueDpInputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        pos_label_list=None,
        dep_label_list=None,
    ):
        if max_length is None:
            max_length = tokenizer.max_len#AttributeError: 'BertTokenizerFast' object has no attribute 'max_len'

        pos_label_map = {label: i for i, label in enumerate(pos_label_list)}# 45
        dep_label_map = {label: i for i, label in enumerate(dep_label_list)}# 38

        SENT_ID = 0

        token_list = []
        pos_list = []
        head_list = []
        dep_list = []

        features = []
        for i, example in enumerate(tqdm(examples)):
            # at the end of the loop
            if i == len(examples) - 1:
                token_list.append(example.token)
                pos_list.append(example.pos.split("+")[-1])  # 맨 뒤 pos정보만 사용
                head_list.append(int(example.head))
                dep_list.append(example.dep)

            # if sentence index is changed or end of the loop
            if SENT_ID != example.sent_id or i == len(examples) - 1:
                SENT_ID = example.sent_id
                encoded = tokenizer.encode_plus(
                    " ".join(token_list),
                    None,
                    add_special_tokens=True,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )

                ids, mask = encoded["input_ids"], encoded["attention_mask"]
                # CLS token을 위한 padding 처리일것 같음
                bpe_head_mask = [0]# 각 DP 분석 단위(문장)별 subwords masking 정보(head)
                bpe_tail_mask = [0]# "상동" (tail): head와 tail은 word가 아니라 subword(tokenized )의 위치에 대응
                head_ids = [-1]# head_mapping 
                dep_ids = [-1]# DEPREL mapping, -1에 해당하는 id 값이 없음(0-37)
                pos_ids = [-1]  # --> CLS token # POS mapping, -1에 해당하는 id 값이 없음(0-45)
                for token, head, dep, pos in zip(# 단어(아마 split(' '))에 가까운 문자열을 subword(with tokenizer)로 mapping
                    token_list, head_list, dep_list, pos_list
                ):
                    bpe_len = len(tokenizer.tokenize(token))# DP 단위내 subword의 위치 정보 반영에 사용
                    head_token_mask = [1] + [0] * (bpe_len - 1)# 첫번째 subwords를 제외한 나머지 subwords masking
                    tail_token_mask = [0] * (bpe_len - 1) + [1]# 마지막 subwords를 제외한 나머지 subwords masking
                    bpe_head_mask.extend(head_token_mask)# head subwords masking 정보 축적
                    bpe_tail_mask.extend(tail_token_mask)# head subwords masking 정보 축적

                    head_mask = [head] + [-1] * (bpe_len - 1)
                    head_ids.extend(head_mask)# reference 정보 축적(head), parsing 단위별(ex 문장)
                    dep_mask = [dep_label_map[dep]] + [-1] * (bpe_len - 1)# dependecy label mapping
                    dep_ids.extend(dep_mask)# DEPREL 정보 축적, parsing 단위별(ex 문장)
                    pos_mask = [pos_label_map[pos]] + [-1] * (bpe_len - 1)# POS label mapping
                    pos_ids.extend(pos_mask)# POS 정보 축적, parsing 단위별(ex 문장)
                # DP 대상 + subword 위치 정보가 반영됨 + SEP token을 위한 masking 처리일것 같음
                # len([i for i in ids if i not in [1]]) == torch.tensor(mask).sum() 
                bpe_head_mask.append(0)#                == len(bpe_head_mask)
                bpe_tail_mask.append(0)#                == len(bpe_tail_mask)
                head_ids.append(-1)#                    == len(head_ids)
                dep_ids.append(-1)#                     == len(dep_ids)
                pos_ids.append(-1)  # END token         == len(pos_ids)
                if len(bpe_head_mask) > max_length:
                    bpe_head_mask = bpe_head_mask[:max_length]
                    bpe_tail_mask = bpe_tail_mask[:max_length]
                    head_ids = head_ids[:max_length]
                    dep_ids = dep_ids[:max_length]
                    pos_ids = pos_ids[:max_length]# end token 없음?

                else:# 왜 padding index가 같지 않지?
                    bpe_head_mask.extend(
                        [0] * (max_length - len(bpe_head_mask))
                    )  # padding by max_len
                    bpe_tail_mask.extend(
                        [0] * (max_length - len(bpe_tail_mask))
                    )  # padding by max_len
                    head_ids.extend(
                        [-1] * (max_length - len(head_ids))
                    )  # padding by max_len
                    dep_ids.extend(
                        [-1] * (max_length - len(dep_ids))
                    )  # padding by max_len
                    pos_ids.extend([-1] * (max_length - len(pos_ids)))

                feature = KlueDpInputFeatures(
                    guid=example.guid,
                    ids=ids,# len([i for i in ids if i not in [1]]) == torch.tensor(mask).sum()
                    mask=mask,
                    bpe_head_mask=bpe_head_mask,
                    bpe_tail_mask=bpe_tail_mask,
                    head_ids=head_ids,
                    dep_ids=dep_ids,
                    pos_ids=pos_ids,
                )
                features.append(feature)

                token_list = []
                pos_list = []
                head_list = []
                dep_list = []

            # always add token-level examples
            token_list.append(example.token)
            pos_list.append(example.pos.split("+")[-1])  # 맨 뒤 pos정보만 사용
            head_list.append(int(example.head))
            dep_list.append(example.dep)#DEPREL, type of the arc connecting the head and the current token.

        return features

    # def _create_dataset(self, data_dir: str, data_filename: str, mode) -> Dataset:
    def _create_dataset(self, data_dir: str, mode) -> Dataset:
        # Load data features from cache or dataset file
        cached_file_name = f'cached_DP_{mode}_features_{self.hparams.model_type}_{self.hparams.max_seq_length}'

        cached_features_file = os.path.join(data_dir, cached_file_name)
        
        if mode == 'train':
            batch_size = self.hparams.train_batch_size 
        elif mode == 'dev':
            batch_size = self.hparams.train_batch_size
        elif mode == 'test':
            batch_size = self.hparams.train_batch_size
        
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            if mode == 'train':
                data_filename = 'klue-dp-v1.1_train.tsv'
            elif mode == 'dev':
                data_filename = 'klue-dp-v1.1_dev.tsv'
            elif mode == 'test':
                data_filename = 'klue-dp-v1.1_dev_sample_10.tsv'

            file_path = os.path.join(data_dir, data_filename)

            logger.info("Creating features from dataset file at %s", file_path)

            examples = self._create_examples(file_path)
            features = self._convert_features(examples)

            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        bpe_head_mask = torch.tensor(
            [f.bpe_head_mask for f in features], dtype=torch.long
        )
        bpe_tail_mask = torch.tensor(
            [f.bpe_tail_mask for f in features], dtype=torch.long
        )
        head_ids = torch.tensor([f.head_ids for f in features], dtype=torch.long)
        dep_ids = torch.tensor([f.dep_ids for f in features], dtype=torch.long)
        pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)

        return TensorDataset(
            input_ids,
            attention_mask,
            bpe_head_mask,
            bpe_tail_mask,
            head_ids,
            dep_ids,
            pos_ids,
        ), batch_size

    # def get_test_dataset(
    #     self, data_dir: str, data_filename: str = "klue-dp-v1_test.tsv"
    # ) -> TensorDataset:
    #     file_path = os.path.join(data_dir, data_filename)
    #     return self._create_dataset(file_path, 'test')

    def get_dataset(
        self, mode, data_dir: str#, data_filename: str
    ) -> TensorDataset:
        # file_path = os.path.join(data_dir, data_filename)
        return self._create_dataset(data_dir, mode)

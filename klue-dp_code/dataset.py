import os
from typing import List, Optional

import torch
from torch.utils.data import Dataset, TensorDataset
from transformers import PreTrainedTokenizer
from utils import (KlueDpInputExample, KlueDpInputFeatures, get_dp_labels,
                   get_pos_labels)

####
from konlpy.tag import Mecab
m = Mecab()

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
                        for i, token in enumerate(text.split()):
                            pos_info=[pos for char, pos in m.pos(token)][-1]
                            examples.append(
                                KlueDpInputExample(
                                    guid=guid,# ex klue-dp-v1_dev_00001_wikitre
                                    text=text,# 'K팝스타3’ 유희열이 홍정희의 탈락에 눈물을 흘렸다.
                                    sent_id=sent_id,# 분석사례의 순서, 0번째 부터 
                                    token_id=int(i+1),# index(공백단위로 나뉜 문자열의 위치정보, 0은 root)    >> 전부 0으로 주자
                                    token=token,# 공백단위로 나뉜 문자열                                       >> text.split()[...]
                                    pos= pos_info if pos_info in get_pos_labels() else 'NA',# ex)'SS+SL+NNP+SN+SS', 공백단위로 나뉜 문자열의 형태소           >> ''.join(m.pos(text.split()[...])[:,1])
                                    head=0,# ex) 2(int), 공백단위로 나뉜 문자열이 참조하는 문자열의 위치        >> 전부 0으로 주자
                                    dep='',# ex) 'NP'-str, 참조하는 내용,                                 >> ''
                                )
                            )

                else:
                    token_list = []
                    token_list = (# sent_id, index	공백단위 split	LEMMA(형태소단위)	POS(형태소정보)	HEAD(참조 index)	DEPREL(관계정보)
                        [sent_id]
                        + [token.replace("\n", "") for token in line.split("\t")]
                        + ["-", "-"]
                    )
                    # examples.append(
                    #     KlueDpInputExample(
                    #         guid=guid,# ex klue-dp-v1_dev_00001_wikitre
                    #         text=text,# 'K팝스타3’ 유희열이 홍정희의 탈락에 눈물을 흘렸다.
                    #         sent_id=sent_id,# 분석사례의 순서, 0번째 부터 
                    #         token_id=int(token_list[1]),# index(공백단위로 나뉜 문자열의 위치정보, 0은 root)    >> 전부 0으로 주자
                    #         token=token_list[2],# 공백단위로 나뉜 문자열                                       >> text.split()[...]
                    #         pos=token_list[4],# ex)'SS+SL+NNP+SN+SS', 공백단위로 나뉜 문자열의 형태소           >> ''.join(m.pos(text.split()[...])[:,1])
                    #         head=token_list[5],# ex) '2', 공백단위로 나뉜 문자열이 참조하는 문자열의 위치        >> 전부 0으로 주자
                    #         dep=token_list[6],# ex) '동사관계', 참조하는 내용,                                 >> ''
                    #     )
                    # )
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
            max_length = tokenizer.max_len

        pos_label_map = {label: i for i, label in enumerate(pos_label_list)}
        dep_label_map = {label: i for i, label in enumerate(dep_label_list)}

        SENT_ID = 0

        token_list = []
        pos_list = []
        head_list = []
        dep_list = []

        features = []
        for i, example in enumerate(examples):
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

                bpe_head_mask = [0]
                bpe_tail_mask = [0]
                head_ids = [-1]
                dep_ids = [-1]
                pos_ids = [-1]  # --> CLS token

                for token, head, dep, pos in zip(
                    token_list, head_list, dep_list, pos_list
                ):
                    bpe_len = len(tokenizer.tokenize(token))
                    head_token_mask = [1] + [0] * (bpe_len - 1)
                    tail_token_mask = [0] * (bpe_len - 1) + [1]
                    bpe_head_mask.extend(head_token_mask)
                    bpe_tail_mask.extend(tail_token_mask)

                    head_mask = [head] + [-1] * (bpe_len - 1)# 예측해야 하는 값
                    head_ids.extend(head_mask)
                    # dep_mask = [dep_label_map[dep]] + [-1] * (bpe_len - 1)# 각 단어당 하나
                    dep_mask = [-1] * (bpe_len)# 각 단어당 하나# 예측해야 하는 값
                    dep_ids.extend(dep_mask)
                    pos_mask = [pos_label_map[pos]] + [-1] * (bpe_len - 1)
                    # pos_mask = [len(pos_label_map)] + [-1] * (bpe_len - 1)# len(pos_label_map) == pos_padding_idx
                    pos_ids.extend(pos_mask)

                bpe_head_mask.append(0)
                bpe_tail_mask.append(0)
                head_ids.append(-1)
                dep_ids.append(-1)
                pos_ids.append(-1)  # END token
                if len(bpe_head_mask) > max_length:
                    bpe_head_mask = bpe_head_mask[:max_length]
                    bpe_tail_mask = bpe_tail_mask[:max_length]
                    head_ids = head_ids[:max_length]
                    dep_ids = dep_ids[:max_length]
                    pos_ids = pos_ids[:max_length]

                else:
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
                    ids=ids,
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
            dep_list.append(example.dep)

        return features

    def _create_dataset(self, file_path: str) -> Dataset:
        examples = self._create_examples(file_path)
        features = self._convert_features(examples)

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
        )

    def get_test_dataset(
        self, data_dir: str, data_filename: str = "klue-dp-v1_test.tsv"
    ) -> TensorDataset:
        file_path = os.path.join(data_dir, data_filename)
        return self._create_dataset(file_path)

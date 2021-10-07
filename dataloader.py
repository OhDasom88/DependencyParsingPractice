import torch
from dataset import KlueDpDataset
from torch.utils.data import DataLoader
from utils import get_pos_labels, get_dp_labels
import logging

logger = logging.getLogger(__name__)


class KlueDpDataLoader(object):
    def __init__(self, args, tokenizer, data_dir):
        self.args = args
        self.data_dir = data_dir
        self.dataset = KlueDpDataset(args, tokenizer)

    def collate_fn(self, batch):
        # 1. set args
        batch_size = len(batch)
        pos_padding_idx = None if self.args.no_pos else len(get_pos_labels())
        # 2. build inputs : input_ids, attention_mask, bpe_head_mask, bpe_tail_mask
        input_ids = []
        attention_masks = []
        bpe_head_masks = []
        bpe_tail_masks = []
        for batch_id in range(batch_size):
            (
                input_id,# subword 기준 input_id[input_id != 1].shape, self.dataset.tokenizer.convert_ids_to_tokens(input_id)
                attention_mask, #       == attention_mask.sum()
                bpe_head_mask,#     bpe_head_mask.sum()
                bpe_tail_mask,#     == bpe_tail_mask.sum()
                _,
                _,
                _,
            ) = batch[batch_id]
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            bpe_head_masks.append(bpe_head_mask)
            bpe_tail_masks.append(bpe_tail_mask)
        # 2. build inputs : packing tensors
        input_ids = torch.stack(input_ids)# torch.Size([ , max_seq_length])
        attention_masks = torch.stack(attention_masks)# [len([i for i in row if i !=0]) for row in attention_masks ] == [len([i for i in row if i !=1]) for row in input_ids]
        bpe_head_masks = torch.stack(bpe_head_masks)#   [len([i for i in row if i !=0]) for row in bpe_head_masks ] (단어단위 not subword 단위)
        bpe_tail_masks = torch.stack(bpe_tail_masks)#   = [len([i for i in row if i !=0]) for row in bpe_tail_masks ] 
        # 3. token_to_words : set in-batch max_word_length
        max_word_length = max(torch.sum(bpe_head_masks, dim=1)).item()# 각 DP 사례별 unmasked 사례 건수(subwords + special tokens)의 최대값
        # 3. token_to_words : placeholders
        head_ids = torch.zeros(batch_size, max_word_length).long()# 
        type_ids = torch.zeros(batch_size, max_word_length).long()# 
        pos_ids = torch.zeros(batch_size, max_word_length + 1).long()# 첫 자리에 pos_padding_idx를 추가. 
        mask_e = torch.zeros(batch_size, max_word_length + 1).long()# 
        # 3. token_to_words : head_ids, type_ids, pos_ids, mask_e, mask_d
        for batch_id in range(batch_size):
            (
                _,
                _,
                bpe_head_mask,# subword 단위 head mask 정보, (0 or 1)
                _,
                token_head_ids,# -1: padding, 0: root?, DP 대상 문자열의 처음부터 1,2,3 ..
                token_type_ids,# DEPREL #  0,1,2,...37 << {label: i for i, label in enumerate(get_dp_labels())}
                token_pos_ids, # 0,1,2 ... 44 {label: i for i, label in enumerate(get_pos_labels())}
            ) = batch[batch_id]
            head_id = [i for i, token in enumerate(bpe_head_mask) if token == 1]# unmasked head의 index 정보(subwords 기준) , self.dataset.tokenizer.convert_ids_to_tokens(batch[batch_id][0][head_id])
            word_length = len(head_id)# len(head_id) == len(words for DP 대상) 각 head의 위치정보 subword << word 위치정보 
            head_id.extend([0] * (max_word_length - word_length))# padding으로 처리, 0 (subword token의 경우 [CLS]에 대응, ) 으로 처리
            head_ids[batch_id] = token_head_ids[head_id] # 각 word의 첫번째 subword에 해당하는 head 정보, token_head_ids의 0번 값은 'CLS' token의 head 정보인 -1로 mapping, torch.Size([max_word_length]) << torch.Size([max_seq_length]) 
            type_ids[batch_id] = token_type_ids[head_id] # 각 word의 첫번째 subword에 해당하는 type 정보, token_type_ids의 0번 값은 'CLS' token의 type 정보인 -1로 mapping, torch.Size([max_word_length]) << torch.Size([max_seq_length]) 
            if not self.args.no_pos:
                pos_ids[batch_id][0] = pos_padding_idx# 왜 앞에 따로 padding 처리? len(get_pos_labels()), pos 목록에는 해당 index 없음
                pos_ids[batch_id][1:] = token_pos_ids[head_id] # 각 word의 첫번째 subword에 해당하는 pos 정보, token_pos_ids의 0번 값은 'CLS' token의 head 정보인 -1로 mapping, torch.Size([max_word_length]) << torch.Size([max_seq_length]) 
                pos_ids[batch_id][torch.sum(bpe_head_mask) + 1 :] = pos_padding_idx#  torch.sum(bpe_head_mask) == word_length, 첫번째 padding과 head mask 정보를 반영해 POS padding 처리(뒷부분 padding 처리)
            mask_e[batch_id] = torch.LongTensor(# mask_e?
                [1] * (word_length + 1) + [0] * (max_word_length - word_length)# word_length +1 ??
            )
        mask_d = mask_e[:, 1:]# 뭐지?, torch.Size([ , max_word_length]) << torch.Size([ , max_word_length+1])
        # 4. pack everything
        masks = (attention_masks, bpe_head_masks, bpe_tail_masks, mask_e, mask_d)# 0~2: torch.Size([ , max_seq_length]), 3: torch.Size([ , max_word_length+1]), 4: torch.Size([ , max_word_length])
        ids = (head_ids, type_ids, pos_ids)# 0~1: torch.Size([ , max_word_length]),  2: torch.Size([ , max_word_length+1]),

        return input_ids, masks, ids, max_word_length# torch.Size([, max_seq_length]), tuple, tuple, int

    def get_dataloader(self, mode, **kwargs):
        
        dataset = self.dataset.get_dataset(mode, self.data_dir)
        # if mode == 'train':
        #     data_filename = 'klue-dp-v1.1_train.tsv'
        #     dataset = self.dataset.get_dataset(mode, self.data_dir, data_filename)
        #     batch_size = self.args.train_batch_size  
        # elif mode == 'dev':
        #     data_filename = 'klue-dp-v1.1_dev.tsv'
        #     dataset = self.dataset.get_dataset(mode, self.data_dir, data_filename)
        #     batch_size = self.args.eval_batch_size
        # elif mode == 'test':
        #     data_filename = 'klue-dp-v1.1_dev_sample_10.tsv'
        #     dataset = self.dataset.get_dataset(mode, self.data_dir, data_filename)
        #     batch_size = self.args.eval_batch_size
        
        if mode == 'train':
            batch_size = self.args.train_batch_size 
        elif mode == 'dev':
            batch_size = self.args.eval_batch_size
        elif mode == 'test':
            batch_size = self.args.eval_batch_size

        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", batch_size)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            **kwargs
        )

    # 원래 있던건데 학습모델 만들면서 위 함수로 변경 했다가 다시 수정
    def get_test_dataloader(self, data_filename: str = "klue-dp-v1_test.tsv", **kwargs):
        dataset = self.dataset.get_test_dataset(self.data_dir, data_filename)
        return DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            **kwargs
        )
import torch
from dataset import KlueDpDataset
from torch.utils.data import DataLoader
from utils import get_pos_labels
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
                input_id,# subword 기준 input_id[input_id != 1].shape
                attention_mask, #       == attention_mask.sum()
                bpe_head_mask,# bpe_head_mask.sum()
                bpe_tail_mask,# == 1(???) + bpe_tail_mask.sum()
                _,
                _,
                _,
            ) = batch[batch_id]
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            bpe_head_masks.append(bpe_head_mask)
            bpe_tail_masks.append(bpe_tail_mask)
        # 2. build inputs : packing tensors
        input_ids = torch.stack(input_ids)# torch.Size([batch_size, max_length])
        attention_masks = torch.stack(attention_masks)
        bpe_head_masks = torch.stack(bpe_head_masks)
        bpe_tail_masks = torch.stack(bpe_tail_masks)
        # 3. token_to_words : set in-batch max_word_length
        max_word_length = max(torch.sum(bpe_head_masks, dim=1)).item()# 각 DP 사례별 unmasked 사례 건수(subwords + special tokens)의 최대값
        # 3. token_to_words : placeholders
        head_ids = torch.zeros(batch_size, max_word_length).long()
        type_ids = torch.zeros(batch_size, max_word_length).long()
        pos_ids = torch.zeros(batch_size, max_word_length + 1).long()
        mask_e = torch.zeros(batch_size, max_word_length + 1).long()
        # 3. token_to_words : head_ids, type_ids, pos_ids, mask_e, mask_d
        for batch_id in range(batch_size):
            (
                _,
                _,
                bpe_head_mask,
                _,
                token_head_ids,
                token_type_ids,
                token_pos_ids,
            ) = batch[batch_id]
            head_id = [i for i, token in enumerate(bpe_head_mask) if token == 1]# unmasked인 경우 head index 정보?(subwords 기준)
            word_length = len(head_id)# len(head_id) == len(words for DP 대상)각 head의 위치정보는 subword << word 위치정보 
            head_id.extend([0] * (max_word_length - word_length))#padding
            head_ids[batch_id] = token_head_ids[head_id]
            type_ids[batch_id] = token_type_ids[head_id]
            if not self.args.no_pos:
                pos_ids[batch_id][0] = pos_padding_idx
                pos_ids[batch_id][1:] = token_pos_ids[head_id]
                pos_ids[batch_id][torch.sum(bpe_head_mask) + 1 :] = pos_padding_idx
            mask_e[batch_id] = torch.LongTensor(
                [1] * (word_length + 1) + [0] * (max_word_length - word_length)
            )
        mask_d = mask_e[:, 1:]
        # 4. pack everything
        masks = (attention_masks, bpe_head_masks, bpe_tail_masks, mask_e, mask_d)
        ids = (head_ids, type_ids, pos_ids)

        return input_ids, masks, ids, max_word_length

    def get_dataloader(self, mode, **kwargs):
        
        dataset, batch_size = self.dataset.get_dataset(mode, self.data_dir)
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
        
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            **kwargs
        )


import os
import shutil
import logging
from tqdm import tqdm, trange
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from model import AutoModelforKlueDp
import utils
import tarfile
logger = logging.getLogger(__name__)


class Trainer(object):
    # def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None, tokenizer = None):
    def __init__(self, args, dataset,  tokenizer = None):
        self.args = args
        # self.train_dataset = train_dataset
        # self.dev_dataset = dev_dataset
        # self.test_dataset = test_dataset
        self.dataset = dataset

        self.tokenizer = tokenizer

        # self.label_lst = get_labels(args)
        # self.num_labels = len(self.label_lst)
        
        
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        # self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

        # self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]

        # self.config = self.config_class.from_pretrained(args.model_name_or_path,
        #                                                 num_labels=self.num_labels,
        #                                                 finetuning_task=args.task,
        #                                                 id2label={str(i): label for i, label in enumerate(self.label_lst)},
        #                                                 label2id={label: i for i, label in enumerate(self.label_lst)})
        # self.model = self.model_class.from_pretrained(args.model_name_or_path, config=self.config)

        # device setup
        self.num_gpus = torch.cuda.device_count()
        self.use_cuda = self.num_gpus > 0
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # self.model = AutoModelForMaskedLM.from_pretrained("klue/roberta-large")
        config = AutoConfig.from_pretrained("klue/roberta-large")
        self.model = AutoModelforKlueDp(config, args)
        # self.model.load_state_dict("klue/roberta-large", map_location='cpu')
        #  = AutoModelforKlueDp.from_pretrained("klue/roberta-large", config=config, self.args)

        # GPU or CPU
        # self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

        # self.test_texts = None
        # if args.write_pred:
        #     self.test_texts = get_test_texts(args, tokenizer)
        #     # Empty the original prediction files
        #     if os.path.exists(args.pred_dir):
        #         shutil.rmtree(args.pred_dir)

    # def train(self):
    def train(self, wandb=None):
        # train_sampler = RandomSampler(self.train_dataset)
        # train_dataloader = DataLoader(
        #     self.train_dataset
        #     , sampler=train_sampler
        #     , batch_size=self.args.train_batch_size
        # )
        
        # load KLUE-DP-test
        kwargs = {"num_workers": self.num_gpus, "pin_memory": True} if self.use_cuda else {}

        train_dataloader = self.dataset.get_dataloader('train',  **kwargs)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        # logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):# batch:  input_ids, masks, ids, max_word_length
                self.model.train()
                input_ids, masks, ids, max_word_length = batch
                input_ids = input_ids.to(self.device)
                attention_mask, bpe_head_mask, bpe_tail_mask, mask_e, mask_d = (
                    mask.to(self.device) for mask in masks
                )
                head_ids, type_ids, pos_ids = (id.to(self.device) for id in ids)

                batch_size, _ = head_ids.size()
                batch_index = torch.arange(0, batch_size).long()

                out_arc, out_type = self.model(# loss 반환 x
                    bpe_head_mask,
                    bpe_tail_mask,
                    pos_ids,
                    head_ids,
                    max_word_length,
                    mask_e,
                    mask_d,
                    batch_index,
                    input_ids,
                    attention_mask,
                )

                loss_on_heads = torch.nn.functional.cross_entropy(out_arc.view(-1, out_arc.shape[-1]), head_ids.view(-1))
                loss_on_types = torch.nn.functional.cross_entropy(out_type.view(-1, out_type.shape[-1]), type_ids.view(-1))


                loss = loss_on_heads + loss_on_types

                # batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                # inputs = {'input_ids': batch[0],# len([i for i in batch[0][0] if i not in [1]]) == torch.tensor(batch[1][0]).sum()
                #           'attention_mask': batch[1],# (attention_masks, bpe_head_masks, bpe_tail_masks, mask_e, mask_d)
                #           'labels': batch[3]}
                # if self.args.model_type != 'distilkobert':
                #     inputs['token_type_ids'] = batch[2]

                # outputs = self.model(**inputs)
                # loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                    # if wandb !=None:
                    #     wandb.log({'loss':loss})

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate("test", global_step)

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode, step):
        # if mode == 'test':
        #     dataset = self.test_dataset
        # elif mode == 'dev':
        #     dataset = self.dev_dataset
        # else:
        #     raise Exception("Only dev and test dataset available")
        # eval_sampler = SequentialSampler(dataset)
        # eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # load KLUE-DP-test
        kwargs = {"num_workers": self.num_gpus, "pin_memory": True} if self.use_cuda else {}
        eval_dataloader = self.dataset.get_dataloader('dev', **kwargs)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        # logger.info("  Num examples = %d", len(dataset))
        # logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        # inference, 20211002
        predictions = []
        labels = []


        for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            input_ids, masks, ids, max_word_length = batch
            input_ids = input_ids.to(self.device)
            attention_mask, bpe_head_mask, bpe_tail_mask, mask_e, mask_d = (
                mask.to(self.device) for mask in masks
            )
            head_ids, type_ids, pos_ids = (id.to(self.device) for id in ids)

            batch_size, _ = head_ids.size()
            batch_index = torch.arange(0, batch_size).long()

            out_arc, out_type = self.model(# torch.Size([8, 25, 26]), torch.Size([8, 25, 63])
                bpe_head_mask,# torch.Size([8, 128])
                bpe_tail_mask,# torch.Size([8, 128])
                pos_ids,# torch.Size([8, 26])
                head_ids,# torch.Size([8, 25])
                max_word_length,# 25
                mask_e,# torch.Size([8, 26])
                mask_d,# torch.Size([8, 25])
                batch_index,# torch.Size([8])
                input_ids,# torch.Size([8, 128])
                attention_mask,# torch.Size([8, 128])
            )

            heads = torch.argmax(out_arc, dim=2)#torch.Size([8, 25]) << torch.Size([8, 25, 26])
            types = torch.argmax(out_type, dim=2)#torch.Size([8, 25]) << torch.Size([8, 25, 63])

            prediction = (heads, types)# (torch.Size([8, 25]), torch.Size([8, 25]))
            predictions.append(prediction)

            # predictions are valid where labels exist
            label = (head_ids, type_ids)# (torch.Size([8, 25]), torch.Size([8, 25]))
            labels.append(label)

            loss_on_heads = torch.nn.functional.cross_entropy(out_arc.view(-1, out_arc.shape[-1]), head_ids.view(-1))
            eval_loss += loss_on_heads.mean().item()
            # type 안맞음
            # loss_on_types = torch.nn.functional.cross_entropy(out_type.view(-1, out_type.shape[-1]), type_ids.view(-1))
            # eval_loss += loss_on_types.mean().item()
            
            
            # eval_loss += tmp_eval_loss.mean().item()

            # # batch = tuple(t.to(self.device) for t in batch)
            # with torch.no_grad():
            #     inputs = {'input_ids': batch[0],# eval batchsize, max_sequence
            #               'attention_mask': batch[1],# eval batchsize, max_sequence
            #               'labels': batch[3]}## eval batchsize, max_sequence
            #     if self.args.model_type != 'distilkobert':
            #         inputs['token_type_ids'] = batch[2]# eval batchsize, max_sequence
            #     outputs = self.model(**inputs)
            #     tmp_eval_loss, logits = outputs[:2]
            #     eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # # Slot prediction
            # if preds is None:
            #     preds = logits.detach().cpu().numpy()
            #     out_label_ids = inputs["labels"].detach().cpu().numpy()
            # else:
            #     preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            #     out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        
        head_preds, type_preds, head_labels, type_labels = utils.flatten_prediction_and_labels(predictions, labels)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        if self.args.write_pred:
            if not os.path.exists(self.args.output_dir):
                os.mkdir(self.args.output_dir)

            # write results to output_dir
            with open(os.path.join(self.args.output_dir, 'output_{}.csv'.format(step)), "w", encoding="utf8") as f:
                for h, t, hl,tl in zip(head_preds, type_preds, head_labels, type_labels):
                    f.write(" ".join([str(h),str(hl),str(t), str(tl)]) + "\n")

            # with open(os.path.join(self.args.pred_dir, "pred_{}.txt".format(step)), "w", encoding="utf-8") as f:
            #     for text, true_label, pred_label in zip(self.test_texts, out_label_list, preds_list):
            #         for t, tl, pl in zip(text, true_label, pred_label):
            #             f.write("{}\t{}\t{}\n".format(t, tl, pl))
            #         f.write("\n")

            # 20210924
            with open(os.path.join(self.args.output_dir, "report_{}.txt".format(step)), "w", encoding="utf-8") as f:
                f.write(utils.show_report(head_labels, head_preds))
                f.write("\n")
                f.write(utils.show_report(type_labels, type_preds))
                f.write("\n")

        result = utils.compute_metrics(head_labels, head_preds)
        results.update(result)
        result = utils.compute_metrics(type_labels, type_preds)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("\n" + utils.show_report(head_labels, head_preds))  # Get the report for each tag result
        logger.info("\n" + utils.show_report(type_labels, type_preds))  # Get the report for each tag result

        
        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        #20211001
        self.tokenizer.save_pretrained(self.args.model_dir)

        # 20211002?  https://tutorials.pytorch.kr/beginner/saving_loading_models.html#state-dict
        torch.save(self.model.state_dict(), self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    # def load_model(self):
    #     # Check whether model exists
    #     if not os.path.exists(self.args.model_dir):
    #         raise Exception("Model doesn't exists! Train first!")
    #     try:
    #         self.model = self.model_class.from_pretrained(self.args.model_dir)
    #         self.model.to(self.device)
    #         logger.info("***** Model Loaded *****")
    #     except:
    #         raise Exception("Some model files might be missing...")

    def load_model(self):
        # # extract tar.gz
        # model_name = self.args.model_tar_file
        # tarpath = os.path.join(self.args.model_dir, model_name)
        # tar = tarfile.open(tarpath, "r:gz")
        # tar.extractall(path=self.args.model_dir)

        # config = AutoConfig.from_pretrained(os.path.join(self.args.model_dir, "config.json"))
        # model = AutoModelforKlueDp(config, args)
        # model.load_state_dict(torch.load(os.path.join(self.args.model_dir, "dp-model.bin"), map_location='cpu'))
        model = self.model# 학습 모델이 없음 임시
        return model
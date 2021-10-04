import os
import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed#, MODEL_CLASSES, MODEL_PATH_MAP
# from data_loader import load_and_cache_examples
from dataloader import KlueDpDataLoader
from model import AutoModelforKlueDp


def main(args):

    init_logger()
    set_seed(args)  
    
    model_dir = args.model_dir
    data_dir = args.data_dir
    output_dir = args.output_dir

    tokenizer = load_tokenizer(args)
    # train_dataset = None
    # dev_dataset = None
    # test_dataset = None
    # if args.do_train or args.do_eval:
    #     test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    # if args.do_train:
    #     train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    # trainer = Trainer(args, train_dataset, dev_dataset, test_dataset, tokenizer)
    klue_dp_dataset = KlueDpDataLoader(args, tokenizer, data_dir)
    trainer = Trainer(args, klue_dp_dataset, tokenizer)

    if args.do_train:
        trainer.train()
        # trainer.train(wandb)
    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test", "sample")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default=f"{os.path.dirname(__file__)}/model", type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default=f"{os.path.dirname(__file__)}/data", type=str, help="The input data dir")
    parser.add_argument("--output_dir", default=f"{os.path.dirname(__file__)}/output", type=str, help="The prediction file dir")


    parser.add_argument("--write_pred", default=True, action="store_true", help="Write prediction during evaluation")

    parser.add_argument("--model_type", default="roberta-large", type=str)

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=8, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=256, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=512, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", default=True, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # model-specific arguments
    parser = AutoModelforKlueDp.add_arguments(parser)

    args = parser.parse_args()

    main(args)

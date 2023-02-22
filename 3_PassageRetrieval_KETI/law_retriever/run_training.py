# Copyright 2021 san kim
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import os
import time
import json
import shutil
import logging
import functools

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch import optim

import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter


from dataset_retriever import DatasetForRetriever, QueryPassageFormatter
from modeling_retriever import (
    T5SimpleMomentumRetriever,
    T5MeanMomentumRetriever,
    T5SimpleBiEncoderRetriever,
    T5MeanBiEncoderRetriever
)

_logger = logging.getLogger(__name__)


def compute_loss(model, batch, loss_fn, args):
    outputs = model(batch)
    # outputs: query, pos_ctx, neg_ctx - [batch_size, 1, model_dim]

    batch_size = outputs["query"].size(0)
    # in_batch negatives
    pos_scores = torch.mm(outputs["query"].squeeze(1), outputs["pos_ctx"].squeeze(1).t())
    # pos_scores(diagonal): [batch_size, batch_size]

    neg_scores = torch.mm(outputs["query"].squeeze(1), outputs["neg_ctx"].squeeze(1).t())
    # neg_scores: [batch_size, batch_size]

    scores = torch.cat([pos_scores, neg_scores], dim=1)
    # scores: [batch_size, 2 X batch_size]

    if args.momentum:
        # [batch_size, model_dim] x [queue_size, model_dim] = [batch_size, queue_size]
        # queue_neg_scores: [batch_size, queue_size]
        if args.distributed:
            queue_neg_scores = torch.mm(outputs["query"].squeeze(1), model.module.queue.clone().detach().t())
        else:
            queue_neg_scores = torch.mm(outputs["query"].squeeze(1), model.queue.clone().detach().t())


        # temperature for memorized negative samples
        if args.temperature != 1.0:
            queue_neg_scores = queue_neg_scores / args.temperature

        # scores: [batch_size, 2 X batch_size + queue_size]
        scores = torch.cat([scores, queue_neg_scores], dim=1)
        # enqueue positive_ctx. 
        # use the current positive context as negative context on next batch
        if args.distributed:
            model.module.dequeue_and_enqueue(outputs["pos_ctx"].squeeze(1).detach())
            model.module.momentum_update_key_encoder()
        else:
            model.dequeue_and_enqueue(outputs["pos_ctx"].squeeze(1).detach())
            model.momentum_update_key_encoder()

    # first hop supporting passage as ground truth for first hop
    # second hop supporting passage as ground truth for second hop
    target = torch.arange(batch_size).to(outputs["query"].device)

    retrieve_loss = loss_fn(scores, target)

    return retrieve_loss


def retrieval_eval(model, batch):
    outputs = model(batch)
    # outputs: query, pos_ctx, neg_ctx - [batch_size, 1, model_dim]

    batch_size = outputs["query"].size(0)
    # in_batch negatives
    pos_scores = torch.mm(outputs["query"].squeeze(1), outputs["pos_ctx"].squeeze(1).t())
    # pos_scores(diagonal): [batch_size, batch_size]

    neg_scores = torch.mm(outputs["query"].squeeze(1), outputs["neg_ctx"].squeeze(1).t())
    # neg_scores: [batch_size, batch_size]

    scores = torch.cat([pos_scores, neg_scores], dim=1)
    # scores: [batch_size, 2 X batch_size]

    target = torch.arange(batch_size).to(outputs["query"].device)

    # scores: [batch_size, 2 X batch_size]
    ranked = scores.argsort(dim=1, descending=True)
    # [[0.1, 0.3, -0.2, 0.14 ]] -> [[1, 3, 0, 2]] (index of score - descending order)
    idx2ranked_t = ranked.argsort(dim=1)

    # [[1, 3, 0, 2]] -> [[2, 0, 3, 1]] (index to rank)
    rrs = []
    for t, idx2ranked in zip(target, idx2ranked_t):
        rrs.append(1 / (idx2ranked[t].item() + 1))
    
    # reciprocal rank for 1st, 2nd hop
    return {
        "mrr": torch.tensor(np.mean(rrs)).to(outputs["query"].device)
        }


def make_dir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def create_directory_info(args, create_dir=True):

    model_dir = args.output_dir
    if args.dir_suffix is not None:
        model_dir = '_'.join([model_dir, args.dir_suffix])
    weights_dir = os.path.join(model_dir, "weights")
    logs_dir = os.path.join(model_dir, "logs")

    path_info = {
        'model_dir': model_dir,
        'weights_dir': weights_dir,
        'logs_dir': logs_dir,
    }

    if create_dir:
        for k, v in path_info.items():
            make_dir_if_not_exist(v)

    path_info['best_model_path'] = os.path.join(weights_dir, "best_model.pth")
    path_info['ckpt_path'] = os.path.join(weights_dir, "checkpoint.pth")
    return path_info


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename='checkpoint.pth', best_filename='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

def get_env_var(env_var, type_cls, default_val):
    if env_var in os.environ:
        return type_cls(os.environ[env_var])
    return default_val


MODEL_CLS = {
    "T5SimpleMomentumRetriever": {
        "model_cls": T5SimpleMomentumRetriever,
    },
    "T5MeanMomentumRetriever": {
        "model_cls": T5MeanMomentumRetriever,
    },
    "T5SimpleBiEncoderRetriever": {
        "model_cls": T5SimpleBiEncoderRetriever,
    },
    "T5MeanBiEncoderRetriever": {
        "model_cls": T5MeanBiEncoderRetriever,
    },
}

def main():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--pre_trained_model",
                        default="KETI-AIR/ke-t5-small", type=str)
    parser.add_argument("--tokenizer",
                        default="KETI-AIR/ke-t5-small", type=str)
    parser.add_argument("--config_path",
                        default="KETI-AIR/ke-t5-small", type=str)
    parser.add_argument("--model_cls", default="T5MeanBiEncoderRetriever", 
                        choices=["T5SimpleMomentumRetriever", 
                                "T5MeanMomentumRetriever", 
                                "T5SimpleBiEncoderRetriever", 
                                "T5MeanBiEncoderRetriever"],
                        type=str, help="model class")
    parser.add_argument("--dir_suffix",
                        default=None, type=str)
    parser.add_argument("--output_dir",
                        default="output", type=str)

    # resume
    parser.add_argument("--resume", default=None, type=str,
                        help="path to checkpoint.")
    parser.add_argument("--hf_path", default=None, type=str,
                        help="path to score huggingface model")
                    

    # default settings for training, evaluation
    parser.add_argument("--batch_size", default=64,
                        type=int, help="mini batch size")
    parser.add_argument("--workers", default=0, type=int,
                        help="number of workers")
    parser.add_argument("--print_freq", default=50,
                        type=int, help="print frequency")

    # distributed setting
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--local_world_size", type=int, default=1,
                        help="The size of the local worker group.")
    parser.add_argument("--rank", type=int, default=0,
                        help="The rank of the worker within a worker group.")
    parser.add_argument("--world_size", type=int, default=1,
                        help="world size. (num_nodes*num_dev_per_node)")
    parser.add_argument("--distributed", action='store_true',
                        help="is distributed training")

    # encoding
    parser.add_argument("--max_q_len", default=70, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--max_q_sp_len", default=350, type=int)
    parser.add_argument("--max_c_len", default=350, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    # dataset setting
    parser.add_argument("--remove_question_mark", action='store_true',
                        help="remove_question_mark")
    parser.add_argument("--wo_cls_token", action='store_false',
                        help="wo_cls_token")
    parser.add_argument("--wo_sep_token", action='store_false',
                        help="wo_sep_token")
    parser.add_argument("--add_token_type_ids", action='store_false',
                        help="add_token_type_ids")
    parser.add_argument("--cls_token_id", type=int, default=-1,
                        help="cls_token_id")
    parser.add_argument("--sep_token_id", type=int, default=-1,
                        help="sep_token_id")
    
    parser.add_argument("--train_file", type=str,
                        default="./law_data/retriever/law_qa_retriever_clean-train.json")
    parser.add_argument("--dev_file", type=str,
                        default="./law_data/retriever/law_qa_retriever_clean-dev.json")

    # default settings for training
    parser.add_argument("--epochs", default=80, type=int,
                        help="number of epochs for training")
    parser.add_argument("--start_epoch", default=0,
                        type=int, help="start epoch")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--learning_rate", default=1e-4,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup", default=0.2,
                        type=float, help="warm-up proportion for linear scheduling")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--negative_sample", default="hard_negative_ctxs",
                        type=str, choices=[
                            "negative_ctxs", "hard_negative_ctxs"],
                        help="negative sample for training")
    parser.add_argument("--extend_negative", action='store_true',
                        help="extend_negative")
    
    parser.add_argument("--off_scheduling", action='store_false',
                        help="off_scheduling")
    
    # ddp settings for sync
    parser.add_argument("--seed", default=0,
                        type=int, help="seed for torch manual seed")
    parser.add_argument("--deterministic", action='store_true',
                        help="deterministic")

    # momentum retriever
    parser.add_argument("--momentum", action="store_true")
    parser.add_argument("--k", type=int, default=38400,
                        help="memory bank size")
    parser.add_argument("--m", type=float, default=0.999, help="momentum")

    args = parser.parse_args()

    args.local_rank = get_env_var('LOCAL_RANK', int, args.local_rank)
    args.local_world_size = get_env_var('LOCAL_WORLD_SIZE', int, args.local_world_size)
    args.rank = get_env_var('RANK', int, args.rank)
    args.world_size = get_env_var('WORLD_SIZE', int, args.world_size)

    # check world size
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.local_rank = int(os.environ['LOCAL_RANK'])
    
    # create directory and summary logger
    best_score = 0
    summary_logger = None
    if args.local_rank == 0 or not args.distributed:
        path_info = create_directory_info(args)
        summary_logger = SummaryWriter(path_info["logs_dir"])
    path_info = create_directory_info(args, create_dir=False)

    # if the world size is bigger than 1, init process group(sync)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
    
    # device
    device = torch.device('cuda')

    # deterministic seed
    if args.deterministic:
        torch.manual_seed(args.seed)
        data_seed = args.seed
    else:
        data_seed = torch.randint(9999, (1,), device=device, requires_grad=False)
        if args.distributed:
            data_seed = broadcast(data_seed)
        data_seed = data_seed.cpu().item()
        _logger.info("[rank {}]seed for data: {}".format(args.rank if args.distributed else 0, data_seed))

    # update batch_size per a device
    args.batch_size = int(
        args.batch_size / args.gradient_accumulation_steps)

    # get model class
    model_cls_cfg = MODEL_CLS[args.model_cls]
    model_cls = model_cls_cfg["model_cls"]

    # create model
    model = model_cls(args)
    if model_cls._RETRIEVER_TYPE == "momentum":
        args.momentum = True
    model = model.cuda()

    # get optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay
    )

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # create query formatter
    qp_formatter = QueryPassageFormatter(
        tokenizer,
        max_q_len=args.max_q_len,
        max_q_sp_len=args.max_q_sp_len,
        max_c_len=args.max_c_len,
        remove_question_mark=args.remove_question_mark,
        add_cls_token=args.wo_cls_token,
        add_sep_token=args.wo_sep_token,
        add_token_type_ids=args.add_token_type_ids,
        cls_token_id=args.cls_token_id,
        sep_token_id=args.sep_token_id,
    )

    # create dataset
    train_dataset = DatasetForRetriever(
        qp_formatter, 
        args.train_file,
        negative_sample=args.negative_sample,
        extend_negative=args.extend_negative,
        )
    test_dataset = DatasetForRetriever(
        qp_formatter, 
        args.dev_file,
        extend_negative=args.extend_negative,
        max_num_of_data=500
        )

    collate_fn = test_dataset.get_collate_fn()
    
    
    # create sampler for distributed data loading without redundant
    train_sampler = None
    test_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            seed=data_seed)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset,
            seed=data_seed)

    # create data loader
    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=(train_sampler is None),
                            num_workers=args.workers,
                            sampler=train_sampler,
                            collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             sampler=test_sampler,
                             collate_fn=collate_fn)
    
    
    # learning rate scheduler
    scheduler = None
    if args.off_scheduling:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            epochs=args.epochs,
            last_epoch=-1,
            steps_per_epoch=int(len(train_loader)/args.gradient_accumulation_steps),
            pct_start=args.warmup,
            anneal_strategy="linear"
        )


    # wrap model using DDP
    if args.distributed:
        model = DDP(model,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank)
    
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                _logger.info("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(
                    args.resume, map_location=lambda storage, loc: storage.cuda(args.local_rank))
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint and scheduler is not None:
                    scheduler.load_state_dict(checkpoint['scheduler'])

                if args.local_rank == 0 or not args.distributed:
                    best_score = checkpoint['best_score']
                _logger.info("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            elif args.resume.lower()=='true':
                args.resume = path_info['ckpt_path']
                resume()
            elif args.resume.lower()=='best':
                args.resume = path_info['best_model_path']
                resume()
            else:
                _logger.info("=> no checkpoint found at '{}'".format(args.resume))
        resume()
    

    # save model as huggingface model
    if args.hf_path:
        if args.hf_path.lower()=='default':
            args.hf_path = os.path.join(path_info["model_dir"], "hf")

        if args.local_rank == 0 and args.distributed:
            model.module.save_pretrained(args.hf_path)
            _logger.info('hf model is saved in {}'.format(args.hf_path))
        elif not args.distributed:
            model.save_pretrained(args.hf_path)
            _logger.info('hf model is saved in {}'.format(args.hf_path))
        exit()


    # run training
    for epoch in range(args.start_epoch, args.epochs):
        # set epoch to train sampler 
        # for different order of example between epochs
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # training
        train(train_loader, model, optimizer, scheduler, epoch, args, summary_logger=summary_logger)

        scores = validate(test_loader, model, epoch, args)

        if args.local_rank == 0 or not args.distributed:
            curr_best = max(scores['mrr'].item(), best_score)
            is_best = curr_best > best_score
            if is_best:
                best_score = curr_best
                best_result = {k: v.item() for k, v in scores.items()}
                best_result["epoch"] = epoch
                with open(os.path.join(path_info["model_dir"], "best_score.json"), "w") as f:
                    json.dump(best_result, f, indent=4)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_score': best_score,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else scheduler,
            }, is_best,
                path_info["ckpt_path"],
                path_info["best_model_path"])

            summary_logger.add_scalar('eval/loss',
                                    scores['loss'], epoch)
            summary_logger.add_scalar('eval/mrr',
                                scores['mrr'], epoch)
    
    hf_dir = os.path.join(path_info["model_dir"], "hf")
    if args.local_rank == 0 and args.distributed:
        model.module.save_pretrained(hf_dir)
        tokenizer.save_pretrained(hf_dir)
        _logger.info('hf model is saved in {}'.format(hf_dir))
    elif not args.distributed:
        model.save_pretrained(hf_dir)
        tokenizer.save_pretrained(hf_dir)
        _logger.info('hf model is saved in {}'.format(hf_dir))


def train(train_loader, model, optimizer, scheduler, epoch, args, summary_logger=None):
    loss_fn = CrossEntropyLoss(ignore_index=-1)

    # calc batch time
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    steps_per_epoch = len(train_loader)

    # switch to train mode
    model.train()
    end = time.time()

    # zero grad
    optimizer.zero_grad()

    for step_inbatch, batch in enumerate(train_loader):
        # compute loss
        loss = compute_loss(model, batch, loss_fn, args)

        # backward pass            
        loss.backward()
        if (step_inbatch + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            # schedule learning rate
            if scheduler is not None:
                scheduler.step()

        losses.update(loss)

        global_step = epoch*steps_per_epoch + step_inbatch
        if (global_step + 1) % args.print_freq == 0:
            with torch.no_grad():
                batch_time.update((time.time() - end)/args.print_freq)
                end = time.time()

                avg_loss = reduce_tensor(losses.avg, args)

                mrr = retrieval_eval(model, batch)
                avg_mrr = reduce_tensor(mrr["mrr"], args)

                if args.local_rank == 0 or not args.distributed:

                    summary_logger.add_scalar('train/loss',
                                      avg_loss, global_step)
                    summary_logger.add_scalar('train/mrr',
                                      avg_mrr, global_step)

                    score_log = "loss\t{:.3f}\t mrr\t{:.3f}\n".format(
                        avg_loss.item(), avg_mrr.item()
                    )

                    _logger.info('-----Training----- \nEpoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Speed {3:.3f} ({4:.3f})\t'.format(
                              epoch, step_inbatch, steps_per_epoch,
                              args.batch_size/batch_time.val,
                              args.batch_size/batch_time.avg,
                              batch_time=batch_time)+score_log)


def validate(eval_loader, model, epoch, args):
    # loss function
    loss_fn = CrossEntropyLoss(ignore_index=-1)

    steps_per_epoch = len(eval_loader)

    # score meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    mrr_meter = AverageMeter()    

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for step_inbatch, batch in enumerate(eval_loader):
            loss = compute_loss(model, batch, loss_fn, args)
            mrr = retrieval_eval(model, batch)
            losses.update(loss)
            mrr_meter.update(mrr["mrr"])

            if step_inbatch % args.print_freq == (args.print_freq - 1):
                batch_time.update((time.time() - end)/min(args.print_freq, step_inbatch + 1))
                end = time.time()

                avg_loss = reduce_tensor(losses.avg, args)
                avg_mrr = reduce_tensor(mrr_meter.avg, args)

                if args.local_rank == 0 or not args.distributed:

                    score_log = "loss\t{:.3f}\t mrr\t{:.3f}\n".format(
                        avg_loss.item(), avg_mrr.item()
                    )

                    _logger.info('-----Evaluation----- \nEpoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Speed {3:.3f} ({4:.3f})\t'.format(
                              epoch, step_inbatch, steps_per_epoch,
                              min(args.print_freq, step_inbatch + 1)/batch_time.val,
                              min(args.print_freq, step_inbatch + 1)/batch_time.avg,
                              batch_time=batch_time)+score_log)
    
    scores = {
        "loss": reduce_tensor(losses.avg, args),
        "mrr": reduce_tensor(mrr_meter.avg, args)
    }
    score_log = "loss\t{:.3f}\t mrr\t{:.3f}\n".format(
                        scores["loss"].item(), scores["mrr"].item()
                    )
    
    if args.local_rank == 0 or not args.distributed:
        _logger.info('-----Evaluation----- \nEpoch: [{0}]\t'.format(
                                epoch)+score_log)

    return scores



def broadcast(tensors, rank=0):
    rt = tensors.clone()
    torch.distributed.broadcast(rt, rank)
    return rt

def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def reduce_sum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt




if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)

    main()



# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=8 training_retriever.py \
# --tokenizer KETI-AIR/ke-t5-small \
# --pre_trained_model KETI-AIR/ke-t5-small \
# --config_path KETI-AIR/ke-t5-small \
# --model_cls T5MeanBiEncoderRetriever \
# --extend_negative \
# --dir_suffix ext \
# --train_file ./law_data/retriever/law_qa_retriever_clean-train.json \
# --dev_file ./law_data/retriever/law_qa_retriever_clean-dev.json

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=8 training_retriever.py \
# --tokenizer KETI-AIR/ke-t5-small \
# --pre_trained_model KETI-AIR/ke-t5-small \
# --config_path KETI-AIR/ke-t5-small \
# --model_cls T5MeanBiEncoderRetriever \
# --extend_negative \
# --dir_suffix ext_e160 \
# --epochs 160 \
# --train_file ./law_data/retriever/law_qa_retriever_clean-train.json \
# --dev_file ./law_data/retriever/law_qa_retriever_clean-dev.json


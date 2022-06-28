
# !/usr/bin/python
# Copyright 2022 VMware, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import tqdm
import torch
import pathlib
import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tfrecord.torch.dataset import TFRecordDataset
from accelerate import Accelerator
from transformers import BertForPreTraining, get_linear_schedule_with_warmup


class Pretraining_Config:
    def __init__(self):
        # Bert base / large model
        self.is_base = True

        # Name of your bert model
        self.model_name = 'CUSTOM_BERT'

        # How often to save the model checkpoint.
        self.save_intermediate_checkpoints_steps = 25000

        # How often to perform evaluation
        self.eval_point = 10000

        # Training batch_size
        self.train_batch_size = 258

        # Evaluation Batch size
        self.eval_batch_size = 258

        # MSL
        self.max_seq_length = 128
        if self.max_seq_length == 128:
            self.max_predictions_per_seq = 20
        else:
            self.max_predictions_per_seq = 75

        # If you want to split your training tf_record file into train, eval
        # mention the train split percentage
        self.split_ratio = None

        # Maximum number of eval steps to evaluate the data for
        self.max_eval_steps = 10000

        # save checkpoint during training
        self.save_intermediate_checkpoints = True

        # number of training steps
        self.num_train_steps = 10000

        # number of warmup steps
        self.num_warmup_steps = 10

        # learning rate
        self.learning_rate = 1e-5

        # initialization checkpoint, with a bert directory or huggingface model
        self.init_checkpoint = ''
        

        # Tf_record file
        self.input_file = "./input/demo_MSL128.tfrecord"

        # Evaluation tf_record file
        # (if not provided will perform eval based on training set)
        self.eval_file = "./input/demo_MSL128.tfrecord"

        # Checkpoints saving directory
        self.output_dir = './ckpts'

        # File which logs the training results at each evaluation step
        self.log_csv = './eval_results.csv'

        # Number of gpus to run the training on
        self.num_gpu = 3

    def __str__(self) -> str:
        hp_string = '\t'
        hp_string += 'model_name:\t'+str(self.model_name)+'\n\t'
        hp_string += 'is_base:\t'+str(self.is_base)+'\n\t'

        # hp_string+='\n\t'
        hp_string += 'max_seq_length:\t'+str(self.max_seq_length)+'\n\t'
        hp_string += 'max_predictions_per_seq:\t' + \
                     str(self.max_predictions_per_seq)+'\n\t'
        hp_string += 'num_train_steps:\t'+str(self.num_train_steps)+'\n\t'
        hp_string += 'num_warmup_steps:\t'+str(self.num_warmup_steps)+'\n\t'
        hp_string += 'learning_rate:\t'+str(self.learning_rate)+'\n\t'
        hp_string += 'train_batch_size:\t'+str(self.train_batch_size)+'\n\t'
        hp_string += 'save_intermediate_checkpoints:\t' + \
                     str(self.save_intermediate_checkpoints)+'\n\t'
        hp_string += 'save_intermediate_checkpoints_steps:\t' + \
                     str(self.save_intermediate_checkpoints_steps)+'\n\t'
        hp_string += 'eval_batch_size:\t'+str(self.eval_batch_size)+'\n\t'
        hp_string += 'max_eval_steps:\t'+str(self.max_eval_steps)+'\n\t'
        hp_string += 'eval_point:\t'+str(self.eval_point)+'\n\t'
        hp_string += 'split_ratio:\t'+str(self.split_ratio)+'\n\t'

        # hp_string+='\n\t'
        hp_string += 'init_checkpoint:\t'+str(self.init_checkpoint)+'\n\t'
        hp_string += 'input_file:\t'+str(self.input_file)+'\n\t'
        hp_string += 'eval_file:\t'+str(self.eval_file)+'\n\t'

        # hp_string+='\n\t'
        hp_string += 'log_csv:\t'+str(self.log_csv)+'\n\t'
        hp_string += 'output_dir:\t'+str(self.output_dir)+'\n\t'

        # hp_string+='\n\t'
        hp_string += 'num_gpu:\t'+str(self.num_gpu)+'\n\t'

        return hp_string


class BertPretrainDataset(Dataset):
    "add class comment"
    def __init__(self):
        self.masked_lm_positions = []
        self.masked_lm_weights = []
        self.segment_ids = []
        self.masked_lm_ids = []
        self.input_mask = []
        self.next_sentence_labels = []
        self.input_ids = []
        self.mlm_label = []

    def __getitem__(self, index):
        mlmp = self.masked_lm_positions[index]
        mlmw = self.masked_lm_weights[index]
        mlmi = self.masked_lm_ids[index]
        si = self.segment_ids[index]
        im = self.input_mask[index]
        nsl = self.next_sentence_labels[index]
        ii = self.input_ids[index]
        mlml = self.mlm_label[index]
        return {'masked_lm_positions': mlmp, 'masked_lm_weights': mlmw,
                'segment_ids': si,
                'masked_lm_ids': mlmi, 'input_mask': im,
                'next_sentence_labels': nsl,
                'input_ids': ii, 'mlm_label': mlml}

    def __len__(self):
        return len(self.input_mask)

    def load_tfrecord(self, file, bs=512, accelerator=None):
        """ Load the pre_training tfrecord file

        Args:
            file (str): pretraining tfrecord file created using bert
                        create_pretraining_data.py
            bs (int, optional): Batch size. Defaults to 512.
            accelerator (accelerate.accelerator, optional): The huggingface\
                                        accelerator object . Defaults to None.
        """
        dataset = TFRecordDataset(file, None, transform=mlm_targets)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=bs)
        count = 0
        try:
            for x in train_loader:
                for y in range(0, x['masked_lm_positions'].shape[0]):
                    self.masked_lm_positions.append(x['masked_lm_positions']
                                                    [y])
                    self.masked_lm_weights.append(x['masked_lm_weights'][y])
                    self.segment_ids.append(x['segment_ids'][y])
                    self.masked_lm_ids.append(x['masked_lm_ids'][y])
                    self.input_mask.append(x['input_mask'][y])
                    self.next_sentence_labels.append(x['next_sentence_labels']
                                                     [y])
                    self.input_ids.append(x['input_ids'][y])
                    self.mlm_label.append(x['mlm_label'][y])
                count += x['masked_lm_positions'].shape[0]
                if accelerator:
                    accelerator.print('\r loading data point number:\t' +
                                      str(count), end='')
        except Exception as e:
            accelerator.print(str(e))
            return
        if accelerator:
            accelerator.print('...\n')


def setup_model(pretraining_config, logger):
    """ Load the base model to perform pretraining on

    Args:
        pretraining_config (Pretraining_Config):
                    Config file with pretraining parameters
        logger (_type_): _description_

    Returns:
        BertForPreTraining model
    """
    logger.info('SETTING UP MODEL: {} \t from config_path: {} '
                .format(pretraining_config.is_base,
                        pretraining_config.init_checkpoint))
    if not pretraining_config.init_checkpoint:
        if pretraining_config.is_base:
            model = BertForPreTraining.from_pretrained('bert-base-uncased')
        else:
            model = BertForPreTraining.from_pretrained('bert-large-uncased')
    else:
        model = BertForPreTraining.from_pretrained(pretraining_config.
                                                   init_checkpoint)
    return model


def load_dataset(tf_record_path, batch_size, shuffle=False):
    """Load tfrecord dataset

    Args:
        tf_record_path (str): file path to the tf_record file
        batch_size (int): batch_size
        shuffle (bool, optional): Shuffle the tfrecord vectors.
                                 Defaults to False.

    Returns:
        torch.utils.data.DataLoader: Returns a data loader
    """
    index_path = None
    if shuffle:
        dataset = TFRecordDataset(tf_record_path, index_path,
                                  transform=mlm_targets,
                                  shuffle_queue_size=1024)
    else:
        dataset = TFRecordDataset(tf_record_path, index_path,
                                  transform=mlm_targets)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return loader


def load_dataset_split(tf_record_path, train_batch_size, eval_batch_size,
                       split_ratio, accelerator=None):
    """ Load dataset and split it into training / evaluation file

    Args:
        tf_record_path (str): file path to the tf_record file
        train_batch_size (int): batch_size for the train loader
        eval_batch_size (int): batch_size for eval loader
        split_ratio (float): decides what percent/100 of the dataset
                             goes into training (value<1)
        accelerator (Accelerate.accelerator, optional):
                                        HuggingFace accelerator class
                                                    Defaults to None.

    Returns:
        torch.utils.dataloader, torch.utils.dataloader: train_loader,
                                                        eval_loader
    """

    ds_loader_size = 2048
    full_dataset = BertPretrainDataset()
    full_dataset.load_tfrecord(tf_record_path, bs=ds_loader_size,
                               accelerator=accelerator)

    train_len = int(len(full_dataset)*split_ratio)
    eval_len = len(full_dataset)-train_len
    train_set, eval_set = torch.utils.data.random_split(full_dataset,
                                                        [train_len, eval_len],
                                                        generator=torch.
                                                        Generator().
                                                        manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=train_batch_size)
    eval_loader = torch.utils.data.DataLoader(eval_set,
                                              batch_size=eval_batch_size)
    return train_loader, eval_loader


def mlm_targets(data):
    """ Transforms tfrecord file inputs into a consumable format
    """
    target = np.ones(data['input_ids'].shape, dtype=np.int64)
    target = target*-100
    target[data['masked_lm_positions']] = data['masked_lm_ids']
    data['mlm_label'] = target
    return data


def mlm_accuracy(prediction_logits, ground_truth, masked_positions):
    """Computes the Masked Language Modelling accuracy

    Args:
        prediction_logits (torch tensor): torch tensor output by the bert model
        ground_truth (torch tensor): ground truth values at the masked position
        masked_positions (torch tensor): indices of the input tensor which
                                        were masked for pretraining

    Returns:
        float: MLM accuracy
    """

    predictions = torch.argmax(prediction_logits, 2)
    predict = torch.zeros(predictions.shape[0],
                          masked_positions.shape[1]).to(predictions.device)
    for i in range(predictions.shape[0]):
        predict[i] = torch.index_select(predictions[i], 0, masked_positions[i])
    matches = torch.sum(predict == ground_truth, dim=1)/ground_truth.shape[1]
    accuracy = torch.mean(matches)
    return accuracy.item()


def nsp_accuracy(predictions, ground_truth):
    """Computes the Next sentence prediction accuracy

    Args:
        predictions (torch tensor): torch tensor output by the bert model
        ground_truth (torch tensor): ground truth value

    Returns:
        float: NSP accuracy
    """
    predicts = torch.argmax(predictions, dim=1)
    ground_truth = ground_truth.squeeze()
    return (torch.sum((predicts == ground_truth))/predictions.shape[0]).item()


def train(pretraining_config, model, train_data, optim, scheduler, accelerator,
          logger, eval_data=None):
    """ Training function

    Args:
        pretraining_config (Pretraining_config): Pretraining_config with
                                                 pretraining parameters
        model (BertForPreTraining): bert model class
        train_data (torch.utils.data.dataloader): dataloader
        optim (torch.optim.AdamW): AdamW optimizer
        scheduler (get_linear_schedule_with_warmup): scheduler for learning
                                                     rate
        accelerator (accelerate.Accelerator): HuggingFace's accelerator
        logger (logging): logger
        eval_data (torch.utils.data.dataloader, optional):
                    Evaluation dataloader. Defaults to None.

    Returns:
        list[dict{logged_results}]: returns the evaluation results
    """
    step_count = 0
    TOTAL_STEPS = None
    # get total steps
    if pretraining_config.num_train_steps:
        TOTAL_STEPS = pretraining_config.num_train_steps +\
                      pretraining_config.num_warmup_steps
    # prepare model for training
    model.train()
    logger.info('-*'*100)
    logger.info('TRAINING')
    accelerator.print("_TRAINING BERT MODEL_")
    accelerator.print(pretraining_config.model_name)
    logging_list = []
    losses = []
    # setup tqdm progress bars
    if TOTAL_STEPS:
        progress_bar = tqdm.auto.tqdm(total=TOTAL_STEPS, desc="TRAINING:\t",
                                      disable=not accelerator.is_main_process,
                                      position=0)
    else:
        progress_bar = tqdm.auto.tqdm(total=len(train_data),
                                      desc="TRAINING:\t",
                                      disable=not accelerator.is_main_process,
                                      position=0)
    write_bar = tqdm.auto.tqdm(total=0, bar_format='{desc}',
                               disable=not accelerator.is_main_process,
                               position=1)

    for batch in train_data:
        if TOTAL_STEPS:
            if step_count == TOTAL_STEPS:
                break
        input_ids = batch['input_ids']
        attention_mask = batch['input_mask']
        token_type_ids = batch['segment_ids']
        next_sentence_label = batch['next_sentence_labels']
        target = batch['mlm_label']

        # get the model output
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       token_type_ids=token_type_ids,
                       labels=target, next_sentence_label=next_sentence_label)

        # compute loss and perform backprop
        loss = output.loss
        accelerator.backward(loss)
        optim.step()
        scheduler.step()
        optim.zero_grad()

        losses.append(output.loss.item())
        step_count += 1
        progress_bar.update(1)
        write_bar.set_description_str('Training Loss : {} \r'.
                                      format(output.loss.item()))

        # save checkpoint
        if pretraining_config.save_intermediate_checkpoints:
            if (step_count-pretraining_config.num_warmup_steps) % (
                pretraining_config.save_intermediate_checkpoints_steps) == 0 and (
                    step_count-pretraining_config.num_warmup_steps) != 0:
                accelerator.wait_for_everyone()
                accelerator.print('Saving Checkpoint')
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(os.path.join(
                        pretraining_config.output_dir, 'Steps_' +
                        str(step_count-pretraining_config.num_warmup_steps)),
                        save_function=accelerator.save)

        # perform evaluation
        if (step_count-pretraining_config.num_warmup_steps) % \
                (pretraining_config.eval_point) == 0:
            eval_outputs = evaluate(pretraining_config, model, eval_data,
                                    accelerator, logger,
                                    step_count -
                                    pretraining_config.num_warmup_steps)

            if accelerator.is_main_process:
                # stores evaluation results
                log_dict = {}
                log_dict['MODEL_NAME'] = pretraining_config.model_name
                log_dict['STEPS'] = step_count - pretraining_config.\
                    num_warmup_steps
                log_dict['WARMUP_STEPS'] = pretraining_config.num_warmup_steps
                log_dict['LR'] = pretraining_config.learning_rate
                log_dict['BATCH_SIZE'] = pretraining_config.train_batch_size
                log_dict["TRAIN_STEPS"] = pretraining_config.\
                    train_batch_size*(step_count-pretraining_config.
                                      num_warmup_steps)
                log_dict['TRAIN_LOSS'] = np.mean(losses)
                log_dict['EVAL_LOSS'] = eval_outputs[1]
                log_dict['EVAL_NSP_A'] = eval_outputs[2]
                log_dict['EVAL_MLM_A'] = eval_outputs[3]
                logging_list.append(log_dict)
            logger.info('\n')
            logger.info('\t Train_loss: {} \t after Train_steps: {}'.
                        format(np.mean(losses),
                               step_count-pretraining_config.num_warmup_steps))

    logger.info('\n')
    logger.info('\t Train_loss: {} \t Train_steps: {}'.format(
                np.mean(losses), step_count))

    # eval at end of training
    eval_outputs = evaluate(pretraining_config, model, eval_data, accelerator,
                            logger,
                            step_count-pretraining_config.num_warmup_steps)

    if accelerator.is_main_process:
        # stores evaluation results
        log_dict = {}
        log_dict['MODEL_NAME'] = pretraining_config.model_name
        log_dict['STEPS'] = step_count - pretraining_config.num_warmup_steps
        log_dict['WARMUP_STEPS'] = pretraining_config.num_warmup_steps
        log_dict['LR'] = pretraining_config.learning_rate
        log_dict['BATCH_SIZE'] = pretraining_config.train_batch_size
        log_dict["TRAIN_STEPS"] = pretraining_config.train_batch_size*(
                                  step_count-pretraining_config.
                                  num_warmup_steps)
        log_dict['TRAIN_LOSS'] = np.mean(losses)
        log_dict['EVAL_LOSS'] = eval_outputs[1]
        log_dict['EVAL_NSP_A'] = eval_outputs[2]
        log_dict['EVAL_MLM_A'] = eval_outputs[3]
        logging_list.append(log_dict)

    accelerator.wait_for_everyone()

    # save at the end of training
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(os.path.join(pretraining_config.
                                        output_dir, 'Steps_'+str(
                                            step_count-pretraining_config.
                                            num_warmup_steps)),
                                        save_function=accelerator.save)
    torch.cuda.empty_cache()
    return logging_list


def evaluate(pretraining_config, model, data, accelerator, logger,
             step_number=0):
    """ Code to perform model evaluation

    Args:
        pretraining_config (Pretraining_config): Pretraining_config with
                                                 pretraining parameters
        model (BertForPreTraining): bert model class
        data (torch.utils.data.dataloader): eval dataloader
        optim (torch.optim.AdamW): AdamW optimizer
        scheduler (get_linear_schedule_with_warmup): scheduler for learning
                                                     rate
        accelerator (accelerate.Accelerator): HuggingFace's accelerator
        logger (logging): logger
        step_number (int, optional): Step at which eval was called.
                                    Defaults to 0.

    Returns:
        list[floats]: metrics
    """
    step_count = 0
    TOTAL_STEPS = None

    if pretraining_config.max_eval_steps:
        TOTAL_STEPS = pretraining_config.max_eval_steps
    model.eval()

    # setup tqdm bars
    if TOTAL_STEPS:
        progress_bar = tqdm.auto.tqdm(total=TOTAL_STEPS, desc="EVALUATING:\t",
                                      disable=not accelerator.is_main_process,
                                      position=2)
    elif pretraining_config.eval_file:
        progress_bar = tqdm.auto.tqdm(desc="EVALUATING:\t",
                                      disable=not accelerator.is_main_process,
                                      position=2)
    else:
        progress_bar = tqdm.auto.tqdm(total=len(data), desc="EVALUATING:\t",
                                      disable=not accelerator.is_main_process,
                                      position=2)

    write_bar = tqdm.auto.tqdm(total=0, bar_format='{desc}',
                               disable=not accelerator.is_main_process,
                               position=3)
    mlm_accuracies = []
    nsp_accuracies = []
    losses = []
    logger.info('\t'+'-&-'*30)
    logger.info('\t EVALUATION')

    for batch in data:
        if TOTAL_STEPS:
            if step_count == TOTAL_STEPS:
                break
        input_ids = batch['input_ids']
        attention_mask = batch['input_mask']
        token_type_ids = batch['segment_ids']
        next_sentence_label = batch['next_sentence_labels']
        target = batch['mlm_label']

        # model predictions
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       token_type_ids=token_type_ids,
                       labels=target, next_sentence_label=next_sentence_label)

        prediction_logits = accelerator.gather(output.prediction_logits)
        seq_relationship_logits = accelerator.gather(output.
                                                     seq_relationship_logits)

        masked_positions = accelerator.gather(batch['masked_lm_positions'])
        masked_positions = accelerator.gather(masked_positions)

        # compute and collect metrics
        mlm_a = mlm_accuracy(prediction_logits, accelerator.gather(
                             batch['masked_lm_ids']), masked_positions)

        mlm_accuracies.append(mlm_a)
        nsp_a = nsp_accuracy(seq_relationship_logits,
                             accelerator.gather(batch['next_sentence_labels']))
        nsp_accuracies.append(nsp_a)
        losses.append(output.loss.item())
        step_count += 1

        progress_bar.update(1)

    logger.info('\n')
    logger.info('\t Evaluating at STEP NUMBER: {} \tEval_MLM_accuracy : {}\
                 \tEval_NSP_accuracy: {} \t \
                Eval_loss: {}\t Eval_steps: {}'.format(step_number,
                                                       np.mean(mlm_accuracies),
                                                       np.mean(nsp_accuracies),
                                                       np.mean(losses),
                                                       step_count))

    write_bar.set_description_str('\t Evaluating after TRAINING STEP: {}\t\
                                  Eval_MLM_accuracy : {}\tEval_NSP_accuracy:\
                                  {}\t Eval_steps: {}'.
                                  format(step_number, np.mean(mlm_accuracies),
                                         np.mean(nsp_accuracies), step_count))
    torch.cuda.empty_cache()

    # return the metrics
    return step_number, np.mean(losses), np.mean(nsp_accuracies),\
        np.mean(mlm_accuracies)


def run_pretraining(pretraining_config):
    """ Pretraining function launched through accelerator notebook launcher

    Args:
        pretraining_config (Pretraining_config)
    """
    # Initial Declarations
    logging.basicConfig(
                        level=logging.INFO,
                        format='{%(asctime)s:%(filename)s:%(lineno)d}\
                                 %(levelname)s - %(message)s',
                        handlers=[
                                  logging.FileHandler(
                                      filename=pretraining_config.model_name +
                                      '_pretraining.log')])
    logger = logging.getLogger('LOGGER_NAME')

    accelerator = Accelerator(split_batches=True)
    accelerator.print("PRETRAINING_CONFIG PARAMS:\n"+str(pretraining_config))
    device = accelerator.device
    accelerator.print("DEVICES IN USE:")
    print(device)

    torch.cuda.empty_cache()

    # setup the model
    model = setup_model(pretraining_config, logger)

    # set up the dataloaders
    if not pretraining_config.split_ratio and not pretraining_config.eval_file:
        accelerator.print('PREPARING DATASET, it may take time to load the\
                          dataset and split it into train, test sets.')
        train_loader = load_dataset(pretraining_config.input_file,
                                    pretraining_config.train_batch_size,
                                    shuffle=True)
        eval_loader = load_dataset(pretraining_config.input_file,
                                   pretraining_config.eval_batch_size,
                                   shuffle=False)
    elif pretraining_config.eval_file:
        train_loader = load_dataset(pretraining_config.input_file,
                                    pretraining_config.train_batch_size,
                                    shuffle=True)
        eval_loader = load_dataset(pretraining_config.eval_file,
                                   pretraining_config.eval_batch_size,
                                   shuffle=False)
    else:
        train_loader, eval_loader = load_dataset_split(
            pretraining_config.input_file, pretraining_config.train_batch_size,
            pretraining_config.eval_batch_size, pretraining_config.split_ratio,
            accelerator)
    accelerator.print("DATASET PREPARED")

    # initalize the optimizer
    optim = torch.optim.AdamW(model.parameters(),
                              lr=pretraining_config.learning_rate,
                              betas=(0.9, 0.999), eps=1e-6)
    logger.info("Working on the device:\t {}".format(device))
    model, optim, data, eval_data = accelerator.prepare(model, optim,
                                                        train_loader,
                                                        eval_loader)
    # setup the LR scheduler
    if pretraining_config.num_train_steps:
        sc = get_linear_schedule_with_warmup(optim,
                                             num_warmup_steps=pretraining_config.num_warmup_steps,
                                             num_training_steps=pretraining_config.num_train_steps
                                             )
    else:
        sc = get_linear_schedule_with_warmup(optim,
                                             num_warmup_steps=pretraining_config.num_warmup_steps,
                                             num_training_steps=len(
                                                    train_loader))

    scheduler = sc
    start_time = time.time()

    # train the model
    log_list = train(pretraining_config, model, data, optim, scheduler,
                     accelerator, logger, eval_data)
    end_time = time.time()

    # convert the metrics into dataframe
    df = pd.DataFrame(log_list)
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    accelerator.print('SAVING THE TRAINING LOG CSV')

    # save the csv
    if accelerator.is_main_process:
        if os.path.exists(pretraining_config.log_csv):
            df.to_csv(pretraining_config.log_csv, mode='a', header=False)
        else:
            dirname = os.path.dirname(pretraining_config.log_csv)
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            df.to_csv(pretraining_config.log_csv, header=True)
        accelerator.print("PRETRAINING TIME:\t"+str(end_time-start_time))
        logger.info("PRETRAINING TIME:\t"+str(end_time-start_time))

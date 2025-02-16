# import torch
# import torchvision
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# from tqdm import tqdm
# import timm
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# # from torch.utils.data import DataLoader
# # from torch import nn
# from pytorch_transformers import AdamW, WEIGHTS_NAME, WarmupLinearSchedule
from pytorch_transformers import AdamW, WEIGHTS_NAME
# import csv
# import numpy as np
# import os
# import logging
from fp16 import FP16_Module, FP16_Optimizer
# from parallel import DataParallelModel, DataParallelCriterion
# from collections import OrderedDict
from utils import *
from settings import args, init_logging, DOMAIN_DICT, CONFIG_CLASS
from settings import TOKENIZER, SPECIAL_TOKEN_IDS, FILL_VAL, SAVE_NAME, FINAL_SAVE_NAME, TOKENS_WEIGHT, CONFIG_NAME
from scheduler import AnnealingLR
from regularizers import REG_TYPES, REG_TYPE_KEYS, Weight_Regularized_AdamW, Weight_Regularized_SGD
# from torch.nn import CrossEntropyLoss
logger = logging.getLogger(__name__)
# import sys

# from transformers import GPT2Model, GPT2Tokenizer



def Train(domain_ids, model):
    domains = [args.domains[domain_id] for domain_id in domain_ids]
    print("domains:", domains)
    logger.info("start to train { domain: %s}" % (domains))
    logger.info('args = {}'.format(str(args)))
    model_dir = get_model_dir(domains)
    make_dir(model_dir)

    train_dataset = [DOMAIN_DICT[d]["train"] for d in domains]
    val_dataset = [DOMAIN_DICT[d]["eval"] for d in domains]
    train_extra_data = []
    print("train_dataset:", train_dataset)

    print("domain_ids:", domain_ids)
    print("domain_ids[0]:", domain_ids[0])

    print("current domain:", domains[0])
    # if "lll" in args.seq_train_type:
    if "lll" in args.seq_train_type and domain_ids[0] > 0:
        get_real_data(domains[0], train_extra_data, accum=False, encode=True)
        print("hello real data")
        args.memory_data.append(train_extra_data)
        print("extra data:", train_extra_data)
        print("len of extra data:", len(train_extra_data))

    #     train_extra_data = []


    # LLL generate extra data
    # if "lll" in args.seq_train_type and domain_ids[0] > 0 and not args.skip_tasks:
    # if "lll" in args.seq_train_type and domain_ids[0] > 0:
    # # if "lll" in args.seq_train_type:
    #     print("hello next domain")
    #     prev_domain = args.domains[domain_ids[0]-1]
    #     with torch.no_grad():
    #         create_extra_data(domains[0], prev_domain, model, train_extra_data)
    #         print("train_extra_data:", train_extra_data)


    logger.info('extra training data size: {}'.format(len(train_extra_data)))
    # return model


    if not model:
        # which_model_to_load = model_dir if os.path.isfile(os.path.join(model_dir, FINAL_SAVE_NAME)) else args.model_name
        model = MODEL_CLASS.from_pretrained(args.model_name).cuda()
        print("TOKENIZER type:", type(TOKENIZER))
        model.resize_token_embeddings(len(TOKENIZER))
        # if not args.fp32:
        #     model = FP16_Module(model).cuda()

    gen_token = get_gen_token(domains[0])
    TOKENIZER.add_tokens([gen_token])
    TOKENIZER.save_pretrained(model_dir)
    SPECIAL_TOKENS[domains[0]] = gen_token
    SPECIAL_TOKEN_IDS[domains[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
    print("special token ID", SPECIAL_TOKEN_IDS[domains[0]])
    logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[domains[0]]))
    MODEL_CONFIG.vocab_size = len(TOKENIZER)
    MODEL_CONFIG.to_json_file(os.path.join(model_dir,CONFIG_NAME))
    global TOKENS_WEIGHT
    if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
        TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))

    model.resize_token_embeddings(len(TOKENIZER))


    ############################ Data preparation   #####################################
    print("train_extra_data size:", len(train_extra_data))
    max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
    # train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
    train_dataloader = create_dataloader(train_qadata, "train", 10)
    val_dataloader = create_dataloader(val_qadata, "train", 10)

    ############################ Data preparation   #####################################


    logger.info('len of train dataset: {} , max train batch size {} , num of opt steps: {}'.format(
        len(train_qadata), max_train_batch_size, n_train_optimization_steps))



    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


    ############################ Set Optimizer #####################################
    print("args.seq_train_type in REG_TYPE_KEYS:", args.seq_train_type in REG_TYPE_KEYS)
    print("REG_TYPE_KEYS:", REG_TYPE_KEYS)
    optimizer = FP16_Optimizer(optimizer, static_loss_scale=None, dynamic_loss_scale=True,
                                   dynamic_loss_args={'scale_window': 100, 'min_scale': 1, 'delayed_shift': 2})

    ############################ Set Optimizer and scheduler #####################################
    print("optimizer:", optimizer)

    scheduler = AnnealingLR(optimizer, start_lr=args.learning_rate,
                            warmup_iter=int(args.n_warmup_ratio * len(train_qadata)),
                            num_iters=int(n_train_optimization_steps), decay_style=args.decay_style)

    ############################ Set Loss Function   #####################################
    # train_loss_fct = DataParallelCriterion(CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT),
    #                                        args.device_ids)


    train_loss_fct = nn.CrossEntropyLoss()

    ############################ Set regularizer   #####################################
    print("args.seq_train_type in REG_TYPE_KEYS:", args.seq_train_type in REG_TYPE_KEYS)
    if args.seq_train_type in REG_TYPE_KEYS:
        print("hello regularizer")
        copy_train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
        prev_domain = args.domains[domain_ids[0] - 1]
        regularizer = REG_TYPES[args.seq_train_type](model, [copy_train_dataloader], domains[0],
                                                     prev_domain)
        regularizer.task_start_do()
        print("start regularizer")
        print("regularizer.dataloaders:", regularizer.dataloaders)
    ############################ Set Loss Function   #####################################
    tot_n_steps = 0
    train_once = TrainStep(model, optimizer, scheduler)
    # if "lll" in args.seq_train_type and [0] != 0:
    #     gem_step = GEMStep(model, parallel_model, train_loss_fct, optimizer)

    # set early stop
    early_stopping = EarlyStopping(logger=logger, patience=2, verbose=True)

    model.train()

############################ Training Loop start #####################################
    print("real n_train_epochs:", n_train_epochs, "current domain:", domains[0])
    print("len(train_dataloader.dataset):", len(train_dataloader.dataset))
    logger.info('len of train dataset: {} , current domain: {} , n_train_epochs: {}'.format(
        len(train_qadata), domains[0], n_train_epochs))
    n = 0

    for ep in range(n_train_epochs):
        
        print("train_dataloader:", train_dataloader)
        for n_steps, (_, _, cqa, _, Y, gen_X, gen_Y) in enumerate(train_dataloader):
            print("n:", n)
            n = n + 1
            print("n_steps:", n_steps)

            n_inputs = sum(_cqa.shape[0] for _cqa in cqa)


            print("cqa:",cqa)
            print("cqa[0]:", cqa[0])
            print("type(cqa):", type(cqa))
            print("len(cqa):", len(cqa))
            print("type(train_dataloader):", type(train_dataloader))
            print("batch_size:", train_dataloader.batch_size)
            # cqa[0] = cqa[0].cuda()
            # print(cqa[0].dtype)
        #     break
        # break
            # output = model(cqa[0])
            # print(output)
            # break
            # print(Y)
            # print(gen_X)
            # print(gen_Y)

            # for i in range(len(cqa)):
            #     cqa[i] = (cqa[i].to(args.device_ids[i]),)
            #     Y[i] = Y[i].to(args.device_ids[i])
            #     gen_X[i] = (gen_X[i].to(args.device_ids[i]),)
            #     gen_Y[i] = gen_Y[i].to(args.device_ids[i])

            # transfer original data to GRU
            cqa[0] = cqa[0].cuda()
            Y[0] = Y[0].cuda()
            gen_X[0] = gen_X[0].cuda()
            gen_Y[0] = gen_Y[0].cuda()


            losses = get_losses(model, cqa[0], Y[0], gen_X[0], gen_Y[0], train_loss_fct)
            print("train loss:",losses, "step:", n_steps)
            print("type of losses:", type(losses))
        #     break
        # break

            loss = sum(losses)
            print("train loss:", loss)
            print("loss type:", type(loss))
            # if "lll" in args.seq_train_type and domain_ids[0] != 0:
            #     gem_step(domain_ids[0])

            # train once
            train_once(loss, n_inputs)
        #     break
        # break
    #
    #         qa_loss = losses[0].item() * n_inputs
    #         lm_loss = losses[1].item() * n_inputs
    #         cum_loss += (qa_loss + lm_loss)
    #         cum_qa_loss += qa_loss
    #         cum_lm_loss += lm_loss
    #         cur_n_inputs += n_inputs
    #
    #         if (n_steps + 1) % args.logging_steps == 0:
    #             logger.info(
    #                 'progress {:.3f} , lr {:.1E} , loss {:.3f} , qa loss {:.3f} , lm loss {:.3f} , avg batch size {:.1f}'.format(
    #                     ep + cur_n_inputs / len(train_qadata), scheduler.get_lr(), cum_loss / cur_n_inputs,
    #                     cum_qa_loss / cur_n_inputs, cum_lm_loss / cur_n_inputs,
    #                     cur_n_inputs / (n_steps + 1)
    #                 ))
    #
    #     torch.save(model.state_dict(), os.path.join(model_dir, SAVE_NAME + str(ep + 1)))
        tot_n_steps += (n_steps + 1)
        print("tot_n_steps:", tot_n_steps)


        # loss_after_val = evaluate(model, val_dataloader)
        # early_stopping(loss_after_val, model, model_dir)
        # # print("after early stop")
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        epoch_model_name = FINAL_SAVE_NAME + str(domains[0]) + str(ep)
        print("current epoch number:", ep)
        torch.save(model.state_dict(), os.path.join(args.model_dir_root, epoch_model_name))

    #
    #
    #     logger.info(
    #         'epoch {}/{} done , tot steps {} , lr {:.1E} , loss {:.2f} , qa loss {:.2f} , lm loss {:.2f} , avg batch size {:.1f}'.format(
    #             ep + 1, n_train_epochs, tot_n_steps, scheduler.get_lr(), cum_loss / cur_n_inputs,
    #             cum_qa_loss / cur_n_inputs, cum_lm_loss / cur_n_inputs, cur_n_inputs / (n_steps + 1)
    #         ))
    #     ############################ Training Loop End #####################################
    #
    ## task end do for reg ###
    print("args.seq_train_type in REG_TYPE_KEYS:", args.seq_train_type in REG_TYPE_KEYS)
    if args.seq_train_type in REG_TYPE_KEYS:
        regularizer.task_end_do()
        print("end regularizer")

    #
    # ### save the model ###

    # torch.save(model.state_dict(), os.path.join(model_dir, FINAL_SAVE_NAME))
    torch.save(model.state_dict(), os.path.join(args.model_dir_root, FINAL_SAVE_NAME))
    print("path:", os.path.join(args.model_dir_root, FINAL_SAVE_NAME))
    print("model:", model)

    return model




if __name__ == '__main__':
    make_dir(args.model_dir_root)
    # print(args.model_dir_root)
    # init_logging('/Users/erichuang/PycharmProjects/ELLAM/Model_Directory/log_train.txt')
    init_logging(os.path.join(args.model_dir_root[3:], 'log_train.txt'))
    init_logging(os.path.join(args.model_dir_root, 'log_train.txt'))
    ### start to train ###
    model = None

    # # load the model
    # model_dir = "../Model_Directory"
    # print("model_dir:", model_dir)
    # # model_path = os.path.join(model_dir, 'model-{}'.format(ep+1))
    # model_path = "../Model_Directory/model-finish"
    # config_path = os.path.join(model_dir, CONFIG_NAME)
    # print("config_path:", config_path)
    #
    # # gen_token = get_gen_token(domain_load)
    # # TOKENIZER.add_tokens([gen_token])
    # # SPECIAL_TOKENS[domain_load] = gen_token
    # # SPECIAL_TOKEN_IDS[domain_load] = TOKENIZER.convert_tokens_to_ids(gen_token)
    #
    # #### load the model ####
    # model_config = CONFIG_CLASS.from_json_file(config_path)
    # model = MODEL_CLASS(model_config).cuda().eval()
    # state_dict = torch.load(model_path, map_location='cuda:0')  # 此处load state_DICT 参数
    # print("model_path:", model_path)
    # model.load_state_dict(state_dict)
    # print("trained model:", model)


    logger.info("----------------------------------------LOGGER START----------------------------------------")
    if args.seq_train_type == "lll":
        train_step = 0
        for domain_id in range(len(args.domains)):
            print("trainstep:", train_step)
            model = Train([domain], model)
            train_step += 1
        # Train(list(range(len(args.domains))), model)
    else:
        pass


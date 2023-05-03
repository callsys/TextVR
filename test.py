import argparse

import pandas as pd
import torch
import transformers
from sacred import Experiment
from tqdm import tqdm

import data_loader.data_loader as module_data
import model.metric as module_metric
import model.model as module_arch
from model.model import sim_matrix
from parse_config import ConfigParser
from trainer.trainer import verbose
from utils.util import state_dict_data_parallel_fix
import numpy as np
import os

from params import params

ex = Experiment('test')


def _format_ocr_info(data,tokenizer,device):
    '''
    tokenize the ocr information
    '''
    max_ocr_l = 50
    ocr_info = []
    for item in data['meta']['ocr_info']:
        ocr_info.append(eval(item))

    tmp1 = []
    tmp2 = []
    tmp3 = []
    for item in ocr_info:
        tmp1.append(item['valid_l'])
        tmp2.extend(item['gt_ocr'])
        tmp3.extend(item['gt_ocr_score'])
    all_valid_l = torch.LongTensor(tmp1).to(device)
    all_gt_ocr_token = tokenizer(tmp2, return_tensors='pt', padding=True, truncation=True)
    all_input_ids = all_gt_ocr_token.data['input_ids'].to(device)
    all_attention_mask = all_gt_ocr_token.data['attention_mask'].to(device)
    all_gt_score = torch.from_numpy(np.array(tmp3).astype(np.float32))

    all_input_ids = all_input_ids.reshape(-1, max_ocr_l, all_input_ids.shape[-1])
    all_attention_mask = all_attention_mask.reshape(-1, max_ocr_l, all_attention_mask.shape[-1])
    all_gt_score = all_gt_score.reshape(-1, max_ocr_l)

    data['ocr'] = dict(all_valid_l=all_valid_l,
                       all_input_ids=all_input_ids,
                       all_attention_mask=all_attention_mask,
                       all_gt_score=all_gt_score)
    return data


if params['version'] == 'offline':
    def _format_ocr_info(data):
        ocr_info = []
        for item in data['meta']['ocr_info']:
            ocr_info.append(eval(item))

        data['ocr'] = ocr_info
        return data
    @ex.main
    def run():
        # setup data_loader instances
        config._config['data_loader']['args']['split'] = args.split
        config._config['data_loader']['args']['tsfm_split'] = 'test'  # set transform to test split to remove augmentations
        config._config['data_loader']['args']['shuffle'] = False
        config._config['data_loader']['args']['batch_size'] = 1
        config._config['data_loader']['args']['sliding_window_stride'] = args.sliding_window_stride

        data_loader = config.initialize('data_loader', module_data)

        # tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

        # build model architecture
        model = config.initialize('arch', module_arch)

        # get function handles of loss and metrics
        metric_fns = [getattr(module_metric, met) for met in config['metrics']]

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        meta_arr = []
        print(len(data_loader))
        with torch.no_grad():
            sims = []
            for i, data in tqdm(tqdm(enumerate(data_loader))):
                # leave this for now since not doing anything on the gpu
                meta_arr.append(data['meta'])

                data['ocr_info'] = [eval(el) for el in data['meta']['ocr_info']]

                sim = model(data, return_embeds=True)
                sims.append(sim)

        mask = None


        if args.split != 'train':  # because train is usually too big
            sims = torch.cat(sims,0)
            sims = sims.numpy()

            nested_metrics = {}
            for metric in metric_fns:
                metric_name = metric.__name__
                res = metric(sims, query_masks=mask)
                verbose(epoch=0, metrics=res, name="", mode=metric_name)
                nested_metrics[metric_name] = res

elif params['version'] == 'ocr':

    def _format_ocr_info(data,tokenizer,device):
        '''
        tokenize the ocr information
        '''
        max_ocr_l = params['ocr_token_l']
        ocr_info = []
        for item in data['meta']['ocr_info']:
            ocr_info.append(eval(item))

        tmp1 = []
        tmp2 = []
        tmp3 = []
        for item in ocr_info:
            tmp1.append(item['valid_l'])
            tmp2.extend(item['gt_ocr'])
            tmp3.extend(item['gt_pos'])
        all_valid_l = torch.LongTensor(tmp1).to(device)
        all_gt_ocr_token = tokenizer(tmp2, return_tensors='pt', padding=True, truncation=True)
        all_input_ids = all_gt_ocr_token.data['input_ids'].to(device)
        all_attention_mask = all_gt_ocr_token.data['attention_mask'].to(device)
        all_gt_pos = torch.from_numpy(np.array(tmp3).astype(np.float32))

        all_input_ids = all_input_ids.reshape(-1, max_ocr_l, all_input_ids.shape[-1])
        all_attention_mask = all_attention_mask.reshape(-1, max_ocr_l, all_attention_mask.shape[-1])
        all_gt_pos = all_gt_pos.reshape(-1, max_ocr_l, all_gt_pos.shape[-1])

        data['ocr'] = dict(all_valid_l=all_valid_l,
                           all_input_ids=all_input_ids,
                           all_attention_mask=all_attention_mask,
                           all_gt_pos=all_gt_pos)
        return data

    
    @ex.main
    def run():
        # setup data_loader instances
        config._config['data_loader']['args']['split'] = args.split
        config._config['data_loader']['args'][
            'tsfm_split'] = 'test'  # set transform to test split to remove augmentations
        config._config['data_loader']['args']['shuffle'] = False
        config._config['data_loader']['args']['batch_size'] = args.batch_size
        config._config['data_loader']['args']['sliding_window_stride'] = args.sliding_window_stride

        data_loader = config.initialize('data_loader', module_data)

        tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

        # build model architecture
        model = config.initialize('arch', module_arch)

        # get function handles of loss and metrics
        metric_fns = [getattr(module_metric, met) for met in config['metrics']]


        if config.resume is not None:
            print('Loading checkpoint: {} ...'.format(config.resume))
            checkpoint = torch.load(config.resume)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict, strict=False)
        else:
            print('Using random weights')

        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        meta_arr = []
        text_embed_arr = []
        vid_embed_arr = []
        print(len(data_loader))
        with torch.no_grad():
            for i, data in tqdm(tqdm(enumerate(data_loader))):
#             for dl_idx, dl in enumerate(data_loader):
#                 for data in tqdm(dl, desc=f"Validating dl{dl_idx}"):
#                     print(data)
                    # leave this for now since not doing anything on the gpu
                meta_arr.append(data['meta'])
                if tokenizer is not None:
                    data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)

                    data = _format_ocr_info(data, tokenizer, device)
                data['text'] = {key: val.cuda() for key, val in data['text'].items()}
#                 if isinstance(data['video'], list):
#                     data['video'] = [x.to(device) for x in data['video']]
#                 else:
#                     data['video'] = data['video'].to(device)       

                text_embed, vid_embed = model(data, return_embeds=True)
                text_embed_arr.append(text_embed.cpu().detach())
                vid_embed_arr.append(vid_embed.cpu().detach())

        text_embeds = torch.cat(text_embed_arr)
        vid_embeds = torch.cat(vid_embed_arr)

        mask = None
        if data_loader.dataset.sliding_window_stride != -1:
            cpu_vid_embeds = vid_embeds
            cpu_text_embeds = text_embeds

            li_vid_embeds = [x for x in cpu_vid_embeds]
            li_txt_embeds = [x for x in cpu_text_embeds]
            videoids = pd.Series([x['paths'] for x in meta_arr]).explode()
            raw_caps = pd.Series([x['raw_captions']] for x in meta_arr).explode().explode()
            vid_df = pd.DataFrame({'videoid': videoids, 'vid_embed': li_vid_embeds, 'txt_embed': li_txt_embeds,
                                   'captions': raw_caps})
            new_vid_embeds = []
            new_txt_embeds = []
            for vid in vid_df['videoid'].unique():
                tdf = vid_df[vid_df['videoid'] == vid]
                tvembeds = torch.stack(tdf['vid_embed'].values.tolist())
                tvembeds = tvembeds.mean(dim=0)
                new_vid_embeds.append(tvembeds)

                for cap in tdf['captions'].unique():
                    cdf = vid_df[vid_df['captions'] == cap]
                    ttembeds = torch.stack(cdf['txt_embed'].values.tolist())
                    new_txt_embeds.append(ttembeds[0])

            vid_embeds = torch.stack(new_vid_embeds)
            text_embeds = torch.stack(new_txt_embeds)

        if args.split != 'train':  # because train is usually too big
            sims = sim_matrix(text_embeds, vid_embeds)
            sims = sims.numpy()

            nested_metrics = {}
            for metric in metric_fns:
                metric_name = metric.__name__
                res = metric(sims) # , query_masks=mask
                verbose(epoch=0, metrics=res, name="", mode=metric_name)
                nested_metrics[metric_name] = res
                
                
            txt_embeds_save_fp = os.path.join("./work_dirs", f'sim_path_ocr_TextVR.npy')
            np.save(txt_embeds_save_fp, sims)
            
            
        # if config.config['visualizer']:
        #    raise NotImplementedError
        if args.save_feats is not None:
            vid_embeds = vid_embeds.cpu().detach().numpy()
            text_embeds = text_embeds.cpu().detach().numpy()
            vid_embeds_save_fp = os.path.join(args.save_feats, f'vid_embeds_{data_loader.dataset.split}.npy')
            txt_embeds_save_fp = os.path.join(args.save_feats, f'txt_embeds_{data_loader.dataset.split}.npy')

            np.save(vid_embeds_save_fp, vid_embeds)
            np.save(txt_embeds_save_fp, text_embeds)

            videoids = pd.Series([x['paths'] for x in meta_arr]).explode()
            videoids.to_csv(os.path.join(args.save_feats, f'ids_{data_loader.dataset.split}.csv'), index=False)

elif params['version'] == 'video':
    
    
    @ex.main
    def run():
        # setup data_loader instances
        config._config['data_loader']['args']['split'] = args.split
        config._config['data_loader']['args'][
            'tsfm_split'] = 'test'  # set transform to test split to remove augmentations
        config._config['data_loader']['args']['shuffle'] = False
        config._config['data_loader']['args']['batch_size'] = args.batch_size
        config._config['data_loader']['args']['sliding_window_stride'] = args.sliding_window_stride

        data_loader = config.initialize('data_loader', module_data)

        tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

        # build model architecture
        model = config.initialize('arch', module_arch)

        # get function handles of loss and metrics
        metric_fns = [getattr(module_metric, met) for met in config['metrics']]

        
        
        if config.resume is not None:
            print('Loading checkpoint: {} ...'.format(config.resume))
            checkpoint = torch.load(config.resume)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict, strict=False)
        else:
            print('Using random weights')

        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        meta_arr = []
        text_embed_arr = []
        vid_embed_arr = []
        print(len(data_loader))
        with torch.no_grad():
            for i, data in tqdm(tqdm(enumerate(data_loader))):
#             for dl_idx, dl in enumerate(data_loader):
#                 for data in tqdm(dl, desc=f"Validating dl{dl_idx}"):
#                     print(data)
                    # leave this for now since not doing anything on the gpu
                meta_arr.append(data['meta'])
                if tokenizer is not None:
                    data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)

                data['text'] = {key: val.cuda() for key, val in data['text'].items()}
                if isinstance(data['video'], list):
                    data['video'] = [x.to(device) for x in data['video']]
                else:
                    data['video'] = data['video'].to(device)       

                text_embed, vid_embed = model(data, return_embeds=True)
                text_embed_arr.append(text_embed.cpu().detach())
                vid_embed_arr.append(vid_embed.cpu().detach())

        text_embeds = torch.cat(text_embed_arr)
        vid_embeds = torch.cat(vid_embed_arr)

        mask = None
        if data_loader.dataset.sliding_window_stride != -1:
            cpu_vid_embeds = vid_embeds
            cpu_text_embeds = text_embeds

            li_vid_embeds = [x for x in cpu_vid_embeds]
            li_txt_embeds = [x for x in cpu_text_embeds]
            videoids = pd.Series([x['paths'] for x in meta_arr]).explode()
            raw_caps = pd.Series([x['raw_captions']] for x in meta_arr).explode().explode()
            vid_df = pd.DataFrame({'videoid': videoids, 'vid_embed': li_vid_embeds, 'txt_embed': li_txt_embeds,
                                   'captions': raw_caps})
            new_vid_embeds = []
            new_txt_embeds = []
            for vid in vid_df['videoid'].unique():
                tdf = vid_df[vid_df['videoid'] == vid]
                tvembeds = torch.stack(tdf['vid_embed'].values.tolist())
                tvembeds = tvembeds.mean(dim=0)
                new_vid_embeds.append(tvembeds)

                for cap in tdf['captions'].unique():
                    cdf = vid_df[vid_df['captions'] == cap]
                    ttembeds = torch.stack(cdf['txt_embed'].values.tolist())
                    new_txt_embeds.append(ttembeds[0])

            vid_embeds = torch.stack(new_vid_embeds)
            text_embeds = torch.stack(new_txt_embeds)

        if args.split != 'train':  # because train is usually too big
            sims = sim_matrix(text_embeds, vid_embeds)
            sims = sims.numpy()

            nested_metrics = {}
            for metric in metric_fns:
                metric_name = metric.__name__
                res = metric(sims) # , query_masks=mask
                verbose(epoch=0, metrics=res, name="", mode=metric_name)
                nested_metrics[metric_name] = res
                
                
            txt_embeds_save_fp = os.path.join("./work_dirs", f'sim_path_vision_TextVR.npy')
            np.save(txt_embeds_save_fp, sims)
            
            
        # if config.config['visualizer']:
        #    raise NotImplementedError
        if args.save_feats is not None:
            vid_embeds = vid_embeds.cpu().detach().numpy()
            text_embeds = text_embeds.cpu().detach().numpy()
            vid_embeds_save_fp = os.path.join(args.save_feats, f'vid_embeds_{data_loader.dataset.split}.npy')
            txt_embeds_save_fp = os.path.join(args.save_feats, f'txt_embeds_{data_loader.dataset.split}.npy')

            np.save(vid_embeds_save_fp, vid_embeds)
            np.save(txt_embeds_save_fp, text_embeds)

            videoids = pd.Series([x['paths'] for x in meta_arr]).explode()
            videoids.to_csv(os.path.join(args.save_feats, f'ids_{data_loader.dataset.split}.csv'), index=False)
            
else:
    def _format_ocr_info(data,tokenizer,device):
        '''
        tokenize the ocr information
        '''
        max_ocr_l = params['ocr_token_l']
        ocr_info = []
        for item in data['meta']['ocr_info']:
            ocr_info.append(eval(item))


        tmp1 = []
        tmp2 = []
        tmp3 = []
        for item in ocr_info:
            tmp1.append(item['valid_l'])
            tmp2.extend(item['gt_ocr'])
            tmp3.extend(item['gt_pos'])
        all_valid_l = torch.LongTensor(tmp1).to(device)
        all_gt_ocr_token = tokenizer(tmp2, return_tensors='pt', padding=True, truncation=True)
        all_input_ids = all_gt_ocr_token.data['input_ids'].to(device)
        all_attention_mask = all_gt_ocr_token.data['attention_mask'].to(device)
        all_gt_pos = torch.from_numpy(np.array(tmp3).astype(np.float32))

        all_input_ids = all_input_ids.reshape(-1, max_ocr_l, all_input_ids.shape[-1])
        all_attention_mask = all_attention_mask.reshape(-1, max_ocr_l, all_attention_mask.shape[-1])
        all_gt_pos = all_gt_pos.reshape(-1, max_ocr_l, all_gt_pos.shape[-1])

        data['ocr'] = dict(all_valid_l=all_valid_l,
                           all_input_ids=all_input_ids,
                           all_attention_mask=all_attention_mask,
                           all_gt_pos=all_gt_pos)
        return data

    
    
    
    @ex.main
    def run():
        # setup data_loader instances
        config._config['data_loader']['args']['split'] = args.split
        config._config['data_loader']['args'][
            'tsfm_split'] = 'test'  # set transform to test split to remove augmentations
        config._config['data_loader']['args']['shuffle'] = False
        config._config['data_loader']['args']['batch_size'] = args.batch_size
        config._config['data_loader']['args']['sliding_window_stride'] = args.sliding_window_stride

        data_loader = config.initialize('data_loader', module_data)

        tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

        # build model architecture
        model = config.initialize('arch', module_arch)

        # get function handles of loss and metrics
        metric_fns = [getattr(module_metric, met) for met in config['metrics']]

        
        
        if config.resume is not None:
            print('Loading checkpoint: {} ...'.format(config.resume))
            checkpoint = torch.load(config.resume)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict, strict=False)
        else:
            print('Using random weights')

        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        meta_arr = []
        text_embed_arr = []
        vid_embed_arr = []
        print(len(data_loader))
        with torch.no_grad():
            for i, data in tqdm(tqdm(enumerate(data_loader))):
#             for dl_idx, dl in enumerate(data_loader):
#                 for data in tqdm(dl, desc=f"Validating dl{dl_idx}"):
#                     print(data)
                    # leave this for now since not doing anything on the gpu
                meta_arr.append(data['meta'])
                if tokenizer is not None:
                    data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)

                    data = _format_ocr_info(data, tokenizer, device)
                data['text'] = {key: val.cuda() for key, val in data['text'].items()}
                if isinstance(data['video'], list):
                    data['video'] = [x.to(device) for x in data['video']]
                else:
                    data['video'] = data['video'].to(device)       

                text_embed, _, vid_embed = model(data, return_embeds=True)
                text_embed_arr.append(text_embed.cpu().detach())
                vid_embed_arr.append(vid_embed.cpu().detach())

        text_embeds = torch.cat(text_embed_arr)
        vid_embeds = torch.cat(vid_embed_arr)

        mask = None
        if data_loader.dataset.sliding_window_stride != -1:
            cpu_vid_embeds = vid_embeds
            cpu_text_embeds = text_embeds

            li_vid_embeds = [x for x in cpu_vid_embeds]
            li_txt_embeds = [x for x in cpu_text_embeds]
            videoids = pd.Series([x['paths'] for x in meta_arr]).explode()
            raw_caps = pd.Series([x['raw_captions']] for x in meta_arr).explode().explode()
            vid_df = pd.DataFrame({'videoid': videoids, 'vid_embed': li_vid_embeds, 'txt_embed': li_txt_embeds,
                                   'captions': raw_caps})
            new_vid_embeds = []
            new_txt_embeds = []
            for vid in vid_df['videoid'].unique():
                tdf = vid_df[vid_df['videoid'] == vid]
                tvembeds = torch.stack(tdf['vid_embed'].values.tolist())
                tvembeds = tvembeds.mean(dim=0)
                new_vid_embeds.append(tvembeds)

                for cap in tdf['captions'].unique():
                    cdf = vid_df[vid_df['captions'] == cap]
                    ttembeds = torch.stack(cdf['txt_embed'].values.tolist())
                    new_txt_embeds.append(ttembeds[0])

            vid_embeds = torch.stack(new_vid_embeds)
            text_embeds = torch.stack(new_txt_embeds)

        if args.split != 'train':  # because train is usually too big
            sims = sim_matrix(text_embeds, vid_embeds)
            sims = sims.numpy()

            nested_metrics = {}
            for metric in metric_fns:
                metric_name = metric.__name__
                res = metric(sims) # , query_masks=mask
                verbose(epoch=0, metrics=res, name="", mode=metric_name)
                nested_metrics[metric_name] = res
                
            if args.sim_path is not None:
                # txt_embeds_save_fp = os.path.join("./work_dirs", f'sim_path_vision_TextVR.npy')
                np.save(args.sim_path, sims)
            
            
        # if config.config['visualizer']:
        #    raise NotImplementedError
        if args.save_feats is not None:
            vid_embeds = vid_embeds.cpu().detach().numpy()
            text_embeds = text_embeds.cpu().detach().numpy()
            vid_embeds_save_fp = os.path.join(args.save_feats, f'vid_embeds_{data_loader.dataset.split}.npy')
            txt_embeds_save_fp = os.path.join(args.save_feats, f'txt_embeds_{data_loader.dataset.split}.npy')

            np.save(vid_embeds_save_fp, vid_embeds)
            np.save(txt_embeds_save_fp, text_embeds)

            videoids = pd.Series([x['paths'] for x in meta_arr]).explode()
            videoids.to_csv(os.path.join(args.save_feats, f'ids_{data_loader.dataset.split}.csv'), index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default='configs/TextVR_fusion.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('--save_feats', default=None,
                      help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
    args.add_argument('--sim_path', default=None,
                      help='path to store sim_matrix.')
    args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=64, type=int,
                      help='size of batch')
    config = ConfigParser(args, test=True)
    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    ex.run()

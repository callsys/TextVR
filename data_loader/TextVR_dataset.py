'''
video datasets for zyz anns
'''


import json
import traceback

import cv2
import av
import os
import numpy as np
import torch
import random
from PIL import Image
from torchvision import transforms
import decord
import matplotlib.pyplot as plt
from base.base_dataset import TextVideoDataset
from params import params
import time
import tqdm
import concurrent
from concurrent.futures import ProcessPoolExecutor


if params['version'] == 'offline':
    class ZYZOCRanns_datasets(TextVideoDataset):
        def __init__(self,
                     dataset_name,
                     text_params,
                     video_params,
                     data_dir,
                     metadata_dir=None,
                     split='train',
                     tsfms=None,
                     cut=None,
                     subsample=1,
                     sliding_window_stride=-1,
                     reader='decord'
                     ):
            self.dataset_name = dataset_name
            self.text_params = text_params
            self.video_params = video_params
            # check for environment variables
            self.data_dir = os.path.expandvars(data_dir)
            if metadata_dir is not None:
                self.metadata_dir = os.path.expandvars(metadata_dir)
            else:
                self.metadata_dir = self.data_dir
            self.split = split
            self.transforms = tsfms
            self.cut = cut
            self.subsample = subsample
            self.sliding_window_stride = sliding_window_stride
            self.video_reader = video_reader[reader]
            self.label_type = 'caption'
            self._load_metadata()
            self.max_ocr_l = 600
            self.lower = True



        def _load_metadata(self):
            data_dir = '../'
            if self.dataset_name == 'MSRVTT_zyzocranns_official':
                anns_path = os.path.join(data_dir,'msrvtt.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'official_split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            elif self.dataset_name == 'MSRVTT_zyzocranns_oneka':
                anns_path = os.path.join(data_dir,'msrvtt/msrvtt.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'oneka_split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
            elif "MSVD" in self.dataset_name:
                anns_path = os.path.join(data_dir,'msvd/msvd.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            elif "DIDEMO" in self.dataset_name:
                anns_path = os.path.join(data_dir,'didemo/didemo.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            elif "WEBIVD" in self.dataset_name:
                anns_path = os.path.join(data_dir,'webvid/webvid.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
            elif 'VITVR_zyzocranns' == self.dataset_name:
                # anns_path = os.path.join(data_dir, 'vitvr/vitvr_822.json')
                anns_path = params['anns']
                with open(anns_path, 'r') as fr:
                    zyz_anns = json.load(fr)
                    if not os.path.exists(zyz_anns[0]['path']):
                        for ann in zyz_anns:
                            ann['path'] = ann['path'].replace('/mmu-ocr/pub', '/mmu-ocr')
                    self.zyz_anns = zyz_anns
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            else:
                dataset = self.dataset_name.split('_')[0]
                anns_path = os.path.join(data_dir, f'{dataset}/{dataset}.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']


            l_train = len(self.train_anns)
            l_val = len(self.val_anns)
            l_test = len(self.test_anns)
            self.split_sizes = {'train': l_train, 'val': l_val, 'test': l_test}

            if self.split == 'train':
                self.metadata = self.train_anns
            elif self.split == 'val':
                self.metadata = self.val_anns
            else:
                self.metadata = self.test_anns

            captions_all = [el['captions_info'][0]['caption'] for el in self.test_anns]

            self.captions_all = captions_all

            for ann in self.metadata:
                ann['captions_all'] = captions_all

            self._load_ocr_infos()

        def _load_ocr_infos(self):
            for el in self.metadata:
                ocr_path = el['videos_info']['ocr_path']
                # ocr_path = os.path.join('data/vitvr_json/',*ocr_path.split('/')[-2:])
                # ocr_path = ocr_path.replace('/share/mmu-ocr/datasets/zyz_anns/frozen-in-time-ocr/cache','cache')
                if not os.path.exists(ocr_path):
                    print(f'{ocr_path} not exists.')
                    continue
                with open(ocr_path, 'r') as fr:
                    ocr_file = json.load(fr)
                el['ocr_info'] = ocr_file

        def ocr_format(self, sample):
            ocr_info = sample['ocr_info']
            gt_text_list = []
            gt_score_list = []

            for key, value in ocr_info.items():
                box = value['recommend_box']
                gt_score = value['score']
                gt_text = value['text']

                if ' ' in gt_text:
                    gt_texts = gt_text.split(' ')
                else:
                    gt_texts = [gt_text]

                for text in gt_texts:
                    ord_text = [ord(el) for el in text]
                    if len(ord_text)<1:
                        continue
                    if max(ord_text)>123: # remove chinese
                        continue

                    tracks = value['tracks']
                    if self.lower:
                        out_text = text.strip().lower()
                    else:
                        out_text = text.strip()
                    gt_text_list.append(out_text)
                    gt_score_list.append(gt_score)

            valid_l = len(gt_text_list)
            if valid_l > self.max_ocr_l:
                gt_text_list = gt_text_list[:self.max_ocr_l]
                gt_score_list = gt_score_list[:self.max_ocr_l]
            else:
                gt_text_list = gt_text_list + [' '] * (self.max_ocr_l - valid_l)
                gt_score_list = gt_score_list + [0] * (self.max_ocr_l - valid_l)
            return dict(gt_ocr=gt_text_list, gt_ocr_score=gt_score_list, valid_l=valid_l)

        def _get_caption(self, sample):
            caption_sample = self.text_params.get('caption_sample', "rand")
            if self.split in ['train'] and caption_sample == "rand":
                captions_info = sample['captions_info']
                caption = random.choice(captions_info)['caption']
            else:
                caption = sample['captions_info'][0]['caption']

            if self.lower:
                caption = caption.lower()
            return caption

        def __getitem__(self, item):
            item = item % len(self.metadata)
            sample = self.metadata[item]
            caption = self._get_caption(sample)

            ocr_info = self.ocr_format(sample)

            meta_arr = {'dataset': self.dataset_name, 'ocr_info': str(ocr_info)}
            data = {'video': '', 'text': caption, 'meta': meta_arr, 'captions_all':self.captions_all}
            return data

elif params['version'] == 'ocr':
    class ZYZOCRanns_datasets(TextVideoDataset):
        def __init__(self,
                     dataset_name,
                     text_params,
                     video_params,
                     data_dir,
                     metadata_dir=None,
                     split='train',
                     tsfms=None,
                     cut=None,
                     subsample=1,
                     sliding_window_stride=-1,
                     reader='decord'
                     ):
            self.dataset_name = dataset_name
            self.text_params = text_params
            self.video_params = video_params
            # check for environment variables
            self.data_dir = os.path.expandvars(data_dir)
            if metadata_dir is not None:
                self.metadata_dir = os.path.expandvars(metadata_dir)
            else:
                self.metadata_dir = self.data_dir
            self.split = split
            self.transforms = tsfms
            self.cut = cut
            self.subsample = subsample
            self.sliding_window_stride = sliding_window_stride
            self.video_reader = video_reader[reader]
            self.label_type = 'caption'
            self.topl = params['ocr_token_l']  # input the top l score words
            self.min_ocr_l = 3  # min word length
            self.lower = True
            # self._load_metadata()
            cache_name = os.path.join('cache', dataset_name+ '_' + split + '.json')
            if os.path.exists(cache_name):
                with open(cache_name,'r') as fr:
                    self.metadata = json.load(fr)
            else:
                self._load_metadata()
                with open(cache_name,'w') as fw:
                    json.dump(self.metadata,fw)

            # if self.sliding_window_stride != -1:
            #     if self.split != 'test':
            #         raise ValueError('Fixing frame sampling is for test time only. can remove but...')
            #     self._fix_temporal_samples()

        def _load_metadata(self):
            data_dir = '../'
            if self.dataset_name == 'MSRVTT_zyzocranns_official':
                anns_path = os.path.join(data_dir, 'msrvtt.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'official_split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            elif self.dataset_name == 'MSRVTT_zyzocranns_oneka':
                anns_path = os.path.join(data_dir, 'msrvtt/msrvtt.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'oneka_split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
            elif "MSVD" in self.dataset_name:
                anns_path = os.path.join(data_dir, 'msvd/msvd.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            elif "DIDEMO" in self.dataset_name:
                anns_path = os.path.join(data_dir, 'didemo/didemo.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            elif "WEBIVD" in self.dataset_name:
                anns_path = os.path.join(data_dir, 'webvid/webvid.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
            elif "YOUCOOK2SEG" in self.dataset_name:
#                 anns_path = os.path.join(data_dir, 'youcook2seg/youcook2seg.json')
                anns_path = params['anns']
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
            elif 'VITVR_zyzocranns' == self.dataset_name:
                # anns_path = os.path.join(data_dir, 'vitvr/vitvr_822.json')
                anns_path = params['anns']
                with open(anns_path, 'r') as fr:
                    zyz_anns = json.load(fr)
                    if not os.path.exists(zyz_anns[0]['path']):
                        for ann in zyz_anns:
                            ann['path'] = ann['path'].replace('/mmu-ocr/pub', '/mmu-ocr')
                    self.zyz_anns = zyz_anns
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            else:
                dataset = self.dataset_name.split('_')[0]
                anns_path = os.path.join(data_dir, f'{dataset}/{dataset}.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']

            l_train = len(self.train_anns)
            l_val = len(self.val_anns)
            l_test = len(self.test_anns)
            self.split_sizes = {'train': l_train, 'val': l_val, 'test': l_test}

            if self.split == 'train':
                self.metadata = self.train_anns
            elif self.split == 'val':
                self.metadata = self.val_anns
            else:
                self.metadata = self.test_anns
                
                
            if 'ocr_info' not in self.metadata[0]:
                self.norm_flag = True
                self._load_ocr_infos()
            else:
                print('='*50)
                print('load existing ocr info...')
                print('='*50)
                self.norm_flag = False # do not normalize the meta data

        def _load_ocr_infos(self):
            for el in self.metadata:
                ocr_path = el['videos_info']['ocr_path']
                # ocr_path = os.path.join('data/vitvr_json/', *ocr_path.split('/')[-2:])
                # ocr_path = ocr_path.replace('/mmu-ocr/datasets/zyz_anns/ocr_anns','./data/MSRVTT')
                if not os.path.exists(ocr_path):
                    print(f'{ocr_path} not exists.')
                    continue
                with open(ocr_path, 'r') as fr:
                    ocr_file = json.load(fr)
                el['ocr_info'] = ocr_file
            self.metadata = [self.ocr_format(el) for el in self.metadata]

        def ocr_format(self, sample):
            ocr_info = sample['ocr_info']
            gt_text_list = []
            gt_score_list = []
            gt_ocr_list = []

            if len(ocr_info)!=0:
                for key, value in ocr_info.items():
                    gt_score = float(value['score'])
                    gt_text = value['text']

                    if ' ' in gt_text:
                        gt_texts = gt_text.split(' ')
                    else:
                        gt_texts = [gt_text]

                    for text in gt_texts:
                        ord_text = [ord(el) for el in text]
                        if len(ord_text) < 1:
                            continue
                        if max(ord_text) > 123:  # remove chinese
                            continue
                        if text.strip() in gt_text_list:
                            continue
                        if len(text) < self.min_ocr_l:
                            continue

                        tracks = value['tracks']
                        
                        if self.lower:
                            out_text = text.strip().lower()
                        else:
                            out_text = text.strip()
                        gt_text_list.append(out_text)
                        gt_ocr_list.append((out_text, gt_score))
                gt_ocr_list = sorted(gt_ocr_list, key=lambda x: x[1], reverse=True)
                gt_ocr_list = gt_ocr_list[:self.topl]
                gt_text_list = [el[0] for el in gt_ocr_list]
                gt_score_list = [el[1] for el in gt_ocr_list]

            valid_l = len(gt_text_list)
            if valid_l > self.topl:
                gt_text_list = gt_text_list[:self.topl]
                gt_score_list = gt_score_list[:self.topl]
            else:
                gt_text_list = gt_text_list + [' '] * (self.topl - valid_l)
                gt_score_list = gt_score_list + [0] * (self.topl - valid_l)

            sample['ocr_info'] = dict(gt_ocr=gt_text_list, gt_ocr_score=gt_score_list, valid_l=valid_l)
            return sample

        def _get_caption(self, sample):
            caption_sample = self.text_params.get('caption_sample', "rand")
            if self.split in ['train'] and caption_sample == "rand":
                captions_info = sample['captions_info']
                caption = random.choice(captions_info)['caption']
            else:
                caption = sample['captions_info'][0]['caption']
            if self.lower:
                caption = caption.lower()
            return caption

        def __getitem__(self, item):
            item = item % len(self.metadata)
            sample = self.metadata[item]
            caption = self._get_caption(sample)

            ocr_info = sample['ocr_info']


            meta_arr = {'raw_captions': caption, 'paths': '', 'dataset': self.dataset_name,
                        'ocr_info': str(ocr_info)}
            data = {'video': '', 'text': caption, 'meta': meta_arr}
            return data

elif params['version'] == 'video':
    class ZYZOCRanns_datasets(TextVideoDataset):
        def __init__(self,
                     dataset_name,
                     text_params,
                     video_params,
                     data_dir,
                     metadata_dir=None,
                     split='train',
                     tsfms=None,
                     cut=None,
                     subsample=1,
                     sliding_window_stride=-1,
                     reader='decord'
                     ):
            self.reader = reader
            self.dataset_name = dataset_name
            self.text_params = text_params
            self.video_params = video_params
            # check for environment variables
            self.data_dir = os.path.expandvars(data_dir)
            if metadata_dir is not None:
                self.metadata_dir = os.path.expandvars(metadata_dir)
            else:
                self.metadata_dir = self.data_dir
            self.split = split
            self.transforms = tsfms
            self.cut = cut
            self.subsample = subsample
            self.sliding_window_stride = sliding_window_stride
            self.video_reader = video_reader[reader]
            self.label_type = 'caption'
            self.topl = 20  # input the top l score words
            self.min_ocr_l = 3  # min word length
            self._load_metadata()
            # self._cache_videos()


            # data augmentation pipelines used in OA-Transformer
            if self.split == 'train':
                self.transforms = transforms.Compose(
                    [
                        transforms.Resize(size=(242, 242)),
                        transforms.RandomCrop(size=(224, 224)),
                        # add for data augmentation
                        transforms.RandomApply([
                            transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.3),
                        # transforms.RandomApply([base_augmentation.GaussianBlur([.1, 2.])], p=0.5), # PIL
                        # transforms.ToTensor(), # PIL
                        # transforms.RandomResizedCrop(size=(224, 224)),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )
            else:
                self.transforms = transforms.Compose(
                    [
                        transforms.Resize(size=(224, 224)),
                        # transforms.RandomResizedCrop(size=(224, 224)),
                        # transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )

            print('-'*50)
            print(self.transforms)
            print('-'*50)


            # if self.sliding_window_stride != -1:
            #     if self.split != 'test':
            #         raise ValueError('Fixing frame sampling is for test time only. can remove but...')
            #     self._fix_temporal_samples()

        def _load_metadata(self):
            data_dir = '../'
            if self.dataset_name == 'MSRVTT_zyzocranns_official':
                anns_path = os.path.join(data_dir, 'msrvtt.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'official_split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            elif self.dataset_name == 'MSRVTT_zyzocranns_oneka':
                anns_path = os.path.join(data_dir, 'msrvtt/msrvtt.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'oneka_split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
            elif "MSVD" in self.dataset_name:
                anns_path = os.path.join(data_dir, 'msvd/msvd.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            elif "DIDEMO" in self.dataset_name:
                anns_path = os.path.join(data_dir, 'didemo/didemo.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            elif "WEBIVD" in self.dataset_name:
                anns_path = os.path.join(data_dir, 'webvid/webvid.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
            elif "YOUCOOK2SEG" in self.dataset_name:
                anns_path = os.path.join(data_dir, 'youcook2seg/youcook2seg.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
            elif 'VITVR_zyzocranns' == self.dataset_name:
                # anns_path = os.path.join(data_dir, 'vitvr/vitvr_822.json')
                anns_path = params['anns']
                with open(anns_path, 'r') as fr:
                    zyz_anns = json.load(fr)
                    if not os.path.exists(zyz_anns[0]['path']):
                        for ann in zyz_anns:
                            ann['path'] = ann['path'].replace('/mmu-ocr/pub', '/mmu-ocr')
                    self.zyz_anns = zyz_anns
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            else:
                dataset = self.dataset_name.split('_')[0]
                anns_path = os.path.join(data_dir, f'{dataset}/{dataset}.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']

            l_train = len(self.train_anns)
            l_val = len(self.val_anns)
            l_test = len(self.test_anns)
            self.split_sizes = {'train': l_train, 'val': l_val, 'test': l_test}

            if self.split == 'train':
                self.metadata = self.train_anns
            elif self.split == 'val':
                self.metadata = self.val_anns
            else:
                self.metadata = self.test_anns

        def _get_video_path(self, sample):
            path = sample['path']
            # path = '/home/zyz/code/SynthText/tmp/demo.mp4'
            return path, os.path.basename(path)

        def _get_caption(self, sample):
            caption_sample = self.text_params.get('caption_sample', "rand")
            if self.split in ['train'] and caption_sample == "rand":
                captions_info = sample['captions_info']
                caption = random.choice(captions_info)['caption']
            else:
                caption = sample['captions_info'][0]['caption']
            return caption

        def __getitem__(self, item):
            item = item % len(self.metadata)
            sample = self.metadata[item]
            video_fp, rel_fp = self._get_video_path(sample)
            caption = self._get_caption(sample)

            frame_sample = 'rand'
            # frame_sample = 'uniform'
            fix_start = None
            if self.split == 'test':
                frame_sample = 'uniform'
            # if self.sliding_window_stride != -1:
            #     fix_start = sample['fix_start']

            try:
                if os.path.isfile(video_fp):
                    if self.reader == 'imgs':
                        imgs, idxs, vlen = self.video_reader(sample, self.video_params['num_frames'], frame_sample,
                                                             fix_start=fix_start)
                    else:
                        imgs, idxs, vlen = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample,
                                                             fix_start=fix_start)
                else:
                    print(f"Warning: missing video file {video_fp}.")
                    assert False
            except Exception as e:
                print("video is error in: {}".format(rel_fp))
                new_item = random.randint(1, len(self.metadata))
                return self.__getitem__(new_item)

            if self.transforms is not None:
                imgs = self.transforms(imgs)

            final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                                 self.video_params['input_res']])
            final[:imgs.shape[0]] = imgs

            meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name,
                        }
            data = {'video': final, 'text': caption, 'meta': meta_arr}
            return data

elif params['version'] == 'fusion':
    class ZYZOCRanns_datasets(TextVideoDataset):
        def __init__(self,
                     dataset_name,
                     text_params,
                     video_params,
                     data_dir,
                     metadata_dir=None,
                     split='train',
                     tsfms=None,
                     cut=None,
                     subsample=1,
                     sliding_window_stride=-1,
                     reader='decord'
                     ):
            self.reader = reader
            self.dataset_name = dataset_name
            self.text_params = text_params
            self.video_params = video_params
            # check for environment variables
            self.data_dir = os.path.expandvars(data_dir)
            if metadata_dir is not None:
                self.metadata_dir = os.path.expandvars(metadata_dir)
            else:
                self.metadata_dir = self.data_dir
            self.split = split
            self.transforms = tsfms
            self.cut = cut
            self.subsample = subsample
            self.sliding_window_stride = sliding_window_stride
            print(f'current video reader : {reader}')
            self.video_reader = video_reader[reader]
            self.label_type = 'caption'
            self.topl = params['ocr_token_l']  # input the top l score words
            self.min_ocr_l = 3  # min word length
#             self._load_metadata()
            # self._cache_videos()
            self.lower = True
            
#             cache_name = os.path.join('cache', dataset_name+ '_' + split + '.json')
#             if os.path.exists(cache_name):
#                 with open(cache_name,'r') as fr:
#                     self.metadata = json.load(fr)
#             else:
#                 self._load_metadata()
#                 with open(cache_name,'w') as fw:
#                     json.dump(self.metadata,fw)
            self._load_metadata()

            # data augmentation pipelines used in OA-Transformer
            if self.split == 'train':
                self.transforms = transforms.Compose(
                    [
                        transforms.Resize(size=(242, 242)),
                        transforms.RandomCrop(size=(224, 224)),
                        # add for data augmentation
                        transforms.RandomApply([
                            transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.3),
                        # transforms.RandomApply([base_augmentation.GaussianBlur([.1, 2.])], p=0.5), # PIL
                        # transforms.ToTensor(), # PIL
                        # transforms.RandomResizedCrop(size=(224, 224)),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )
            else:
                self.transforms = transforms.Compose(
                    [
                        transforms.Resize(size=(224, 224)),
                        # transforms.RandomResizedCrop(size=(224, 224)),
                        # transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )

            print('-'*50)
            print(self.transforms)
            print('-'*50)


            # if self.sliding_window_stride != -1:
            #     if self.split != 'test':
            #         raise ValueError('Fixing frame sampling is for test time only. can remove but...')
            #     self._fix_temporal_samples()

        def _load_metadata(self):
#             data_dir = '../'
            
            if self.dataset_name == 'MSRVTT_zyzocranns_official':
                anns_path = os.path.join(data_dir, 'msrvtt.json')
                with open(anns_path, 'r') as fr:
                    zyz_anns = json.load(fr)
                    split = 'official_split'
                train_anns = [el for el in zyz_anns if el['videos_info'][split] == 'train']
                val_anns = [el for el in zyz_anns if el['videos_info'][split] == 'test']
                test_anns = [el for el in zyz_anns if el['videos_info'][split] == 'test']
            elif self.dataset_name == 'MSRVTT_zyzocranns_oneka':
                anns_path = os.path.join(data_dir, 'msrvtt/msrvtt.json')
                with open(anns_path, 'r') as fr:
                    zyz_anns = json.load(fr)
                    split = 'oneka_split'
                train_anns = [el for el in zyz_anns if el['videos_info'][split] == 'train']
                val_anns = [el for el in zyz_anns if el['videos_info'][split] == 'val']
                test_anns = [el for el in zyz_anns if el['videos_info'][split] == 'val']
            elif "MSVD" in self.dataset_name:
                anns_path = os.path.join(data_dir, 'msvd/msvd.json')
                with open(anns_path, 'r') as fr:
                    zyz_anns = json.load(fr)
                    split = 'split'
                train_anns = [el for el in zyz_anns if el['videos_info'][split] == 'train']
                val_anns = [el for el in zyz_anns if el['videos_info'][split] == 'test']
                test_anns = [el for el in zyz_anns if el['videos_info'][split] == 'test']
            elif "DIDEMO" in self.dataset_name:
                anns_path = os.path.join(data_dir, 'didemo/didemo.json')
                with open(anns_path, 'r') as fr:
                    zyz_anns = json.load(fr)
                    split = 'split'
                train_anns = [el for el in zyz_anns if el['videos_info'][split] == 'train']
                val_anns = [el for el in zyz_anns if el['videos_info'][split] == 'test']
                test_anns = [el for el in zyz_anns if el['videos_info'][split] == 'test']
            elif "WEBIVD" in self.dataset_name:
                anns_path = os.path.join(data_dir, 'webvid/webvid.json')
                with open(anns_path, 'r') as fr:
                    zyz_anns = json.load(fr)
                    split = 'split'
                train_anns = [el for el in zyz_anns if el['videos_info'][split] == 'train']
                val_anns = [el for el in zyz_anns if el['videos_info'][split] == 'val']
                test_anns = [el for el in zyz_anns if el['videos_info'][split] == 'val']
            elif "YOUCOOK2SEG" in self.dataset_name:
#                 anns_path = os.path.join(data_dir, 'youcook2seg/youcook2seg.json')
                anns_path = params['anns']
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
                test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
            elif self.dataset_name == 'TextVR':
                # anns_path = os.path.join(data_dir, 'vitvr/vitvr_822.json')
                if self.split == 'train':
                    anns_path = params['train_anns']
                else:
                    anns_path = params['val_anns']
                with open(anns_path, 'r') as fr:
                    zyz_anns = json.load(fr)
                    if not os.path.exists(zyz_anns[0]['path']):
                        for i,ann in enumerate(zyz_anns):
                            zyz_anns[i]['path'] = os.path.join(self.data_dir,"Video",zyz_anns[i]['path'])
                            
                    split = 'split'
                train_anns = [el for el in zyz_anns if el['videos_info'][split] == 'train']
                val_anns = [el for el in zyz_anns if el['videos_info'][split] == 'test']
                test_anns = [el for el in zyz_anns if el['videos_info'][split] == 'test']
            else:
                dataset = self.dataset_name.split('_')[0]
                anns_path = os.path.join(data_dir, f'{dataset}/{dataset}.json')
                with open(anns_path, 'r') as fr:
                    zyz_anns = json.load(fr)
                    split = 'split'
                train_anns = [el for el in zyz_anns if el['videos_info'][split] == 'train']
                val_anns = [el for el in zyz_anns if el['videos_info'][split] == 'val']
                test_anns = [el for el in zyz_anns if el['videos_info'][split] == 'val']

            l_train = len(train_anns)
            l_val = len(val_anns)
            l_test = len(test_anns)
            self.split_sizes = {'train': l_train, 'val': l_val, 'test': l_test}

            if self.split == 'train':
                self.metadata = train_anns
            elif self.split == 'val':
                self.metadata = val_anns
            else:
                self.metadata = test_anns
            
            if 'ocr_info' not in self.metadata[0]:
                self.norm_flag = True
                self._load_ocr_infos()
            else:
                print('load existing ocr info...')
                self.norm_flag = False # do not normalize the meta data

        def _load_ocr_infos(self):
            for el in self.metadata:
                ocr_path = el['videos_info']['Kwai_OCR']
#                 if not os.path.exists(ocr_path):
#                     ocr_path = ocr_path.replace('/mmu-ocr','/mmu-ocr/pub')
                ocr_path = os.path.join(self.data_dir,"Kwai_VideoOCR",ocr_path)
                if not os.path.exists(ocr_path):
                    print(f'{ocr_path} not exists.')
                    continue
                
                with open(ocr_path, 'r') as fr:
                    ocr_file = json.load(fr)
                el['ocr_info'] = ocr_file
            self.metadata = [self.ocr_format(el) for el in self.metadata]

        def ocr_format(self, sample):
            ocr_info = sample['ocr_info']
            gt_text_list = []
            gt_pos_list = []
            gt_ocr_list = []

            if len(ocr_info)!=0:
                for key, value in ocr_info.items():
                    gt_score = float(value['score'])
                    gt_text = value['text']

                    if ' ' in gt_text:
                        gt_texts = gt_text.split(' ')
                    else:
                        gt_texts = [gt_text]

                    for text in gt_texts:
                        ord_text = [ord(el) for el in text]
                        if len(ord_text) < 1:
                            continue
                        if max(ord_text) > 123:  # remove chinese
                            continue
                        if text.strip() in gt_text_list:
                            continue
                        if len(text) < self.min_ocr_l:
                            continue

                        tracks = value['tracks']
                        keys = np.array([int(el) for el in tracks.keys()])
                        values = np.array([el[:-2] for el in tracks.values()])
                        # vec : [st, ed, ed-st, mx, my, mw, mh, edx-stx, edy-sty, edw/stw, edh/sth]
                        # norm : [l, l, l, w, h, w, h, w, h, 1, 1, 1]
                        pos = np.zeros(12)
                        pos[0] = keys[0]
                        pos[1] = keys[-1]
                        pos[2] = len(keys)
                        pos[3:5] = values.reshape(-1,4,2).mean((0,1))
                        pos[5:7] = values.reshape(-1,4,2).mean(0).max(0)-values.reshape(-1,4,2).mean(0).min(0)
                        pos[7:9] = (values.reshape(-1,4,2)[-1]-values.reshape(-1,4,2)[-1][0]).mean(0)
                        edwh = values.reshape(-1,4,2)[-1].max(0)-values.reshape(-1,4,2)[-1].min(0)
                        stwh = values.reshape(-1,4,2)[0].max(0)-values.reshape(-1,4,2)[0].min(0)
                        pos[9:11] = edwh/(stwh+1e-6)
                        pos[-1] = gt_score
                        
                        if self.lower:
                            out_text = text.strip().lower()
                        else:
                            out_text = text.strip()

                        gt_text_list.append(out_text)
                        gt_ocr_list.append((out_text, pos))
                gt_ocr_list = sorted(gt_ocr_list, key=lambda x: x[1][-1], reverse=True)
                gt_ocr_list = gt_ocr_list[:self.topl]
                gt_text_list = [el[0] for el in gt_ocr_list]
                gt_pos_list = [el[1] for el in gt_ocr_list]

            valid_l = len(gt_text_list)
            if valid_l > self.topl:
                gt_text_list = gt_text_list[:self.topl]
                gt_pos_list = gt_pos_list[:self.topl]
            else:
                gt_text_list = gt_text_list + [' '] * (self.topl - valid_l)
                gt_pos_list = gt_pos_list + [np.zeros(12)] * (self.topl - valid_l)

            sample['ocr_info'] = dict(gt_ocr=gt_text_list, gt_pos=np.array(gt_pos_list).astype(np.float32).tolist(), valid_l=valid_l)
            return sample

        def _get_video_path(self, sample):
            path = sample['path']
            if not os.path.exists(path):
                    path = path.replace('/mmu-ocr','/mmu-ocr/pub')
            # path = '/home/zyz/code/SynthText/tmp/demo.mp4'
            return path, os.path.basename(path)

        def _get_caption(self, sample):
            caption_sample = self.text_params.get('caption_sample', "rand")
            if self.split in ['train'] and caption_sample == "rand":
                captions_info = sample['captions_info']
                caption = random.choice(captions_info)['caption']
            else:
                caption = sample['captions_info'][0]['caption']
            if self.lower:
                caption = caption.lower()
            return caption

        def _norm_pos(self, pos, w, h, l):
            norm_vec = np.array([l, l, l, w, h, w, h, w, h, 1, 1, 1])
            pos = pos/norm_vec[None]
            return pos.tolist()

        def __getitem__(self, item):
            item = item % len(self.metadata)
            sample = self.metadata[item]
            video_fp, rel_fp = self._get_video_path(sample)
            caption = self._get_caption(sample)

            frame_sample = 'rand'
            # frame_sample = 'uniform'
            fix_start = None
            if self.split == 'test':
                frame_sample = 'uniform'
            # if self.sliding_window_stride != -1:
            #     fix_start = sample['fix_start']

            try:
                if os.path.isfile(video_fp):
                    if self.reader == 'imgs':
                        imgs, idxs, vlen = self.video_reader(sample, self.video_params['num_frames'], frame_sample,
                                                             fix_start=fix_start)
                    else:
                        imgs, idxs, vlen = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample,
                                                             fix_start=fix_start)
                else:
                    print(f"Warning: missing video file {video_fp}.")
                    assert False
            except Exception as e:
                print("video is error in: {}".format(rel_fp))
                new_item = random.randint(1, len(self.metadata))
                return self.__getitem__(new_item)

            ocr_info = sample['ocr_info']

            # normalize the pos vec
            if self.norm_flag:
                _, _, h, w = imgs.shape
                ocr_info['gt_pos'] = self._norm_pos(np.array(ocr_info['gt_pos']), w, h , vlen)

            if self.transforms is not None:
                imgs = self.transforms(imgs)

            final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                                 self.video_params['input_res']])
            final[:imgs.shape[0]] = imgs

            meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name,
                        'ocr_info': str(ocr_info)}
            data = {'video': final, 'text': caption, 'meta': meta_arr}
            return data


else:
    class ZYZOCRanns_datasets(TextVideoDataset):
        def __init__(self,
                     dataset_name,
                     text_params,
                     video_params,
                     data_dir,
                     metadata_dir=None,
                     split='train',
                     tsfms=None,
                     cut=None,
                     subsample=1,
                     sliding_window_stride=-1,
                     reader='decord'
                     ):
            self.reader = reader
            self.dataset_name = dataset_name
            self.text_params = text_params
            self.video_params = video_params
            # check for environment variables
            self.data_dir = os.path.expandvars(data_dir)
            if metadata_dir is not None:
                self.metadata_dir = os.path.expandvars(metadata_dir)
            else:
                self.metadata_dir = self.data_dir
            self.split = split
            self.transforms = tsfms
            self.cut = cut
            self.subsample = subsample
            self.sliding_window_stride = sliding_window_stride
            self.video_reader = video_reader[reader]
            self.label_type = 'caption'
            self.topl = 20  # input the top l score words
            self.min_ocr_l = 3  # min word length
            self._load_metadata()
            # self._cache_videos()


            # data augmentation pipelines used in OA-Transformer
            if self.split == 'train':
                self.transforms = transforms.Compose(
                    [
                        transforms.Resize(size=(242, 242)),
                        transforms.RandomCrop(size=(224, 224)),
                        # add for data augmentation
                        transforms.RandomApply([
                            transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.3),
                        # transforms.RandomApply([base_augmentation.GaussianBlur([.1, 2.])], p=0.5), # PIL
                        # transforms.ToTensor(), # PIL
                        # transforms.RandomResizedCrop(size=(224, 224)),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )
            else:
                self.transforms = transforms.Compose(
                    [
                        transforms.Resize(size=(224, 224)),
                        # transforms.RandomResizedCrop(size=(224, 224)),
                        # transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )

            print('-'*50)
            print(self.transforms)
            print('-'*50)


            # if self.sliding_window_stride != -1:
            #     if self.split != 'test':
            #         raise ValueError('Fixing frame sampling is for test time only. can remove but...')
            #     self._fix_temporal_samples()

        def _load_metadata(self):
            data_dir = '../'
            if self.dataset_name == 'MSRVTT_zyzocranns_official':
                anns_path = os.path.join(data_dir, 'msrvtt.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'official_split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            elif self.dataset_name == 'MSRVTT_zyzocranns_oneka':
                anns_path = os.path.join(data_dir, 'msrvtt/msrvtt.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'oneka_split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
            elif "MSVD" in self.dataset_name:
                anns_path = os.path.join(data_dir, 'msvd/msvd.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            elif "DIDEMO" in self.dataset_name:
                anns_path = os.path.join(data_dir, 'didemo/didemo.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            elif "WEBIVD" in self.dataset_name:
                anns_path = os.path.join(data_dir, 'webvid/webvid.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
            elif 'VITVR_zyzocranns' == self.dataset_name:
                # anns_path = os.path.join(data_dir, 'vitvr/vitvr_822.json')
                anns_path = 'data/vitvr_830.json'
                with open(anns_path, 'r') as fr:
                    zyz_anns = json.load(fr)
                    if not os.path.exists(zyz_anns[0]['path']):
                        for ann in zyz_anns:
                            ann['path'] = ann['path'].replace('/mmu-ocr/pub', '/mmu-ocr')
                    self.zyz_anns = zyz_anns
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            else:
                dataset = self.dataset_name.split('_')[0]
                anns_path = os.path.join(data_dir, f'{dataset}/{dataset}.json')
                with open(anns_path, 'r') as fr:
                    self.zyz_anns = json.load(fr)
                    split = 'split'
                self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
                self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
                self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']

            l_train = len(self.train_anns)
            l_val = len(self.val_anns)
            l_test = len(self.test_anns)
            self.split_sizes = {'train': l_train, 'val': l_val, 'test': l_test}

            if self.split == 'train':
                self.metadata = self.train_anns
            elif self.split == 'val':
                self.metadata = self.val_anns
            else:
                self.metadata = self.test_anns

            self._load_ocr_infos()

        def _load_ocr_infos(self):
            for el in self.metadata:
                ocr_path = el['videos_info']['ocr_path']
                ocr_path = os.path.join('data/vitvr_json/', *ocr_path.split('/')[-2:])
                # ocr_path = ocr_path.replace('/mmu-ocr/datasets/zyz_anns/ocr_anns','./data/MSRVTT')
                if not os.path.exists(ocr_path):
                    print(f'{ocr_path} not exists.')
                    continue
                with open(ocr_path, 'r') as fr:
                    ocr_file = json.load(fr)
                el['ocr_info'] = ocr_file
            self.metadata = [self.ocr_format(el) for el in self.metadata]

        def ocr_format(self, sample):
            ocr_info = sample['ocr_info']
            gt_text_list = []
            gt_pos_list = []
            gt_ocr_list = []

            if len(ocr_info)!=0:
                for key, value in ocr_info.items():
                    gt_score = float(value['score'])
                    gt_text = value['text']

                    if ' ' in gt_text:
                        gt_texts = gt_text.split(' ')
                    else:
                        gt_texts = [gt_text]

                    for text in gt_texts:
                        ord_text = [ord(el) for el in text]
                        if len(ord_text) < 1:
                            continue
                        if max(ord_text) > 123:  # remove chinese
                            continue
                        if text.strip() in gt_text_list:
                            continue
                        if len(text) < self.min_ocr_l:
                            continue

                        tracks = value['tracks']
                        keys = np.array([int(el) for el in tracks.keys()])
                        values = np.array([el[:-2] for el in tracks.values()])
                        # vec : [st, ed, ed-st, mx, my, mw, mh, edx-stx, edy-sty, edw/stw, edh/sth]
                        # norm : [l, l, l, w, h, w, h, w, h, 1, 1, 1]
                        pos = np.zeros(12)
                        pos[0] = keys[0]
                        pos[1] = keys[-1]
                        pos[2] = len(keys)
                        pos[3:5] = values.reshape(-1,4,2).mean((0,1))
                        pos[5:7] = values.reshape(-1,4,2).mean(0).max(0)-values.reshape(-1,4,2).mean(0).min(0)
                        pos[7:9] = (values.reshape(-1,4,2)[-1]-values.reshape(-1,4,2)[-1][0]).mean(0)
                        edwh = values.reshape(-1,4,2)[-1].max(0)-values.reshape(-1,4,2)[-1].min(0)
                        stwh = values.reshape(-1,4,2)[0].max(0)-values.reshape(-1,4,2)[0].min(0)
                        pos[9:11] = edwh/(stwh+1e-6)
                        pos[-1] = gt_score

                        gt_text_list.append(text.strip())
                        gt_ocr_list.append((text.strip(), pos))
                gt_ocr_list = sorted(gt_ocr_list, key=lambda x: x[1][-1], reverse=True)
                gt_ocr_list = gt_ocr_list[:self.topl]
                gt_text_list = [el[0] for el in gt_ocr_list]
                gt_pos_list = [el[1] for el in gt_ocr_list]

            valid_l = len(gt_text_list)
            if valid_l > self.topl:
                gt_text_list = gt_text_list[:self.topl]
                gt_pos_list = gt_pos_list[:self.topl]
            else:
                gt_text_list = gt_text_list + [' '] * (self.topl - valid_l)
                gt_pos_list = gt_pos_list + [np.zeros(12)] * (self.topl - valid_l)

            sample['ocr_info'] = dict(gt_ocr=gt_text_list, gt_pos=np.array(gt_pos_list).astype(np.float32), valid_l=valid_l)
            return sample

        def _get_video_path(self, sample):
            path = sample['path']
#             path = '/home/zyz/code/SynthText/tmp/demo.mp4'
            return path, os.path.basename(path)

        def _get_caption(self, sample):
            caption_sample = self.text_params.get('caption_sample', "rand")
            if self.split in ['train'] and caption_sample == "rand":
                captions_info = sample['captions_info']
                caption = random.choice(captions_info)['caption']
            else:
                caption = sample['captions_info'][0]['caption']
            return caption

        def __getitem__(self, item):
            item = item % len(self.metadata)
            sample = self.metadata[item]
            video_fp, rel_fp = self._get_video_path(sample)
            caption = self._get_caption(sample)

            frame_sample = 'rand'
            # frame_sample = 'uniform'
            fix_start = None
            if self.split == 'test':
                frame_sample = 'uniform'
            # if self.sliding_window_stride != -1:
            #     fix_start = sample['fix_start']

            try:
                if os.path.isfile(video_fp):
                    if self.reader == 'imgs':
                        imgs, idxs, vlen = self.video_reader(sample, self.video_params['num_frames'], frame_sample,
                                                             fix_start=fix_start)
                    else:
                        imgs, idxs, vlen = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample,
                                                             fix_start=fix_start)
                else:
                    print(f"Warning: missing video file {video_fp}.")
                    assert False
            except Exception as e:
                print("video is error in: {}".format(rel_fp))
                new_item = random.randint(1, len(self.metadata))
                return self.__getitem__(new_item)

            ocr_info = sample['ocr_info']

            if self.transforms is not None:
                imgs = self.transforms(imgs)

            final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                                 self.video_params['input_res']])
            final[:imgs.shape[0]] = imgs

            meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name,
                        'ocr_info': str(ocr_info)}
            data = {'video': final, 'text': caption, 'meta': meta_arr}
            return data





class ZYZOCRanns_datasets_youcook2seg(TextVideoDataset):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfms=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='decord'
                 ):
        self.dataset_name = dataset_name
        self.text_params = text_params
        self.video_params = video_params
        # check for environment variables
        self.data_dir = os.path.expandvars(data_dir)
        if metadata_dir is not None:
            self.metadata_dir = os.path.expandvars(metadata_dir)
        else:
            self.metadata_dir = self.data_dir
        self.split = split
        self.transforms = tsfms
        self.cut = cut
        self.subsample = subsample
        self.sliding_window_stride = sliding_window_stride
        self.video_reader = video_reader[reader]
        self.label_type = 'caption'
        self._load_metadata()
        self.max_ocr_l = 50

        # data augmentation pipelines used in OA-Transformer
        if self.split == 'train':
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(size=(242, 242)),
                    transforms.RandomCrop(size=(224, 224)),
                    # add for data augmentation
                    transforms.RandomApply([
                        transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.3),
                    # transforms.RandomApply([base_augmentation.GaussianBlur([.1, 2.])], p=0.5), # PIL
                    # transforms.ToTensor(), # PIL
                    # transforms.RandomResizedCrop(size=(224, 224)),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    # transforms.RandomResizedCrop(size=(224, 224)),
                    # transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        print('-'*50)
        print(self.transforms)
        print('-'*50)


        if self.sliding_window_stride != -1:
            if self.split != 'test':
                raise ValueError('Fixing frame sampling is for test time only. can remove but...')
            self._fix_temporal_samples()





    def _load_ocr_infos(self):
        return


    def ocr_format(self, sample, imgs, idxs, vlen):
        b, c, h, w = imgs.shape
        ocr_info = sample['ocr_info']
        gt_text_list = []
        gt_score_list = []

        for key, value in ocr_info.items():
            box = value['recommend_box']
            gt_score = value['score']
            gt_text = value['text']
            tracks = value['tracks']
            gt_text_list.append(gt_text)
            gt_score_list.append(gt_score)

        valid_l = len(gt_text_list)
        if valid_l>self.max_ocr_l:
            gt_text_list = gt_text_list[:self.max_ocr_l]
            gt_score_list = gt_score_list[:self.max_ocr_l]
        else:
            gt_text_list = gt_text_list + [' ']*(self.max_ocr_l-valid_l)
            gt_score_list = gt_score_list + [0]*(self.max_ocr_l-valid_l)
        return dict(gt_ocr=gt_text_list,gt_ocr_score=gt_score_list,valid_l = valid_l)

    def _get_caption(self, sample):
        caption_sample = self.text_params.get('caption_sample', "rand")
        if self.split in ['train'] and caption_sample == "rand":
            captions_info = sample['captions_info']
            caption = random.choice(captions_info)['caption']
        else:
            caption = sample['captions_info'][0]['caption']
        return caption

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        frame_sample = 'rand'
        # frame_sample = 'uniform'
        fix_start = None
        if self.split == 'test':
            frame_sample = 'uniform'
        # if self.sliding_window_stride != -1:
        #     fix_start = sample['fix_start']

        try:
            if os.path.isfile(video_fp):
                imgs, idxs, vlen = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample,
                                               fix_start=fix_start)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            print("video is error in: {}".format(rel_fp))
            new_item = random.randint(1, len(self.metadata))
            return self.__getitem__(new_item)

        ocr_info = self.ocr_format(sample, imgs, idxs, vlen)

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs

        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name, 'ocr_info': str(ocr_info)}
        data = {'video': final, 'text': caption, 'meta': meta_arr}
        return data


def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs


def read_clip_frames_cv2(video_path, num_frames, sample='rand', fix_start=None):
    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')
    cap.release()
    return frames


def read_frames_cv2(video_path, num_frames, sample='rand', fix_start=None):
    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')

    frames = torch.stack(frames).float() / 255
    cap.release()
    return frames, success_idxs, vlen


def read_frames_av(video_path, num_frames, sample='rand', fix_start=None):
    reader = av.open(video_path)
    try:
        frames = []
        frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    except (RuntimeError, ZeroDivisionError) as exception:
        print('{}: WEBM reader cannot open {}. Empty '
              'list returned.'.format(type(exception).__name__, video_path))
    vlen = len(frames)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = torch.stack([frames[idx] for idx in frame_idxs]).float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs, vlen


def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs)
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs, vlen


def cache_video_name(cache_dir, video_path, ext = 'npy'):
    basename = '__'.join(video_path.split('/')[-3:])
    return os.path.join(cache_dir, basename).replace('mp4',ext)


def cache_video(path):
    try:
        cache_name = cache_video_name('./cache', path)
        if os.path.exists(cache_name):
            return
        cap = cv2.VideoCapture(path)
        assert (cap.isOpened())
        frames = []
        while True:
            exist, frame = cap.read()  # 读取一帧数据
            if not exist:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.moveaxis(frame,2,0)
            frames.append(frame)
        frames = np.stack(frames, 0)
        np.save(cache_name, frames)
    except:
        traceback.print_exc()
        print(path)
    return


def cache_videos(paths):
    nworkers = 32
    todos = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
        for path in paths:
            future = executor.submit(cache_video, path)
            todos.append(future)

        for future in concurrent.futures.as_completed(todos):
            try:
                print(future.result())
            except ZeroDivisionError as e:
                print(e.__repr__())


def read_frames_imgs(meta, num_frames, sample='rand', fix_start=None):

    try:
        source_dir = 'cache/vitvr'

        sample_dir = meta['videos_info']['source'] + '__' + os.path.basename(meta['path'])[:-4]

        dir_ = os.path.join(source_dir, sample_dir)

        try:
            assert os.path.exists(dir_)
        except:
            print(f'{dir_} not exists.')
            raise ValueError

        imgs = sorted([os.path.join(dir_,el) for el in os.listdir(dir_) if el.endswith('jpg')])

        vlen = len(imgs)

        frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)

        imgs = [imgs[idx] for idx in frame_idxs]

        frames = [cv2.imread(el)[:,:,::-1] for el in imgs]

        frames = np.stack(frames,0)

        frames = torch.from_numpy(frames)

        frames = frames.float() / 255
        frames = frames.permute(0, 3, 1, 2)
        return frames, frame_idxs, vlen
    except:
        return read_frames_decord(meta['path'], num_frames, sample, fix_start=fix_start)


def read_frames_decord_cache(video_path, num_frames, sample='rand', fix_start=None):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)

    cache_dir = './cache'
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    cache_name = cache_video_name(cache_dir, video_path)
    if os.path.exists(cache_name):
        frames = np.load(cache_name)
    else:
        frames = video_reader.get_batch(list(range(vlen)))
        np.save(cache_name, frames.numpy())
    frame_idxs = np.array(frame_idxs,dtype=np.long)
    frames = torch.from_numpy(frames[frame_idxs])

    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs, vlen


def get_video_len(video_path):
    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen



video_reader = {
    'av': read_frames_av,
    'cv2': read_frames_cv2,
    'decord': read_frames_decord,
    'imgs': read_frames_imgs,
}

if __name__ == '__main__':
    anns_path = '../vitvr/vitvr_830.json'
    with open(anns_path, 'r') as fr:
        zyz_anns = json.load(fr)
        if not os.path.exists(zyz_anns[0]['path']):
            for ann in zyz_anns:
                ann['path'] = ann['path'].replace('/mmu-ocr/pub', '/mmu-ocr')
    split = 'split'
    paths = [el['path'] for el in zyz_anns if el['videos_info'][split] == 'train']
    cache_videos(paths)

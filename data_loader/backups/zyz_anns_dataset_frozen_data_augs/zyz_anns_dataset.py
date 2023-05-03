'''
video datasets for zyz anns
'''


import json
import os
import random

import torch
from PIL import Image
from torchvision import transforms

from base.base_dataset import TextVideoDataset


class ZYZanns_datasets(TextVideoDataset):
    def _load_metadata(self):
        data_dir = '/share/mmu-ocr/datasets/zyz_anns'
        test_for_validation = True
        if self.dataset_name == 'MSRVTT_zyzanns_official':
            anns_path = os.path.join(data_dir,'msrvtt/msrvtt.json')
            with open(anns_path, 'r') as fr:
                self.zyz_anns = json.load(fr)
                split = 'official_split'
            self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
            self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
        elif self.dataset_name == 'MSRVTT_zyzanns_oneka':
            anns_path = os.path.join(data_dir,'msrvtt/msrvtt.json')
            with open(anns_path, 'r') as fr:
                self.zyz_anns = json.load(fr)
                split = 'oneka_split'
            self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
            self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
            self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
        elif self.dataset_name == 'MSVD_zyzanns':
            anns_path = os.path.join(data_dir,'msvd/msvd.json')
            with open(anns_path, 'r') as fr:
                self.zyz_anns = json.load(fr)
                split = 'split'
            self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
            self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
        elif self.dataset_name == 'DIDEMO_zyzanns':
            anns_path = os.path.join(data_dir,'didemo/didemo.json')
            with open(anns_path, 'r') as fr:
                self.zyz_anns = json.load(fr)
                split = 'split'
            self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
            self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
            self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'test']
        elif self.dataset_name == 'WEBVID_zyzanns':
            anns_path = os.path.join(data_dir,'webvid/webvid.json')
            with open(anns_path, 'r') as fr:
                self.zyz_anns = json.load(fr)
                split = 'split'
            self.train_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'train']
            self.val_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
            self.test_anns = [el for el in self.zyz_anns if el['videos_info'][split] == 'val']
        else:
            raise


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
        new_path = path.replace('/mmu-ocr','/share/mmu-ocr')
        return new_path, os.path.basename(new_path)

    def _get_caption(self, sample):
        caption_sample = self.text_params.get('caption_sample', "rand")
        if self.split in ['train', 'val'] and caption_sample == "rand":
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

        video_loading = self.video_params.get('loading', 'strict')
        frame_sample = 'rand'
        fix_start = None
        if self.split == 'test':
            frame_sample = 'uniform'
        # if self.sliding_window_stride != -1:
        #     fix_start = sample['fix_start']

        try:
            if os.path.isfile(video_fp):
                imgs, idxs = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample,
                                               fix_start=fix_start)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            print("video is error in: {}".format(rel_fp))
            new_item = random.randint(1, len(self.metadata))
            return self.__getitem__(new_item)
            # if video_loading == 'strict':
            #     raise ValueError(
            #         f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            # else:
            #     print("video is error in: {}".format(rel_fp))
            #     new_item = random.randint(1, len(self.metadata))
            #     return self.__getitem__(new_item)
            #     # imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
            #     # imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs

        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': final, 'text': caption, 'meta': meta_arr}
        return data


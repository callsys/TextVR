'''
video datasets for zyz anns
'''


import json
import cv2
import av
import os
import numpy as np
import torch
import random
from PIL import Image
from torchvision import transforms
import decord

from base.base_dataset import TextVideoDataset


class ZYZanns_datasets(TextVideoDataset):
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

        # data augmentation pipelines used in OA-Transformer
        if self.split == 'train':
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(size=(238, 238)),
                    transforms.RandomCrop(size=(224, 224)),
                    # add for data augmentation
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    # transforms.RandomApply([base_augmentation.GaussianBlur([.1, 2.])], p=0.5), # PIL
                    # transforms.ToTensor(), # PIL
                    # transforms.RandomResizedCrop(size=(224, 224)),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(size=(238, 238)),
                    transforms.CenterCrop(size=(224, 224)),
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

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs

        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
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
    'decord': read_frames_decord
}
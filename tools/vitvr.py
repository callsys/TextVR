import copy
import os
import json
import numpy as np
import tqdm
import random
import concurrent
from concurrent.futures import ProcessPoolExecutor
import cv2

ocr_dir = '../vitvr/vitvr_json'

json_path = '../vitvr/vitvr_830.json'



with open(json_path,'r') as fr:
    files = json.load(fr)

print(len(files))


class Matcher():
    def __init__(self,files):
        self.files = files
        self.ocr_dir = ocr_dir
        self.mat = np.zeros((len(files), 3))
        self.kwords = set()

    def metric(self):
        total = len(self.files)
        valid = self.mat[:, -1]!=-1
        mat = self.mat[valid]

        print(f'valid data : {valid.sum()}/{total}')
        for i in range(1,10):
            print(f'{i} words matched videos : {(mat[:,-1]>=i).sum()} / {len(mat)}')

        print(f'n match recall {(mat[:,-1]/mat[:,0]).mean()}')

    def match(self, lower=True, minl=2):



        for i,file in enumerate(tqdm.tqdm(files)):
            try:
                captions = [el['caption'] for el in file['captions_info']]
                kwords = []
                for caption in captions:
                    idx = caption.index(':')
                    tmp = caption[:idx]
                    tmp = tmp.strip().split(' ')
                    kwords.extend(tmp)
                if lower:
                    kwords = [el.lower().strip(',') for el in kwords]
                kwords = set(kwords)
                self.kwords = self.kwords.union(kwords)

                ocr_path = file['videos_info']['ocr_path']
                ocr_path = os.path.join(self.ocr_dir,*ocr_path.split('/')[-2:])
                assert os.path.exists(ocr_path)
                with open(ocr_path,'r') as fr:
                    ocr_file = json.load(fr)
                owords = []
                ocr_lines = [el['text'] for el in ocr_file.values()]
                for line in ocr_lines:
                    owords.extend(line.strip().split(' '))
                if lower:
                    owords = [el.lower() for el in owords]
                owords = [el for el in owords if len(el)>minl]
                owords = set(owords)

                self.mat[i,0] = len(kwords)
                self.mat[i,1] = len(owords)
                self.mat[i,2] = len(kwords.intersection(owords))
            except:
                self.mat[i] = -1

        self.metric()

        return self.mat

    def match2(self, lower=True, minl=2):



        for i,file in enumerate(tqdm.tqdm(files)):
            try:
                captions = [el['caption'] for el in file['captions_info']]
                kwords = []
                for caption in captions:
                    tmp = caption.strip().split(' ')
                    kwords.extend(tmp)
                if lower:
                    kwords = [el.lower().strip(',') for el in kwords]
                kwords = set(kwords)
                self.kwords = self.kwords.union(kwords)

                ocr_path = file['videos_info']['ocr_path']
                ocr_path = os.path.join(self.ocr_dir,*ocr_path.split('/')[-2:])
                assert os.path.exists(ocr_path)
                with open(ocr_path,'r') as fr:
                    ocr_file = json.load(fr)
                owords = []
                ocr_lines = [el['text'] for el in ocr_file.values()]
                for line in ocr_lines:
                    owords.extend(line.strip().split(' '))
                if lower:
                    owords = [el.lower() for el in owords]
                owords = [el for el in owords if len(el)>minl]
                owords = set(owords)

                self.mat[i,0] = len(kwords)
                self.mat[i,1] = len(owords)
                self.mat[i,2] = len(kwords.intersection(owords))
            except:
                self.mat[i] = -1

        self.metric()

        return self.mat

    def mod_file(self, save_file = '../data/vitvr_822.json'):

        nfiles = []
        for i,file in enumerate(tqdm.tqdm(self.files)):
            try:
                kwords = []
                for i, caption in enumerate(file['captions_info']):
                    ncaption = caption['caption']
                    idx = ncaption.index(':')
                    if i==0:
                        tmp = caption['caption'][:idx]
                        tmp = tmp.strip().split(' ')
                        kwords.extend(tmp)

                    ncaption = ncaption[idx+1:].strip()
                    caption['caption'] = ncaption



                kwords = set(kwords)

                ocr_path = file['videos_info']['ocr_path']
                ocr_path = os.path.join(self.ocr_dir, *ocr_path.split('/')[-2:])
                assert os.path.exists(ocr_path)
                with open(ocr_path, 'r') as fr:
                    ocr_file = json.load(fr)
                owords = []
                ocr_lines = [el['text'] for el in ocr_file.values()]
                for line in ocr_lines:
                    owords.extend(line.strip().split(' '))
                owords = set(owords)

                if len(kwords.intersection(owords))>2:
                    file['videos_info']['split'] = 'test'
                else:
                    file['videos_info']['split'] = 'train'

                nfiles.append(file)
            except:
                continue

        with open(save_file,'w') as fw:
            json.dump(nfiles,fw)

        return

    def mod_random(self, save_file = '../data/vitvr_830_act2.json'):

        nfiles = []
        # for i,file in enumerate(tqdm.tqdm(self.files)):
        #     try:
        #         if random.random()<0.7:
        #             file['videos_info']['split'] = 'train'
        #         else:
        #             file['videos_info']['split'] = 'test'
        #
        #
        #         nfiles.append(file)
        #     except:
        #         continue

        for i,file in enumerate(tqdm.tqdm(self.files)):
            try:
                if 'Act' in file['path']:
                    if random.random() < 0.5:
                        file['videos_info']['split'] = 'train'
                    else:
                        file['videos_info']['split'] = 'test'
                else:
                    if random.random()<0.8:
                        file['videos_info']['split'] = 'train'
                    else:
                        file['videos_info']['split'] = 'test'


                nfiles.append(file)
            except:
                continue

        with open(save_file,'w') as fw:
            json.dump(nfiles,fw)

        return

    def add_actcaptions(self, save_file = '../data/vitvr_830_act.json'):
        path_list = ['../data/act_train.json', '../data/act_val1.json', '../data/act_val2.json']

        all_files = dict()
        for path in path_list:
            with open(path,'r') as fr:
                file = json.load(fr)
            all_files.update(file)

        nfiles = []
        for file in tqdm.tqdm(self.files):
            if 'Act' in file['path']:
                match_act = [(k,v) for k,v in all_files.items() if k in file['path']]
                assert len(match_act)==1
                texts = match_act[0][1]['sentences']
                for i,text in enumerate(texts):
                    file['captions_info'].insert(0,{'caption':text, 'caption_info':{'duration':0,'desc_id':100+i}})

            nfiles.append(file)

        with open(save_file,'w') as fw:
            json.dump(nfiles,fw)

        return


class Cacher():
    def __init__(self,files):
        self.files = files
        self.nworkers = 64
        self.save_dir = './cache/vitvr'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.total = len(self.files)
        self.finish = 0

    def cache(self):
        for file in files:
            try:
                file['path'] = file['path'].replace('/mmu-ocr/pub','/mmu-ocr')
                assert os.path.exists(file['path'])
            except:
                print(file)
                raise ValueError
        todos = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.nworkers) as executor:
            for file in files:
                future = executor.submit(self._cache, file)
                todos.append(future)

            for future in concurrent.futures.as_completed(todos):
                try:
                    print(future.result())
                except ZeroDivisionError as e:
                    print(e.__repr__())

    def _cache(self, file):
        path = file['path']
        save_dir = file['videos_info']['source'] + '__' + os.path.basename(path)[:-4]
        save_dir = os.path.join(self.save_dir, save_dir)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        video = cv2.VideoCapture(path)
        fps = int(video.get(cv2.CAP_PROP_FPS))

        count = 0
        while True:
            exist, frame = video.read()
            if not exist:
                break

            # frame = frame[:,:,::-1] # bgr to rgb

            name = os.path.join(save_dir, '{:0>8d}.jpg'.format(count))
            cv2.imwrite(name, frame)
            count += 1
        video.release()

        self.finish = self.finish + 1
        print(f'finished : {self.finish} | {self.total}')




if __name__ == '__main__':
    # matcher = Matcher(files)
    # matcher.match2()
    # matcher.mod_random()
    # matcher.mod_file()
    # matcher.add_actcaptions()

    cacher = Cacher(files)
    cacher.cache()

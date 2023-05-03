import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import transformers

from base import BaseModel
from model.video_transformer import SpaceTimeTransformer
from utils.util import state_dict_data_parallel_fix
from einops import repeat,rearrange
from params import params
import math
import einops
import numpy as np
version = params['version']


if version == 'base':
    class FrozenInTime(BaseModel):
        def __init__(self,
                     video_params,
                     text_params,
                     projection_dim=256,
                     load_checkpoint=None,
                     projection='minimal',
                     load_temporal_fix='zeros'):
            super().__init__()

            self.video_params = video_params
            self.text_params = text_params
            self.load_temporal_fix = load_temporal_fix
            if not text_params['pretrained']:
                raise NotImplementedError("Huggingface text models require pretrained init.")

            self.text_model = AutoModel.from_pretrained(text_params['model'])
            self.text_model.train()

            pretrained = video_params['pretrained']
            if video_params['model'] == "SpaceTimeTransformer":
                num_frames = video_params.get('num_frames', 4)
                time_init = video_params.get('time_init', 'zeros')
                attention_style = video_params.get('attention_style', 'frozen-in-time')
                arch_config = video_params.get('arch_config', 'base_patch16_224')
                vit_init = video_params.get('vit_init', 'imagenet-21k')
                if arch_config == 'base_patch16_224':
                    vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
                    model = SpaceTimeTransformer(num_frames=num_frames,
                                                time_init=time_init,
                                                attention_style=attention_style)
                else:
                    raise NotImplementedError

                model.head = nn.Identity()
                model.pre_logits = nn.Identity()
                ftr_dim = model.embed_dim
                if load_checkpoint in ["", None]:
                    vit_checkpoint = vit_model.state_dict()
                    model.load_state_dict(vit_checkpoint, strict=False)
                self.video_model = model
            else:
                raise NotImplementedError(f"{video_params['model']} not implemented")

            # for backwards compatibility (old models)
            self.video_model.fc = nn.Identity()

            # Project to a common embedding
            if projection == 'minimal':
                txt_proj = nn.Sequential(nn.ReLU(),
                                         nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                         )

                vid_proj = nn.Sequential(
                    nn.Linear(ftr_dim, projection_dim)
                )
            elif projection == '':
                txt_proj = nn.Identity()
                vid_proj = nn.Identity()
            else:
                raise NotImplementedError
            self.txt_proj = txt_proj
            self.vid_proj = vid_proj

            if load_checkpoint not in ["", None]:
                checkpoint = torch.load(load_checkpoint)
                state_dict = checkpoint['state_dict']
                new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
                new_state_dict = self._inflate_positional_embeds(new_state_dict)
                self.load_state_dict(new_state_dict, strict=True)

        def set_device(self, device):
            self.device = device

        def forward(self, data, return_embeds=True):

            text_data = data['text']
            video_data = data['video']

            text_embeddings = self.compute_text(text_data)
            video_embeddings = self.compute_video(video_data)

            if return_embeds:
                return text_embeddings, video_embeddings

            return sim_matrix(text_embeddings, video_embeddings)

        def compute_text(self, text_data):
            if self.text_params['model'].startswith('bert'):
                text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                    'pooler_output']
            elif self.text_params['model'].startswith('distilbert'):
                text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
            else:
                raise NotImplementedError
            text_embeddings = self.txt_proj(text_embeddings)
            return text_embeddings

        def compute_video(self, video_data):
            video_embeddings = self.video_model(video_data)
            video_embeddings = self.vid_proj(video_embeddings)
            return video_embeddings

        def _inflate_positional_embeds(self, new_state_dict):
            # allow loading of timesformer with fewer num_frames
            curr_keys = list(self.state_dict().keys())
            if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
                load_temporal_embed = new_state_dict['video_model.temporal_embed']
                load_num_frames = load_temporal_embed.shape[1]
                curr_num_frames = self.video_params['num_frames']
                embed_dim = load_temporal_embed.shape[2]

                if load_num_frames != curr_num_frames:
                    if load_num_frames > curr_num_frames:
                        print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                              f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                        new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                    else:
                        print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                              f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                        if self.load_temporal_fix == 'zeros':
                            new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                            new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                        elif self.load_temporal_fix in ['interp', 'bilinear']:
                            # interpolate
                            # unsqueeze so pytorch thinks its an image
                            mode = 'nearest'
                            if self.load_temporal_fix == 'bilinear':
                                mode = 'bilinear'
                            load_temporal_embed = load_temporal_embed.unsqueeze(0)
                            new_temporal_embed = F.interpolate(load_temporal_embed,
                                                               (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                        else:
                            raise NotImplementedError
                    new_state_dict['video_model.temporal_embed'] = new_temporal_embed
            # allow loading with smaller spatial patches. assumes custom border crop, to append the
            # border patches to the input sequence
            if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
                load_pos_embed = new_state_dict['video_model.pos_embed']
                load_num_patches = load_pos_embed.shape[1]
                curr_pos_embed = self.state_dict()['video_model.pos_embed']
                if load_num_patches != curr_pos_embed.shape[1]:
                    raise NotImplementedError(
                        'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

            return new_state_dict

elif version == 'offline':
    class FrozenInTime(BaseModel):
        def __init__(self,
                     *args,
                     **kwargs):
            super().__init__()


        def set_device(self, device):
            self.device = device

        def forward(self, data, return_embeds=True):

            valid_l = data['ocr_info'][0]['valid_l']

            ocr = data['ocr_info'][0]['gt_ocr'][:valid_l]
            # ['word1','word2','word3']

            ocr_score = data['ocr_info'][0]['gt_ocr_score'][:valid_l]

            ocr = list(zip(ocr,ocr_score))
            ocr = sorted(ocr,key=lambda x:x[1],reverse=True)
            ocr = [el[0] for el in ocr][:20]

            captions_all = data['captions_all']
            # ['word4 word5 word6']

            sim_matrix = torch.zeros((1,len(captions_all)))
            min_l = 3
            ocr = [el for el in ocr if len(el)>min_l]
            # ocr_score = [el for el in ocr_score if len(el)>min_l]
            ocr = set(ocr)

            for i,caption in enumerate(captions_all):
                caps = [el for el in caption[0].strip().split() if len(el)>min_l]
                caps = set(caps)
                match_words = ocr.intersection(caps)
                if len(match_words)>0:
                    sim_matrix[0,i] = len(match_words)

            return sim_matrix


    # class FrozenInTime(BaseModel):
    #     def __init__(self,
    #                  *args,
    #                  **kwargs):
    #         super().__init__()
    #
    #
    #     def set_device(self, device):
    #         self.device = device
    #
    #     def phoc(self, words):
    #         phoc_words = []
    #         for word in words:
    #             try:
    #                 phoc_word = phoc(word)
    #             except:
    #                 phoc_word = np.zeros((1,604))
    #             phoc_words.append(phoc_word)
    #         return torch.from_numpy(np.concatenate(phoc_words,0))
    #
    #     def forward(self, data, return_embeds=True):
    #
    #         topn = 10
    #
    #         valid_l = data['ocr_info'][0]['valid_l']
    #
    #         ocr = data['ocr_info'][0]['gt_ocr'][:valid_l]
    #
    #         ocr_score = data['ocr_info'][0]['gt_ocr_score'][:valid_l]
    #
    #         captions_all = data['captions_all']
    #
    #         sim_matrix = torch.zeros((1,len(captions_all),topn))
    #         min_l = 1
    #         ocr = [el for el in ocr if len(el)>min_l]
    #         # ocr_score = [el for el in ocr_score if len(el)>min_l]
    #         ocr = set(ocr)
    #
    #         phoc_ocr = self.phoc(ocr)
    #
    #         for i,caption in enumerate(captions_all):
    #             caps = [el.strip(',.-') for el in caption[0].strip().split() if len(el)>min_l]
    #             caps = set(caps)
    #             phoc_caps = self.phoc(caps)
    #
    #             phoc_dis = torch.abs(phoc_ocr[:,None]-phoc_caps[None])
    #             phoc_dis = phoc_dis.reshape(-1,604).sum(-1).topk(1,largest=False)
    #
    #             # match_words = ocr.intersection(caps)
    #             # if len(match_words)>0:
    #             sim_matrix[0,i] = phoc_dis.values
    #
    #         return sim_matrix

elif version == 'ocr':
    class SelfAttention(nn.Module):

        def __init__(self, hidden_size, num_attention_heads, dropout_prob):
            super(SelfAttention, self).__init__()
            if hidden_size % num_attention_heads != 0:  # 整除
                raise ValueError(
                    "The hidden size (%d) is not a multiple of the number of attention "
                    "heads (%d)" % (hidden_size, num_attention_heads))
            # 参数定义
            self.num_attention_heads = num_attention_heads  # 8
            self.attention_head_size = int(hidden_size / num_attention_heads)  # 16  每个注意力头的维度
            self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
            # all_head_size = 128 即等于hidden_size, 一般自注意力输入输出前后维度不变

            # query, key, value 的线性变换（上述公式2）
            self.query = nn.Linear(hidden_size, self.all_head_size)  # 128, 128
            self.key = nn.Linear(hidden_size, self.all_head_size)
            self.value = nn.Linear(hidden_size, self.all_head_size)

        def transpose_for_scores(self, x):
            # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # [bs, seqlen, 8, 16]
            x = x.view(*new_x_shape)  #
            return x.permute(0, 2, 1, 3)  # [bs, 8, seqlen, 16]

        def forward(self, hidden_states):

            # 线性变换
            mixed_query_layer = self.query(hidden_states)  # [bs, seqlen, hid_size]
            mixed_key_layer = self.key(hidden_states)  # [bs, seqlen, hid_size]
            mixed_value_layer = self.value(hidden_states)  # [bs, seqlen, hid_size]

            query_layer = self.transpose_for_scores(mixed_query_layer)  # [bs, 8, seqlen, 16]
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)  # [bs, 8, seqlen, 16]

            # Take the dot product between "query" and "key" to get the raw attention scores.
            # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [bs, 8, seqlen, seqlen]

            # 将注意力转化为概率分布，即注意力权重
            attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]

            # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
            context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [bs, seqlen, 128]
            context_layer = context_layer.view(*new_context_layer_shape)
            return context_layer


    class FrozenInTime(BaseModel):
        def __init__(self,
                     video_params,
                     text_params,
                     projection_dim=256,
                     load_checkpoint=None,
                     projection='minimal',
                     load_temporal_fix='zeros'):
            super().__init__()
            # projection_dim = 32

            self.video_params = video_params
            self.text_params = text_params
            self.load_temporal_fix = load_temporal_fix
            if not text_params['pretrained']:
                raise NotImplementedError("Huggingface text models require pretrained init.")

            self.text_model = AutoModel.from_pretrained(text_params['model'])
            self.text_model.train()

            self.ocr_model = AutoModel.from_pretrained(text_params['model'])
            self.ocr_model.train()

            self.fusion_model = SelfAttention(768,8,0)
            self.fusion_model.train()

            self.query_token = nn.Parameter(torch.zeros(1,768))

            # Project to a common embedding
            if projection == 'minimal':
                txt_proj = nn.Sequential(nn.ReLU(),
                                         nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                         )

                fusion_proj = nn.Sequential(nn.ReLU(),
                                         nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                         )


            elif projection == '':
                txt_proj = nn.Identity()
                fusion_proj = nn.Identity()
            else:
                raise NotImplementedError
            self.txt_proj = txt_proj
            self.fusion_proj = fusion_proj


            if load_checkpoint not in ["", None]:
                checkpoint = torch.load(load_checkpoint)
                state_dict = checkpoint['state_dict']
                new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
                new_state_dict = self._inflate_positional_embeds(new_state_dict)
                self.load_state_dict(new_state_dict, strict=True)



        def set_device(self, device):
            self.device = device

        def forward(self, data, return_embeds=True):

            text_data = data['text']
            ocr_data = data['ocr']

            text_embeddings = self.compute_text(text_data)

            ocr_embeddings = self.compute_ocr(ocr_data)

            return text_embeddings, ocr_embeddings

        def compute_text(self, text_data):
            if self.text_params['model'].startswith('bert'):
                text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                    'pooler_output']
            elif self.text_params['model'].startswith('distilbert'):
                text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
            else:
                raise NotImplementedError
            text_embeddings = self.txt_proj(text_embeddings)
            return text_embeddings

        def compute_ocr(self, ocr_data):
            fusion_embeddings = []
            all_valid_l = ocr_data['all_valid_l']
            all_input_ids = ocr_data['all_input_ids']
            all_attention_mask = ocr_data['all_attention_mask']
            for valid_l, input_ids, attenion_mask in zip(all_valid_l, all_input_ids, all_attention_mask):
                valid_l = max(1,valid_l)
                query_feat = self.query_token
                valid_input_ids = input_ids[:valid_l]
                valid_attention_mask = attenion_mask[:valid_l]
                ocr_feats = dict(input_ids=valid_input_ids, attention_mask=valid_attention_mask)
                ocr_feats = self.ocr_model(**ocr_feats).last_hidden_state[:, 0, :]
                fusion_feats = torch.cat([query_feat, ocr_feats], 0)
                fusion_embedding = query_feat + self.fusion_model(fusion_feats[None])[:,0,:]
                fusion_embeddings.append(fusion_embedding)
            fusion_embeddings = torch.cat(fusion_embeddings, 0)
            fusion_embeddings = self.fusion_proj(fusion_embeddings)
            return fusion_embeddings

        def _inflate_positional_embeds(self, new_state_dict):
            # allow loading of timesformer with fewer num_frames
            curr_keys = list(self.state_dict().keys())
            if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
                load_temporal_embed = new_state_dict['video_model.temporal_embed']
                load_num_frames = load_temporal_embed.shape[1]
                curr_num_frames = self.video_params['num_frames']
                embed_dim = load_temporal_embed.shape[2]

                if load_num_frames != curr_num_frames:
                    if load_num_frames > curr_num_frames:
                        print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                              f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                        new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                    else:
                        print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                              f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                        if self.load_temporal_fix == 'zeros':
                            new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                            new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                        elif self.load_temporal_fix in ['interp', 'bilinear']:
                            # interpolate
                            # unsqueeze so pytorch thinks its an image
                            mode = 'nearest'
                            if self.load_temporal_fix == 'bilinear':
                                mode = 'bilinear'
                            load_temporal_embed = load_temporal_embed.unsqueeze(0)
                            new_temporal_embed = F.interpolate(load_temporal_embed,
                                                               (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                        else:
                            raise NotImplementedError
                    new_state_dict['video_model.temporal_embed'] = new_temporal_embed
            # allow loading with smaller spatial patches. assumes custom border crop, to append the
            # border patches to the input sequence
            if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
                load_pos_embed = new_state_dict['video_model.pos_embed']
                load_num_patches = load_pos_embed.shape[1]
                curr_pos_embed = self.state_dict()['video_model.pos_embed']
                if load_num_patches != curr_pos_embed.shape[1]:
                    raise NotImplementedError(
                        'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

            return new_state_dict

elif version == 'video':
    class FrozenInTime(BaseModel):
        def __init__(self,
                     video_params,
                     text_params,
                     projection_dim=256,
                     load_checkpoint=None,
                     projection='minimal',
                     load_temporal_fix='zeros'):
            super().__init__()

            self.video_params = video_params
            self.text_params = text_params
            self.load_temporal_fix = load_temporal_fix
            if not text_params['pretrained']:
                raise NotImplementedError("Huggingface text models require pretrained init.")

            self.text_model = AutoModel.from_pretrained(text_params['model'])
            self.text_model.train()

            pretrained = video_params['pretrained']
            if video_params['model'] == "SpaceTimeTransformer":
                num_frames = video_params.get('num_frames', 4)
                time_init = video_params.get('time_init', 'zeros')
                attention_style = video_params.get('attention_style', 'frozen-in-time')
                arch_config = video_params.get('arch_config', 'base_patch16_224')
                vit_init = video_params.get('vit_init', 'imagenet-21k')
                if arch_config == 'base_patch16_224':
                    vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
                    model = SpaceTimeTransformer(num_frames=num_frames,
                                                time_init=time_init,
                                                attention_style=attention_style)
                else:
                    raise NotImplementedError

                model.head = nn.Identity()
                model.pre_logits = nn.Identity()
                ftr_dim = model.embed_dim
                if load_checkpoint in ["", None]:
                    vit_checkpoint = vit_model.state_dict()
                    model.load_state_dict(vit_checkpoint, strict=False)
                self.video_model = model
            else:
                raise NotImplementedError(f"{video_params['model']} not implemented")

            # for backwards compatibility (old models)
            self.video_model.fc = nn.Identity()

            # Project to a common embedding
            if projection == 'minimal':
                txt_proj = nn.Sequential(nn.ReLU(),
                                         nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                         )

                vid_proj = nn.Sequential(
                    nn.Linear(ftr_dim, projection_dim)
                )
            elif projection == '':
                txt_proj = nn.Identity()
                vid_proj = nn.Identity()
            else:
                raise NotImplementedError
            self.txt_proj = txt_proj
            self.vid_proj = vid_proj

            if load_checkpoint not in ["", None]:
                checkpoint = torch.load(load_checkpoint)
                state_dict = checkpoint['state_dict']
                new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
                new_state_dict = self._inflate_positional_embeds(new_state_dict)
                self.load_state_dict(new_state_dict, strict=True)

        def set_device(self, device):
            self.device = device

        def forward(self, data, return_embeds=True):

            text_data = data['text']
            video_data = data['video']

            text_embeddings = self.compute_text(text_data)
            video_embeddings = self.compute_video(video_data)

            if return_embeds:
                return text_embeddings, video_embeddings

            return sim_matrix(text_embeddings, video_embeddings)

        def compute_text(self, text_data):
            if self.text_params['model'].startswith('bert'):
                text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                    'pooler_output']
            elif self.text_params['model'].startswith('distilbert'):
                text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
            else:
                raise NotImplementedError
            text_embeddings = self.txt_proj(text_embeddings)
            return text_embeddings

        def compute_video(self, video_data):
            video_embeddings = self.video_model(video_data)
            video_embeddings = self.vid_proj(video_embeddings)
            return video_embeddings

        def _inflate_positional_embeds(self, new_state_dict):
            # allow loading of timesformer with fewer num_frames
            curr_keys = list(self.state_dict().keys())
            if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
                load_temporal_embed = new_state_dict['video_model.temporal_embed']
                load_num_frames = load_temporal_embed.shape[1]
                curr_num_frames = self.video_params['num_frames']
                embed_dim = load_temporal_embed.shape[2]

                if load_num_frames != curr_num_frames:
                    if load_num_frames > curr_num_frames:
                        print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                              f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                        new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                    else:
                        print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                              f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                        if self.load_temporal_fix == 'zeros':
                            new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                            new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                        elif self.load_temporal_fix in ['interp', 'bilinear']:
                            # interpolate
                            # unsqueeze so pytorch thinks its an image
                            mode = 'nearest'
                            if self.load_temporal_fix == 'bilinear':
                                mode = 'bilinear'
                            load_temporal_embed = load_temporal_embed.unsqueeze(0)
                            new_temporal_embed = F.interpolate(load_temporal_embed,
                                                               (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                        else:
                            raise NotImplementedError
                    new_state_dict['video_model.temporal_embed'] = new_temporal_embed
            # allow loading with smaller spatial patches. assumes custom border crop, to append the
            # border patches to the input sequence
            if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
                load_pos_embed = new_state_dict['video_model.pos_embed']
                load_num_patches = load_pos_embed.shape[1]
                curr_pos_embed = self.state_dict()['video_model.pos_embed']
                if load_num_patches != curr_pos_embed.shape[1]:
                    raise NotImplementedError(
                        'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

            return new_state_dict

elif version == 'fusion':
    class SelfAttention(nn.Module):

        def __init__(self, hidden_size, num_attention_heads, dropout_prob):
            super(SelfAttention, self).__init__()
            if hidden_size % num_attention_heads != 0:  # 整除
                raise ValueError(
                    "The hidden size (%d) is not a multiple of the number of attention "
                    "heads (%d)" % (hidden_size, num_attention_heads))
            # 参数定义
            self.num_attention_heads = num_attention_heads  # 8
            self.attention_head_size = int(hidden_size / num_attention_heads)  # 16  每个注意力头的维度
            self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
            # all_head_size = 128 即等于hidden_size, 一般自注意力输入输出前后维度不变

            # query, key, value 的线性变换（上述公式2）
            self.query = nn.Linear(hidden_size, self.all_head_size)  # 128, 128
            self.key = nn.Linear(hidden_size, self.all_head_size)
            self.value = nn.Linear(hidden_size, self.all_head_size)

        def transpose_for_scores(self, x):
            # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # [bs, seqlen, 8, 16]
            x = x.view(*new_x_shape)  #
            return x.permute(0, 2, 1, 3)  # [bs, 8, seqlen, 16]

        def forward(self, hidden_states):

            # 线性变换
            mixed_query_layer = self.query(hidden_states)  # [bs, seqlen, hid_size]
            mixed_key_layer = self.key(hidden_states)  # [bs, seqlen, hid_size]
            mixed_value_layer = self.value(hidden_states)  # [bs, seqlen, hid_size]

            query_layer = self.transpose_for_scores(mixed_query_layer)  # [bs, 8, seqlen, 16]
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)  # [bs, 8, seqlen, 16]

            # Take the dot product between "query" and "key" to get the raw attention scores.
            # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [bs, 8, seqlen, seqlen]

            # 将注意力转化为概率分布，即注意力权重
            attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]

            # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
            context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [bs, seqlen, 128]
            context_layer = context_layer.view(*new_context_layer_shape)
            return context_layer


   
    class FrozenInTime(BaseModel):
        def __init__(self,
                     video_params,
                     text_params,
                     projection_dim=256,
                     load_checkpoint=None,
                     projection='minimal',
                     load_temporal_fix='zeros'):
            super().__init__()

            self.video_params = video_params
            self.text_params = text_params
            self.load_temporal_fix = load_temporal_fix
            if not text_params['pretrained']:
                raise NotImplementedError("Huggingface text models require pretrained init.")

            self.text_model = AutoModel.from_pretrained(text_params['model'])
            self.text_model.train()

            self.ocr_model = AutoModel.from_pretrained(text_params['model'])
            self.ocr_model.train()

            self.fusion_model = SelfAttention(768,8,0)
            self.fusion_model.train()

            self.query_token = nn.Parameter(torch.zeros(1,768))

            self.type_embedding = nn.Embedding(3,768)

            self.ocr_pos_embeds = nn.Linear(12,768)

            pretrained = video_params['pretrained']
            if video_params['model'] == "SpaceTimeTransformer":
                num_frames = video_params.get('num_frames', 4)
                time_init = video_params.get('time_init', 'zeros')
                attention_style = video_params.get('attention_style', 'frozen-in-time')
                arch_config = video_params.get('arch_config', 'base_patch16_224')
                vit_init = video_params.get('vit_init', 'imagenet-21k')
                if arch_config == 'base_patch16_224':
                    vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
                    model = SpaceTimeTransformer(num_frames=num_frames,
                                                 time_init=time_init,
                                                 attention_style=attention_style)
                else:
                    raise NotImplementedError

                model.head = nn.Identity()
                model.pre_logits = nn.Identity()
                ftr_dim = model.embed_dim
                if load_checkpoint in ["", None]:
                    vit_checkpoint = vit_model.state_dict()
                    model.load_state_dict(vit_checkpoint, strict=False)
                self.video_model = model
            else:
                raise NotImplementedError(f"{video_params['model']} not implemented")

            # for backwards compatibility (old models)
            self.video_model.fc = nn.Identity()

            # Project to a common embedding
            if projection == 'minimal':
                txt_proj = nn.Sequential(nn.ReLU(),
                                         nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                         )

                vid_proj = nn.Sequential(
                    nn.Linear(ftr_dim, projection_dim)
                )

                fusion_proj = nn.Sequential(nn.ReLU(),
                                            nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                            )
            elif projection == '':
                txt_proj = nn.Identity()
                vid_proj = nn.Identity()
                fusion_proj = nn.Identity()
            else:
                raise NotImplementedError
            self.txt_proj = txt_proj
            self.vid_proj = vid_proj
            self.fusion_proj = fusion_proj

            if load_checkpoint not in ["", None]:
                checkpoint = torch.load(load_checkpoint)
                state_dict = checkpoint['state_dict']
                new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
                new_state_dict = self._inflate_positional_embeds(new_state_dict)
                self.load_state_dict(new_state_dict, strict=True)

        def set_device(self, device):
            self.device = device

        def forward(self, data, return_embeds=True):

            text_data = data['text']
            video_data = data['video']
            ocr_data = data['ocr']

            text_embeddings = self.compute_text(text_data)

            video_embeddings, video_feats = self.compute_video(video_data)

            ocr_feats = self.compute_ocr(ocr_data)

            fusion_embeddings = self.compute_fusion(video_feats, ocr_feats)

            if return_embeds:
                return text_embeddings, video_embeddings, fusion_embeddings

            return sim_matrix(text_embeddings, video_embeddings)

        def compute_text(self, text_data):
            if self.text_params['model'].startswith('bert'):
                text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                    'pooler_output']
            elif self.text_params['model'].startswith('distilbert'):
                text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
            else:
                raise NotImplementedError
            text_embeddings = self.txt_proj(text_embeddings)
            return text_embeddings

        def compute_video(self, video_data):
            video_features = self.video_model(video_data)
            video_embeddings = self.vid_proj(video_features)
            return video_embeddings, video_features

        def compute_ocr(self, ocr_data):
            ocr_feats = []
            all_valid_l = ocr_data['all_valid_l']
            all_input_ids = ocr_data['all_input_ids']
            all_attention_mask = ocr_data['all_attention_mask']
            all_pos = ocr_data['all_gt_pos']
            for valid_l, input_ids, attenion_mask, pos in zip(all_valid_l, all_input_ids, all_attention_mask, all_pos):
                valid_l = max(1,valid_l) # ' '
                valid_input_ids = input_ids[:valid_l]
                valid_attention_mask = attenion_mask[:valid_l]
                valid_pos = pos[:valid_l].to(input_ids.device)

                valid_pos_embeds = self.ocr_pos_embeds(valid_pos)

                ocr_feat = dict(input_ids=valid_input_ids, attention_mask=valid_attention_mask)

                # ocr_feat = ocr_feat + ocr_pos_embed (tracks, [st, ed ,ed-st, mx, my, mw, mh]) mlp
                ocr_feat = self.ocr_model(**ocr_feat).last_hidden_state[:, 0, :]

                # with pos embed or not
                ocr_feat = ocr_feat + valid_pos_embeds
                ocr_feats.append(ocr_feat)
            return ocr_feats

        def compute_fusion(self, video_feats, ocr_feats):
            fusion_feats = []
            for video_feat, ocr_feat in zip(video_feats, ocr_feats):
                video_feat = video_feat[None]

                # make modality embeddings
                lv = len(video_feat)
                lo = len(ocr_feat)
                type_embeddings = torch.zeros((1+lv+lo)).to(video_feat).to(torch.long)
                type_embeddings[:1] = 0
                type_embeddings[1:1+lv] = 1
                type_embeddings[1+lv:] = 2
                type_embeddings = self.type_embedding(type_embeddings)

                # make input features
                fusion_feat = torch.cat([self.query_token,video_feat,ocr_feat],0)
                fusion_feat = fusion_feat + type_embeddings

                # fusion features
                fusion_feat = self.query_token + self.fusion_model(fusion_feat[None])[:,0,:]
                fusion_feats.append(fusion_feat)

            fusion_feats = torch.cat(fusion_feats, 0)
            fusion_embeddings = self.fusion_proj(fusion_feats)
            return fusion_embeddings


        def _inflate_positional_embeds(self, new_state_dict):
            # allow loading of timesformer with fewer num_frames
            curr_keys = list(self.state_dict().keys())
            if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
                load_temporal_embed = new_state_dict['video_model.temporal_embed']
                load_num_frames = load_temporal_embed.shape[1]
                curr_num_frames = self.video_params['num_frames']
                embed_dim = load_temporal_embed.shape[2]

                if load_num_frames != curr_num_frames:
                    if load_num_frames > curr_num_frames:
                        print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                              f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                        new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                    else:
                        print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                              f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                        if self.load_temporal_fix == 'zeros':
                            new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                            new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                        elif self.load_temporal_fix in ['interp', 'bilinear']:
                            # interpolate
                            # unsqueeze so pytorch thinks its an image
                            mode = 'nearest'
                            if self.load_temporal_fix == 'bilinear':
                                mode = 'bilinear'
                            load_temporal_embed = load_temporal_embed.unsqueeze(0)
                            new_temporal_embed = F.interpolate(load_temporal_embed,
                                                               (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                        else:
                            raise NotImplementedError
                    new_state_dict['video_model.temporal_embed'] = new_temporal_embed
            # allow loading with smaller spatial patches. assumes custom border crop, to append the
            # border patches to the input sequence
            if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
                load_pos_embed = new_state_dict['video_model.pos_embed']
                load_num_patches = load_pos_embed.shape[1]
                curr_pos_embed = self.state_dict()['video_model.pos_embed']
                if load_num_patches != curr_pos_embed.shape[1]:
                    raise NotImplementedError(
                        'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

            return new_state_dict

elif version == 'mmspacetimetransformer':

    class FrozenInTime(BaseModel):
        def __init__(self,
                     video_params,
                     text_params,
                     projection_dim=256,
                     load_checkpoint=None,
                     projection='minimal',
                     load_temporal_fix='zeros'):
            super().__init__()

            self.video_params = video_params
            self.text_params = text_params
            self.load_temporal_fix = load_temporal_fix
            if not text_params['pretrained']:
                raise NotImplementedError("Huggingface text models require pretrained init.")

            self.text_model = AutoModel.from_pretrained(text_params['model'])
            self.text_model.train()

            self.ocr_model = AutoModel.from_pretrained(text_params['model'])
            self.ocr_model.train()

            pretrained = video_params['pretrained']
            if video_params['model'] == "SpaceTimeTransformer":
                num_frames = video_params.get('num_frames', 4)
                time_init = video_params.get('time_init', 'zeros')
                attention_style = video_params.get('attention_style', 'frozen-in-time')
                arch_config = video_params.get('arch_config', 'base_patch16_224')
                vit_init = video_params.get('vit_init', 'imagenet-21k')
                if arch_config == 'base_patch16_224':
                    vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
                    model = SpaceTimeTransformer(num_frames=num_frames,
                                                 time_init=time_init,
                                                 attention_style=attention_style)
                else:
                    raise NotImplementedError

                model.head = nn.Identity()
                model.pre_logits = nn.Identity()
                ftr_dim = model.embed_dim
                if load_checkpoint in ["", None]:
                    vit_checkpoint = vit_model.state_dict()
                    model.load_state_dict(vit_checkpoint, strict=False)
                self.video_model = model
            else:
                raise NotImplementedError(f"{video_params['model']} not implemented")

            # for backwards compatibility (old models)
            self.video_model.fc = nn.Identity()

            # Project to a common embedding
            if projection == 'minimal':
                txt_proj = nn.Sequential(nn.ReLU(),
                                         nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                         )

                vid_proj = nn.Sequential(
                    nn.Linear(ftr_dim, projection_dim)
                )

            elif projection == '':
                txt_proj = nn.Identity()
                vid_proj = nn.Identity()
                fusion_proj = nn.Identity()
            else:
                raise NotImplementedError
            self.txt_proj = txt_proj
            self.vid_proj = vid_proj

            if load_checkpoint not in ["", None]:
                checkpoint = torch.load(load_checkpoint)
                state_dict = checkpoint['state_dict']
                new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
                new_state_dict = self._inflate_positional_embeds(new_state_dict)
                self.load_state_dict(new_state_dict, strict=True)

        def set_device(self, device):
            self.device = device

        def forward(self, data, return_embeds=True):

            text_data = data['text']
            video_data = data['video']
            ocr_data = data['ocr']

            text_embeddings = self.compute_text(text_data)

            fusion_embeddings = self.compute_fusion(video_data, ocr_data)

            if return_embeds:
                return text_embeddings, fusion_embeddings, fusion_embeddings

            return sim_matrix(text_embeddings, fusion_embeddings)

        def compute_text(self, text_data):
            if self.text_params['model'].startswith('bert'):
                text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                    'pooler_output']
            elif self.text_params['model'].startswith('distilbert'):
                text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
            else:
                raise NotImplementedError
            text_embeddings = self.txt_proj(text_embeddings)
            return text_embeddings

        def compute_fusion(self, video_data, ocr_data):
            ocr_features = self.compute_ocr(ocr_data)
            video_features = self.video_model(video_data, ocr_features)
            video_embeddings = self.vid_proj(video_features)
            return video_embeddings

        def compute_ocr(self, ocr_data):
            ocr_feats = []
            all_valid_l = ocr_data['all_valid_l']
            all_input_ids = ocr_data['all_input_ids']
            all_attention_mask = ocr_data['all_attention_mask']
            for valid_l, input_ids, attention_mask in zip(all_valid_l, all_input_ids, all_attention_mask):
                # valid_l = max(1,valid_l)
                # valid_input_ids = input_ids[:valid_l]
                # valid_attention_mask = attenion_mask[:valid_l]
                ocr_feat = dict(input_ids=valid_input_ids, attention_mask=attention_mask)
                ocr_feat = self.ocr_model(**ocr_feat).last_hidden_state[:, 0, :]
                ocr_feats.append(ocr_feat)
            return torch.stack(ocr_feats,0)

        def _inflate_positional_embeds(self, new_state_dict):
            # allow loading of timesformer with fewer num_frames
            curr_keys = list(self.state_dict().keys())
            if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
                load_temporal_embed = new_state_dict['video_model.temporal_embed']
                load_num_frames = load_temporal_embed.shape[1]
                curr_num_frames = self.video_params['num_frames']
                embed_dim = load_temporal_embed.shape[2]

                if load_num_frames != curr_num_frames:
                    if load_num_frames > curr_num_frames:
                        print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                              f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                        new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                    else:
                        print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                              f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                        if self.load_temporal_fix == 'zeros':
                            new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                            new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                        elif self.load_temporal_fix in ['interp', 'bilinear']:
                            # interpolate
                            # unsqueeze so pytorch thinks its an image
                            mode = 'nearest'
                            if self.load_temporal_fix == 'bilinear':
                                mode = 'bilinear'
                            load_temporal_embed = load_temporal_embed.unsqueeze(0)
                            new_temporal_embed = F.interpolate(load_temporal_embed,
                                                               (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                        else:
                            raise NotImplementedError
                    new_state_dict['video_model.temporal_embed'] = new_temporal_embed
            # allow loading with smaller spatial patches. assumes custom border crop, to append the
            # border patches to the input sequence
            if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
                load_pos_embed = new_state_dict['video_model.pos_embed']
                load_num_patches = load_pos_embed.shape[1]
                curr_pos_embed = self.state_dict()['video_model.pos_embed']
                if load_num_patches != curr_pos_embed.shape[1]:
                    raise NotImplementedError(
                        'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

            return new_state_dict

# class FrozenInTime_tvalign(BaseModel):
#     def __init__(self,
#                  video_params,
#                  text_params,
#                  projection_dim=256,
#                  load_checkpoint=None,
#                  projection='minimal',
#                  load_temporal_fix='zeros'):
#         super().__init__()
#
#         self.video_params = video_params
#         self.text_params = text_params
#         self.load_temporal_fix = load_temporal_fix
#         if not text_params['pretrained']:
#             raise NotImplementedError("Huggingface text models require pretrained init.")
#
#         self.text_model = AutoModel.from_pretrained(text_params['model'])
#         self.text_model.train()
#
#         pretrained = video_params['pretrained']
#         if video_params['model'] == "SpaceTimeTransformer":
#             num_frames = video_params.get('num_frames', 4)
#             time_init = video_params.get('time_init', 'zeros')
#             attention_style = video_params.get('attention_style', 'frozen-in-time')
#             arch_config = video_params.get('arch_config', 'base_patch16_224')
#             vit_init = video_params.get('vit_init', 'imagenet-21k')
#             if arch_config == 'base_patch16_224':
#                 vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
#                 model = SpaceTimeTransformer(num_frames=num_frames,
#                                             time_init=time_init,
#                                             attention_style=attention_style)
#             else:
#                 raise NotImplementedError
#
#             model.head = nn.Identity()
#             model.pre_logits = nn.Identity()
#             ftr_dim = model.embed_dim
#             if load_checkpoint in ["", None]:
#                 vit_checkpoint = vit_model.state_dict()
#                 model.load_state_dict(vit_checkpoint, strict=False)
#             self.video_model = model
#         else:
#             raise NotImplementedError(f"{video_params['model']} not implemented")
#
#         # for backwards compatibility (old models)
#         self.video_model.fc = nn.Identity()
#
#
#         self.tv_align = nn.Sequential(nn.ReLU(),
#                                      nn.Linear(projection_dim, projection_dim),
#                                      )
#
#         # Project to a common embedding
#         if projection == 'minimal':
#             txt_proj = nn.Sequential(nn.ReLU(),
#                                      nn.Linear(self.text_model.config.hidden_size, projection_dim),
#                                      )
#
#             vid_proj = nn.Sequential(
#                 nn.Linear(ftr_dim, projection_dim)
#             )
#         elif projection == '':
#             txt_proj = nn.Identity()
#             vid_proj = nn.Identity()
#         else:
#             raise NotImplementedError
#         self.txt_proj = txt_proj
#         self.vid_proj = vid_proj
#
#         if load_checkpoint not in ["", None]:
#             checkpoint = torch.load(load_checkpoint)
#             state_dict = checkpoint['state_dict']
#             new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
#             new_state_dict = self._inflate_positional_embeds(new_state_dict)
#             self.load_state_dict(new_state_dict, strict=True)
#
#     def set_device(self, device):
#         self.device = device
#
#     def forward(self, data, return_embeds=True):
#
#         text_data = data['text']
#         video_data = data['video']
#
#         text_embeddings = self.compute_text(text_data)
#         video_embeddings = self.compute_video(video_data)
#
#         if return_embeds:
#             return text_embeddings, video_embeddings
#
#         return sim_matrix(text_embeddings, video_embeddings)
#
#     def compute_text(self, text_data):
#         if self.text_params['model'].startswith('bert'):
#             text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
#                 'pooler_output']
#         elif self.text_params['model'].startswith('distilbert'):
#             text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
#         else:
#             raise NotImplementedError
#         text_embeddings = self.txt_proj(text_embeddings)
#         text_embeddings = self.tv_align(text_embeddings)
#         return text_embeddings
#
#     def compute_video(self, video_data):
#         video_embeddings = self.video_model(video_data)
#         video_embeddings = self.vid_proj(video_embeddings)
#         video_embeddings = self.tv_align(video_embeddings)
#         return video_embeddings
#
#     def _inflate_positional_embeds(self, new_state_dict):
#         # allow loading of timesformer with fewer num_frames
#         curr_keys = list(self.state_dict().keys())
#         if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
#             load_temporal_embed = new_state_dict['video_model.temporal_embed']
#             load_num_frames = load_temporal_embed.shape[1]
#             curr_num_frames = self.video_params['num_frames']
#             embed_dim = load_temporal_embed.shape[2]
#
#             if load_num_frames != curr_num_frames:
#                 if load_num_frames > curr_num_frames:
#                     print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
#                           f'### loading weights, filling in the extras via {self.load_temporal_fix}')
#                     new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
#                 else:
#                     print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
#                           f'### loading weights, filling in the extras via {self.load_temporal_fix}')
#                     if self.load_temporal_fix == 'zeros':
#                         new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
#                         new_temporal_embed[:, :load_num_frames] = load_temporal_embed
#                     elif self.load_temporal_fix in ['interp', 'bilinear']:
#                         # interpolate
#                         # unsqueeze so pytorch thinks its an image
#                         mode = 'nearest'
#                         if self.load_temporal_fix == 'bilinear':
#                             mode = 'bilinear'
#                         load_temporal_embed = load_temporal_embed.unsqueeze(0)
#                         new_temporal_embed = F.interpolate(load_temporal_embed,
#                                                            (curr_num_frames, embed_dim), mode=mode).squeeze(0)
#                     else:
#                         raise NotImplementedError
#                 new_state_dict['video_model.temporal_embed'] = new_temporal_embed
#         # allow loading with smaller spatial patches. assumes custom border crop, to append the
#         # border patches to the input sequence
#         if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
#             load_pos_embed = new_state_dict['video_model.pos_embed']
#             load_num_patches = load_pos_embed.shape[1]
#             curr_pos_embed = self.state_dict()['video_model.pos_embed']
#             if load_num_patches != curr_pos_embed.shape[1]:
#                 raise NotImplementedError(
#                     'Loading models with different spatial resolution / patch number not yet implemented, sorry.')
#
#         return new_state_dict


#
# def sim_matrix(a, b, eps=1e-8):
#     """
#     added eps for numerical stability
#     """
#     a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
#     a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
#     b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
#     sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
#     return sim_mt


# class FrozenInTime_v1(BaseModel):
#     def __init__(self,
#                  video_params,
#                  text_params,
#                  projection_dim=256,
#                  load_checkpoint=None,
#                  projection='minimal',
#                  load_temporal_fix='zeros'):
#         super().__init__()
#
#         self.video_params = video_params
#         self.text_params = text_params
#         self.load_temporal_fix = load_temporal_fix
#         if not text_params['pretrained']:
#             raise NotImplementedError("Huggingface text models require pretrained init.")
#
#         self.text_model = AutoModel.from_pretrained(text_params['model'])
#         self.text_model.train()
#
#         self.ocr_model = AutoModel.from_pretrained(text_params['model'])
#         self.ocr_query = nn.Parameter(torch.zeros(1,768))
#         self.ocr_model.train()
#         self.ocr_proj = nn.Sequential(nn.ReLU(),nn.Linear(self.text_model.config.hidden_size, projection_dim))
#         self.ocr_video_fusion_layer = nn.Linear(projection_dim,projection_dim)
#
#         pretrained = video_params['pretrained']
#         if video_params['model'] == "SpaceTimeTransformer":
#             num_frames = video_params.get('num_frames', 4)
#             time_init = video_params.get('time_init', 'zeros')
#             attention_style = video_params.get('attention_style', 'frozen-in-time')
#             arch_config = video_params.get('arch_config', 'base_patch16_224')
#             vit_init = video_params.get('vit_init', 'imagenet-21k')
#             if arch_config == 'base_patch16_224':
#                 vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
#                 model = SpaceTimeTransformer(num_frames=num_frames,
#                                             time_init=time_init,
#                                             attention_style=attention_style)
#             else:
#                 raise NotImplementedError
#
#             model.head = nn.Identity()
#             model.pre_logits = nn.Identity()
#             ftr_dim = model.embed_dim
#             if load_checkpoint in ["", None]:
#                 vit_checkpoint = vit_model.state_dict()
#                 model.load_state_dict(vit_checkpoint, strict=False)
#             self.video_model = model
#         else:
#             raise NotImplementedError(f"{video_params['model']} not implemented")
#
#         # for backwards compatibility (old models)
#         self.video_model.fc = nn.Identity()
#
#         # Project to a common embedding
#         if projection == 'minimal':
#             txt_proj = nn.Sequential(nn.ReLU(),
#                                      nn.Linear(self.text_model.config.hidden_size, projection_dim),
#                                      )
#
#             vid_proj = nn.Sequential(
#                 nn.Linear(ftr_dim, projection_dim)
#             )
#         elif projection == '':
#             txt_proj = nn.Identity()
#             vid_proj = nn.Identity()
#         else:
#             raise NotImplementedError
#         self.txt_proj = txt_proj
#         self.vid_proj = vid_proj
#
#         if load_checkpoint not in ["", None]:
#             checkpoint = torch.load(load_checkpoint)
#             state_dict = checkpoint['state_dict']
#             new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
#             new_state_dict = self._inflate_positional_embeds(new_state_dict)
#             self.load_state_dict(new_state_dict, strict=True)
#
#     def set_device(self, device):
#         self.device = device
#
#     def forward(self, data, return_embeds=True):
#
#         text_data = data['text']
#         video_data = data['video']
#         ocr_data = data['ocr']
#
#         text_embeddings = self.compute_text(text_data)
#         video_embeddings = self.compute_video(video_data)
#         ocr_embeddings = self.compute_ocr(ocr_data)
#         ocr_video_embeddings = self.ocr_video_fusion(video_embeddings, ocr_embeddings)
#
#         if return_embeds:
#             return text_embeddings, video_embeddings, ocr_video_embeddings
#
#         return sim_matrix(text_embeddings, video_embeddings)
#
#     def compute_text(self, text_data):
#         if self.text_params['model'].startswith('bert'):
#             text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
#                 'pooler_output']
#         elif self.text_params['model'].startswith('distilbert'):
#             text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
#         else:
#             raise NotImplementedError
#         text_embeddings = self.txt_proj(text_embeddings)
#         return text_embeddings
#
#     def compute_ocr(self, ocr_data):
#         ocr_embeddings = []
#         all_valid_l = ocr_data['all_valid_l']
#         all_input_ids = ocr_data['all_input_ids']
#         all_attention_mask = ocr_data['all_attention_mask']
#         for valid_l,input_ids,attenion_mask in zip(all_valid_l,all_input_ids,all_attention_mask):
#             if valid_l==0:
#                 ocr_feats = self.ocr_query
#             else:
#                 valid_input_ids = input_ids[:valid_l]
#                 valid_attention_mask = attenion_mask[:valid_l]
#                 ocr_token = dict(input_ids=valid_input_ids,attention_mask=valid_attention_mask)
#                 ocr_feats = self.text_model(**ocr_token).last_hidden_state[:,0,:]
#                 ocr_feats = torch.cat([self.ocr_query,ocr_feats],0)
#             ocr_embedding = self.ocr_model(inputs_embeds=ocr_feats[None]).last_hidden_state[:,0,:]
#             ocr_embeddings.append(ocr_embedding)
#         ocr_embeddings = torch.cat(ocr_embeddings,0)
#         ocr_embeddings = self.ocr_proj(ocr_embeddings)
#         return ocr_embeddings
#
#     def ocr_video_fusion(self, video_embeddings, ocr_embeddings):
#         return self.ocr_video_fusion_layer(video_embeddings+ocr_embeddings)
#
#     def compute_video(self, video_data):
#         video_embeddings = self.video_model(video_data)
#         video_embeddings = self.vid_proj(video_embeddings)
#         return video_embeddings
#
#     def _inflate_positional_embeds(self, new_state_dict):
#         # allow loading of timesformer with fewer num_frames
#         curr_keys = list(self.state_dict().keys())
#         if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
#             load_temporal_embed = new_state_dict['video_model.temporal_embed']
#             load_num_frames = load_temporal_embed.shape[1]
#             curr_num_frames = self.video_params['num_frames']
#             embed_dim = load_temporal_embed.shape[2]
#
#             if load_num_frames != curr_num_frames:
#                 if load_num_frames > curr_num_frames:
#                     print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
#                           f'### loading weights, filling in the extras via {self.load_temporal_fix}')
#                     new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
#                 else:
#                     print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
#                           f'### loading weights, filling in the extras via {self.load_temporal_fix}')
#                     if self.load_temporal_fix == 'zeros':
#                         new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
#                         new_temporal_embed[:, :load_num_frames] = load_temporal_embed
#                     elif self.load_temporal_fix in ['interp', 'bilinear']:
#                         # interpolate
#                         # unsqueeze so pytorch thinks its an image
#                         mode = 'nearest'
#                         if self.load_temporal_fix == 'bilinear':
#                             mode = 'bilinear'
#                         load_temporal_embed = load_temporal_embed.unsqueeze(0)
#                         new_temporal_embed = F.interpolate(load_temporal_embed,
#                                                            (curr_num_frames, embed_dim), mode=mode).squeeze(0)
#                     else:
#                         raise NotImplementedError
#                 new_state_dict['video_model.temporal_embed'] = new_temporal_embed
#         # allow loading with smaller spatial patches. assumes custom border crop, to append the
#         # border patches to the input sequence
#         if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
#             load_pos_embed = new_state_dict['video_model.pos_embed']
#             load_num_patches = load_pos_embed.shape[1]
#             curr_pos_embed = self.state_dict()['video_model.pos_embed']
#             if load_num_patches != curr_pos_embed.shape[1]:
#                 raise NotImplementedError(
#                     'Loading models with different spatial resolution / patch number not yet implemented, sorry.')
#
#         return new_state_dict


# class FrozenInTime_v2(BaseModel):
#     def __init__(self,
#                  video_params,
#                  text_params,
#                  projection_dim=256,
#                  load_checkpoint=None,
#                  projection='minimal',
#                  load_temporal_fix='zeros'):
#         super().__init__()
#
#         self.video_params = video_params
#         self.text_params = text_params
#         self.load_temporal_fix = load_temporal_fix
#         if not text_params['pretrained']:
#             raise NotImplementedError("Huggingface text models require pretrained init.")
#
#         self.text_model = AutoModel.from_pretrained(text_params['model'])
#         self.text_model.train()
#
#         self.ocr_model = AutoModel.from_pretrained(text_params['model'])
#         self.ocr_model.train()
#         self.ocr_proj = nn.Sequential(nn.ReLU(),nn.Linear(self.text_model.config.hidden_size, projection_dim))
#         self.video_ebd = nn.Parameter(torch.zeros(1,768))
#         self.ocr_ebd = nn.Parameter(torch.zeros(1,768))
#
#         pretrained = video_params['pretrained']
#         if video_params['model'] == "SpaceTimeTransformer":
#             num_frames = video_params.get('num_frames', 4)
#             time_init = video_params.get('time_init', 'zeros')
#             attention_style = video_params.get('attention_style', 'frozen-in-time')
#             arch_config = video_params.get('arch_config', 'base_patch16_224')
#             vit_init = video_params.get('vit_init', 'imagenet-21k')
#             if arch_config == 'base_patch16_224':
#                 vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
#                 model = SpaceTimeTransformer(num_frames=num_frames,
#                                             time_init=time_init,
#                                             attention_style=attention_style)
#             else:
#                 raise NotImplementedError
#
#             model.head = nn.Identity()
#             model.pre_logits = nn.Identity()
#             ftr_dim = model.embed_dim
#             if load_checkpoint in ["", None]:
#                 vit_checkpoint = vit_model.state_dict()
#                 model.load_state_dict(vit_checkpoint, strict=False)
#             self.video_model = model
#         else:
#             raise NotImplementedError(f"{video_params['model']} not implemented")
#
#         # for backwards compatibility (old models)
#         self.video_model.fc = nn.Identity()
#
#         # Project to a common embedding
#         if projection == 'minimal':
#             txt_proj = nn.Sequential(nn.ReLU(),
#                                      nn.Linear(self.text_model.config.hidden_size, projection_dim),
#                                      )
#
#             vid_proj = nn.Sequential(
#                 nn.Linear(ftr_dim, projection_dim)
#             )
#         elif projection == '':
#             txt_proj = nn.Identity()
#             vid_proj = nn.Identity()
#         else:
#             raise NotImplementedError
#         self.txt_proj = txt_proj
#         self.vid_proj = vid_proj
#
#         if load_checkpoint not in ["", None]:
#             checkpoint = torch.load(load_checkpoint)
#             state_dict = checkpoint['state_dict']
#             new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
#             new_state_dict = self._inflate_positional_embeds(new_state_dict)
#             self.load_state_dict(new_state_dict, strict=True)
#
#     def set_device(self, device):
#         self.device = device
#
#     def forward(self, data, return_embeds=True):
#
#         text_data = data['text']
#         video_data = data['video']
#         ocr_data = data['ocr']
#
#         text_embeddings = self.compute_text(text_data)
#         video_feats, video_embeddings = self.compute_video(video_data)
#         ocr_embeddings = self.compute_ocr(ocr_data, video_feats.detach())
#
#         if return_embeds:
#             return text_embeddings, video_embeddings, ocr_embeddings
#
#         return sim_matrix(text_embeddings, video_embeddings)
#
#     def compute_text(self, text_data):
#         if self.text_params['model'].startswith('bert'):
#             text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
#                 'pooler_output']
#         elif self.text_params['model'].startswith('distilbert'):
#             text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
#         else:
#             raise NotImplementedError
#         text_embeddings = self.txt_proj(text_embeddings)
#         return text_embeddings
#
#     def compute_ocr(self, ocr_data, video_feats):
#         ocr_embeddings = []
#         all_valid_l = ocr_data['all_valid_l']
#         all_input_ids = ocr_data['all_input_ids']
#         all_attention_mask = ocr_data['all_attention_mask']
#         for valid_l,input_ids,attenion_mask,video_feat in zip(all_valid_l,all_input_ids,all_attention_mask, video_feats):
#             video_feat = video_feat[None]
#             if valid_l==0:
#                 ocr_feats = video_feat
#                 ocr_embedding = ocr_feats
#             else:
#                 valid_input_ids = input_ids[:valid_l]
#                 valid_attention_mask = attenion_mask[:valid_l]
#                 ocr_token = dict(input_ids=valid_input_ids,attention_mask=valid_attention_mask)
#                 ocr_feats = self.text_model(**ocr_token).last_hidden_state[:,0,:]
#                 l_ocr_feats = ocr_feats.shape[0]
#                 modality_embed = torch.cat([self.video_ebd,repeat(self.ocr_ebd,'a b -> (a c) b',c=l_ocr_feats)],0)
#                 ocr_feats = torch.cat([video_feat,ocr_feats],0)
#                 input_feats = modality_embed+ocr_feats
#                 ocr_embedding = video_feat + self.ocr_model(inputs_embeds=input_feats[None]).last_hidden_state[:,0,:]
#             ocr_embeddings.append(ocr_embedding)
#         ocr_embeddings = torch.cat(ocr_embeddings,0)
#         ocr_embeddings = self.vid_proj(ocr_embeddings)
#         return ocr_embeddings
#
#     def compute_video(self, video_data):
#         pre_video_embeddings = self.video_model(video_data)
#         video_embeddings = self.vid_proj(pre_video_embeddings)
#         return pre_video_embeddings,video_embeddings
#
#     def _inflate_positional_embeds(self, new_state_dict):
#         # allow loading of timesformer with fewer num_frames
#         curr_keys = list(self.state_dict().keys())
#         if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
#             load_temporal_embed = new_state_dict['video_model.temporal_embed']
#             load_num_frames = load_temporal_embed.shape[1]
#             curr_num_frames = self.video_params['num_frames']
#             embed_dim = load_temporal_embed.shape[2]
#
#             if load_num_frames != curr_num_frames:
#                 if load_num_frames > curr_num_frames:
#                     print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
#                           f'### loading weights, filling in the extras via {self.load_temporal_fix}')
#                     new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
#                 else:
#                     print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
#                           f'### loading weights, filling in the extras via {self.load_temporal_fix}')
#                     if self.load_temporal_fix == 'zeros':
#                         new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
#                         new_temporal_embed[:, :load_num_frames] = load_temporal_embed
#                     elif self.load_temporal_fix in ['interp', 'bilinear']:
#                         # interpolate
#                         # unsqueeze so pytorch thinks its an image
#                         mode = 'nearest'
#                         if self.load_temporal_fix == 'bilinear':
#                             mode = 'bilinear'
#                         load_temporal_embed = load_temporal_embed.unsqueeze(0)
#                         new_temporal_embed = F.interpolate(load_temporal_embed,
#                                                            (curr_num_frames, embed_dim), mode=mode).squeeze(0)
#                     else:
#                         raise NotImplementedError
#                 new_state_dict['video_model.temporal_embed'] = new_temporal_embed
#         # allow loading with smaller spatial patches. assumes custom border crop, to append the
#         # border patches to the input sequence
#         if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
#             load_pos_embed = new_state_dict['video_model.pos_embed']
#             load_num_patches = load_pos_embed.shape[1]
#             curr_pos_embed = self.state_dict()['video_model.pos_embed']
#             if load_num_patches != curr_pos_embed.shape[1]:
#                 raise NotImplementedError(
#                     'Loading models with different spatial resolution / patch number not yet implemented, sorry.')
#
#         return new_state_dict


# class FrozenInTime_v3(BaseModel):
#     def __init__(self,
#                  video_params,
#                  text_params,
#                  projection_dim=256,
#                  load_checkpoint=None,
#                  projection='minimal',
#                  load_temporal_fix='zeros'):
#         super().__init__()
#
#         self.video_params = video_params
#         self.text_params = text_params
#         self.load_temporal_fix = load_temporal_fix
#         if not text_params['pretrained']:
#             raise NotImplementedError("Huggingface text models require pretrained init.")
#
#         self.text_model = AutoModel.from_pretrained(text_params['model'])
#         self.text_model.train()
#
#         self.ocr_model = AutoModel.from_pretrained(text_params['model'])
#         self.ocr_query = nn.Parameter(torch.zeros(1,768))
#         self.ocr_model.train()
#         self.ocr_proj = nn.Sequential(nn.ReLU(),nn.Linear(self.text_model.config.hidden_size, projection_dim))
#         self.ocr_video_fusion_layer = nn.Linear(projection_dim,projection_dim)
#
#         pretrained = video_params['pretrained']
#         if video_params['model'] == "SpaceTimeTransformer":
#             num_frames = video_params.get('num_frames', 4)
#             time_init = video_params.get('time_init', 'zeros')
#             attention_style = video_params.get('attention_style', 'frozen-in-time')
#             arch_config = video_params.get('arch_config', 'base_patch16_224')
#             vit_init = video_params.get('vit_init', 'imagenet-21k')
#             if arch_config == 'base_patch16_224':
#                 vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
#                 model = SpaceTimeTransformer(num_frames=num_frames,
#                                             time_init=time_init,
#                                             attention_style=attention_style)
#             else:
#                 raise NotImplementedError
#
#             model.head = nn.Identity()
#             model.pre_logits = nn.Identity()
#             ftr_dim = model.embed_dim
#             if load_checkpoint in ["", None]:
#                 vit_checkpoint = vit_model.state_dict()
#                 model.load_state_dict(vit_checkpoint, strict=False)
#             self.video_model = model
#         else:
#             raise NotImplementedError(f"{video_params['model']} not implemented")
#
#         # for backwards compatibility (old models)
#         self.video_model.fc = nn.Identity()
#
#         # Project to a common embedding
#         if projection == 'minimal':
#             txt_proj = nn.Sequential(nn.ReLU(),
#                                      nn.Linear(self.text_model.config.hidden_size, projection_dim),
#                                      )
#
#             vid_proj = nn.Sequential(
#                 nn.Linear(ftr_dim, projection_dim)
#             )
#         elif projection == '':
#             txt_proj = nn.Identity()
#             vid_proj = nn.Identity()
#         else:
#             raise NotImplementedError
#         self.txt_proj = txt_proj
#         self.vid_proj = vid_proj
#
#         if load_checkpoint not in ["", None]:
#             checkpoint = torch.load(load_checkpoint)
#             state_dict = checkpoint['state_dict']
#             new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
#             new_state_dict = self._inflate_positional_embeds(new_state_dict)
#             self.load_state_dict(new_state_dict, strict=True)
#
#     def set_device(self, device):
#         self.device = device
#
#     def forward(self, data, return_embeds=True):
#
#         text_data = data['text']
#         video_data = data['video']
#         ocr_data = data['ocr']
#
#         text_embeddings = self.compute_text(text_data)
#         video_feats, video_embeddings = self.compute_video(video_data)
#         ocr_embeddings = self.compute_ocr(ocr_data, video_feats)
#
#         if return_embeds:
#             return text_embeddings, video_embeddings, ocr_embeddings
#
#         return sim_matrix(text_embeddings, video_embeddings)
#
#     def compute_text(self, text_data):
#         if self.text_params['model'].startswith('bert'):
#             text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
#                 'pooler_output']
#         elif self.text_params['model'].startswith('distilbert'):
#             text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
#         else:
#             raise NotImplementedError
#         text_embeddings = self.txt_proj(text_embeddings)
#         return text_embeddings
#
#     def compute_ocr(self, ocr_data, video_feats):
#         ocr_embeddings = []
#         all_valid_l = ocr_data['all_valid_l']
#         all_input_ids = ocr_data['all_input_ids']
#         all_attention_mask = ocr_data['all_attention_mask']
#         for valid_l,input_ids,attenion_mask,video_feat in zip(all_valid_l,all_input_ids,all_attention_mask, video_feats):
#             video_feat = video_feat[None]
#             if valid_l==0:
#                 ocr_feats = video_feat
#             else:
#                 valid_input_ids = input_ids[:valid_l]
#                 valid_attention_mask = attenion_mask[:valid_l]
#                 ocr_token = dict(input_ids=valid_input_ids,attention_mask=valid_attention_mask)
#                 ocr_feats = self.text_model(**ocr_token).last_hidden_state[:,0,:]
#                 ocr_feats = torch.cat([video_feat,ocr_feats],0)
#             ocr_embedding = video_feat + self.ocr_model(inputs_embeds=ocr_feats[None]).last_hidden_state[:,0,:]
#             ocr_embeddings.append(ocr_embedding)
#         ocr_embeddings = torch.cat(ocr_embeddings,0)
#         ocr_embeddings = self.vid_proj(ocr_embeddings)
#         return ocr_embeddings
#
#     def compute_video(self, video_data):
#         pre_video_embeddings = self.video_model(video_data)
#         video_embeddings = self.vid_proj(pre_video_embeddings)
#         return pre_video_embeddings,video_embeddings
#
#     def _inflate_positional_embeds(self, new_state_dict):
#         # allow loading of timesformer with fewer num_frames
#         curr_keys = list(self.state_dict().keys())
#         if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
#             load_temporal_embed = new_state_dict['video_model.temporal_embed']
#             load_num_frames = load_temporal_embed.shape[1]
#             curr_num_frames = self.video_params['num_frames']
#             embed_dim = load_temporal_embed.shape[2]
#
#             if load_num_frames != curr_num_frames:
#                 if load_num_frames > curr_num_frames:
#                     print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
#                           f'### loading weights, filling in the extras via {self.load_temporal_fix}')
#                     new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
#                 else:
#                     print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
#                           f'### loading weights, filling in the extras via {self.load_temporal_fix}')
#                     if self.load_temporal_fix == 'zeros':
#                         new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
#                         new_temporal_embed[:, :load_num_frames] = load_temporal_embed
#                     elif self.load_temporal_fix in ['interp', 'bilinear']:
#                         # interpolate
#                         # unsqueeze so pytorch thinks its an image
#                         mode = 'nearest'
#                         if self.load_temporal_fix == 'bilinear':
#                             mode = 'bilinear'
#                         load_temporal_embed = load_temporal_embed.unsqueeze(0)
#                         new_temporal_embed = F.interpolate(load_temporal_embed,
#                                                            (curr_num_frames, embed_dim), mode=mode).squeeze(0)
#                     else:
#                         raise NotImplementedError
#                 new_state_dict['video_model.temporal_embed'] = new_temporal_embed
#         # allow loading with smaller spatial patches. assumes custom border crop, to append the
#         # border patches to the input sequence
#         if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
#             load_pos_embed = new_state_dict['video_model.pos_embed']
#             load_num_patches = load_pos_embed.shape[1]
#             curr_pos_embed = self.state_dict()['video_model.pos_embed']
#             if load_num_patches != curr_pos_embed.shape[1]:
#                 raise NotImplementedError(
#                     'Loading models with different spatial resolution / patch number not yet implemented, sorry.')
#
#         return new_state_dict


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt



if __name__ == "__main__":
    pass

import copy
import os
import json
import numpy as np
import torch

from model.metric import *
from model.model import sim_matrix
import tqdm


def t2v_metrics(sims, query_masks=None):
    """Compute retrieval metrics from a similarity matrix.

    Args:
        sims (th.Tensor): N x M matrix of similarities between embeddings, where
             x_{i,j} = <text_embd[i], vid_embed[j]>
        query_masks (th.Tensor): mask any missing queries from the dataset (two videos
             in MSRVTT only have 19, rather than 20 captions)

    Returns:
        (dict[str:float]): retrieval metrics
    """
    assert sims.ndim == 2, "expected a matrix"
    num_queries, num_vids = sims.shape
    dists = -sims
    sorted_dists = np.sort(dists, axis=1)# small to large in axis 1, sort scores for videos, small is better

    # The indices are computed such that they slice out the ground truth distances
    # from the psuedo-rectangular dist matrix
    queries_per_video = num_queries // num_vids
    gt_idx = [[np.ravel_multi_index([ii, jj], (num_queries, num_vids))
               for ii in range(jj * queries_per_video, (jj + 1) * queries_per_video)]
              for jj in range(num_vids)]
    gt_idx = np.array(gt_idx)
    gt_dists = dists.reshape(-1)[gt_idx.reshape(-1)]
    gt_dists = gt_dists[:, np.newaxis]
    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    # --------------------------------
    # NOTE: Breaking ties
    # --------------------------------
    # We sometimes need to break ties (in general, these should occur extremely rarely,
    # but there are pathological cases when they can distort the scores, such as when
    # the similarity matrix is all zeros). Previous implementations (e.g. the t2i
    # evaluation function used
    # here: https://github.com/niluthpol/multimodal_vtt/blob/master/evaluation.py and
    # here: https://github.com/linxd5/VSE_Pytorch/blob/master/evaluation.py#L87) generally
    # break ties "optimistically".  However, if the similarity matrix is constant this
    # can evaluate to a perfect ranking. A principled option is to average over all
    # possible partial orderings implied by the ties. See # this paper for a discussion:
    #    McSherry, Frank, and Marc Najork,
    #    "Computing information retrieval performance measures efficiently in the presence
    #    of tied scores." European conference on information retrieval. Springer, Berlin,
    #    Heidelberg, 2008.
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.145.8892&rep=rep1&type=pdf

    break_ties = "optimistically"
    #break_ties = "averaging"

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            _, idx = np.unique(rows, return_index=True)
            cols = cols[idx]
        elif break_ties == "averaging":
            # fast implementation, based on this code:
            # https://stackoverflow.com/a/49239335
            locs = np.argwhere((sorted_dists - gt_dists) == 0)

            # Find the split indices
            steps = np.diff(locs[:, 0])
            splits = np.nonzero(steps)[0] + 1
            splits = np.insert(splits, 0, 0)

            # Compute the result columns
            summed_cols = np.add.reduceat(locs[:, 1], splits)
            counts = np.diff(np.append(splits, locs.shape[0]))
            avg_cols = summed_cols / counts
            if False:
                print("Running slower code to verify rank averaging across ties")
                # slow, but more interpretable version, used for testing
                avg_cols_slow = [np.mean(cols[rows == idx]) for idx in range(num_queries)]
                assert np.array_equal(avg_cols, avg_cols_slow), "slow vs fast difference"
                print("passed num check")
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    if cols.size != num_queries:
        import ipdb;
        ipdb.set_trace()
    assert cols.size == num_queries, msg

    if False:
        # overload mask to check that we can recover the scores for single-query
        # retrieval
        print("DEBUGGING MODE")
        query_masks = np.zeros_like(query_masks)
        query_masks[:, 0] = 1  # recover single query score

    if query_masks is not None:
        # remove invalid queries
        assert query_masks.size == num_queries, "invalid query mask shape"
        cols = cols[query_masks.reshape(-1).astype(np.bool)]
        assert cols.size == query_masks.sum(), "masking was not applied correctly"
        # update number of queries to account for those that were missing
        num_queries = query_masks.sum()

    if False:
        # sanity check against old logic for square matrices
        gt_dists_old = np.diag(dists)
        gt_dists_old = gt_dists_old[:, np.newaxis]
        _, cols_old = np.where((sorted_dists - gt_dists_old) == 0)
        assert np.array_equal(cols_old, cols), "new metric doesn't match"

    return cols2metrics(cols, num_queries), np.array(cols,np.int32)


def v2t_metrics(sims, query_masks=None):
    """Compute retrieval metrics from a similarity matrix.

    Args:
        sims (th.Tensor): N x M matrix of similarities between embeddings, where
             x_{i,j} = <text_embd[i], vid_embed[j]>
        query_masks (th.Tensor): mask any missing captions from the dataset

    Returns:
        (dict[str:float]): retrieval metrics

    NOTES: We find the closest "GT caption" in the style of VSE, which corresponds
    to finding the rank of the closest relevant caption in embedding space:
    github.com/ryankiros/visual-semantic-embedding/blob/master/evaluation.py#L52-L56
    """
    # switch axes of text and video
    sims = sims.T

    if False:
        # experiment with toy example
        sims = np.ones((3, 3))
        sims[0, 0] = 2
        sims[1, 1:2] = 2
        sims[2, :] = 2
        query_masks = None

    assert sims.ndim == 2, "expected a matrix"
    num_queries, num_caps = sims.shape
    dists = -sims
    caps_per_video = num_caps // num_queries
    break_ties = "averaging"

    MISSING_VAL = 1E8
    query_ranks = []
    for ii in range(num_queries):
        row_dists = dists[ii, :]
        if query_masks is not None:
            # Set missing queries to have a distance of infinity.  A missing query
            # refers to a query position `n` for a video that had less than `n`
            # captions (for example, a few MSRVTT videos only have 19 queries)
            row_dists[np.logical_not(query_masks.reshape(-1))] = MISSING_VAL

        # NOTE: Using distance subtraction to perform the ranking is easier to make
        # deterministic than using argsort, which suffers from the issue of defining
        # "stability" for equal distances.  Example of distance subtraction code:
        # github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/train.py
        sorted_dists = np.sort(row_dists)

        min_rank = np.inf
        for jj in range(ii * caps_per_video, (ii + 1) * caps_per_video):
            if row_dists[jj] == MISSING_VAL:
                # skip rankings of missing captions
                continue
            ranks = np.where((sorted_dists - row_dists[jj]) == 0)[0]
            if break_ties == "optimistically":
                rank = ranks[0]
            elif break_ties == "averaging":
                # NOTE: If there is more than one caption per video, its possible for the
                # method to do "worse than chance" in the degenerate case when all
                # similarities are tied.  TODO(Samuel): Address this case.
                rank = ranks.mean()
            if rank < min_rank:
                min_rank = rank
        query_ranks.append(min_rank)
    query_ranks = np.array(query_ranks)

    # sanity check against old version of code
    if False:
        sorted_dists = np.sort(dists, axis=1)
        gt_dists_old = np.diag(dists)
        gt_dists_old = gt_dists_old[:, np.newaxis]
        rows_old, cols_old = np.where((sorted_dists - gt_dists_old) == 0)
        if rows_old.size > num_queries:
            _, idx = np.unique(rows_old, return_index=True)
            cols_old = cols_old[idx]
        num_diffs = (1 - (cols_old == query_ranks)).sum()
        msg = f"new metric doesn't match in {num_diffs} places"
        assert np.array_equal(cols_old, query_ranks), msg

        # visualise the distance matrix
        import sys
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
        from zsvision.zs_iterm import zs_dispFig  # NOQA
        plt.matshow(dists)
        zs_dispFig()

    return cols2metrics(query_ranks, num_queries), np.array(query_ranks,np.int32)


class ResultsCompare():
    def __init__(self,
                 pre_results = 'zyz_utils/tmp/pre_results/',
                 aft_results = 'zyz_utils/tmp/after_results/',
                 anns_path = '/share/mmu-ocr/datasets/zyz_anns/youcook2seg/youcook2seg.json',
                 split = 'train'):
        self.save_name = './change_results.json'
        self.anns_path = anns_path
        self.id_file = os.path.join(pre_results,f'ids_{split}.csv')
        self.save_format = {'video_name':'','t2v':dict(),'v2t':dict(),'combine':dict()}
        pre_txt_embeds = os.path.join(pre_results,f'txt_embeds_{split}.npy')
        pre_vid_embeds = os.path.join(pre_results, f'vid_embeds_{split}.npy')

        aft_txt_embeds = os.path.join(aft_results, f'txt_embeds_{split}.npy')
        aft_vid_embeds = os.path.join(aft_results, f'vid_embeds_{split}.npy')

        self.pre_txt_embeds = np.load(pre_txt_embeds)
        self.pre_vid_embeds = np.load(pre_vid_embeds)
        self.aft_txt_embeds = np.load(aft_txt_embeds)
        self.aft_vid_embeds = np.load(aft_vid_embeds)



    def metric(self,source = 'pre'):
        if source == 'pre':
            txt_embeds = self.pre_txt_embeds
            vid_embeds = self.pre_vid_embeds
        else:
            txt_embeds = self.aft_txt_embeds
            vid_embeds = self.aft_vid_embeds
        sims = sim_matrix(torch.from_numpy(txt_embeds), torch.from_numpy(vid_embeds))
        sims = sims.numpy()
        t2v_res, t2v_ranks = t2v_metrics(sims, query_masks=None)
        v2t_res, v2t_ranks = v2t_metrics(sims, query_masks=None)
        return t2v_res, v2t_res, t2v_ranks,v2t_ranks

    def compare_rank(self):
        _, _, pre_t2v_ranks, pre_v2t_ranks = self.metric('pre')
        _, _, aft_t2v_ranks, aft_v2t_ranks = self.metric('aft')
        t2v_ranks_change = -(aft_t2v_ranks-pre_t2v_ranks)
        v2t_ranks_change = -(aft_v2t_ranks-pre_v2t_ranks)
        combine_ranks_change = ((t2v_ranks_change+v2t_ranks_change)/2).astype(np.float32)
        outputs = (pre_t2v_ranks.tolist(), pre_v2t_ranks.tolist(), aft_t2v_ranks.tolist(), aft_v2t_ranks.tolist(),
                   t2v_ranks_change.tolist(), v2t_ranks_change.tolist(),combine_ranks_change.tolist())
        return outputs

    def main(self):
        save_list = []
        outputs = self.compare_rank()
        with open(self.anns_path, 'r') as fr:
            videos = json.load(fr)
        with open(self.id_file, 'r') as fr:
            ids = fr.readlines()
            ids = [el.strip() for el in ids[1:]]
        for id,el1,el2,el3,el4,el5,el6,el7 in zip(tqdm.tqdm(ids),*outputs):
            try:
                video = [el for el in videos if id in el['path']]
                assert len(video)==1
                video = video[0]
            except:
                print(id)
                raise
            save_format = copy.deepcopy(self.save_format)
            save_format['video_name'] = id
            save_format['t2v']['pre_rank'] = el1
            save_format['t2v']['aft_rank'] = el3
            save_format['t2v']['rc'] = el5
            save_format['v2t']['pre_rank'] = el2
            save_format['v2t']['aft_rank'] = el4
            save_format['v2t']['rc'] = el6
            save_format['combine']['rc'] = el7
            video['ranks_info'] = save_format
            save_list.append(video)

        with open(self.save_name, 'w') as fw:
            json.dump(save_list,fw)


    def rank_select(self,save_json = './rank_select.json',thr=1):
        with open(self.save_name, 'r') as fr:
            videos = json.load(fr)
        selected_videos = []
        for video in videos:
            if video['ranks_info']['combine']['rc']>thr:
                selected_videos.append(video)
        print(f"Select {len(selected_videos)} videos from {len(videos)} with thr {thr}.")
        with open(save_json,'w') as fw:
            json.dump(selected_videos,fw)



if __name__ == '__main__':
    RC = ResultsCompare()
    RC.main()
    RC.rank_select(thr=1)
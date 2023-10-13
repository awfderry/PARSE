import torch
import numpy as np
from tqdm import tqdm
from atom3d.datasets import load_dataset
from collapse.utils import pdb_from_fname
import argparse
import parse
import utils

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./data/datasets/af2_dark_proteome/full')
    parser.add_argument('--db', type=str, default='./data/csa_site_db_nn.pkl', help='Reference database embeddings')
    parser.add_argument('--function_sets', type=str, default='./data/csa_function_sets_nn.pkl', help='Reference function sets')
    parser.add_argument('--background', type=str, default='./data/function_score_dists.pkl', help='Function-specific background distributions')
    parser.add_argument('--cutoff', type=float, default=0.001, help='FDR cutoff for reporting results')
    parser.add_argument('--out_path', type=str, default='./data/results/parse_af2_dark.pkl', help='Output path for results (pickle)')
    parser.add_argument('--split_id', type=int, default=0)
    parser.add_argument('--num_splits', type=int, default=1)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, 'lmdb')

    db = utils.deserialize(args.db)
    function_sets = utils.deserialize(args.function_sets)
    background_dists = utils.deserialize(args.background)

    if args.num_splits > 1:
        out_path = args.out_path.replace('.pkl', f'_{args.split_id}.pkl')

        split_idx = np.array_split(np.arange(len(dataset)), args.num_splits)[args.split_id - 1]
        print(f'Processing split {args.split_id} with {len(split_idx)} examples...')

        dataset = torch.utils.data.Subset(dataset, split_idx)
    else:
        out_path = args.out_path
        print(f'Processing full dataset with {len(dataset)} examples...')

    results = {}
    for pdb_data in tqdm(dataset):
        protein = pdb_data['id']
        rnk = parse.compute_rank_df(pdb_data, db)
        result = parse.parse(rnk, function_sets, background_dists, args.cutoff)
        results[protein] = result.copy()
        
    utils.serialize(results)

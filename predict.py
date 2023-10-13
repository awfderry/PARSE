from collapse import process_pdb, embed_protein, initialize_model
from atom3d.datasets import load_dataset
import argparse
import time
import parse
import utils

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, default=None, help='Input PDB file')
    parser.add_argument('--precomputed_id', type=str, default=None, help='ID for accessing precomputed embeddings')
    parser.add_argument('--precomputed_lmdb', type=str, default=None, help='Precomputed embeddings in LMDB format')
    parser.add_argument('--chain', type=str, default=None, help='Input PDB chain to annotate')
    parser.add_argument('--db', type=str, default='./data/csa_site_db_nn.pkl', help='Reference database embeddings')
    parser.add_argument('--function_sets', type=str, default='./data/csa_function_sets_nn.pkl', help='Reference function sets')
    parser.add_argument('--background', type=str, default='./data/function_score_dists.pkl', help='Function-specific background distributions')
    parser.add_argument('--cutoff', type=float, default=0.001, help='FDR cutoff for reporting results')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU to embed proteins')
    args = parser.parse_args()
    
    start = time.time()
    device = 'cuda' if args.use_gpu else 'cpu'
    
    db = utils.deserialize(args.db)
    function_sets = utils.deserialize(args.function_sets)
    background_dists = utils.deserialize(args.background)
    
    if args.pdb:
        model = initialize_model(device=device)
        pdb_df = process_pdb(args.pdb, chain=args.chain, include_hets=False)
        embed_data = embed_protein(pdb_df, model, device, include_hets=False)
        embed_data['id'] = args.pdb
        print(f'Time to embed PDB: {time.time() - start:.2f} seconds')

    elif args.precomputed_id and args.precomputed_lmdb:
        pdb_dataset = load_dataset(args.precomputed_lmdb, 'lmdb')
        idx = pdb_dataset.ids_to_indices([args.precomputed_id])[0]
        embed_data = pdb_dataset[idx]
    
    else:
        raise Exception('Must provide either PDB file (--pdb) or ID and LMDB dataset for precomputed embeddings (--precomputed_id, --precomputed_lmdb)')
    
    rnk = parse.compute_rank_df(embed_data, db)
    results = parse.parse(rnk, function_sets, background_dists, cutoff=args.cutoff)
    print(results)
    print(f'Finished in {time.time() - start:.2f} seconds')
    
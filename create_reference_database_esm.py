import numpy as np
import pandas as pd
import os
import argparse
# from fastdist import fastdist
import pickle
from tqdm import tqdm
import torch
import collections as col
from Bio import BiopythonDeprecationWarning
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", BiopythonDeprecationWarning)
    import Bio.PDB.Polypeptide as Poly

import scipy.spatial
from collapse import data
from torch_geometric.loader import DataLoader
from torch.utils.data import IterableDataset
from collapse import initialize_model, atom_info

def embed_esm(df, model, device):
    assert len(df) > 0
    chain_sequences, chain_residues = get_chain_sequences(df)
    batch_labels, batch_strs, batch_tokens = batch_converter(chain_sequences)
    # batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        try:
            results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise(e)
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            return
    embeddings = results["representations"][33].cpu()
    outdata = dict()
    for ch_idx, chain in enumerate(chain_residues):
        for res_idx in range(len(chain)):
            emb = embeddings[ch_idx,res_idx,:]
            resid = chain_residues[ch_idx][res_idx]
            # outdata['chains'].append(ch_idx)
            # outdata['resids'].append(resid)
            # outdata['embeddings'].append(emb)
            outdata[resid] = emb
    # outdata['embeddings'] = np.stack(outdata['embeddings'], 0)
    return outdata

def get_chain_sequences(df):
    """Return list of tuples of (id, sequence) for different chains of monomers in a given dataframe."""
    # Keep only CA of standard residues
    df = df[df['name'] == 'CA'].drop_duplicates()
    df = df[df['resname'].apply(lambda x: Poly.is_aa(x, standard=True))]
    df['resname'] = df['resname'].apply(Poly.three_to_one)
    chain_sequences = []
    chain_residues = []
    for c, chain in df.groupby(['ensemble', 'subunit', 'structure', 'model', 'chain']):
        seq = ''.join(chain['resname'])
        chain_sequences.append((str(c[2])+'_'+str(c[-1]), seq))
        chain_residues.append([seq[i]+str(r) for i, r in enumerate(chain['residue'].tolist())])
    return chain_sequences, chain_residues

class ESMNNDataset(IterableDataset):
    '''
    Yields graphs from a dictionary of xyz coordinates defining PDB sites, plus neighbors within 3.5A.
    '''

    def __init__(self, dataset, pdb_dir, train_mode=True, model=None, device='cpu'):
        self.dataset = dataset #pd.read_csv(dataset, converters={'locs': lambda x: eval(x)})
        self.pdb_dir = pdb_dir
        self.model = model
        self.device = device
        self.train_mode = train_mode
        if model is None:
            raise Exception('Please pass valid initialized ESM model')

    def __iter__(self):
        if 'pdb' in self.dataset.columns:
            type = 'pdb'
        elif 'uniprot' in self.dataset.columns:
            type = 'uniprot'
        for it, (name, df) in enumerate(self.dataset.groupby(type)):
            if it % 100 == 0:
                print(f'processing protein {it} of {self.dataset[type].nunique()}')
            if type == 'pdb':
                pdb = name[:4]
                chain = name[4:]
                fp = os.path.join(self.pdb_dir, pdb[1:3], 'pdb' + pdb + '.ent.gz')
            elif type == 'uniprot':
                chain = 'A'
                fp = os.path.join(self.pdb_dir, f'AF-{name}-F1-model_v2.pdb.gz')
            if not os.path.exists(fp):
                print('skipping PDB', name)
                continue
            atom_df = data.process_pdb(fp, chain=chain, include_hets=False)
            if len(atom_df) == 0:
                print('skipping PDB', name, 'chain', chain)
                continue
            esm_data = embed_esm(atom_df.copy(), self.model, self.device)
            if esm_data is None:
                print('skipping PDB', name, 'chain', chain, 'due to OOM')
                continue
            kd_tree = scipy.spatial.cKDTree(atom_df[['x', 'y', 'z']].to_numpy())

            for r, (site, _, locs, source, desc) in df.iterrows():
                resids = self._get_neighbors(atom_df.copy(), kd_tree, locs)
                for resid in resids:
                    emb = esm_data[resid]
                    yield emb, resid, name, source, desc

    def _get_neighbors(self, df, kd_tree, resnums):
        # df_ca = df[df.name=='CA'].reset_index()
        df = df.reset_index()
        df_centers = df[df.residue.isin(resnums)]
        pt_idx = kd_tree.query_ball_point(df_centers[['x', 'y', 'z']].to_numpy(), r=3.5, p=2.0)
        pt_idx = [pt for x in pt_idx for pt in x]
        df_nn = df.iloc[pt_idx, :]
        nn_resids = (df_nn['resname'].apply(atom_info.aa_to_letter) + df_nn['residue'].astype(str)).unique()
        return nn_resids

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('embedding_outfile', type=str)
    parser.add_argument('funcsets_outfile', type=str)
    parser.add_argument('--source', type=str, default='M-CSA')
    parser.add_argument('--pdb_dir', type=str, default='/scratch/users/aderry/pdb')
    parser.add_argument('--use_neighbors', action='store_true')
    args = parser.parse_args()
    
    # os.makedirs(args.outfile, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    
    src_dataset = pd.read_csv(args.dataset, converters={'locs': lambda x: eval(x)})
    src_dataset = src_dataset[src_dataset['source'] == args.source]
    
    dataset = ESMNNDataset(src_dataset, args.pdb_dir, train_mode=False, model=model, device=device)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print('Computing embeddings...')
    all_emb = []
    prosite_labels = []
    all_pdb = []
    all_sites = []
    all_sources = []
    all_resids = []
    with torch.no_grad():
        for emb, resid, pdb, source, desc in loader:
            all_emb.append(emb.squeeze().cpu().numpy())
            all_pdb.append(pdb[0])
            all_sites.append(desc[0])
            all_sources.append(source[0])
            all_resids.append(resid[0])
         
    print('Saving...')
    all_emb = np.stack(all_emb)
    outdata = {'embeddings': all_emb.copy(), 'pdbs': all_pdb, 'resids': all_resids, 'sites': all_sites, 'sources': all_sources}
    pdb_resids = [x+'_'+y for x,y in zip(all_pdb, all_resids)]
    
    with open(args.embedding_outfile, 'wb') as f:
        pickle.dump(outdata, f)
        
    fn_lists = col.defaultdict(set)
    for fn, site in zip(all_sites, pdb_resids):
        fn_lists[f'{fn}'].add(str(site))
    with open(args.funcsets_outfile, 'wb') as f:
        pickle.dump({k: list(v) for k,v in fn_lists.items()}, f)

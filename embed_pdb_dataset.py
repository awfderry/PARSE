import numpy as np
import pandas as pd
import os
import argparse
import collections as col
import torch
import Bio.PDB.Polypeptide as Poly
from collapse.data import EmbedTransform
from atom3d.filters.filters import first_model_filter
from atom3d.datasets import load_dataset, make_lmdb_dataset
import atom3d.util.file as fi
from collapse import initialize_model, atom_info

def get_chain_sequences(df):
    """Return list of tuples of (id, sequence) for different chains of monomers in a given dataframe."""
    # Keep only CA of standard residues
    df = df[df['name'] == 'CA'].drop_duplicates()
    df = df[df['resname'].apply(lambda x: Poly.is_aa(x, standard=True))]
    df['resname'] = df['resname'].apply(Poly.three_to_one)
    chain_sequences = []
    chain_residues = []
    chain_confidences = []
    for c, chain in df.groupby(['ensemble', 'subunit', 'structure', 'model', 'chain']):
        seq = ''.join(chain['resname'])
        chain_sequences.append((str(c[2])+'_'+str(c[-1]), seq))
        chain_residues.append([seq[i]+str(r) for i, r in enumerate(chain['residue'].tolist())])
        chain_confidences.append(chain['bfactor'].tolist())
    return chain_sequences, chain_residues, chain_confidences


def embed_esm(df, model, batch_converter, device):
    chain_sequences, chain_residues, confidences = get_chain_sequences(df)
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
    embeddings = results["representations"][33]
    embeddings = embeddings.cpu().numpy()
    outdata = col.defaultdict(list)
    for ch_idx, chain in enumerate(chain_residues):
        for res_idx in range(len(chain)):
            emb = embeddings[ch_idx,res_idx,:]
            resid = chain_residues[ch_idx][res_idx]
            outdata['chains'].append(ch_idx)
            outdata['resids'].append(resid)
            outdata['embeddings'].append(emb)
            outdata['confidence'].append(confidences[ch_idx][res_idx])
    outdata['embeddings'] = np.stack(outdata['embeddings'], 0)
    return outdata

class ESMTransform(object):
    '''
    Embeds processed PDB files in LMDB dataset using ESM-2
    '''

    def __init__(self, model, batch_converter, include_hets=True, env_radius=10.0, device='cpu'):
        self.model = model
        self.include_hets = include_hets
        self.env_radius = env_radius
        self.device = device
        self.batch_converter = batch_converter

    def __call__(self, elem):
        atom_df = elem['atoms']
        if not self.include_hets:
            atom_df = atom_df[atom_df.resname.isin(atom_info.aa)].reset_index(drop=True)
        try:
            atom_df = first_model_filter(atom_df)
            atom_df = atom_df[atom_df.resname != 'HOH']
            atom_df = atom_df[atom_df.element != 'H'].reset_index(drop=True)
            if not self.include_hets:
                atom_df = atom_df[atom_df.resname.isin(atom_info.aa)].reset_index(drop=True)
        except:
            return

        if len(atom_df) == 0:
            return
        outdata = embed_esm(atom_df, self.model, self.batch_converter, self.device)
        if outdata is None:
            return
        elem['resids'] = outdata['resids']
        elem['confidence'] = outdata['confidence']
        elem['chains'] = outdata['chains']
        elem['embeddings'] = outdata['embeddings']

        return elem


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--split_id', type=int, default=0)
    parser.add_argument('--filetype', type=str, default='pdb')
    parser.add_argument('--encoder', type=str, default='COLLAPSE')
    parser.add_argument('--num_splits', type=int, default=1)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_files = fi.find_files(args.data_dir, args.filetype)
    # names = [str(x).split('/')[-1].split('-')[1] for x in all_files]
    

    if args.encoder.upper() == 'COLLAPSE':
        model = initialize_model(device=device)
        transform = EmbedTransform(model, device=device, include_hets=False, compute_res_graph=False)
    elif args.encoder.upper() == 'ESM':
        model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        batch_converter = alphabet.get_batch_converter()
        model = model.to(device)
        model.eval()
        transform = ESMTransform(model, batch_converter, device=device, include_hets=False)
    else:
        raise Exception('Valid encoders are COLLAPSE and ESM')
    dataset = load_dataset(args.data_dir, args.filetype, transform=transform)
    
    if args.num_splits > 1:
        out_path = os.path.join(args.out_dir, f'tmp_{args.split_id}')
        os.makedirs(out_path, exist_ok=True)
    
        split_idx = np.array_split(np.arange(len(all_files)), args.num_splits)[args.split_id - 1]
        print(f'Processing split {args.split_id} with {len(split_idx)} examples...')
        dataset = torch.utils.data.Subset(dataset, split_idx)
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = args.out_dir
    
    make_lmdb_dataset(dataset, out_path, serialization_format='pkl', filter_fn=lambda x: (x is None))

import numpy as np
import pandas as pd
import os
import argparse
from fastdist import fastdist
import pickle
from tqdm import tqdm
import torch
import collections as col
from collapse.data import SiteDataset, SiteNNDataset
from torch_geometric.loader import DataLoader
from collapse import initialize_model, atom_info

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

src_dataset = pd.read_csv(args.dataset, converters={'locs': lambda x: eval(x)})
src_dataset = src_dataset[src_dataset['source'] == args.source]

if args.use_neighbors:
    dataset = SiteNNDataset(src_dataset, args.pdb_dir, train_mode=False)
else:
    dataset = SiteDataset(src_dataset, args.pdb_dir, train_mode=False)
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

model = initialize_model(device=device)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

print('Computing embeddings...')
all_emb = []
prosite_labels = []
all_pdb = []
all_sites = []
all_sources = []
all_resids = []
with torch.no_grad():
    for g, pdb, source, desc in loader:
        g = g.to(device)
        embeddings, _ = model.online_encoder(g, return_projection=False)
        all_emb.append(embeddings.squeeze().cpu().numpy())
        all_pdb.append(pdb[0])
        all_sites.append(desc[0])
        all_sources.append(source[0])
        all_resids.append(g.resid[0])
     
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

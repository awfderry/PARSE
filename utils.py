import pickle
import numpy as np

def serialize(obj, path):
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)

def deserialize(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)

def pdb_from_fname(fname):
    af_flag = False
    if fname.endswith('.ent.gz'):
        pdb = fname[3:7]
    elif fname.endswith('.pdb'):
        pdb = fname[:-4]
    elif 'AF' in fname:
        af_flag = True
        pdb = fname.split('-')[1]
    return pdb, af_flag

def get_db_site_map(rnk):
    res_match = np.array([x.split('_')[1][0] for x in rnk['site']]) == rnk['location'].str[0]
    return dict(zip(rnk['site'], rnk['location'])), dict(zip(rnk['site'], res_match))

def calc_rmsd(A, B):
    D = len(A[0])
    N = len(A)
    rmsd = 0.0
    for v, w in zip(A, B):
        rmsd += sum([(v[i] - w[i])**2.0 for i in range(D)])
    return np.sqrt(rmsd/N)

def align(A, B):
    A -= A.mean(axis=0)
    B -= B.mean(axis=0)
    
    C = np.dot(np.transpose(A), B)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    A_rot = np.dot(A, U)
    rmsd = calc_rmsd(A_rot, B)
    return U, A.mean(axis=0), B.mean(axis=0), rmsd
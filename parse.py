import pickle
import os
import pandas as pd
import numpy as np
import blitzgsea as gsea
from scipy.stats import false_discovery_control
from fastdist import fastdist
import utils

def enrichment(signature, library, min_size=5, max_size=4000, processes=4, center=True):
    """Computes enrichment scores. Adapted from blitzgsea.gsea (https://github.com/MaayanLab/blitzgsea)

    :param signature: ranked list with 2 columns: id and value
    :type signature: pd.DataFrame
    :param library: function sets to test for enrichment
    :type library: dict
    :param min_size: minimum set size to consider, defaults to 5
    :type min_size: int, optional
    :param max_size: maximum set size to consider, defaults to 4000
    :type max_size: int, optional
    :param processes: number of processes, defaults to 4
    :type processes: int, optional
    :param center: whether to mean-center values, defaults to True
    :type center: bool, optional
    
    :return: results dataframe with 4 columns: function, enrichment score, set size, leading edge residues
    :rtype: pd.DataFrame
    """
    signature = signature.copy()
    signature.columns = ["i","v"]

    sig_hash = hash(signature.to_string())
    signature = signature.sort_values("v", ascending=False).set_index("i")
    signature = signature[~signature.index.duplicated(keep='first')]
    
    if center:
        signature.loc[:,"v"] -= np.median(signature.loc[:,"v"])

    abs_signature = np.array(np.abs(signature.loc[:,"v"]))
    
    signature_map = {}
    for i,h in enumerate(signature.index):
        signature_map[h] = i

    gsets = []

    keys = list(library.keys())
    signature_genes = set(signature.index)

    ess = []
    set_size = []
    legeness = []

    for k in keys:
        stripped_set = gsea.strip_gene_set(signature_genes, library[k])
        if len(stripped_set) >= min_size and len(stripped_set) <= max_size:
            gsets.append(k)
            gsize = len(stripped_set)
            rs, es = gsea.enrichment_score(abs_signature, signature_map, stripped_set)
            legenes = gsea.get_leading_edge(rs, signature, stripped_set, signature_map)

            ess.append(float(es))
            set_size.append(gsize)
            legeness.append(legenes)

    res =  pd.DataFrame([gsets, np.array(ess), np.array(set_size), np.array(legeness)]).T
    res.columns = ["function", "score", "func_nres", "ref_sites"]
    res["function"] = res['function'].astype("str")
    res["score"] = res['score'].astype("float")
    res["func_nres"] = res['func_nres'].astype("int")
    res["ref_sites"] = res['ref_sites'].astype("str")
    
    return res.sort_values("score", ascending=False)

def get_pval(df, function_score_dists):
    """compute empirical p-value from background distribution
    
    :param df: row of enrichment results dataframe
    :type df: pd.Series
    :param function_score_dists: function-specific background distributions: keys = function name, values = list of scores
    :type function_score_dists: dict
    
    :return: empirical p-value
    :rtype: float
    """
    backg = np.array(function_score_dists[df['function']])
    empirical_p = np.mean(backg > df['score'])
    return empirical_p

def parse(rank_df, function_sets, background_dists, cutoff=0.001):
    """Main wrapper for PARSE, given ranked list of sites
    
    :param rank_df: ranked list dataframe with at least 2 columns: site, score
    :type rank_df: pd.DataFrame
    :param function_sets: function sets to test for enrichment
    :type function_sets: dict
    :param background_dists: function-specific background distributions: keys = function name, values = list of scores
    :type background_dists: dict
    :cutoff: empirical FDR cutoff for results, defaults to 0.001
    :type cutoff: float, optional
    
    :return: enrichment results dataframe with 8 columns: function, enrichment score, set size, leading edge residues, empirical p-value, empirical FDR, reference DB sites, query hit sites
    :rtype: pd.DataFrame
    """
    in_df = rank_df[['site', 'score']]
    in_df.columns = [0, 1]
    result = enrichment(in_df, function_sets)
    result['empirical_pval'] = result.apply(lambda x: get_pval(x, function_score_dists=background_dists), axis=1)
    result['empirical_FDR'] = false_discovery_control(result['empirical_pval'], method='bh')
    result['ref_sites'] = [x.split(',') for x in np.nan_to_num(result['ref_sites'].tolist(), '')]
    site_map, res_match = utils.get_db_site_map(rank_df)
    result['hit_sites'] = [[site_map.get(x, 'N/A') for x in l] for l in result['ref_sites']]
    # result['res_match'] = [[res_match.get(x, 'N/A') for x in l] for l in result['ref_sites']]
    return result[result['empirical_FDR'] < cutoff]
    
def compute_rank_df(pdb_data, db):
    """Compute ranked list of DB sites based on distance to query residues

    :param pdb_data: query data dictionary returned by COLLAPSE embedding (see collapse.embed_protein)
    :type pdb_data: dict
    :param db: database embedding dictionary, must have keys `pdb`, `resids`, `embeddings`
    :type db: dict
    
    :return: ranked list of DB sites in dataframe format, sorted by cosine similarity
    :rtype: pd.DataFrame
    """
    pdb_resids = [x+'_'+y for x,y in zip(db['pdbs'], db['resids'])]
    
    pdb_id, af_flag = utils.pdb_from_fname(pdb_data["id"])

    resids = np.array(pdb_data['resids'])
    chains = np.array(pdb_data['chains'])
    embeddings = np.array(pdb_data['embeddings'])
    confidences = np.array(pdb_data['confidence'])

    if af_flag:
        # print('Removing low confidence residues')
        high_conf_idx = confidences >= 70
        resids = resids[high_conf_idx]
        chains = chains[high_conf_idx]
        embeddings = embeddings[high_conf_idx]

    cosines = fastdist.cosine_matrix_to_matrix(embeddings, db['embeddings'])  # (n_res, n_db)
    
    max_site_idx = np.argmax(cosines, axis=0)
    max_values = np.amax(cosines, 0)

    out_df = pd.DataFrame({'site': pdb_resids, 'score': max_values, 'location': resids[max_site_idx]})
    out_df = out_df.sort_values('score', ascending=False)
    out_df = out_df.drop_duplicates('site').reset_index(drop=True)
    
    return out_df
from itertools import combinations
from Bio import SeqIO
from Bio import pairwise2
import os
import time
import numpy as np
import argparse
import queue, threading
import multiprocessing

def calculate_identity(seq1, seq2):
    # Perform global alignment using the Needleman-Wunsch algorithm
    alignments = pairwise2.align.globalxx(seq1, seq2)
    
    # Use the best alignment (the first one in the list)
    best_alignment = alignments[0]
    
    # Get the number of matches from the alignment
    matches = best_alignment[2]
    
    # Compute identity as matches divided by the maximum sequence length
    identity = matches / max(len(seq1), len(seq2))
    return identity

def cluster_sequences(seq_identity_dic, seq_ids, threshold):
    """
    Clusters sequences based on sequence identity threshold after alignment.
    Args:
        sequences (list): List of sequences to be clustered.
        threshold (float): Sequence identity threshold for clustering.
    Returns:
        list of lists: Clusters of sequences.
    """
    clusters = []
    
    for seq_id in seq_ids:
        # Try to place the sequence in an existing cluster
        added_to_cluster = False
        for cluster in clusters:
            # Compare with the first sequence in the cluster
            if seq_identity_dic[(seq_id, cluster[0])] >= threshold:
                cluster.append(seq_id)
                added_to_cluster = True
                break
        
        # If no cluster was found, create a new one
        if not added_to_cluster:
            clusters.append([seq_id])
    
    return clusters

def process(q):
    while True:
        p= q.get(block=True)
        if p is None:
            return
        i, seq_id1, seq_aa1, seq_id2, seq_aa2 = p[0], p[1], p[2], p[3], p[4]
        
        if np.log10(i+1)%1 == 0:
            print (f"[{time.ctime()}] {i+1} sequence pairs processed.")

        seq_identity = calculate_identity(seq_aa1, seq_aa2)
        results[(seq_id1, seq_id2)] = seq_identity
        results[(seq_id2, seq_id1)] = seq_identity


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--fasta_path", type=str, required=True, help="Path of input file with fasta format")
    parser.add_argument("--seq_id_threshold", type=float, required=True, help="Threshold of sequence identity for clustering")
    parser.add_argument("--output_path", type=str, required=True, help="Path of output file")
    args = parser.parse_args()

    f = open(args.fasta_path, 'r')
    data = f.read().split(">")
    f.close()
    seq_dic = {}
    for da in data:
        if len(da) == 0: continue
        da = da.strip().split("\n")
        seq_id, seq_aa = da[0], "".join(da[1:])
        seq_dic[seq_id] = seq_aa


    the_queue = multiprocessing.Queue()  # Queue stores the iterables
    manager = multiprocessing.Manager()
    results = manager.dict()
    
    pair_num = 0
    seq_ids = seq_dic.keys()
    for i, seq_id1 in enumerate(seq_ids):
        for j, seq_id2 in enumerate(seq_ids):
            if i >= j: continue
            the_queue.put([pair_num, seq_id1, seq_dic[seq_id1], seq_id2, seq_dic[seq_id2]])
            pair_num += 1
    pool = multiprocessing.Pool(os.cpu_count()-1, process, (the_queue, ))
    print ("Number of sequence pairs:", pair_num)
    
    # None to end each process
    for _i in range(os.cpu_count()):
        the_queue.put(None)
    
    # Closing the queue and the pool
    the_queue.close()
    the_queue.join_thread()
    pool.close()
    pool.join()
            
    # Compute identity for all pairs and cluster sequences
    clusters = cluster_sequences(results, seq_ids, args.seq_id_threshold)
    
    # Save the clusters
    fo = open(args.output_path, 'w')
    print ("\t".join(["Cluster_ID", "Seq_ID", "AA_Sequence"]), file=fo)
    
    for i, cluster in enumerate(clusters):
        for seq_id in cluster:
            print ("\t".join([str(el) for el in [i, seq_id, seq_dic[seq_id]]]), file=fo)
    fo.close()
    
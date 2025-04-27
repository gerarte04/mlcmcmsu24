import numpy as num
from sklearn.metrics import pairwise as pair_wise


def silhouette_score(data_vectors, cluster_assignments):
    '''
    Calculate clustering quality score based on object silhouettes
    data_vectors: 2D array of feature vectors
    cluster_assignments: 1D array of cluster IDs
    returns: silhouette coefficient value
    '''
    data_vectors = num.array(data_vectors, dtype=float)
    cluster_assignments = num.array(cluster_assignments, dtype=int)

    num_objects = data_vectors.shape[0]
    unique_clusters = num.unique(cluster_assignments)

    if num_objects < 2 or len(unique_clusters) < 2:
        return 0.0

    distance_matrix = pair_wise.pairwise_distances(data_vectors)
    cluster_groups = {cid: num.where(cluster_assignments == cid)[0] for cid in unique_clusters}

    scores = num.zeros(num_objects)

    for obj_idx in range(num_objects):
        current_cluster = cluster_assignments[obj_idx]
        cluster_members = cluster_groups[current_cluster]

        if cluster_members.size == 1:
            continue  # score remains 0 for single-object clusters

        # Calculate average intra-cluster distance
        mask = cluster_members != obj_idx
        avg_intra_dist = num.mean(distance_matrix[obj_idx, cluster_members[mask]])

        # Find nearest neighboring cluster
        min_inter_dist = num.inf
        for cid, members in cluster_groups.items():
            if cid == current_cluster:
                continue
            current_dist = num.mean(distance_matrix[obj_idx, members])
            if current_dist < min_inter_dist:
                min_inter_dist = current_dist

        denominator = max(avg_intra_dist, min_inter_dist)
        if denominator > 0:
            scores[obj_idx] = (min_inter_dist - avg_intra_dist) / denominator

    return float(num.mean(scores))


def bcubed_score(ground_truth, predictions):
    '''
    Calculate B-Cubed F1 score for clustering results
    ground_truth: true cluster assignments
    predictions: predicted cluster assignments
    returns: harmonic mean of precision and recall
    '''
    gt = num.array(ground_truth).flatten()
    pred = num.array(predictions).flatten()

    if gt.size == 0 or pred.size == 0:
        raise ValueError("Input arrays cannot be empty")
    if gt.shape != pred.shape:
        raise ValueError("Arrays must have identical dimensions")

    gt_matrix = gt[:, num.newaxis] == gt
    pred_matrix = pred[:, num.newaxis] == pred

    correct_matches = gt_matrix & pred_matrix

    pred_counts = pred_matrix.sum(1)
    gt_counts = gt_matrix.sum(1)

    correct_counts = correct_matches.sum(1)

    prec = num.zeros_like(correct_counts, dtype=float)
    mask = pred_counts > 0
    prec[mask] = correct_counts[mask] / pred_counts[mask]

    rec = num.zeros_like(correct_counts, dtype=float)
    mask = gt_counts > 0
    rec[mask] = correct_counts[mask] / gt_counts[mask]

    avg_prec = num.mean(prec)
    avg_rec = num.mean(rec)

    if avg_prec + avg_rec == 0:
        return 0.0

    return 2 * (avg_prec * avg_rec) / (avg_prec + avg_rec)

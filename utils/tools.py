import os
import csv
import json
import copy
import random
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from hdbscan import HDBSCAN
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
import gc
from sentence_transformers import SentenceTransformer, util
import torch.backends.cudnn as cudnn
from sklearn.mixture import GaussianMixture

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    size = min(y_pred.size, y_true.size)
    for i in range(size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred, known_lab):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    ind_map = {j: i for i, j in ind}
    
    old_acc = 0
    total_old_instances = 0
    for i in known_lab:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances
    
    new_acc = 0
    total_new_instances = 0
    for i in range(len(np.unique(y_true))):
        if i not in known_lab:
            new_acc += w[ind_map[i], i]
            total_new_instances += sum(w[:, i])
    if total_new_instances == 0:
        new_acc = 0
    else:
        new_acc /= total_new_instances
    return (round(acc*100, 2), round(old_acc*100, 2), round(new_acc*100, 2))

def clustering_score(y_true, y_pred, known_lab, evaluate_head_middle_tail=True):
    Acc, Known, Novel = clustering_accuracy_score(y_true, y_pred, known_lab)
    result = {
            'Acc': Acc,
            'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
            'H-Score': round(2 * Known * Novel / (Known + Novel), 2),
            'Known': Known,
            'Novel': Novel
            }
    
    if evaluate_head_middle_tail:
        print('Evaluating head/middle/tail...')
        # Get class frequencies in y_true
        unique_classes, class_counts = np.unique(y_true, return_counts=True)
        
        # Sort classes by frequency (descending order - most frequent first)
        sorted_indices = np.argsort(class_counts)[::-1]
        sorted_classes = unique_classes[sorted_indices]
        
        # Split into thirds
        n_classes = len(sorted_classes)
        head_size = n_classes // 3
        middle_size = n_classes // 3
        tail_size = n_classes - head_size - middle_size  # Handle remainder
        
        head_classes = set(sorted_classes[:head_size])
        middle_classes = set(sorted_classes[head_size:head_size + middle_size])
        tail_classes = set(sorted_classes[head_size + middle_size:])
        
        # Get alignment mapping (same as used in clustering_accuracy_score)
        ind, w = hungray_aligment(y_true, y_pred)
        
        # Calculate metrics for each group
        for group_name, group_classes in [('head', head_classes), ('middle', middle_classes), ('tail', tail_classes)]:
            if len(group_classes) > 0:
                # Get indices of samples belonging to this group
                group_mask = np.isin(y_true, list(group_classes))
                
                if np.sum(group_mask) > 0:
                    y_true_group = y_true[group_mask]
                    y_pred_group = y_pred[group_mask]
                    
                    # Calculate accuracy using the same Hungarian alignment method
                    group_total = len(y_true_group)
                    group_correct = 0
                    for i, j in ind:
                        # Count how many instances in this group have pred=i and true=j
                        group_correct += np.sum((y_pred_group == i) & (y_true_group == j))
                    
                    acc_group = (group_correct / group_total) * 100 if group_total > 0 else 0
                    
                    # Calculate NMI and ARI for the subgroup (these are permutation-invariant)
                    nmi_group = normalized_mutual_info_score(y_true_group, y_pred_group) * 100
                    ari_group = adjusted_rand_score(y_true_group, y_pred_group) * 100
                    
                    result[f'Acc_{group_name}'] = round(acc_group, 2)
                    result[f'NMI_{group_name}'] = round(nmi_group, 2)
                    result[f'ARI_{group_name}'] = round(ari_group, 2)
                else:
                    result[f'Acc_{group_name}'] = 0.0
                    result[f'NMI_{group_name}'] = 0.0
                    result[f'ARI_{group_name}'] = 0.0
            else:
                result[f'Acc_{group_name}'] = 0.0
                result[f'NMI_{group_name}'] = 0.0
                result[f'ARI_{group_name}'] = 0.0
    
    return result

def mask_tokens(inputs, tokenizer,\
    special_tokens_mask=None, mlm_probability=0.15):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs==0)] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class view_generator:
    def __init__(self, tokenizer, rtr_prob, seed):
        set_seed(seed)
        self.tokenizer = tokenizer
        self.rtr_prob = rtr_prob
    
    def random_token_replace(self, ids):
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        ids, _ = mask_tokens(ids, self.tokenizer, mlm_probability=0.25)
        random_words = torch.randint(len(self.tokenizer), ids.shape, dtype=torch.long)
        indices_replaced = torch.where(ids == mask_id)
        ids[indices_replaced] = random_words[indices_replaced]
        return ids

    def shuffle_tokens(self, ids):
        view_pos = []
        for inp in torch.unbind(ids):
            new_ids = copy.deepcopy(inp)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(inp, already_has_special_tokens=True)
            sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]
            inds = np.arange(len(sent_tokens_inds))
            np.random.shuffle(inds)
            shuffled_inds = sent_tokens_inds[inds]
            inp[sent_tokens_inds] = new_ids[shuffled_inds]
            view_pos.append(new_ids)
        view_pos = torch.stack(view_pos, dim=0)
        return view_pos

def measure_interpretability(predictions, references, args):
    ## Compute Similarity Matrix
    # Load the Sentence Transformer model with error handling
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        print("Attempting to fix URL scheme issue...")
        
        # Try to fix the URL scheme issue by setting environment variables
        import os
        os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
        os.environ['HF_HUB_URL'] = 'https://huggingface.co'
        
        try:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            print("Using fallback approach - skipping semantic similarity computation")
            # Return default values as fallback
            K = len(references)
            coverage_score = 0.5  # Default fallback value
            uniformity_score = 0.5  # Default fallback value
            semantic_matching_score = 0.5  # Default fallback value
            informativeness_score = 0.25  # Default fallback value
            
            print('Fallback Interpretability Scores:')
            print('Coverage Score: ', coverage_score)
            print('Uniformity Score: ', uniformity_score)
            print('Semantic Matching Score: ', semantic_matching_score)
            print('Informativeness Score: ', informativeness_score)
            
            # Save the fallback scores
            interpretability_scores = {'Coverage Score': coverage_score, 'Uniformity Score': uniformity_score, 'Semantic Matching Score': semantic_matching_score, 'Informativeness Score': informativeness_score}
            
            save_results_path = './analysis/interpretability'
            file_name = f'interpretability_score_{args.experiment_name}.csv'
            results_path = os.path.join(save_results_path, file_name)

            if not os.path.exists(save_results_path):
                os.makedirs(save_results_path)

            # Handle missing arguments gracefully
            names = ['interpret_sampling_strategy', 'interpret_num_representatives', 'llm', 'label_setting', 'labeled_ratio', 'labeled_shot', 'known_cls_ratio', 'evaluation_epoch', 'experiment_name', 'running_method']
            var = []
            for name in names:
                if hasattr(args, name):
                    var.append(getattr(args, name))
                else:
                    var.append('N/A')  # Default value for missing attributes
            
            print('Key Hyperparameters and Values:')
            for i in range(len(names)):
                print(names[i], ':', var[i])
            vars_dict = {k:v for k,v in zip(names, var)}
            results = dict(interpretability_scores,**vars_dict)
            keys = list(results.keys())
            values = list(results.values())

            if not os.path.exists(results_path):
                ori = []
                ori.append(values)
                df1 = pd.DataFrame(ori,columns = keys)
                df1.to_csv(results_path,index=False)
            else:
                df1 = pd.read_csv(results_path)
                new = pd.DataFrame(results,index=[1])
                # df1 = df1.append(new,ignore_index=True)
                df1 = pd.concat([df1, new], ignore_index=True)
                df1.to_csv(results_path,index=False)
            
            print('Fallback Interpretability Scores Saved to ', results_path)
            return coverage_score, uniformity_score, semantic_matching_score, informativeness_score

    # Compute embeddings for both lists of sentences
    prediction_embeddings = model.encode(predictions, convert_to_tensor=True)
    reference_embeddings = model.encode(references, convert_to_tensor=True)

    # Compute pairwise cosine similarity
    similarity_matrix = util.pytorch_cos_sim(prediction_embeddings, reference_embeddings)

    # Convert similarity matrix to a numpy array for easier handling if necessary
    similarity_matrix = similarity_matrix.cpu().numpy()

    # Cleanup to free GPU memory
    del model, prediction_embeddings, reference_embeddings
    torch.cuda.empty_cache()
    gc.collect()

    # return the index of the maximum value in each row
    max_indices = np.argmax(similarity_matrix, axis=1).tolist()
    print('Similarity Matrix: ', similarity_matrix)
    print('Matching Index: ', max_indices)

    ## Compute Diverse Metrics
    K = len(references)
    # Coverage Score: The percentage of unique items in the list
    coverage_score = len(set(max_indices)) / K
    print('Coverage Score: ', coverage_score)

    # Uniformity Score: how evenly the list covers all the items, max score is 1
    counts = [max_indices.count(i) for i in range(K)] # num of times each cateogry are mapped to
    ratio = [count / len(max_indices) for count in counts] # ratio of each category are mapped to
    uniformity_score = -sum([r * np.log(r) for r in ratio if r > 0]) / np.log(K)  # calculate entropy, only for non-zero ratios
    print('Uniformity Score: ', uniformity_score)

    # Semantic Matching Score: how well the list matches the reference list in terms of semantic similarity
    max_scores = np.max(similarity_matrix, axis=1)
    semantic_matching_score = np.mean(max_scores)
    print('Semantic Matching Score: ', semantic_matching_score)

    # Informativeness Score: consider both the semantic matching score and the uniformity score
    informativeness_score = semantic_matching_score * uniformity_score
    print('Informativeness Score: ', informativeness_score)

    print('\n### More Detailed Interpretability Results ###')
    print(f'[#Unique Mapped Categories/#Total Categories]: [{len(set(max_indices))}/{K}]')
    print('Unique Mapped Categories: ', set(max_indices))
    print('Counts of Each Mapped Categories: ', counts)
    
    # Show some good and bad cases: good cases: top 5 highest similarity scores, bad cases: top 5 lowest similarity scores
    top_k = 10
    print('Top K Highest Similarity Scores and References and Corresponding Predictions: ')
    top_k_indices, top_k_scores = zip(*sorted(enumerate(max_scores), key=lambda x: x[1], reverse=True)[:top_k])
    for i, (index, score) in enumerate(zip(top_k_indices, top_k_scores)):
        print(f'{i+1}. Similarity Score: {score:.3f}, Reference: {references[max_indices[index]]}, Prediction: {predictions[index]}')
    print('Top K Lowest Similarity Scores and References and Corresponding Predictions: ')
    bottom_k_indices, bottom_k_scores = zip(*sorted(enumerate(max_scores), key=lambda x: x[1], reverse=False)[:top_k])
    for i, (index, score) in enumerate(zip(bottom_k_indices, bottom_k_scores)):
        print(f'{i+1}. Similarity Score: {score:.3f}, Reference: {references[max_indices[index]]}, Prediction: {predictions[index]}')

    # Save the interpretability scores
    interpretability_scores = {'Coverage Score': coverage_score, 'Uniformity Score': uniformity_score, 'Semantic Matching Score': semantic_matching_score, 'Informativeness Score': informativeness_score}

    save_results_path = './analysis/interpretability'
    file_name = f'interpretability_score_{args.experiment_name}.csv'
    results_path = os.path.join(save_results_path, file_name)

    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)

    names = ['interpret_sampling_strategy', 'interpret_num_representatives', 'llm', 'label_setting', 'labeled_ratio', 'labeled_shot', 'known_cls_ratio', 'evaluation_epoch', 'experiment_name', 'running_method']
    var = []
    for name in names:
        if hasattr(args, name):
            var.append(getattr(args, name))
        else:
            var.append('N/A')  # Default value for missing attributes
    
    print('Key Hyperparameters and Values:')
    for i in range(len(names)):
        print(names[i], ':', var[i])
    vars_dict = {k:v for k,v in zip(names, var)}
    results = dict(interpretability_scores,**vars_dict)
    keys = list(results.keys())
    values = list(results.values())

    if not os.path.exists(results_path):
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori,columns = keys)
        df1.to_csv(results_path,index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results,index=[1])
        # df1 = df1.append(new,ignore_index=True)
        df1 = pd.concat([df1, new], ignore_index=True)
        df1.to_csv(results_path,index=False)
    
    print('Interpretability Scores Saved to ', results_path)

    return coverage_score, uniformity_score, semantic_matching_score, informativeness_score

def perform_clustering(features, n_clusters, algorithm='kmeans', random_state=42, **kwargs):
    """
    Perform clustering using different algorithms while maintaining compatibility with cluster_centers_.
    
    Args:
        features: Input features to cluster
        n_clusters: Number of clusters
        algorithm: Clustering algorithm ('kmeans', 'hdbscan', 'spectral', 'gmm')
        random_state: Random seed
        **kwargs: Additional arguments for specific algorithms
    
    Returns:
        clustering_result: Object with labels_ and cluster_centers_ attributes
    """
    print(f"Performing clustering with {algorithm} algorithm")

    if algorithm == 'kmeans':
        clustering_result = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs).fit(features)
    
    elif algorithm == 'hdbscan':
        # HDBSCAN doesn't require n_clusters, but we can use min_cluster_size to control
        min_cluster_size = kwargs.get('min_cluster_size', max(2, len(features) // (n_clusters * 10)))
        clustering_result = HDBSCAN(min_cluster_size=min_cluster_size, **kwargs).fit(features)
        
        # HDBSCAN may not find exactly n_clusters, so we need to handle this
        unique_labels = np.unique(clustering_result.labels_)
        if -1 in unique_labels:  # -1 indicates noise points in HDBSCAN
            unique_labels = unique_labels[unique_labels != -1]
        
        # If we have fewer clusters than expected, assign noise points to nearest clusters
        if len(unique_labels) < n_clusters:
            print(f"Warning: HDBSCAN found {len(unique_labels)} clusters instead of {n_clusters}")
            # For noise points, assign to nearest cluster center
            if -1 in clustering_result.labels_:
                noise_indices = np.where(clustering_result.labels_ == -1)[0]
                if len(unique_labels) > 0:
                    # Calculate cluster centers for existing clusters
                    cluster_centers = []
                    for label in unique_labels:
                        cluster_points = features[clustering_result.labels_ == label]
                        cluster_centers.append(np.mean(cluster_points, axis=0))
                    cluster_centers = np.array(cluster_centers, dtype=features.dtype)
                    
                    # Assign noise points to nearest cluster
                    for noise_idx in noise_indices:
                        distances = np.linalg.norm(cluster_centers - features[noise_idx], axis=1)
                        nearest_cluster = unique_labels[np.argmin(distances)]
                        clustering_result.labels_[noise_idx] = nearest_cluster
        
        # Ensure we have exactly n_clusters by potentially merging or splitting
        unique_labels = np.unique(clustering_result.labels_)
        if len(unique_labels) != n_clusters:
            # Simple approach: reassign labels to have exactly n_clusters
            # This is a simplified solution - in practice, you might want more sophisticated handling
            kmeans_fallback = KMeans(n_clusters=n_clusters, random_state=random_state).fit(features)
            clustering_result.labels_ = kmeans_fallback.labels_
            clustering_result.cluster_centers_ = kmeans_fallback.cluster_centers_
        else:
            # Calculate cluster centers for HDBSCAN
            cluster_centers = []
            for label in unique_labels:
                cluster_points = features[clustering_result.labels_ == label]
                cluster_centers.append(np.mean(cluster_points, axis=0))
            clustering_result.cluster_centers_ = np.array(cluster_centers)
    
    elif algorithm == 'spectral':
        clustering_result = SpectralClustering(n_clusters=n_clusters, random_state=random_state, **kwargs).fit(features)
        
        # SpectralClustering doesn't have cluster_centers_, so we need to calculate them
        unique_labels = np.unique(clustering_result.labels_)
        cluster_centers = []
        for label in unique_labels:
            cluster_points = features[clustering_result.labels_ == label]
            cluster_centers.append(np.mean(cluster_points, axis=0))
        clustering_result.cluster_centers_ = np.array(cluster_centers)
    
    elif algorithm == 'gmm':
        # Gaussian Mixture Model clustering
        # GMM provides means_ which we can use as cluster centers
        clustering_result = GaussianMixture(n_components=n_clusters, random_state=random_state, **kwargs).fit(features)
        
        # GMM doesn't have labels_ by default, so we need to predict them
        clustering_result.labels_ = clustering_result.predict(features)
        
        # GMM has means_ attribute which serves as cluster centers
        # Ensure the cluster centers have the same dtype as the input features
        clustering_result.cluster_centers_ = clustering_result.means_.astype(features.dtype)
    
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    
    print(f"Clustering Finished")
    return clustering_result
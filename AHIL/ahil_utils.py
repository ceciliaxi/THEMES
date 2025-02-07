import numpy as np
import itertools


# Split the sub-trajectories based on clustering results
def ClusterPartitionSubTrajs(df_list, all_feats, pred_labs_list, clus_num=6):
    clus_demos = {i: [] for i in range(clus_num)} # Recording the subtrajectories belonging to each cluster
    clus_returns = {i: [] for i in range(clus_num)} # Recording the returns belonging to each cluster
    orig_traj_dict, all_groups_split = {}, []

    # Collect the subtrajectories
    sub_demos = {'trajs': [], 'returns': []}
    sub_demos_num = 0 # Subtrajectory number
    
    sub_demos_lab = [] # Labels (actions) for subtrajectories
    for idx in range(len(pred_labs_list)): # For each trajectory
        # --------------------------------------------------------
        # Group the partitioned trajectory (cluster index, occurance times)
        lst = pred_labs_list[idx]
        groups = [(k, sum(1 for _ in g)) for k, g in itertools.groupby(lst)] 
        all_groups_split.append(groups)

        # Get the split index and the corresponding cluster labels
        split_idx = np.cumsum([0] + [groups[i][1] for i in range(len(groups))]) # Splitting timestamp index
        clus_labs = [groups[i][0] for i in range(len(groups))] # Aggregating repetative clustering labels

        # --------------------------------------------------------
        # Generate subtrajectory demos
        tmp_demos_lab = [] # Cluster labels for subtrajectories
        comp_idx = split_idx + [len(pred_labs_list)] # Splitting timestamp index (Incorporate the end index)
        inner_demos_num = 0 # Counter for subtrajectories inner the current trajectory
        
        for sub_idx in range(len(comp_idx)-1): 
            start_idx, end_idx = comp_idx[sub_idx], comp_idx[sub_idx+1]
            tmp_clus = clus_labs[sub_idx]
            tmp_demo_df = df_list[idx].iloc[start_idx:end_idx] # Slice the dataframe
                    
            # Get the trajectories with state-action pairs
            tmp_s = [np.array(tmp_demo_df.iloc[k][all_feats]) for k in range(len(tmp_demo_df))]
            tmp_a = [np.array([tmp_demo_df.iloc[k]['Action'].astype('uint32')]) for k in
                     range(len(tmp_demo_df))]
            tmp_trajs = [(tmp_s[k], tmp_a[k]) for k in range(len(tmp_s))]
            clus_demos[tmp_clus].append(tmp_trajs)

            # Check the outcome of the visits (here set as nan)
            tmp_returns = np.nan
            clus_returns[tmp_clus].append(tmp_returns)

            # Combine all demos and returns
            sub_demos['trajs'].append(tmp_trajs)
            sub_demos['returns'].append(tmp_returns)

            # Record the location in originial trajectory 
            orig_traj_dict[sub_demos_num] = (idx, sub_idx, tmp_clus)
            tmp_demos_lab.append(tmp_clus)
            
            sub_demos_num += 1 # Counter for overall subtrajectories
            inner_demos_num += 1 # Counter for subtrajectories inner the current trajectory

        sub_demos_lab.append(tmp_demos_lab) # Cluster labels for all subtrajectories
            
    # Check the number of partitioned trajectories
    print('* Num of Partitioned trajs: ', sum([len(clus_demos[i]) for i in range(clus_num)]))
    print('* Sub-traj for each cluster: ')
    print('  - ', {i: len(clus_returns[i]) for i in range(clus_num)})

    return clus_demos, clus_returns, sub_demos, orig_traj_dict, all_groups_split, sub_demos_lab


# Assigninig the clustering labels to each trajectory
def subTrajectoriesbyCluster(df_list, all_feats, clus_labs, clus_num=6): 
    # Distrubute the clustering results to each trajectory
    all_len = [len(i) for i in df_list]
    tmp_len = [0] + list(np.cumsum(all_len)) # Get the length of each trajectory
    
    split_pred_list = [] # Splitted clustering labels for each trajectory
    for idx in range(len(tmp_len)-1): 
        tmp = clus_labs[tmp_len[idx]:tmp_len[idx+1]]
        split_pred_list.append(tmp)

    # Get the sub-trajectories
    _, _, sub_demos, _, _, sub_demos_lab = ClusterPartitionSubTrajs(df_list, all_feats, split_pred_list, clus_num=clus_num)
    sub_demos_lab = list(itertools.chain.from_iterable(sub_demos_lab))

    return sub_demos, sub_demos_lab
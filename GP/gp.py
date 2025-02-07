import numpy as np
import time

from GP.brd import BayesianRewardDistribution
import sklearn.metrics.pairwise as pair_kernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# trajectory number for each row of data
def trejectory_index(end_episode):
    episode_index = np.zeros((end_episode[-1] + 1, 1), dtype=np.int32)
    j = 0
    for i in range(end_episode[-1] + 1):
        episode_index[i] = j
        if end_episode[j] == i:
            j = j + 1
    return (episode_index)


# transfer features to feature action pairs
def prepare_phi(feature_state, action):
    possible_actions = np.amax(action) + 1
    state_shape = feature_state.shape
    phi_dim = (state_shape[0], possible_actions * state_shape[1])
    phi = np.zeros(phi_dim)
    phi_list = [np.zeros(phi_dim) for _ in range(possible_actions)]
    dim = state_shape[1]
    for i, state_value in enumerate(feature_state):
        # action_state = np.asscalar(action[i])
        action_state = action[i].item()
        phi[i, action_state * dim: (action_state + 1) * dim] = state_value
    for j in range(possible_actions):
        phi_list[j][:, j * dim: (j + 1) * dim] = feature_state

    return phi, phi_list


# D is a matrix from episode --> states
def d_matrix(episode_index):
    r = 0
    L1 = len(np.unique(episode_index))
    L2 = len(episode_index)
    D = np.zeros((L1, L2))
    for c in range(L2 - 1):
        D[r, c] = 1
        if episode_index[c + 1] != episode_index[c]:
            r = r + 1
    D[L1 - 1, L2 - 1] = 1
    return (D)


# Estimate the delayed reward by inferred immediate reward
def delayed_reward_episode(e_r_R, episode_index):
    delayed_reward_estimate = []
    for i in range(episode_index[-1][0] + 1):
        delayed_reward_estimate.append(np.sum(e_r_R[episode_index == i]))
    return np.expand_dims(np.asarray(delayed_reward_estimate), axis=1)


def gp_rewards(df, id_feat, sel_feats, GP_reward_path, learn_gp=True):
    if learn_gp: 
        # Infer rewards on the EHR dataset
        Croos_valid_fold = 5

        # Initialization
        reward = np.array(df.reward)

        # Remove the end state row from data
        size_traj = df.groupby(id_feat).size().tolist()
        sample_index = []
        sample_start = 0
        sample_end = -1
        for s in size_traj:
            sample_end += s
            sample_index.extend(list(range(sample_start, sample_end)))
            sample_start = sample_end + 1

        # calculate delayed rewards
        reward_index = sorted(list(set(df.index) - set(sample_index)))
        delayed_reward = np.expand_dims(reward[reward_index], axis=1)

        # Slice the procedural states data without the final state
        pro_df = df.loc[sample_index, :]
        pro_df = pro_df.reset_index(drop=True)
        action = np.array(pro_df.Action)

        # get end_episodes
        pro_traj = pro_df.groupby(id_feat).size().tolist()
        userID_index = []
        end_idx = -1
        for s in pro_traj:
            end_idx += s
            userID_index.append(end_idx)
        end_episode = sorted(list(set(userID_index)))

        feature_state = np.array(pro_df[sel_feats])
        pyi_len = len(pro_df)
        val_act = pro_df.Action.unique()

        # transfer features to feature action pairs
        print("Prepare phi")
        phi, phi_list = prepare_phi(feature_state, action)

        # trajectory number for each row of data
        print("Episode_index")
        episode_index = trejectory_index(end_episode)

        # build D matrix
        D_transform = np.float32(d_matrix(episode_index))

        # grid search hyper parameters
        H_P_length_scale_search = [20] #[1000, 100, 10, 1, 0.1, 0.01, 0.001]
        H_P_sigma_search = [1] #[5.0e-2, 1.0e-1, 1.0, 1.0e+1]
        error_value = np.zeros((len(H_P_length_scale_search), len(H_P_sigma_search)))
        for index_length_scale, value_length_scale in enumerate(H_P_length_scale_search):
            for index_sigma, value_sigma in enumerate(H_P_sigma_search):
                # build cross validation folds
                chunk_fold = np.array_split(range(episode_index[-1][0] + 1), Croos_valid_fold)
                cross_distributed_rewards_list = []
                for ii in range(Croos_valid_fold):
                    # training set
                    training_episode_index = np.concatenate([x for k, x in enumerate(chunk_fold) if k != ii], axis=0)
                    training_sample_index = np.where(episode_index == training_episode_index)[0]
                    # test set
                    testing_episode_index = np.array([x for k, x in enumerate(chunk_fold) if k == ii])
                    testing_sample_index = np.where(episode_index == testing_episode_index)[0]

                    # intiate the BRD class
                    # build kernel (It can be a sklearn class or a numpy kernel)
                    # brd_kernel = RBF(length_scale=value_length_scale)
                    # brd_class = BayesianRewardDistribution(kernel=brd_kernel, alpha=value_sigma)
                    t_kernel = time.time()
                    Cr_fit = pair_kernel.pairwise_kernels(X=np.float32(phi[training_sample_index]), metric='rbf',
                                                        gamma=1.0 / (2.0 * (value_length_scale ** 2)))
        #                 print("building kernel for fitting took", time.time() - t_kernel, "seconds")

                    brd_class = BayesianRewardDistribution(kernel=Cr_fit, alpha=value_sigma)

                    # fit the model
                    t_fit = time.time()
                    brd_class.fit(phi[training_sample_index], delayed_reward[training_episode_index],
                                D_transform[training_episode_index][:, training_sample_index])
        #                 print("Fitting BRD took", time.time() - t_fit, "seconds")

                    # predict test data

                    # distributed_rewards_temp = brd_class.predict(phi[testing_sample_index], D_transform[training_episode_index][:, training_sample_index], K=brd_kernel)
                    # build the kernel
                    t_kernel = time.time()
                    Cr_predict = pair_kernel.pairwise_kernels(X=np.float32(phi[testing_sample_index]),
                                                            Y=np.float32(phi[training_sample_index]), metric='rbf',
                                                            gamma=1.0 / (2.0 * (value_length_scale ** 2)))
        #                 print("building kernel for prediction took", time.time() - t_kernel, "seconds")

                    t_predict = time.time()
                    distributed_rewards_temp = brd_class.predict(phi[testing_sample_index],
                                                                D_transform[training_episode_index][:,
                                                                training_sample_index], K=Cr_predict)
        #                 print("Predicting BRD took", time.time() - t_predict, "seconds")
                    # clean memory
                    brd_class = None
                    Cr_fit = None
                    Cr_predict = None

                    # append immediate rewards
                    cross_distributed_rewards_list.append(distributed_rewards_temp)

                # estimate the error
                cross_distributed_rewards = np.concatenate(cross_distributed_rewards_list)
                delayed_reward_estimate = delayed_reward_episode(cross_distributed_rewards, episode_index)
                dif = delayed_reward_estimate - delayed_reward
                s_dif = np.square(dif)
                error_value[index_length_scale][index_sigma] = np.sum(s_dif)
                print("Error value for length_scale:", index_length_scale, "and sigma: ", index_sigma, "is equal to: ",
                    error_value[index_length_scale][index_sigma])

        print("The cross validation error matrix is as follows: ")
        print(error_value)
        # np.savetxt("error_value.csv", error_value, delimiter=",")

        # find the best hyperparameters
        min_index = np.where(error_value == np.min(error_value))
        length_scale_index = int(min_index[0][0])
        sigma_index = int(min_index[1][0])
        H_P_length = H_P_length_scale_search[length_scale_index]
        H_P_sigma = H_P_sigma_search[sigma_index]

        # build kernel (It can be a sklearn class or a numpy kernel)
        t_kernel = time.time()
        Cr = pair_kernel.pairwise_kernels(np.float32(phi), metric='rbf', gamma=1.0 / (2.0 * (value_length_scale ** 2)))
        #     print("building kernel for final fitting and prediction took", time.time() - t_kernel, "seconds")
        # brd_kernel = RBF(length_scale=H_P_length)

        # intiate the BRD class
        brd_class = BayesianRewardDistribution(kernel=Cr, alpha=H_P_sigma)

        # fit the model
        t_fit = time.time()
        brd_class.fit(phi, delayed_reward, D_transform)
        #     print("Fitting BRD took", time.time() - t_fit, "seconds")

        # Infer rewards
        t_predict = time.time()
        distributed_rewards = brd_class.predict(phi, D_transform, K=Cr)
        #     print("predicting BRD took", time.time() - t_predict, "seconds")

        # np.savetxt("distributed_rewards.csv", distributed_rewards, delimiter=",")
        #     print(distributed_rewards)
        np.save(GP_reward_path, distributed_rewards) 
        
    else:
        distributed_rewards = np.load(GP_reward_path)

    return distributed_rewards
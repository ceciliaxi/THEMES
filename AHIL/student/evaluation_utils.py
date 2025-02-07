import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from sklearn.metrics import jaccard_score

from agent import OAStableAgent
from testing.__head__ import *
from env import CustomizeEnv
import copy


# Make an expert agent
# OAStableAgent --> BaseAgent
def make_agent(config,
) -> OAStableAgent:
    env = config['ENV'] # REVISED HERE: <-- gym.make(config['ENV'])
    trajs_path       = get_trajs_path(config, 'expert')
    algorithm        = config['EXPERT_ALG']
    base_path        = config['BASE_PATH']

    return OAStableAgent(
        env          = env       ,
        trajs_path   = trajs_path,
        algorithm    = algorithm ,
        base_path = base_path,
    )

# Rollout to test the model
# NOTE: Only for Gym environments
def RolloutTest(num_episodes, student, config, n=1, reform_mode='original', avg_patterns='weighted'):
    teacher = make_agent(config)
    teacher.load_pretrained()
    return student.gym_test(config, num_episodes, teacher, n, reform_mode, avg_patterns=avg_patterns)


# Rollout the learned policy
def RolloutToTest(student, config, run_seed=1):
    state_trans_para = config['STATE_TRANSFORM'][config['ENV']]

    if config['ENV'] == 'MountainCar_v2' or config['ENV'] == 'MountainCar_v3':
        config_rollout = copy.deepcopy(config)
        config_rollout['ENV'] = 'MountainCar-v0'

        # Set the parameters to transform the states
        reform_n, reform_mode = state_trans_para['reform_n'], state_trans_para['reform_mode']

    # Rollout the policy in environment
    rollout_result_list = []
    for rollIdx in range(10):
        rollout_result = RolloutTest(config['NUM_TRAJS_VALID'], student, config_rollout, reform_n, reform_mode)
        print("Rollout results for run %s (match_rate, mean_reward, std_reward): \n %s" %
              (run_seed, rollout_result))
        rollout_result_list.append(rollout_result[1])

    print('Avg Rewards: ', np.mean(rollout_result_list))

    return rollout_result_list

# ---------------------------------------------------------------------------------------------------
# Calculate the ACC, Recall, Precision, and F1-score
def ModelEvaluate(real_lab, pre_lab, labels_list, avg_patterns='weighted', verbo=False): # 'weighted'/'micro'/‘macro’
    conf_matrix = confusion_matrix(real_lab, pre_lab, labels=labels_list)
    acc = accuracy_score(real_lab, pre_lab)
    recall = recall_score(real_lab, pre_lab, labels=labels_list, average=avg_patterns)
    precision = precision_score(real_lab, pre_lab, labels=labels_list, average=avg_patterns)
    fscore = f1_score(real_lab, pre_lab, labels=labels_list, average=avg_patterns)

    if verbo:
        print('Performance Measurements:')
        print('Confusion matrix: \n', conf_matrix)
        print('Accuracy: ', acc)
        print('Recall: ', recall)
        print('Precision: ', precision)
        print('F-score: ', fscore)
    return conf_matrix, [acc, recall, precision, fscore]


# Plot the AUC-ROC curve
def PlotAUCROC(y_real, y_prob, n_classes, colors=['blue', 'red', 'green'],
               legends=['0', '1', '2'], verbo=False):
    # Prepare the data
    y_test = np.zeros((len(y_real), np.max(y_real) + 1))
    y_test[np.arange(len(y_real)), y_real] = 1
    y_score = np.array(y_prob)

    # Calculate the AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    print('AUC: ', np.mean(list(roc_auc.values())))

    if verbo:
        # Plot the AUC-ROC curve
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color,
                     label=legends[i] + '(AUC = {1:0.2f})'
                                        ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic for multi-class data')
        plt.legend(loc="lower right")
        plt.show()
    return roc_auc, np.mean(list(roc_auc.values()))


def AUCScore(y_real, y_pred, y_prob, avg_patterns='weighted', verbo=True):
    nrCl = sorted(list(set(y_real).union(set(y_pred))))
    nrCl_dict = {nrCl[i]: i for i in range(len(nrCl))}
    conv_y_real = [nrCl_dict[y_real[i]] for i in range(len(y_real))]
    conv_y_prob = np.array(y_prob)
    conv_y_prob = conv_y_prob[:, nrCl]

    y_binary = Vector2Oonehot(conv_y_real, len(nrCl))
    y_scores = np.array(([list(i) for i in conv_y_prob]))

    if np.shape(y_binary)[1] == 2:
        AUC = roc_auc_score(y_binary, y_scores)
    else:
        AUC = roc_auc_score(y_binary, y_scores, average=avg_patterns)
    # try:
    #     AUC = roc_auc_score(y_binary, y_scores, average=avg_patterns)
    # except ValueError:
    #     AUC = np.nan
    #     pass
    if verbo:
        print('AUC: ', AUC)
    return AUC


# Calculate the average precision score
def APScore(y_real, y_pred, y_prob, avg_patterns='weighted', verbo=True):
    nrCl = sorted(list(set(y_real).union(set(y_pred))))
    nrCl_dict = {nrCl[i]: i for i in range(len(nrCl))}
    conv_y_real = [nrCl_dict[y_real[i]] for i in range(len(y_real))]
    conv_y_prob = np.array(y_prob)
    conv_y_prob = conv_y_prob[:, nrCl]

    y_binary = Vector2Oonehot(conv_y_real, len(nrCl))
    y_scores = np.array(([list(i) for i in conv_y_prob]))

    if np.shape(y_binary)[1] == 2:
        APR = average_precision_score(y_binary, y_scores)
    else:
        APR = average_precision_score(y_binary, y_scores, average=avg_patterns)
    # try:
    #     APR = average_precision_score(y_binary, y_scores, average=avg_patterns)
    # except ValueError:
    #     APR = np.nan
    #     pass
    if verbo:
        print('APR: ', APR)
    return APR


# Calculate the average precision score
# def APScore(y_real, y_prob, n_classes, avg_patterns='weighted', verbo=False):
#     nrCl = sorted(list(set(y_real)))
#     nrCl_dict = {nrCl[i]: i for i in range(len(nrCl))}
#     conv_y_real = [nrCl_dict[y_real[i]] for i in range(len(y_real))]
#
#     y_binary = Vector2Oonehot(conv_y_real, len(nrCl))
#     y_scores = np.array(([list(i) for i in y_prob]))

    # # For each class
    # precision = dict()
    # recall = dict()
    # average_precision = dict()
    # for i in range(n_classes):
    #     precision[i], recall[i], _ = precision_recall_curve(y_binary[:, i], y_scores[:, i])
    #     average_precision[i] = average_precision_score(y_binary[:, i], y_scores[:, i])
    #
    # # A "micro-average": quantifying score on all classes jointly
    # precision["micro"], recall["micro"], _ = precision_recall_curve(y_binary.ravel(),
    #                                                                 y_scores.ravel())
    # average_precision["micro"] = average_precision_score(y_binary, y_scores,
    #                                                    average="micro")
    # print('APR: ', average_precision["micro"])
    #
    # if verbo:
    #     plt.figure()
    #     plt.step(recall['micro'], precision['micro'], where='post')
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.ylim([0.0, 1.05])
    #     plt.xlim([0.0, 1.0])
    #     plt.title(
    #         'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    #             .format(average_precision["micro"]))
    # return average_precision["micro"]


# Calculate the Jaccard Score
def JaccardScore(y_real, y_pred, avg_patterns='weighted'):
    jscore = jaccard_score(y_real, y_pred, average=avg_patterns)
    if avg_patterns == None:
        jscore = np.mean(jscore)

    print('Jaccard Score: ', jscore)
    return jscore


# Convert a vector to onehot
def Vector2Oonehot(y, n_classes):
    onehot_y = np.zeros((len(y), n_classes))
    for i in range(len(y)):
        onehot_y[i, y[i]] = 1
    return onehot_y


# Check the number of parameters in a model
def CheckModelParaNum(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params+=param
    return total_params

# Calculate the BIC & AIC
def postCalculateBIC(model_list, mod_lle, test_demos, weight=1):
    # Number of parameters
    non_zero_params = sum([CheckModelParaNum(model_list[i].qvalue_function) for i in range(len(model_list))])

    # Number of data points
    T = sum([len(test_demos[i]) for i in range(len(test_demos))])

    # Definition of BIC:
    # BIC = kln(n)-2ln(L)
    # - L: maximized llh given the model M, L=p(x|theta,M)
    # - X: observed data
    # - n: number of data points
    # - k: number of parameters
    # AIC = 2ln(n)-2ln(L)

    BIC = weight * non_zero_params * np.log(T) - 2 * mod_lle
    AIC = 2 * weight * non_zero_params - 2 * mod_lle

    return BIC, AIC, [non_zero_params, np.log(T), 2 * mod_lle]


# ---------------------------------------------------------------------------------------------------
# # Save the roll-out results to file
def saveTestResults(
        results_file_path: str,
        repetition_num: int,
        metrics: float
):
    with open(results_file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(tuple(list(metrics)))
        # print(results_file_path)


# Check all saved metrics
def CheckSavedMetrics(results_dir, config):
    results_trial = []
    for run_seed in range(config['NUM_REPETITIONS']):
        tmp_path = results_dir + 'repeat_' + str(run_seed) + '_' + 'results.csv'
        if os.path.exists(tmp_path):
            tmp_result = pd.read_csv(tmp_path, header=None).values
        else:
            tmp_result = []
        results_trial.append(tmp_result)
    return results_trial


# Plot the convergence of loss
def PlotLossConvergence(config, TRIAL, all_losses, loss_name):
    [loss_list, loss_pi_list, loss_rho_list] = all_losses
    plt.figure()
    plt.plot(eval(loss_name + '_list'))
    plt.savefig('testing/results/' + config['ENV'] + '/' + str(config['NUM_TRAJS_GIVEN']) + '/EDMStudent/trial_' +
                str(TRIAL) + '_' + loss_name + '.png')

# Check the convergence of algorithm
def CheckLossConvergence(config, TRIAL, all_losses):
    PlotLossConvergence(config, TRIAL, all_losses, 'loss')
    PlotLossConvergence(config, TRIAL, all_losses, 'loss_pi')
    PlotLossConvergence(config, TRIAL, all_losses, 'loss_rho')


# Check actions in one roll-out
def CheckRolloutActions(config, student, state_trans_para):
    if config['ENV'] in config['GYM_ENV'] + ['MountainCar_v2']:
        reform_n, reform_mode = state_trans_para['reform_n'], state_trans_para['reform_mode']
        # Check the percentage of actions
        tmp_actions = [tmp[1].item() for tmp in student.rollout_reformfeat(config, reform_n, reform_mode)[0]]
        print('Unique actions: ', np.unique(tmp_actions, return_counts=True))


# ---------------------------------------------------------------------------------------------------
# REVISED HERE:
# Convert the current state to another format ('original'/'consecutive'/'skipping')
def state_converter(state_list: np.ndarray,
                     count: int,
                     n: int=1,
                     reform_mode: str='original') -> np.ndarray:
    # Take the current state
    if reform_mode == 'original' or reform_mode == 'none':
        convert_state = state_list[-1]
    # Concatenate n consecutive states as current state
    elif reform_mode == 'consecutive':
        if count == 0:
            tmp_state = [state_list[0]] * n
            convert_state = np.concatenate(tmp_state)
        elif count < n:
            tmp_state = [state_list[-count-1]] * (n-count) + state_list[-count:]
            convert_state = np.concatenate(tmp_state)
        else:
            tmp_state = state_list[-n:]
            convert_state = np.concatenate(tmp_state)
    # Concatenate the current state with the state n steps before
    elif reform_mode == 'skipping':
        if count == 0:
            tmp_state = [state_list[0]] * 2
            convert_state = np.concatenate(tmp_state)
        elif count < n:
            tmp_state = [state_list[0], state_list[-1]]
            convert_state = np.concatenate(tmp_state)
        else:
            tmp_state = [state_list[-n-1], state_list[-1]]
            convert_state = np.concatenate(tmp_state)

    return convert_state



def GetFeatLabs(trajs):
    tmp_features = [[list(trajs[j][i][0]) for i in range(len(trajs[j]))] for j in range(len(trajs))]
    tmp_features = np.concatenate(tmp_features, axis=0)

    tmp_actions = [[list(trajs[j][i][1]) for i in range(len(trajs[j]))] for j in range(len(trajs))]
    tmp_actions = np.concatenate(tmp_actions, axis=0)
    tmp_actions = [i[0] for i in tmp_actions]

    return tmp_features, tmp_actions




# ---------------------------------------------------------------------------------------
# import torch
# from torch.nn.utils.rnn import pad_sequence
#
#
# def evaluate_offline(policy, data, qvalue=None):
#     """
#     Evaluate a policy offline with importance sampling.
#
#     :param policy: Policy
#     :param data: Dataset
#     :param qvalue: QValue
#     """
#
#     def dataset2episodes(X, pad):
#         """
#         :param X: torch.Tensor (len(data), ...)
#         :param pad: padding value
#         :returns: torch.Tensor (num_episodes, max_episode_length, ...)
#         """
#         X = torch.split(X, data.episode_lengths)
#         X = pad_sequence(X, padding_value=pad)
#         return X
#
#     estimators = {}
#
#     evaluation_prob = policy.action_prob(data.state, data.action)
#     ratio = evaluation_prob / data.behavior_prob
#
#     reward = dataset2episodes(data.reward, pad=0)
#     discount = torch.tensor([data.discount ** t
#                              for t in range(reward.size(0))]).view(-1, 1)
#     discounted_reward = reward * discount
#
#     ratio_IS = dataset2episodes(ratio, pad=1)
#     ratio_IS = torch.prod(ratio_IS, dim=0) + 1e-45
#     ep_IS = ratio_IS * torch.sum(discounted_reward, dim=0)
#     IS = ep_IS.mean()
#     WIS = ep_IS.sum() / ratio_IS.sum()
#
#     ratio_PDIS = dataset2episodes(ratio, pad=0)
#     ratio_PDIS = torch.cumprod(ratio_PDIS, dim=0) + 1e-45
#     ep_PDIS = (ratio_PDIS * discounted_reward).sum(dim=0)
#     PDIS = ep_PDIS.mean()
#     weighted_ratio_PDIS = ratio_PDIS / ratio_PDIS.sum(dim=-1, keepdim=True)
#     WPDIS = (weighted_ratio_PDIS * discounted_reward).sum()
#
#     estimators = {"IS": IS.item(),
#                   "WIS": WIS.item(),
#                   "PDIS": PDIS.item(),
#                   "WPDIS": WPDIS.item()}
#
#     if qvalue is not None:
#         Qs = qvalue.Qs(data.state)
#         Q = Qs.gather(1, data.action.view(-1, 1)).view(-1)
#         Qs = dataset2episodes(Qs, pad=0)
#         Q = dataset2episodes(Q, pad=0)
#
#         probs = policy.action_probs(data.state)
#         probs = dataset2episodes(probs, pad=0)
#
#         ep_direct = (Qs[0] * probs[0]).sum(dim=-1)
#         direct = ep_direct.mean()
#
#         next_Qs = qvalue.Qs(data.next_state)
#         next_Qs = dataset2episodes(next_Qs, pad=0)
#
#         next_probs = policy.action_probs(data.next_state)
#         next_probs = dataset2episodes(next_probs, pad=0)
#
#         next_V = (next_Qs * next_probs).sum(dim=-1)
#
#         correction = reward + data.discount * next_V - Q
#         discounted_correction = correction * discount
#         ep_DR = ep_direct + (ratio_PDIS * discounted_correction).sum(dim=0)
#         DR = ep_DR.mean()
#         WDR = (weighted_ratio_PDIS * discounted_correction).sum()
#
#         estimators.update({"direct": direct.item(),
#                            "DR": DR.item(),
#                            "WDR": WDR.item()})
#
#     return estimators


# ------------------------------------------------------------------------------------------------

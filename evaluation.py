import numpy as np 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

from sklearn.metrics import average_precision_score
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt


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
    if verbo:
        print('APR: ', APR)
    return APR


# Convert a vector to onehot
def Vector2Oonehot(y, n_classes):
    onehot_y = np.zeros((len(y), n_classes))
    for i in range(len(y)):
        onehot_y[i, y[i]] = 1
    return onehot_y


# Calculate the Jaccard Score
def JaccardScore(y_real, y_pred, avg_patterns='weighted', verbo=True):
    jscore = jaccard_score(y_real, y_pred, average=avg_patterns)
    if avg_patterns == None:
        jscore = np.mean(jscore)
    if verbo:
        print('Jaccard Score: ', jscore)
    return jscore


# Get overall evluation results
def overall_eval(true_lab, pred_lab, pred_prob, act_num, avg_patterns='binary', verbo=True): 
    _, metrics = ModelEvaluate(true_lab, pred_lab, np.arange(act_num),
                               avg_patterns=avg_patterns, verbo=verbo)

    AUC = AUCScore(np.array(true_lab), np.array(pred_lab), np.array(pred_prob), avg_patterns=avg_patterns, verbo=verbo)
    APR = APScore(np.array(true_lab), np.array(pred_lab), np.array(pred_prob), avg_patterns=avg_patterns, verbo=verbo)
    Jascore = JaccardScore(np.array(true_lab), np.array(pred_lab), avg_patterns=avg_patterns, verbo=verbo)
    metrics.extend([AUC, APR, Jascore])
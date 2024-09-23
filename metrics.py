import numpy as np
from sklearn.metrics import confusion_matrix, recall_score


def gap(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=['API','Black','Hispanic','White'])
    # metrics_per_class = {}
    tpr, tnr = [], []
    for i in range(len(cm)):
        TP = cm[i, i]
        FP = sum(cm[:, i]) - TP
        FN = sum(cm[i, :]) - TP
        TN = sum(cm.sum(axis=1)) - TP - FP - FN
        TPR = TP / float(TP + FN) if (TP + FN) > 0 else 0
        TNR = TN / float(TN + FP) if (TN + FP) > 0 else 0
        # class_label = le.inverse_transform([i])[0]  # Convert index back to original class label
        # metrics_per_class[class_label] = {'TPR': TPR, 'TNR': TNR}
        tpr.append(TPR)
        tnr.append(TNR)

    temp = (np.array(tpr)+np.array(tnr))*0.5
    # print(temp)
    gap = 0.0
    for i in range(len(temp)-1):
        gap += abs(temp[i]-temp[-1])
    gap /= 3
    print("1-GAP: ", round(1-gap, 4))
    return round(1-gap, 4)

def disparate_impact(y_true, y_pred):
    recalls = recall_score(y_true, y_pred, average=None, labels=['API','Black','Hispanic','White'])
    dis = 0.0
    for i in range(len(recalls)-1):
        dis += recalls[i]/recalls[-1]
    dis /= (len(recalls)-1)
    print('Disparate Impact is:',dis)
    return round(dis, 4)
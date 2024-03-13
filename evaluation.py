import numpy as np
import os
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, classification_report

def evaluate(y_true, y_pred, logger):
    labels = ['API', 'Black', 'Hispanic', 'White']
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    logger.info(f'Accuracy: {accuracy:.3f}')
        
    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels)
    print('\nClassification Report:')
    print(class_report)
    logger.info('Classification Report:')
    logger.info('\n'+class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    print('\nConfusion Matrix:')
    print(conf_matrix)
    logger.info('Confusion Matrix:')
    logger.info('\n'+conf_matrix)


def gap(y_true, y_pred, logger):
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
    logger.info("1-GAP: ", round(1-gap, 4))
    return round(1-gap, 4)

def disparate_impact(y_true, y_pred, logger):
    recalls = recall_score(y_true, y_pred, average=None, labels=['API','Black','Hispanic','White'])
    dis = 0.0
    for i in range(len(recalls)-1):
        dis += recalls[i]/recalls[-1]
    dis /= (len(recalls)-1)
    print('Disparate Impact is:',round(dis, 4))
    logger.info('Disparate Impact is:',round(dis, 4))
    return round(dis, 4)

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
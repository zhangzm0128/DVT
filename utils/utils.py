import torch 
import torch.nn as nn
import numpy as np
from sklearn import metrics
from scipy import interp

def format_runtime(time_gap):
    m, s = divmod(time_gap, 60)
    h, m = divmod(m, 60)
    runtime_str = ''
    if h != 0:
        runtime_str = runtime_str + '{}h'.format(int(h))
        runtime_str = runtime_str + '{}m'.format(int(m))
        runtime_str = runtime_str + '{}s'.format(int(s))
    elif m != 0:
        runtime_str = runtime_str + '{}m'.format(int(m))
        runtime_str = runtime_str + '{}s'.format(int(s))
    else:
        runtime_str = runtime_str + '{:.3}s'.format(s)
    return runtime_str

class ClassMetrics:
    def __init__(self, num_labels=10, average='micro'):   
        self.num_labels = num_labels
        self.average = average
                
    def convert_onehot(self, y):
        if len(np.squeeze(y).shape) == 1:
            return np.eye(self.num_labels)[np.squeeze(y).astype(int)]
        elif len(np.squeeze(y).shape) == 2:
            return np.squeeze(y)
        else:
            raise RuntimeError('Plz check data dim: {}'.format(y.shape))
    
    def convert_norm_label(self, y):
        if len(np.squeeze(y).shape) == 2:
            return np.argmax(y, axis=1)
        elif len(np.squeeze(y).shape) == 1:
            return np.squeeze(y)
        else:
            raise RuntimeError('Plz check data dim: {}'.format(y.shape))
        
    def topk_acc(self, y_true, y_pred, k=1):
        # y_pred is classification prob as one-hot label
        y_true = self.convert_norm_label(y_true)
        y_pred = self.convert_onehot(y_pred)
        return metrics.top_k_accuracy_score(y_true, y_pred, k=k, labels=np.arange(self.num_labels))
    
    def recall(self, y_true, y_pred):
        y_true = self.convert_norm_label(y_true)
        y_pred = self.convert_norm_label(y_pred)
        return metrics.recall_score(y_true, y_pred, average=self.average)
        
    def precision(self, y_true, y_pred):
        y_true = self.convert_norm_label(y_true)
        y_pred = self.convert_norm_label(y_pred)
        return metrics.precision_score(y_true, y_pred, average=self.average)
        
    def f1(self, y_true, y_pred):
        y_true = self.convert_norm_label(y_true)
        y_pred = self.convert_norm_label(y_pred)
        return metrics.f1_score(y_true, y_pred, average=self.average)
        
    def roc_auc_each(self, y_true, y_pred):
        fpr, tpr, auc_score = [], [], []
        for x in range(self.num_labels):
            a, b, _ = metrics.roc_curve(y_true[:, x], y_pred[:, x])
            fpr.append(a)
            tpr.append(b)
            auc_score.append(metrics.auc(a, b))
            
        return fpr, tpr, auc_score
    
    def roc_auc(self, y_true, y_pred):
        y_true = self.convert_onehot(y_true)
        y_pred = self.convert_onehot(y_pred)
        
                
        if self.average == 'micro':
            fpr_avg, tpr_avg, _ = metrics.roc_curve(y_true.ravel(), y_pred.ravel())
            auc_score_avg = metrics.auc(fpr_avg, tpr_avg)
            return auc_score_avg
            
        elif self.average == 'macro':
            fpr, tpr, auc_score = self.roc_auc_each(y_true, y_pred)
            
            fpr_avg = np.unique(np.concatenate([fpr[x] for x in range(self.num_labels)]))
            tpr_avg = np.zeros_like(fpr_avg)    
            
            for x in range(self.num_labels):
                tpr_avg += interp(fpr_avg, fpr[x], tpr[x])    
            tpr_avg = tpr_avg / self.num_labels
            
            auc_score_avg = metrics.auc(fpr_avg, tpr_avg)
            return auc_score_avg
            
        elif self.average == None:
            return self.roc_auc_each(y_true, y_pred)
            
    def set_report_metrics(self, metrics_name):
        self.report_metrics = []
        for x in metrics_name:
            kwargs = {}
            if 'Acc-k' in x:
                k = x.split('Acc-k')[1]
                func = self.topk_acc
                kwargs['k'] = int(k)
            elif "Pre" == x:
                func = self.precision
            elif "Recall" == x:
                func = self.recall
            elif "F1" == x:
                func = self.f1
            elif "AUC" == x:
                func = self.roc_auc
            self.report_metrics.append([func, kwargs])
    
    def report(self, y_true, y_pred, with_metric_name=False):
        report_data = []
        for x in self.report_metrics:
            report_data.append(x[0](y_true, y_pred, **x[1]))
        return report_data
        

            
            
                
            
        
        
        
        
        
        
        
        
        
        
        

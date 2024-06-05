"""

Author: Benedikt Lorch, Martin Benes
Affiliation: Universitaet Innsbruck
"""

from enum import Enum
import numpy as np
from sklearn import metrics


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    name = None

    def __init__(self, fmt=':.5f', summary_type=Summary.AVERAGE):
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def update_vector(self, vals):
        self.sum += np.nansum(vals)
        self.count += vals.shape[0]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(name=self.name, **self.__dict__)


class LossMeter(AverageMeter):
    """Computes and stores the average loss value"""
    name = 'loss'


class MAEMeter(AverageMeter):
    """Computes and stores mean absolute error."""
    name = 'mae'

    def __init__(
        self,
        *args,
        multiplier: int = 1,
        masked: bool = None,
        **kw,
    ):
        super().__init__(*args, **kw)
        self.multiplier = multiplier
        self.masked = masked

    def update(self, y_true, y_pred, mask=None):
        #
        if self.masked is True:
            y_true, y_pred = y_true[mask], y_pred[mask]
        elif self.masked is False:
            y_true, y_pred = y_true[~mask], y_pred[~mask]
        #
        resid = (y_true - y_pred)*self.multiplier
        mae = np.nanmean(np.abs(resid))
        super().update(mae)


class CorrMeter(AverageMeter):
    name = 'corr'

    def update(self, y_true, y_pred, mask):
        # get residuals
        dhat_s = y_pred - y_true

        # compute correlation
        a = {'axis': (1, 2, 3)}
        dhat_s_norm = (
            (
                dhat_s -
                dhat_s.mean(**a)[:, None, None, None]
            ) / dhat_s.std(**a)[:, None, None, None])
        y_pred_norm = (
            (
                y_pred -
                y_pred.mean(**a)[:, None, None, None]
            ) / y_pred.std(**a)[:, None, None, None])
        corr = np.sum(dhat_s_norm * y_pred_norm, **a) / (np.prod(y_pred_norm.shape[1:]) - 1)

        # update with average correlation
        super().update_vector(np.abs(corr))


class WSMeter(AverageMeter):
    name = 'ws'

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def update(self, x, x_hat, alphas):
        # process input
        x = x[:, :, 1:-1, 1:-1] * 255.
        x_hat = x_hat[:, :, 1:-1, 1:-1] * 255.
        x_bar = np.round(x).astype('int') ^ 1

        # compute WS estimate
        # y_true_bar = y_true + np.cos(np.deg2rad(y_true*255*180))/255.
        weights = np.ones_like(x) / np.prod(x.shape[1:])
        betas_hat = np.sum(
            weights * (x - x_bar) * (x - x_hat),
            axis=(1, 2, 3),
        )
        betas_hat = np.clip(betas_hat, 0, None)

        # compute reference
        betas = alphas / 2.

        # update with median absolute error
        beta_mae = np.mean(np.abs(betas_hat - betas))
        super().update(beta_mae)


class PerformanceMeter(object):
    """Abstract class for performance meter."""
    def __init__(self, fmt=':4.3f'):
        # self.title = title
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.y_pred = np.array([])
        self.y_true = np.array([])

    def update(self, y_true, y_pred):
        self.y_pred = np.concatenate((self.y_pred, y_pred))
        self.y_true = np.concatenate((self.y_true, y_true))

    @property
    def avg(self):
        raise NotImplementedError

    def __str__(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(name=self.name, avg=self.avg)

    def to_dict(self):
        return {self.name: self.avg}


class AccuracyMeter(PerformanceMeter):
    name = 'accuracy'

    @property
    def avg(self):
        return np.mean(self.y_pred == self.y_true)


class MisclassificationMeter(PerformanceMeter):
    name = 'misclassification'

    @property
    def avg(self):
        return np.mean(self.y_pred != self.y_true)


class PrecisionMeter(PerformanceMeter):
    name = 'precision'

    @property
    def avg(self):
        return ((self.y_pred == 1) & (self.y_true == 1)).sum() / (self.y_pred == 1).sum()


class RecallMeter(PerformanceMeter):
    name = 'recall'

    @property
    def avg(self):
        return ((self.y_pred == 1) & (self.y_true == 1)).sum() / (self.y_true == 1).sum()


class PEMeter(PerformanceMeter):
    """P_E metric, as used in the steganalysis literature.

    Source: https://github.com/DDELab/deepsteganalysis
    """
    name = 'p_e'

    @property
    def avg(self):
        # ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(
            self.y_true,
            self.y_pred,
            pos_label=1,
            drop_intermediate=False,
        )
        # NaNs in output
        if np.isnan(fpr).any() or np.isnan(tpr).any():
            return np.nan
        # P_E
        P = .5*(fpr + (1-tpr))
        return min(P[P > 0])


class PMD5FPMeter(PerformanceMeter):
    """P_MD metric at 5% FPR."""
    name = 'p_md^5fp'

    @property
    def avg(self):
        # ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(
            self.y_true,
            self.y_pred,
            pos_label=1,
            drop_intermediate=False,
        )
        # get tau at fpr < .05
        tau_idx = np.argmax(fpr > .05)
        if fpr[tau_idx] > .05:
            tau_idx -= 1
        #
        return 1 - tpr[tau_idx]  # missed detection at 5% FP


class PredictionWriter(PerformanceMeter):
    """Writer of prediction scores."""
    def __init__(self):
        super().__init__(fmt=None)

    @property
    def avg(self):
        raise RuntimeError('prediction writer has no value')

    def write(self, path):
        with open(path, 'w') as fp:
            fp.write('y_true,y_pred\n')
            for y_true, y_pred in zip(self.y_true, self.y_pred):
                fp.write(f'{y_true},{y_pred}\n')

    def __str__(self):
        raise RuntimeError('prediction writer cannot be printed')


class AUCMeter(PerformanceMeter):
    name = 'auc'

    @property
    def avg(self):
        return metrics.roc_auc_score(self.y_true, self.y_pred)


class wAUCMeter(PerformanceMeter):
    name = 'wauc'

    @property
    def avg(self):
        # ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(
            self.y_true,
            self.y_pred,
            pos_label=1,
            drop_intermediate=False,
        )
        # NaNs in output
        if np.isnan(fpr).any() or np.isnan(tpr).any():
            return np.nan
        # alpha, for which beta=.4
        idx_beta_p4 = np.argmin(tpr < .4)
        alpha_beta_p4 = fpr[idx_beta_p4]
        # split the ROC
        fprA, tprA = fpr[:idx_beta_p4], tpr[:idx_beta_p4]
        fprB, tprB = fpr[idx_beta_p4:], tpr[idx_beta_p4:]
        # integrate
        aucA = metrics.auc(fprA, tprA)
        aucB = metrics.auc(fprB, tprB)
        # sum and weight
        wauc = aucA * 2 + aucB
        # normalize
        wauc = wauc / (1 + alpha_beta_p4)
        return wauc


# def alaska_weighted_auc(y_true, y_valid):
#     """
#     https://www.kaggle.com/anokas/weighted-auc-metric-updated
#     """
#     tpr_thresholds = [0.0, 0.4, 1.0]
#     weights = [2, 1]

#     fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

#     # size of subsets
#     areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

#     # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
#     normalization = np.dot(areas, weights)

#     competition_metric = 0
#     for idx, weight in enumerate(weights):
#         y_min = tpr_thresholds[idx]
#         y_max = tpr_thresholds[idx + 1]
#         mask = (y_min < tpr) & (tpr < y_max)
#         # pdb.set_trace()

#         try:
#             fpr_start = fpr[mask][-1]
#         except IndexError:
#             fpr_start = 0
#         x_padding = np.linspace(fpr_start, 1, 100)

#         x = np.concatenate([fpr[mask], x_padding])
#         y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
#         y = y - y_min  # normalize such that curve starts at y=0
#         score = metrics.auc(x, y)
#         submetric = score * weight
#         best_subscore = (y_max - y_min) * weight
#         competition_metric += submetric

#     return competition_metric / normalization


# class RocAucMeter(PerformanceMeter):
#     name = 'rocauc'

#     def reset(self):
#         self.y_true = np.array([0, 1])
#         self.y_pred = np.array([0.5, 0.5])
#         self.score = 0

#     def update(self, y_true, y_pred):
#         # old version
#         self.y_true = np.hstack((self.y_true, y_true))
#         self.y_pred = np.hstack((self.y_pred, y_pred))
#         self.score = alaska_weighted_auc(self.y_true, self.y_pred)

#     @property
#     def avg(self):
#         return self.score


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def to_str(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


import timm
import torch


def batch_means(x):
    return torch.mean(x, dim=(1, 2, 3), keepdim=True)


def batch_stds(x):
    return torch.std(x, dim=(1, 2, 3), keepdim=True)


class CustomLoss(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device

    def forward(self, *args, **kw):
        raise NotImplementedError

    def to(self, device, *args, **kw):
        super().to(*args, device, **kw)
        self.device = device
        return self


class L1Loss(torch.nn.Module):
    def _error(self, outputs, covers):
        # AE with cover
        return torch.abs(covers - outputs)

    def forward(self, outputs, targets, *args, **kw):
        covers, _ = targets
        # MAE
        return torch.mean(self._error(outputs, covers))


class L2Loss(L1Loss):
    def _error(self, outputs, covers):
        # SE with cover
        return (covers - outputs)**2


class WSLoss(CustomLoss):
    def _error(self, outputs, inputs, betas):
        # flip LSB
        inputs = inputs * 255.
        outputs = outputs * 255.
        inputs_bar = (torch.round(inputs).int() ^ 1).float()

        # weights
        weights = torch.ones_like(inputs) / (torch.numel(inputs)/float(inputs.size(0)))
        # weights = torch.ones(covers.size(), dtype=torch.float32) / 512**2
        if self.device is not None:
            weights = weights.to(self.device)

        # WS estimate
        betas_hat = torch.sum(
            weights * (inputs - inputs_bar) * (inputs - outputs),
            dim=(1, 2, 3)  # sum each image in the batch
        )
        betas_hat = torch.nn.functional.relu(betas_hat)
        # betas_hat = torch.clip(betas_hat, 0, 1)  # clip to [0, inf]

        # print(weights[0, 0, :5, :5])
        # print(inputs[0, 0, :5, :5])
        # print(inputs_bar[0, 0, :5, :5])
        # print(outputs[0, 0, :5, :5])

        # print(
        #     'res:',
        #     torch.mean(torch.abs(inputs - outputs)),
        #     betas_hat[:6],
        #     # betas2_hat[:6],
        #     betas[:6],
        # )

        # MAE with real change rate
        return torch.abs(betas_hat - betas)

    def forward(self, outputs, targets, inputs):
        _, alphas = targets
        # outputs = outputs[:, :, 1:-1, 1:-1]
        # covers = covers[:, :, 1:-1, 1:-1]
        # mask = mask[:, :, 1:-1, 1:-1]
        betas = alphas / 2.

        return torch.mean(self._error(outputs, inputs, betas))
        # return torch.median(self._error(outputs, covers, betas))


class L1WSLoss(torch.nn.Module):
    def __init__(self):
        super(L1WSLoss, self).__init__()
        self.l1_loss = L1Loss()
        self.ws_loss = WSLoss()

    def forward(self, outputs, targets, inputs):
        # get losses
        prediction_mae = self.l1_loss(outputs, targets)
        ws_mae = self.ws_loss(outputs, targets, inputs)

        # #
        # covers, alphas = targets
        # outputs = outputs[:, :, 1:-1, 1:-1]
        # covers = covers[:, :, 1:-1, 1:-1]
        # mask = mask[:, :, 1:-1, 1:-1]
        # betas = alphas / 2.
        # ws_ae = self.ws_loss._error(outputs, covers, betas)
        # prediction_ae = self.l1_loss._error(outputs, covers)
        # return torch.mean(prediction_ae * ws_ae[:, None, None, None]**self.lmbda)

        # return prediction_mae + ws_mae
        return prediction_mae + ws_mae #+ torch.std(ws_ae)
        # return ((self.lmbda) * prediction_mae + (1-self.lmbda) * ws_mae)*2

    def to(self, device, *args, **kw):
        super().to(*args, device, **kw)
        self.device = device
        return self


# class L2WSLoss(torch.nn.Module):
#     def __init__(self, lmbda):
#         super(L2WSLoss, self).__init__()
#         self.l2_loss = L2Loss()
#         self.ws_loss = WSLoss()
#         self.lmbda = lmbda

#     def forward(self, outputs, targets, mask):
#         #
#         prediction_mse = self.l2_loss(outputs, targets, mask)
#         ws_mae = self.ws_loss(outputs, targets, mask)

#         #
#         covers, alphas = targets
#         # outputs = outputs[:, :, 1:-1, 1:-1]
#         covers = covers[:, :, 1:-1, 1:-1]
#         # mask = mask[:, :, 1:-1, 1:-1]
#         betas = alphas / 2.
#         ws_ae = self.ws_loss._error(outputs, covers, betas)
#         # prediction_ae = self.l1_loss._error(outputs, covers)
#         # return torch.mean(prediction_ae * ws_ae[:, None, None, None]**self.lmbda)

#         # return prediction_mse + ws_mse
#         return ((self.lmbda) * prediction_mse + (1-self.lmbda) * torch.mean(ws_ae**2))*2

#     def to(self, device, *args, **kw):
#         super().to(*args, device, **kw)
#         self.device = device
#         return self


# class L1CorrLoss(torch.nn.Module):
#     def __init__(self):
#         super(L1CorrLoss, self).__init__()
#         self.l1_loss = L2Loss()
#         self.ws_loss = WSLoss()

#     def forward(self, outputs, targets, mask):
#         #
#         prediction_mae = self.l1_loss(outputs, targets, mask)
#         ws_mae = self.ws_loss(outputs, targets, mask)

#         # correlation
#         covers, _ = targets
#         # outputs = outputs[:, :, 1:-1, 1:-1]
#         covers = covers[:, :, 1:-1, 1:-1]
#         noise = outputs - covers
#         noise_norm = (noise - batch_means(noise)) / batch_stds(noise)
#         outputs_norm = (outputs - batch_means(outputs)) / batch_stds(outputs)
#         corr = torch.sum(
#             noise_norm * outputs_norm,
#             dim=(1, 2, 3)
#         ) / (torch.numel(outputs) / outputs.size(0))

#         # return prediction_mse + correlation + ws_mse
#         return (
#             prediction_mae + torch.mean(torch.abs(corr)) + ws_mae
#         )
#         # return ((self.lmbda) * prediction_mse + (1-self.lmbda) * ws_mae)*2

#     def to(self, device, *args, **kw):
#         super().to(*args, device, **kw)
#         self.device = device
#         return self

# class wL1Loss(torch.nn.Module):
#     def __init__(self, lbda, push_together: bool = False):
#         super(wL1Loss, self).__init__()
#         self.lbda = torch.tensor(lbda)
#         self.push_together = push_together

#     def forward(self, outputs, targets, mask):
#         N = torch.numel(outputs)
#         mae_1 = torch.sum(torch.abs((outputs - targets)[mask]))
#         mae_0 = torch.sum(torch.abs((outputs - targets)[~mask]))
#         mae = (self.lbda * mae_1 + (1-self.lbda) * mae_0) / N
#         if self.push_together:
#             mae += torch.abs(mae_0 - mae_1) / N
#         return mae

#     def to(self, device, *args, **kw):
#         super().to(*args, device, **kw)
#         self.lbda = self.lbda.to(device)
#         return self


# class L1CorrLoss(torch.nn.Module):

#     def forward(self, outputs, targets, mask):
#         # L1
#         error = outputs - targets
#         mae = torch.mean(torch.abs(error))

#         # correlation
#         mask = mask.float()
#         mask_norm = (mask - batch_means(mask)) / batch_stds(mask)
#         error_norm = (error - batch_means(error)) / batch_stds(error)
#         corr = torch.sum(
#             mask_norm * error_norm,
#             dim=(1, 2, 3)
#         ) / (torch.numel(outputs) / outputs.size(0))
#         # corr = torch.sum(mask_centered * error_centered, dim=(1, 2, 3)) / mask_std / error_std

#         # MAE + correlation
#         return mae + torch.mean(torch.abs(corr))

# class L1WSLoss(torch.nn.Module):
#     def __init__(self, lbda: float):
#         super(L1WSLoss, self).__init__()
#         self.device = None
#         self.lbda = lbda

#     def forward(self, outputs, targets, mask):
#         covers, alphas = targets
#         outputs = outputs[:, :, 1:-1, 1:-1]# * 255
#         covers = covers[:, :, 1:-1, 1:-1]# * 255
#         mask = mask[:, :, 1:-1, 1:-1]
#         if self.device is not None:
#             alphas = alphas.to(self.device)
#         betas = alphas / 2.

#         # L1
#         errors = covers - outputs
#         outputs_ae = torch.abs(errors) * 255
#         outputs_mae = torch.mean(outputs_ae)
#         # outputs_ae_1 = torch.abs(errors[mask])
#         # outputs_ae_0 = torch.abs(errors[~mask])
#         # wmae = (
#         #     self.lbda * torch.sum(ae_1) +
#         #     (1-self.lbda) * torch.sum(ae_0)
#         # ) / torch.numel(outputs)

#         # # correlation
#         # mask = mask.float()
#         # mask_norm = (mask - batch_means(mask)) / batch_stds(mask)
#         # error_norm = (error - batch_means(error)) / batch_stds(error)
#         # corr = torch.sum(
#         #     mask_norm * error_norm,
#         #     dim=(1, 2, 3)
#         # ) / (torch.numel(outputs) / outputs.size(0))

#         # unweighted WS estimate
#         # covers = covers * 255
#         # outputs = outputs * 255
#         weights = (torch.ones_like(covers) / torch.numel(covers))
#         if self.device is not None:
#             weights = weights.to(self.device)
#         # covers_bar = (torch.round(covers*255).int() ^ 1) / 255.
#         # covers_bar = covers + torch.cos(torch.deg2rad(covers*255*180))/255.
#         covers_bar = covers + torch.cos(torch.deg2rad(covers*180*255))/255.  # flip LSB
#         betas_hat = torch.sum(
#             weights * (covers - covers_bar) * (covers - outputs) * 255**2,
#             dim=(1, 2, 3)  # sum each image in the batch
#         )
#         betas_hat = torch.nn.functional.relu(betas_hat)  # clip to [0,inf]
#         # betas_hat = torch.clip(betas_hat, 0, 1)
#         betas_ae = torch.abs(betas_hat - betas)
#         betas_mae = torch.mean(betas_ae)
#         # betas_sle = (torch.log(betas_hat+1) - torch.log(betas+1))**2
#         # betas_rrmse = torch.sqrt((betas_hat-betas)**2/(betas_hat+.5)**2)

#         # # MAE error
#         # return outputs_mae

#         # # wMAE error
#         # return (
#         #     torch.mean(outputs_ae_1 * betas_ae[mask]) +
#         #     torch.mean(outputs_ae_0 * betas_ae[~mask])
#         # )

#         # # WS error
#         # return betas_mae

#         # MAE + WS error
#         alpha = .1
#         return (
#             (alpha) * outputs_mae +
#             # torch.mean(torch.abs(corr)) +
#             (1-alpha) * betas_mae
#         )

#         # # MAE * WS error
#         # return torch.mean(
#         #     outputs_ae * betas_ae[:, None, None, None]
#         # )

#     def to(self, device, *args, **kw):
#         super().to(*args, device, **kw)
#         self.device = device
#         return self

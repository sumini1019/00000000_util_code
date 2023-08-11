import torch
import torch.nn as nn
import pytorch_lightning as pl

class EfficientNet_hemi_compare_v2_cELVO_FE(nn.Module):
    def __init__(self, eff_weight_path=None, feature_extract=False):
        super(EfficientNet_hemi_compare_v2_cELVO_FE, self).__init__()

        from efficientnet.model import EfficientNet_cELVO_FE
        EfficientNet_cELVO_FE = EfficientNet_cELVO_FE.from_name('efficientnet-b4')
        EfficientNet_cELVO_FE._fc = torch.nn.Linear(1792, 1)

        if eff_weight_path:
            EfficientNet_cELVO_FE.load_state_dict(torch.load(eff_weight_path), strict=True)

        self.EfficientNet_cELVO_FE = EfficientNet_cELVO_FE
        # 2022.11.22 - Slice Number 추가되어, 값 변경 (5376 -> 5377)
        self.SENet = SEBlock(c=5377)    # (c=5376)
        self.dropout = nn.Dropout(0.2)
        self.fc = torch.nn.Linear(1792, 1)
        self.feature_extract = feature_extract

    def forward(self, segment, segment_op, slice_num):
        # Target의 -> Out(Feature에 대한 dropout + FC의 output) / Feature 계산
        out, feature = self.EfficientNet_cELVO_FE(segment)  # (b, f)
        # Op의 -> Out(Feature에 대한 dropout + FC의 output) / Feature 계산
        out_op, feature_op = self.EfficientNet_cELVO_FE(segment_op)  # (b, f)
        # Diff Feature 계산 = Target - Op
        df_feature = feature - feature_op

        cat_feature = torch.cat((feature, feature_op), dim=-1)  # (b, f*2)
        cat_feature = torch.cat((cat_feature, df_feature), dim=-1)  # (b, f*3)
        # 2022.11.22 - Slice Number 정보 추가
        cat_feature = torch.cat((cat_feature, slice_num), dim=-1)   # (b, f*3 + 1)

        se_feature = self.SENet(cat_feature.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (b, f)

        if self.feature_extract:
            return se_feature
        else:
            x = self.dropout(se_feature)
            total_out = self.fc(x)
            return se_feature, out, out_op, total_out

class Solver_FE_v2(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Load pretrained model
        model = EfficientNet_hemi_compare_v2_cELVO_FE(eff_weight_path=None)

        model.to(device)

        # Freeze weights
        freeze = 0
        if freeze > 0:  # 0(not freeze), 1, 2, 3, 4
            for name, module in model.named_modules():
                for num in range(0, freeze * 8):
                    if name == '_blocks.{}'.format(num):
                        for param in module.parameters():
                            param.requires_grad = False

        # Model
        self.model = model

        # Criterion
        self.criterion = PolyBCELoss(reduction='mean')

        # Metric
        self.Accuracy = torchmetrics.Accuracy()
        self.Precision = torchmetrics.Precision()
        self.Sensitivity = torchmetrics.Recall()
        self.Specificity = torchmetrics.Specificity()
        self.AUC = torchmetrics.AUROC(pos_label=1)

    def forward(self, x, x_op, slice_num):
        # 2022.09.27 - LSTM에서는, Feature Extraction만 진행하므로, feature extract 여부 보고, output값 1개로 변경
        if self.model.feature_extract == True:
            feature = self.model(x, x_op, slice_num)
            return feature
        else:
            # 1. se_feature : 반구비교 최종 feature
            #                 (SENet의 output) (SENet의 input은, target_f + op_f + diff_f)
            # 2. pred : target의 output값
            # 3. pred_op : op의 output값
            # 4. pred_total : 반구비교 최종 feature(SENet output) 사용해서, dropout+FC 거친 최종 output
            se_feature, pred, pred_op, pred_total = self.model(x, x_op, slice_num)
            return se_feature, pred, pred_op, pred_total

    def training_step(self, batch, batch_idx):
        x = batch['img']
        x_op = batch['img_op']
        slice_num = batch['slice_num']

        y = batch['annot']
        y_op = batch['annot_op']

        # Model 결과
        # 1. se_feature : 반구비교 최종 feature
        #                 (SENet의 output) (SENet의 input은, target_f + op_f + diff_f)
        # 2. pred : target의 output값
        # 3. pred_op : op의 output값
        # 4. pred_total : 반구비교 최종 feature(SENet output) 사용해서, dropout+FC 거친 최종 output
        se_feature, pred, pred_op, pred_total = self(x, x_op, slice_num)
        loss = self.criterion(pred, y)
        loss_op = self.criterion(pred_op, y_op)
        loss_total = self.criterion(pred_total, y)

        loss_sum = (loss + loss_op + loss_total) / 3

        # Tensorboard log
        self.logger.experiment.add_scalar('Train/Loss', loss_sum, self.trainer.global_step)

        # Log for callbacks
        self.log('iteration', self.trainer.global_step, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        return loss_sum

    def validation_step(self, batch, batch_idx):
        x = batch['img']
        x_op = batch['img_op']
        slice_num = batch['slice_num']

        y = batch['annot']
        y_op = batch['annot_op']

        # Model 결과
        # 1. se_feature : 반구비교 최종 feature
        #                 (SENet의 output) (SENet의 input은, target_f + op_f + diff_f)
        # 2. pred : target의 output값
        # 3. pred_op : op의 output값
        # 4. pred_total : 반구비교 최종 feature(SENet output) 사용해서, dropout+FC 거친 최종 output
        se_feature, pred, pred_op, pred_total = self(x, x_op, slice_num)
        loss = self.criterion(pred, y)
        loss_op = self.criterion(pred_op, y_op)
        loss_total = self.criterion(pred_total, y)

        loss_sum = (loss + loss_op + loss_total) / 3

        return {'Loss': loss_sum, 'GT': y.detach(), 'Pred': torch.sigmoid(pred_total).detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['Loss'] for x in outputs]).mean()
        gt = torch.cat([x['GT'] for x in outputs]).long()
        pred = torch.cat([x['Pred'] for x in outputs])

        # Metric
        accuracy = self.Accuracy(pred, gt)
        precision = self.Precision(pred, gt)
        sensitivity = self.Sensitivity(pred, gt)
        specificity = self.Specificity(pred, gt)
        auc = self.AUC(pred, gt)

        # Tensorboard log
        self.logger.experiment.add_scalar('Val/Loss', avg_loss, self.trainer.global_step)
        self.logger.experiment.add_scalar('Val/Accuracy', accuracy, self.trainer.global_step)
        self.logger.experiment.add_scalar('Val/Precision', precision, self.trainer.global_step)
        self.logger.experiment.add_scalar('Val/Sensitivity', sensitivity, self.trainer.global_step)
        self.logger.experiment.add_scalar('Val/Specificity', specificity, self.trainer.global_step)
        self.logger.experiment.add_scalar('Val/AUC', auc, self.trainer.global_step)

        del outputs
        torch.cuda.empty_cache()

        # Log for callbakcs
        self.log('AUC', auc, on_step=False, on_epoch=True, prog_bar=False, logger=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, self.parameters()), opt['lr'])
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                                          T_0=50,
                                                                                          T_mult=2, eta_min=0),
                        'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': 1}

        return [optimizer], [lr_scheduler]


class Solver_EICDMSOI_LSTM(pl.LightningModule):
    def __init__(self, lr=1e-3, path_weight_FE=''):
    # def __init__(self):
        super().__init__()

        # Load pretrained model

        EfficientNet = Solver_FE_v2()
        EfficientNet.load_state_dict(torch.load(path_weight_FE, map_location='cuda:{}'.format(opt['use_gpu_Num'])))
        EfficientNet.model.feature_extract = True
        EfficientNet.eval().to(device)

        # EfficientNet = Solver_FE_v2.load_from_checkpoint(path_weight_FE, map_location='cuda:{}'.format(opt['use_gpu_Num']))
        # EfficientNet.model.feature_extract = True
        # EfficientNet.to(device)

        # Freeze weights
        # - Feature Extractor의 weight는 고정
        for name, module in EfficientNet.named_modules():
            for param in module.parameters():
                param.requires_grad = False

        # Model
        self.EfficientNet = EfficientNet  # FE
        self.LSTM = LSTM(embed_size=1792, LSTM_UNITS=1792)  # LSTM

        # Criterion
        self.criterion = PolyBCELoss(reduction='mean')

        # Metric
        self.Accuracy = torchmetrics.Accuracy()
        self.Precision = torchmetrics.Precision()
        self.Sensitivity = torchmetrics.Recall()
        self.Specificity = torchmetrics.Specificity()
        self.AUC = torchmetrics.AUROC(pos_label=1)

        # Learning rate
        self.learning_rate = lr

    # def load_from_checkpoint(self, checkpoint_path, **kwargs):
    #     state_dict = torch.load(checkpoint_path, **kwargs)
    #     self.load_state_dict(state_dict)

    # def setup(self, lr, path_weight_FE):
    #     # Load pretrained model
    #     EfficientNet = Solver_FE_v2.load_from_checkpoint(path_weight_FE,
    #                                                      map_location='cuda:{}'.format(opt['use_gpu_Num']))
    #
    #     EfficientNet.model.feature_extract = True
    #
    #     EfficientNet.to(device)
    #
    #     # Freeze weights
    #     # - Feature Extractor의 weight는 고정
    #     for name, module in EfficientNet.named_modules():
    #         for param in module.parameters():
    #             param.requires_grad = False
    #
    #     # Model
    #     self.EfficientNet = EfficientNet  # FE
    #     self.LSTM = LSTM(embed_size=1792, LSTM_UNITS=1792)  # LSTM
    #
    #     # Learning rate
    #     self.learning_rate = lr

    def forward(self, x, x_op, slice_num):
        # Feature Extractor로, Feature 획득
        features = EfficientNet_series_hemi_compare_cELVO_FE_SliceNum(self.EfficientNet, x, x_op, slice_num)
        # FE의 output Feature를, LSTM에 input
        out = self.LSTM(features)

        # return out
        return features, out

    def training_step(self, batch, batch_idx):
        self.EfficientNet.eval()

        x = batch['img']
        x_op = batch['img_op']
        slice_num = batch['slice_num']

        y = batch['annot']

        # FE / LSTM 모델 결과
        features, pred = self(x, x_op, slice_num)
        loss = self.criterion(pred, y)

        # Tensorboard log
        self.logger.experiment.add_scalar('Train/Loss', loss, self.trainer.global_step)

        # Log for callbacks
        self.log('iteration', self.trainer.global_step, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        # Empty cache
        if self.trainer.global_step % opt['empty_cache_interval'] == 0:
            torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        self.EfficientNet.eval()

        x = batch['img']
        x_op = batch['img_op']
        slice_num = batch['slice_num']

        y = batch['annot']

        features, pred = self(x, x_op, slice_num)
        loss = self.criterion(pred, y)

        return {'Loss': loss, 'GT': y.detach(), 'Pred': torch.sigmoid(pred).detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['Loss'] for x in outputs]).mean()
        gt = torch.cat([x['GT'] for x in outputs]).long()
        pred = torch.cat([x['Pred'] for x in outputs])

        # Metric
        accuracy = self.Accuracy(pred, gt)
        precision = self.Precision(pred, gt)
        sensitivity = self.Sensitivity(pred, gt)
        specificity = self.Specificity(pred, gt)
        auc = self.AUC(pred, gt)

        # Tensorboard log
        self.logger.experiment.add_scalar('Val/Loss', avg_loss, self.trainer.global_step)
        self.logger.experiment.add_scalar('Val/Accuracy', accuracy, self.trainer.global_step)
        self.logger.experiment.add_scalar('Val/Precision', precision, self.trainer.global_step)
        self.logger.experiment.add_scalar('Val/Sensitivity', sensitivity, self.trainer.global_step)
        self.logger.experiment.add_scalar('Val/Specificity', specificity, self.trainer.global_step)
        self.logger.experiment.add_scalar('Val/AUC', auc, self.trainer.global_step)

        # Empty cache
        del outputs
        torch.cuda.empty_cache()

        # Log for callbakcs
        self.log('AUC', auc, on_step=False, on_epoch=True, prog_bar=False, logger=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, self.parameters()), self.learning_rate)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                                          T_0=200,
                                                                                          T_mult=2, eta_min=0),
                        'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': 1}

        return [optimizer], [lr_scheduler]


# for i in range(0, 5):
#     path_weight_LSTM = opt_weight['path_weight_LSTM_DMS'][i]
#     path_weight_FE = opt_weight['path_weight_FE_DMS'][i]
#
#     model = Solver_EICDMSOI_LSTM_temp.load_from_checkpoint(path_weight_LSTM,
#                                                            map_location='cuda:{}'.format(
#                                                                opt['use_gpu_Num']),
#                                                            path_weight_FE=path_weight_FE)
#
#     new_fn_LSTM = path_weight_LSTM.replace('ckpt', 'pth')
#     torch.save(model.state_dict(), new_fn_LSTM)

# for i in range(0, 5):
#     path_weight_LSTM = opt_weight['path_weight_LSTM_EIC_CHo'][i]
#     path_weight_FE = opt_weight['path_weight_FE_EIC_CHo'][i]
#
#     model = Solver_EICDMSOI_LSTM_temp.load_from_checkpoint(path_weight_LSTM,
#                                                            map_location='cuda:{}'.format(
#                                                                opt['use_gpu_Num']),
#                                                            path_weight_FE=path_weight_FE)
#
#     new_fn_LSTM = path_weight_LSTM.replace('ckpt', 'pth')
#     torch.save(model.state_dict(), new_fn_LSTM)

# for i in range(0, 5):
#     path_weight_LSTM = opt_weight['path_weight_LSTM_EIC_CHx'][i]
#     path_weight_FE = opt_weight['path_weight_FE_EIC_CHx'][i]
#
#     model = Solver_EICDMSOI_LSTM_exclude_hemi_compare.load_from_checkpoint(path_weight_LSTM,
#                                                            map_location='cuda:{}'.format(
#                                                                opt['use_gpu_Num']),
#                                                            path_weight_FE=path_weight_FE)
#
#     new_fn_LSTM = path_weight_LSTM.replace('ckpt', 'pth')
#     torch.save(model.state_dict(), new_fn_LSTM)

# for i in range(0, 5):
#     path_weight_LSTM = opt_weight['path_weight_LSTM_LVO_CHo'][i]
#
#     model = Solver_LVO_LSTM_EIC_CHo_DMS.load_from_checkpoint(
#         path_weight_LSTM,
#         map_location='cuda:{}'.format(opt['use_gpu_Num']),
#         path_weight_LSTM_EIC_CHo=opt_weight['path_weight_LSTM_EIC_CHo'][i],
#         path_weight_LSTM_DMS=opt_weight['path_weight_LSTM_DMS'][i],
#         path_wei
#         use_gpu_Num=opt['use_gpu_Num'],
#         do_inference=True
#     )
#
#     new_fn_LSTM = path_weight_LSTM.replace('ckpt', 'pth')
#     torch.save(model.state_dict(), new_fn_LSTM)

for i in range(0, 5):
    path_weight_LSTM = opt_weight['path_weight_LSTM_LVO_CHx'][i]

    model = Solver_LVO_LSTM_EIC_CHx_DMS.load_from_checkpoint(
        path_weight_LSTM,
        map_location='cuda:{}'.format(opt['use_gpu_Num']),
        path_weight_LSTM_EIC_CHx=opt_weight['path_weight_LSTM_EIC_CHx'][i],
        path_weight_LSTM_DMS=opt_weight['path_weight_LSTM_DMS'][i],
        path_weight_FE_EIC_CHx=opt_weight['path_weight_FE_EIC_CHx'][i],
        path_weight_FE_DMS=opt_weight['path_weight_FE_DMS'][i],
        use_gpu_Num=opt['use_gpu_Num'],
        do_inference=True
    )

    new_fn_LSTM = path_weight_LSTM.replace('ckpt', 'pth')
    torch.save(model.state_dict(), new_fn_LSTM)
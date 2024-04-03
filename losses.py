# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F
from torchvision import transforms
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import random
import sys
import numpy as np

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float, input_size: int, teacher_size: int, weighted_distillation: bool, weight: list,args = None):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.input_size = input_size
        self.teacher_size = teacher_size
        self.weighted_distillation = weighted_distillation
        self.weight = weight
        self.args = args


    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            #print("This is used when there is a teacher model with a distillationhead")
            outputs, outputs_kd = outputs[0], outputs[1]
        try:
            base_loss = self.base_criterion(outputs, labels)
        except:
            labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes = len(self.weight)).cuda()
            base_loss = self.base_criterion(outputs, labels)


        if self.distillation_type == 'none':
            if not self.weighted_distillation:
                token_2_loss = SoftTargetCrossEntropy()(outputs_kd, labels)
            else:
                token_2_loss = torch.nn.CrossEntropyLoss(weight = self.weight)(outputs_kd, labels)
            loss = base_loss * (1 - self.alpha) + token_2_loss * self.alpha
            return loss, base_loss * (1 - self.alpha), token_2_loss * self.alpha 

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher

          
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            pred_t = F.log_softmax(teacher_outputs / T, dim=1)
            if self.weighted_distillation:
                pred_t = pred_t * self.weight
                pred_t = pred_t / pred_t.sum(1)[:, None]

            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                pred_t,
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':

            distillation_targets = teacher_outputs.argmax(dim=1).cuda() #[256]
            if self.args.map_targets:
                distillation_targets = torch.Tensor(np.array(self.args.class_map)[distillation_targets.detach().cpu()]).type(torch.LongTensor).cuda()


            if self.weighted_distillation:
                #print("Weighted Distillation")
                distillation_loss = F.cross_entropy(outputs_kd, distillation_targets, weight = self.weight)
            else:
                distillation_loss = F.cross_entropy(outputs_kd, distillation_targets)

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha

        return loss, base_loss * (1 - self.alpha), distillation_loss * self.alpha






class DistillationLossMultiCrop(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(
        self,
        base_criterion: torch.nn.Module,
        teacher_model: torch.nn.Module,
        distillation_type: str,
        alpha: float,
        tau: float,
        input_size: int,
        teacher_size: int,
        weighted_distillation: bool,
        weight: list,
        args=None,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ["none", "soft", "hard"]
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.input_size = input_size
        self.teacher_size = teacher_size
        self.weighted_distillation = weighted_distillation
        self.weight = weight
        self.args = args

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            # print("This is used when there is a teacher model with a distillationhead")
            outputs, outputs_kd, _, _ = outputs
        try:
            base_loss = self.base_criterion(outputs, labels)
        except:
            labels = torch.nn.functional.one_hot(
                labels.to(torch.int64), num_classes=len(self.weight)
            ).cuda()
            base_loss = self.base_criterion(outputs, labels)

        if self.distillation_type == "none":
            if not self.weighted_distillation:
                token_2_loss = SoftTargetCrossEntropy()(outputs_kd, labels)
            else:
                token_2_loss = torch.nn.CrossEntropyLoss(weight=self.weight)(
                    outputs_kd, labels
                )
            loss = base_loss * (1 - self.alpha) + token_2_loss * self.alpha
            return loss, base_loss * (1 - self.alpha), token_2_loss * self.alpha

        if outputs_kd is None:
            raise ValueError(
                "When knowledge distillation is enabled, the model is "
                "expected to return a Tuple[Tensor, Tensor] with the output of the "
                "class_token and the dist_token"
            )
        # don't backprop throught the teacher

        with torch.no_grad():
            if isinstance(inputs, list):
                teacher_out_global = self.teacher_model(inputs[0])
                teacher_out_local = self.teacher_model(inputs[1])
                teacher_outputs = torch.cat(
                    [teacher_out_global, teacher_out_local], dim=0
                )
            else:
                teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == "soft":
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            pred_t = F.log_softmax(teacher_outputs / T, dim=1)
            if self.weighted_distillation:
                pred_t = pred_t * self.weight
                pred_t = pred_t / pred_t.sum(1)[:, None]

            distillation_loss = (
                F.kl_div(
                    F.log_softmax(outputs_kd / T, dim=1),
                    # We provide the teacher's targets in log probability because we use log_target=True
                    # (as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                    # but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                    pred_t,
                    reduction="sum",
                    log_target=True,
                )
                * (T * T)
                / outputs_kd.numel()
            )
            # We divide by outputs_kd.numel() to have the legacy PyTorch behavior.
            # But we also experiments output_kd.size(0)
            # see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == "hard":
            distillation_targets = teacher_outputs.argmax(dim=1).cuda()  # [256]
            # If the teacher is used only for global crops
            if not self.args.local_global_teacher:
                outputs_kd_final = outputs_kd[: distillation_targets.size(0)]
            else:
                outputs_kd_final = outputs_kd
            if self.args.map_targets:
                distillation_targets = (
                    torch.Tensor(
                        np.array(self.args.class_map)[
                            distillation_targets.detach().cpu()
                        ]
                    )
                    .type(torch.LongTensor)
                    .cuda()
                )
            if self.weighted_distillation:
                # print("Weighted Distillation")
                distillation_loss = F.cross_entropy(
                    outputs_kd_final,
                    distillation_targets,
                    weight=self.weight,
                )
            else:
                distillation_loss = F.cross_entropy(
                    outputs_kd_final,
                    distillation_targets,
                )

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss, base_loss * (1 - self.alpha), distillation_loss * self.alpha
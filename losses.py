"""
Author: Yalala Mohit
Date: 12/10/2023
Course: CS 7180: Advanced Perception, Northeastern University
Description: Contains the loss functions and functions to perform loss computations, both from the OWLV2 paper, and custom loss which GODS proposes.
"""

import os
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from lvis import LVIS, LVISEval
from PIL import ImageDraw, ImageFile
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_area

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.autograd.set_detect_anomaly(True)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    """
    Compute the intersection over union (IoU) of two sets of boxes.

    Args:
    boxes1 (Tensor): The first set of boxes in the format [x0, y0, x1, y1].
    boxes2 (Tensor): The second set of boxes in the same format.

    Returns:
    tuple: A tuple containing the IoU tensor and the union tensor.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
     """
    Compute the Generalized Intersection over Union (GIoU) of two sets of boxes.

    Args:
    boxes1 (Tensor): The first set of boxes in the format [x0, y0, x1, y1].
    boxes2 (Tensor): The second set of boxes in the same format.

    Returns:
    Tensor: A tensor representing the GIoU values.
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


class HungarianMatcher(torch.nn.Module):
   """
    Hungarian Matcher class for matching ground truth boxes with predicted boxes.

    Args:
    n_classes (int): Number of classes.
    cost_class (float): Weight of the classification error in the matching cost.
    cost_bbox (float): Weight of the L1 error of bounding box coordinates in the matching cost.
    cost_giou (float): Weight of the GIoU loss of bounding box in the matching cost.

    Attributes:
    n_classes (int): Number of classes.
    cost_class (float): Weight for classification error.
    cost_bbox (float): Weight for bounding box L1 error.
    cost_giou (float): Weight for GIoU loss.
    device (str): Computation device, 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        n_classes,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.n_classes = n_classes
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    def _get_src_permutation_idx(self, indices):
        """
        Get source permutation index for predicted and target indices.

        Args:
        indices (list): List of tuples with matching indices.

        Returns:
        tuple: Batch and source indices.
        """
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["pred_logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets]).to(self.device)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        indices = [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        ).to(self.device)
        target_classes = torch.full(
            outputs["pred_logits"].shape[:2],
            self.n_classes,
            dtype=torch.int64,
            device=self.device,
        )
        target_classes[idx] = target_classes_o

        return target_classes, indices, idx

class CustomLoss(torch.nn.Module):
    """
    Custom loss module for object detection tasks.

    Args:
    n_classes (int): Number of classes.
    scales (Tensor): Weight scales for each class.

    Attributes:
    matcher (HungarianMatcher): Matcher for predicted and ground truth boxes.
    class_criterion (BCELoss): Binary cross entropy loss for classification.
    background_label (int): Label index for the background class.
    device (str): Computation device, 'cuda' or 'cpu'.
    """
    def __init__(self, n_classes, scales):
        super().__init__()
        self.matcher = HungarianMatcher(n_classes)
        self.class_criterion = torch.nn.BCELoss(reduction="none", weight=scales)
        self.background_label = n_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    def filter_top_objectness(self, logits, boxes, objectness_logits, top_k=300):
        """
        Filter logits and boxes based on top objectness scores while maintaining batch dimensions.
    
        Args:
        logits (torch.Tensor): The logits tensor of shape [batch_size, num_queries, num_classes].
        boxes (torch.Tensor): The bounding boxes tensor of shape [batch_size, num_queries, 4].
        objectness_logits (torch.Tensor): The objectness logits tensor of shape [batch_size, num_queries].
        top_k (int): The number of top objectness scores to keep.
    
        Returns:
        torch.Tensor, torch.Tensor: Tensors of filtered logits and boxes with batch dimensions.
        """
        batch_size, num_queries = logits.size(0), logits.size(1)
    
        # Check if top_k is greater than or equal to num_queries
        if top_k >= num_queries:
            return logits, boxes
    
        filtered_logits_list = []
        filtered_boxes_list = []
    
        for batch_idx in range(batch_size):
            # Calculate objectness scores for the current batch
            objectness_scores = torch.sigmoid(objectness_logits[batch_idx])
    
            # Get the indices of the top objectness scores
            top_indices = torch.topk(objectness_scores, top_k, sorted=False).indices
    
            # Filter logits and boxes based on these indices
            filtered_logits = logits[batch_idx][top_indices]
            filtered_boxes = boxes[batch_idx][top_indices]
    
            filtered_logits_list.append(filtered_logits)
            filtered_boxes_list.append(filtered_boxes)
    
        # Stack the filtered results to maintain batch dimension
        filtered_logits = torch.stack(filtered_logits_list, dim=0)
        filtered_boxes = torch.stack(filtered_boxes_list, dim=0)
    
        return filtered_logits, filtered_boxes


    def logits_processor(self, logits, texts, class_indices_list, num_classes):
        """
        Process logits to match the total number of classes, setting logits for 
        unmentioned classes to a very low value.
    
        Args:
        logits (Tensor): The original logits tensor of shape [batch_size, num_queries, num_text_classes].
        texts (List[List[str]]): List of texts for each image in the batch.
        class_indices_list (List[List[int]]): List of actual class indices for each text in the batch.
        num_classes (int): Total number of classes.
    
        Returns:
        Tensor: Processed logits of shape [batch_size, num_queries, num_classes].
        """
        batch_size, num_queries, _ = logits.shape
        processed_logits = torch.full((batch_size, num_queries, num_classes), float('-inf')).to(logits.device)
    
        for batch_idx, (text_list, class_indices) in enumerate(zip(texts, class_indices_list)):
            # Ensure the class indices list matches the number of non-pad texts
            assert len(class_indices) == len([text for text in text_list if text != '<PAD>']), "Class indices do not match the number of non-pad texts"
    
            # Copy the logits for the existing classes
            for i, class_idx in enumerate(class_indices):
                if class_idx < num_classes:
                    processed_logits[batch_idx, :, class_idx] = logits[batch_idx, :, i]
    
        return processed_logits

    def convert_bbox_format(self, bboxes):
        """
        Convert bounding boxes from (x, y, w, h) format to (x0, y0, x1, y1) format.
    
        Args:
        bboxes (Tensor): A tensor of shape (batch_size, num_boxes, 4) where each box is represented as (x, y, w, h).
    
        Returns:
        Tensor: Converted bounding boxes of shape (batch_size, num_boxes, 4) in (x0, y0, x1, y1) format.
        """
        # Separate the coordinates
        x, y, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    
        # Calculate x0, y0, x1, y1
        x0 = x
        y0 = y
        x1 = x + w
        y1 = y + h
    
        # Stack the coordinates back into the original shape
        converted_bboxes = torch.stack((x0, y0, x1, y1), dim=-1)
    
        return converted_bboxes

    def mask_within_bboxes(self, bounding_boxes, binary_masks):
        """
        Creates a new mask where only the regions within the bounding boxes are preserved.
    
        Parameters:
        - bounding_boxes: Tensor of bounding boxes with each box defined as [X, Y, W, H] for each image in the batch.
        - binary_masks: A batch of binary masks as a 4D tensor of shape (batch_size, 1, height, width).
    
        Returns:
        - A batch of binary masks where regions outside the bounding boxes are set to 0.
        """
        batch_size, height, width = binary_masks.shape
        new_masks = torch.zeros_like(binary_masks)
    
        for i in range(batch_size):
            bounding_boxes[i]= bounding_boxes[i]*960
            for bbox in bounding_boxes[i]:
                X, Y, endX, endY = bbox.int()
    
                endX = torch.clamp(endX, max=width)
                endY = torch.clamp(endY, max=height)
    
                new_masks[i, Y:endY, X:endX] = torch.logical_or(binary_masks[i, Y:endY, X:endX], new_masks[i, Y:endY, X:endX])
    
        return new_masks

    def generate_attention_masks(self, attentions, grid_size=60, upscale_size=(960, 960)):
        """
        Generate attention masks from the last layer of transformer attentions.

        Args:
        attentions (Tensor): Attention weights from a transformer model.
        grid_size (int): The size of the grid to reshape the attention diagonals.
        upscale_size (tuple): The target size to upscale the attention masks.

        Returns:
        Tensor: Upscaled attention masks tensor for each image in the batch.
        """
        attentions = attentions[-1]
        attention_masks = []
        for batch_idx in range(attentions.shape[0]):
            # Process each attention head
            diagonal_attentions = []
            for head in range(attentions.shape[1]):
                # Extract the diagonal for each patch and reshape
                diagonal = torch.diagonal(attentions[batch_idx, head, :-1, :-1], offset=0, dim1=-2, dim2=-1)
                reshaped_diagonal = diagonal.view(grid_size, grid_size)
                diagonal_attentions.append(reshaped_diagonal)
    
            # Average across heads and upscale
            average_diagonal_attention = torch.mean(torch.stack(diagonal_attentions, dim=0), dim=0)
            upscaled_attention = F.interpolate(
                average_diagonal_attention.unsqueeze(0).unsqueeze(0), 
                size=upscale_size, 
                mode='nearest'
            ).squeeze()
            attention_masks.append(upscaled_attention)
        # Stack all attention masks into a single tensor
        attention_masks_tensor = torch.stack(attention_masks, dim=0)
        return attention_masks_tensor
        
    def dice_loss(self, pred_masks, target_masks, smooth=1e-6):
        """
        Compute the Dice loss for a batch of predicted and target masks.

        Args:
        pred_masks (Tensor): Predicted masks, a tensor of shape (batch_size, height, width).
        target_masks (Tensor): Ground truth masks, a tensor of shape (batch_size, height, width).
        smooth (float): A small constant to avoid division by zero.

        Returns:
        Tensor: Dice loss for the batch.
        """
        # Normalize masks if they are in 0-255 range
        if pred_masks.max() > 1:
            pred_masks = pred_masks / 255.0
        if target_masks.max() > 1:
            target_masks = target_masks / 255.0
            
        # Flatten the masks
        pred_masks_flat = pred_masks.view(pred_masks.shape[0], -1).to(self.device)
        target_masks_flat = target_masks.view(target_masks.shape[0], -1).to(self.device)

        # Compute intersection and union
        intersection = (pred_masks_flat * target_masks_flat).sum(dim=1)
        union = pred_masks_flat.sum(dim=1) + target_masks_flat.sum(dim=1)

        # Compute Dice coefficient and loss
        dice_coefficient = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_coefficient

        # Average over the batch
        return dice_loss.mean()
        
    def calculate_no_object_loss(self, outputs, targets, matched_indices, num_classes):
        """
        Calculate the no_object loss for unmatched predictions.

        Args:
            outputs (dict): Model outputs containing 'pred_logits' and 'pred_boxes'.
            targets (list): List of ground truth targets.
            matched_indices (list): List of tuples with matched indices from Hungarian Matcher.
            num_classes (int): Number of classes including the no_object class.

        Returns:
            Tensor: The no_object loss.
        """
        no_object_loss = 0.0
        batch_size = len(targets)

        for batch_idx in range(batch_size):
            # Get all indices for this batch
            all_indices = torch.arange(outputs['pred_logits'].shape[1], device=outputs['pred_logits'].device)
            # Get the indices of predictions that are matched
            matched_pred_indices = matched_indices[batch_idx][0]
            # Find indices of unmatched predictions
            unmatched_pred_indices = torch.ones_like(all_indices).bool()
            unmatched_pred_indices[matched_pred_indices] = False
            unmatched_pred_indices = all_indices[unmatched_pred_indices]

            # Get the logits for unmatched predictions
            unmatched_logits = outputs['pred_logits'][batch_idx][unmatched_pred_indices]

            # Calculate no_object loss (assuming last class is no_object class)
            no_object_target = torch.zeros(unmatched_logits.shape[0], num_classes, device=self.device)
            no_object_target[:, -1] = 1  # Set no_object class to 1

            loss = F.binary_cross_entropy_with_logits(unmatched_logits, no_object_target)
            no_object_loss += loss

        # Normalize the no_object loss by batch size
        no_object_loss /= batch_size

        return no_object_loss

    def calculate_losses(self, outputs, targets, matched_indices, pred_masks, target_masks, attention_masks):
        """
        Calculate various losses for the object detection task.

        Args:
        outputs (dict): Model outputs with 'pred_logits' and 'pred_boxes'.
        targets (list): List of ground truth targets.
        matched_indices (list): List of matched indices between predictions and targets.
        pred_masks (Tensor): Predicted masks for each image.
        target_masks (Tensor): Ground truth masks for each image.
        attention_masks (Tensor): Attention masks generated from transformer attentions.

        Returns:
        dict: Dictionary containing calculated loss values.
        """
        losses = {
            "classification_loss": 0,
            "l1_loss": 0,
            "iou_loss": 0,
            "giou_loss": 0,
            "no_object_loss": 0,
            "dice_loss": 0,
            "attention_loss": 0
        }

        for batch_idx, (pred_idx, target_idx) in enumerate(matched_indices):
            # Extract matched predictions and targets
            pred_logits = outputs["pred_logits"][batch_idx][pred_idx]
            pred_boxes = outputs["pred_boxes"][batch_idx][pred_idx]
            target_labels = targets[batch_idx]["labels"][target_idx]
            target_boxes = targets[batch_idx]["boxes"][target_idx]

            # Ensure the boxes are in the correct shape [num_boxes, 4]
            pred_boxes = pred_boxes.view(-1, 4)
            target_boxes = target_boxes.view(-1, 4).to(self.device)

            # Classification loss
            target_labels_one_hot = F.one_hot(target_labels, num_classes=pred_logits.shape[-1]).float().to(self.device)
            loss_class = F.binary_cross_entropy_with_logits(pred_logits, target_labels_one_hot)
            losses["classification_loss"] += loss_class

            # L1 loss for bounding boxes
            loss_l1 = F.l1_loss(pred_boxes, target_boxes, reduction="none").sum()
            losses["l1_loss"] += loss_l1

            # IoU and GIoU losses
            iou = 1 - box_iou(pred_boxes, target_boxes)[0].diag()
            giou = 1 - generalized_box_iou(pred_boxes, target_boxes).diag()
            losses["iou_loss"] += iou.sum()
            losses["giou_loss"] += giou.sum()

        # Normalize losses by number of matched pairs
        num_matches = sum(len(pred) for pred, _ in matched_indices)
        for key in losses:
            losses[key] /= num_matches
            
        losses["no_object_loss"] = self.calculate_no_object_loss(outputs, targets, matched_indices, self.background_label)
        losses["dice_loss"] = self.dice_loss(pred_masks, target_masks)
        losses["attention_loss"] = self.dice_loss(attention_masks, target_masks)
        return losses

    def forward(self, predicted_classes, predicted_boxes, predicted_object_scores, targets, target_masks, attention):
        """
        Forward pass for the CustomLoss module.

        Args:
        predicted_classes (Tensor): Predicted class logits.
        predicted_boxes (Tensor): Predicted bounding boxes.
        predicted_object_scores (Tensor): Objectness scores for each prediction.
        targets (list): List of ground truth targets.
        target_masks (Tensor): Ground truth masks for each image.
        attention (Tensor): Attention weights from a transformer model.

        Returns:
        dict: Calculated loss values for the predictions.
        """
        filtered_logits, filtered_boxes = self.filter_top_objectness(predicted_classes,
                                                                     predicted_boxes, 
                                                                     predicted_object_scores, 
                                                                     top_k=300)
        scaled_boxes = self.convert_bbox_format(filtered_boxes)
        attention_masks = self.generate_attention_masks(attention)
        predicted_masks = self.mask_within_bboxes(filtered_boxes, target_masks)
        # Format to detr style
        outputs = {
            "pred_logits": filtered_logits,
            "pred_boxes": scaled_boxes,
        }
        target_classes, indices, idx = self.matcher(outputs, targets)
        return self.calculate_losses(outputs, targets, indices, predicted_masks, target_masks, attention_masks)
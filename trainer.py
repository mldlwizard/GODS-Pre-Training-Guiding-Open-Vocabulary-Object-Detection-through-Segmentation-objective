"""
Author: Yalala Mohit
Date: 12/10/2023
Course: CS 7180: Advanced Perception, Northeastern University
Description: Contains the classes and methods for training on the LVIS dataset. It is not essential to download the images, as the dataloader will handle the downloading.
"""

import os
import numpy as np
import PIL.Image as Image
import torch
import torch.optim as optim
from lvis import LVIS, LVISEval
from PIL import Image, ImageDraw, ImageFile
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from trainloader import LvisTrainDataLoader
from transformers import Owlv2ForObjectDetection, Owlv2Processor
ImageFile.LOAD_TRUNCATED_IMAGES = True

class LVISTrainingPipeline:
    """
    A training pipeline for the LVIS dataset using the OwlV2 model.

    Args:
    train_dataloader (DataLoader): DataLoader for the training dataset.
    val_dataloader (DataLoader): DataLoader for the validation dataset.
    model_name (str): The name of the pre-trained OwlV2 model.
    save_path (str): Directory path to save the model checkpoints.
    device (str): The device to use for training (default: 'cuda').
    resume_training (bool): Flag to resume training from the latest checkpoint (default: False).

    Attributes:
    processor (Owlv2Processor): Processor for the OwlV2 model.
    model (Owlv2ForObjectDetection): Pre-trained OwlV2 model for object detection.
    device (str): Device used for training.
    train_dataloader (DataLoader): DataLoader for training data.
    val_dataloader (DataLoader): DataLoader for validation data.
    save_path (str): Path to save model checkpoints.
    writer (SummaryWriter): TensorBoard writer for logging.
    best_loss (float): Best validation loss for saving the best model.
    """

    def __init__(self, train_dataloader, val_dataloader, model_name, save_path, device='cuda', resume_training=False):
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device)
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.save_path = save_path
        self.writer = SummaryWriter()
        self.best_loss = float('inf')

        if resume_training and os.path.exists(os.path.join(save_path, 'latest_model.pth')):
            self.model.load_state_dict(torch.load(os.path.join(save_path, 'latest_model.pth')))

    def train_epoch(self, optimizer):
        """
        Trains the model for one epoch.

        Args:
        optimizer (Optimizer): The optimizer used for training.

        Returns:
        float: Average training loss for the epoch.
        """
        self.model.train()
        total_train_loss = 0.0

        for batch in tqdm(self.train_dataloader, desc="Training"):
            image_ids, image_dims, images, texts, mask_combined, targets = batch
            optimizer.zero_grad()
            inputs = self.processor(text=list(texts), images=images, return_tensors="pt").to(self.device)
            outputs = self.model(input_ids=inputs['input_ids'],
                             pixel_values=inputs['pixel_values'],
                             attention_mask=inputs['attention_mask'],
                             output_attentions=True ,
                             output_hidden_states=False,
                             return_dict=True)
            loss = self.compute_loss(outputs, targets, mask_combined)
            train_loss = sum(loss.values())
            total_train_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()

        return total_train_loss / len(self.train_dataloader)

    def validate_epoch(self):
        """
        Validates the model for one epoch.

        Returns:
        float: Average validation loss for the epoch.
        """
        self.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                image_ids, image_dims, images, texts, mask_combined, targets = batch
                inputs = self.processor(text=list(texts), images=images, return_tensors="pt").to(self.device)
                outputs = self.model(input_ids=inputs['input_ids'],
                             pixel_values=inputs['pixel_values'],
                             attention_mask=inputs['attention_mask'],
                             output_attentions=True ,
                             output_hidden_states=False,
                             return_dict=True)
                loss = self.compute_loss(outputs, targets, mask_combined)
                val_loss = sum(loss.values())
                total_val_loss += val_loss.item()

        return total_val_loss / len(self.val_dataloader)

    def compute_loss(self, outputs, targets, mask_combined):
        """
        Computes the custom loss for the model's outputs.

        Args:
        outputs (ModelOutput): Outputs from the model.
        targets (dict): Target labels and bounding boxes.
        mask_combined (Tensor): Combined mask for the images.

        Returns:
        Tensor: Computed loss.
        """
        attention = outputs.vision_model_output.attentions
        scales = torch.ones(16).to(self.device)
        custom_loss = CustomLoss(16, scales).to(self.device)
        return custom_loss(outputs.logits, 
                           outputs.pred_boxes,
                           outputs.objectness_logits,
                           targets,  
                           mask_combined, 
                           attention)

    def train(self, num_epochs):
        """
        Runs the training process for a specified number of epochs.

        Args:
        num_epochs (int): The number of epochs to train for.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        for epoch in range(num_epochs):
            avg_train_loss = self.train_epoch(optimizer)
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)

            avg_val_loss = self.validate_epoch()
            self.writer.add_scalar('Loss/val', avg_val_loss, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            torch.save(self.model.state_dict(), os.path.join(self.save_path, 'latest_model.pth'))
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))

        self.writer.close()
    

# Parameters
MODEL_NAME = "google/owlv2-base-patch16-finetuned"
IMAGE_DIR = "/work/socialmedia/pytorch/data/val2017/val2017/"
TRAIN_ANNOTATION_PATH = "/work/socialmedia/pytorch/data/lvis_v1_train.json/lvis_v1_train.json"
VAL_ANNOTATION_PATH = "/work/socialmedia/pytorch/data/lvis_v1_val.json/lvis_v1_val.json"
SAVE_PATH = "/home/mohit.y/Owl-Vit_Segmentation/Metadata"
BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
RESUME_TRAINING = True
TRAIN_DATALOADER, VAL_DATALOADER = LvisTrainDataLoader(TRAIN_ANNOTATION_PATH, VAL_ANNOTATION_PATH, IMAGE_DIR, BATCH_SIZE).get_dataloader()
training_pipeline = LVISTrainingPipeline(TRAIN_DATALOADER, VAL_DATALOADER, MODEL_NAME, SAVE_PATH, DEVICE, RESUME_TRAINING)
training_pipeline.train(NUM_EPOCHS)

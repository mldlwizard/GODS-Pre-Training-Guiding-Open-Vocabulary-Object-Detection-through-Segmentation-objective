"""
Author: Yalala Mohit
Date: 12/10/2023
Course: CS 7180: Advanced Perception, Northeastern University
Description: Contains the classes and methods to load the Train and Validation Dataloader for training on the LVIS dataset. It is not essential to download the images, as the dataloader will handle the downloading.
"""

import os
import numpy as np
import PIL.Image as Image
from lvis import LVIS
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.autograd.set_detect_anomaly(True)

class LvisDataset(Dataset):
    """
    A dataset class for LVIS (Large Vocabulary Instance Segmentation) dataset.

    Args:
    annotation_path (str): The path to the annotation file for the LVIS dataset.
    image_dir (str): The directory where the images are stored.
    keep_category (list or None): If provided, filters images to include only those categories. Default is None.

    Attributes:
    lvis (LVIS): An instance of the LVIS class to handle annotations.
    img_dir (str): Directory where images are stored.
    keep_category (list or None): Categories to keep in the dataset.
    img_ids (list): List of image ids after filtering based on categories.
    img_shape (tuple): The target shape to resize images.
    max_text_shape (int): Maximum length for the text descriptions.
    """
    def __init__(self, annotation_path, image_dir, keep_category = None):
        self.lvis = LVIS(annotation_path=annotation_path)
        self.img_dir = image_dir
        self.keep_category = keep_category
        self.img_ids = self.filter_image_ids()
        self.img_shape = (960,960)
        self.max_text_shape = 16

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
        int: The number of images in the dataset.
        """
        return len(self.img_ids)

    def __getitem__(self, idx):
         """
        Get the item at the specified index.

        Args:
        idx (int): The index of the item to retrieve.

        Returns:
        tuple: A tuple containing image_id, image dimensions, image array, padded text, combined mask, and targets.
               Targets include labels and bounding boxes.
        """
        image_id = self.img_ids[idx]
        image_info = self.lvis.load_imgs([image_id])[0]
        image_dim = (image_info['height'], image_info['width'])
        image_path = os.path.join(self.img_dir, image_info['coco_url'].split('/')[-1])
        if not os.path.exists(image_path):
            self.lvis.download(self.img_dir,[image_id])
        image = np.asarray(Image.open(image_path).convert("RGB").resize(size=self.img_shape,
                                                                        resample=Image.Resampling.BILINEAR))
        annotation_ids = self.lvis.get_ann_ids(img_ids=[image_id])
        annotations = self.lvis.load_anns(annotation_ids)
        categories = [self.lvis.load_cats([annotation['category_id']])[0] for annotation in annotations]
        category_ids, text = self.text_to_class_indices([category['name'] for category in categories])
        if len(text) < self.max_text_shape:
            padded_text = text + ['<PAD>'] * (self.max_text_shape - len(text))
        else:
            padded_text = text[:self.max_text_shape]
        bounding_boxes = self.convert_and_scale_bboxes(torch.tensor([annotation['bbox'] for annotation in annotations], 
                                                                    dtype=torch.float32),
                                                      image_dim)
        targets = {"labels": category_ids, "boxes": bounding_boxes}

        mask = [self.lvis.ann_to_mask(ann=annotation) for annotation in annotations]
        if mask:
            mask_combined = (np.logical_or.reduce(mask).astype(np.uint8))*255
        else:
            mask_combined = np.zeros(self.img_shape, dtype=np.uint8)
        mask_combined = np.asarray(Image.fromarray(mask_combined, mode='L').resize(size=self.img_shape,
                                                                        resample=Image.Resampling.BILINEAR))
        return image_id, image_dim, image, padded_text, mask_combined, targets

    def text_to_class_indices(self, text):
        """
        Converts text labels to class indices.

        Args:
        text (list of str): List of text labels.

        Returns:
        tuple: A tuple containing class indices and unique labels.
        """
        # Create a unique set of labels
        unique_labels = list(sorted(set(text)))
        # Create a mapping from label to index
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        # Convert text labels to class indices
        class_indices = torch.tensor([label_to_index[label] for label in text], dtype=torch.int64)

        return class_indices, unique_labels
    
    def filter_image_ids(self):
        """
        Filters image IDs based on category frequency.

        Returns:
        list: List of filtered image IDs.
        """
        if self.keep_category is None:
            return self.lvis.get_img_ids()

        filtered_img_ids = []
        all_categories = self.lvis.load_cats(self.lvis.get_cat_ids())
        category_ids_to_keep = [cat['id'] for cat in all_categories if cat['frequency'] in self.keep_category]

        for img_id in self.lvis.get_img_ids():
            annotations = self.lvis.load_anns(self.lvis.get_ann_ids(img_ids=[img_id]))
            if any(ann['category_id'] in category_ids_to_keep for ann in annotations):
                filtered_img_ids.append(img_id)

        return filtered_img_ids
    
    def convert_and_scale_bboxes(self, bboxes, original_size, target_size=(960, 960)):
        """
        Convert bounding boxes from (x, y, w, h) format to (x0, y0, x1, y1) format and scale them.

        Args:
        bboxes (Tensor): A tensor of shape (num_boxes, 4) where each box is represented as (x, y, w, h).
        original_size (tuple): The original size of the image as (width, height).
        target_size (tuple): The target size to scale the bounding boxes to, default is (960, 960).

        Returns:
        Tensor: Converted and scaled bounding boxes of shape (num_boxes, 4) in (x0, y0, x1, y1) format.
        """
        # Extract original dimensions
        original_height, original_width = original_size
        target_height, target_width = target_size

        # Calculate scale factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        # Separate the coordinates and scale
        x = bboxes[:, 0] * scale_x
        y = bboxes[:, 1] * scale_y
        w = bboxes[:, 2] * scale_x
        h = bboxes[:, 3] * scale_y

        # Convert to (x0, y0, x1, y1) format
        x0 = x
        y0 = y
        x1 = x + w
        y1 = y + h
        
        # Normalize the coordinates to the range [0, 1]
        x0 /= target_width
        y0 /= target_height
        x1 /= target_width
        y1 /= target_height

        # Stack the coordinates back into the original shape
        converted_scaled_bboxes = torch.stack((x0, y0, x1, y1), dim=-1)

        return converted_scaled_bboxes

def custom_collate_fn(batch):
    """
    Custom collate function for the DataLoader to handle batches of data.

    Args:
    batch (list): A list of tuples, each containing elements for a single data point.
                  Each tuple consists of image_id, image_dim, image, text, mask_combined, and targets.

    Returns:
    tuple: A tuple containing batches of image_ids, image_dims, images, texts, masks_combined, and targets.
           The images and masks_combined are collated using default_collate, while other elements are zipped and returned as is.
    """
    # Separate the elements of the batch
    image_ids, image_dims, images, texts, masks_combined, targets = zip(*batch)
    images = default_collate(images)
    masks_combined = default_collate(masks_combined)
    return image_ids, image_dims, images, texts, masks_combined, targets

class LvisTrainDataLoader:
    """
    DataLoader for training and validation datasets using LVIS annotations.

    Args:
    train_annotation_path (str): Path to the training dataset annotation file.
    val_annotation_path (str): Path to the validation dataset annotation file.
    image_dir (str): Directory where the images are located.
    batch_size (int): The size of each batch. Default is 8.
    shuffle (bool): Whether to shuffle the dataset. Default is False.

    Returns:
    tuple: A tuple of two DataLoaders, one for the training dataset and the other for the validation dataset.
    """
    def __init__(self, train_annotation_path, val_annotation_path, image_dir, batch_size=8, shuffle=False):
        self.train_annotation_path = train_annotation_path
        self.val_annotation_path = val_annotation_path
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_dataloader(self):
        return (DataLoader(LvisDataset(annotation_path=self.train_annotation_path,
                            image_dir=self.image_dir), 
                            batch_size=self.batch_size, 
                            shuffle=self.shuffle,
                            collate_fn=custom_collate_fn), 
               DataLoader(LvisDataset(annotation_path=self.val_annotation_path,
                            image_dir=self.image_dir), 
                            batch_size=self.batch_size, 
                            shuffle=self.shuffle,
                            collate_fn=custom_collate_fn))
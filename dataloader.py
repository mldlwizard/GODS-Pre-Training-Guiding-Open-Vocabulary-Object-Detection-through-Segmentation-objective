"""
Author: Yalala Mohit
Date: 12/10/2023
Course: CS 7180: Advanced Perception, Northeastern University
Description: Contains the classes and methods to load the Validation Dataloader performing operations on the LVIS dataset. It is not essential to download the images, as the dataloader will handle the downloading.
"""

import json
import os

import numpy as np
import torch
from lvis import LVIS, LVISEval
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from transformers import Owlv2ForObjectDetection, Owlv2Processor

ImageFile.LOAD_TRUNCATED_IMAGES = True

def custom_collate_fn(batch):
    """
    A custom collate function for PyTorch DataLoader.

    This function processes a batch and organizes the data into separate variables. 
    It collates 'images' and 'masks_combined' using the default collate function, 
    and zips other elements into their respective tuples.

    Args:
    batch (list of tuples): A batch of data. Each tuple in the batch should contain:
        - image_ids (int)
        - image_dims (tuple)
        - images (np.array)
        - texts (list)
        - masks_combined (np.array)

    Returns:
    tuple: Returns a tuple containing:
        - image_ids: A tuple of image IDs.
        - image_dims: A tuple of image dimensions.
        - images: A batch of images collated into a tensor.
        - texts: A tuple of text lists.
        - masks_combined: A batch of combined masks collated into a tensor.
    """
    # Separate the elements of the batch
    image_ids, image_dims, images, texts, masks_combined = zip(*batch)
    images = default_collate(images)
    masks_combined = default_collate(masks_combined)
    return image_ids, image_dims, images, texts, masks_combined


class LvisDataset(Dataset):
    """
    A dataset class for LVIS (Large Vocabulary Instance Segmentation) dataset.

    Args:
    annotation_path (str): Path to the LVIS dataset annotation file.
    image_dir (str): Directory where the images are stored.

    Attributes:
    lvis (LVIS): An LVIS object for handling dataset operations.
    img_dir (str): Directory where images are stored.
    img_ids (list): List of image IDs from the LVIS dataset.
    img_shape (tuple): The shape to which images will be resized.
    max_text_shape (int): Maximum number of categories to be considered per image.
    """

    def __init__(self, annotation_path, image_dir):
        self.lvis = LVIS(annotation_path=annotation_path)
        self.img_dir = image_dir
        self.img_ids = self.lvis.get_img_ids()
        self.img_shape = (960,960)
        self.max_text_shape = 300

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
        int: The length of the dataset.
        """
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset at the specified index.

        Args:
        idx (int): The index of the item.

        Returns:
        tuple: A tuple containing the image ID, image dimensions, image array, 
               padded text list, and combined mask for the image.
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
        text = list(set([category['name'] for category in categories]))
        if len(text) < self.max_text_shape:
            padded_text = text + ['<PAD>'] * (self.max_text_shape - len(text))
        else:
            padded_text = text[:self.max_text_shape]
        bounding_boxes = [annotation['bbox'] for annotation in annotations]
        mask = [self.lvis.ann_to_mask(ann=annotation) for annotation in annotations]
        if mask:
            mask_combined = (np.logical_or.reduce(mask).astype(np.uint8))*255
        else:
            mask_combined = np.zeros(self.img_shape, dtype=np.uint8)
        mask_combined = np.asarray(Image.fromarray(mask_combined, mode='L').resize(size=self.img_shape,
                                                                        resample=Image.Resampling.BILINEAR))
        return image_id, image_dim, image, padded_text, mask_combined

class LvisDataLoader:
    """
    A DataLoader for the LVIS dataset.

    Args:
    annotation_path (str): Path to the annotation file of the LVIS dataset.
    image_dir (str): Path to the directory containing images.
    batch_size (int): Number of items to be included in each batch.
    shuffle (bool): Whether to shuffle the data.

    Attributes:
    annotation_path (str): Path to the LVIS dataset annotation file.
    image_dir (str): Directory where the dataset images are stored.
    batch_size (int): Batch size.
    shuffle (bool): Flag indicating whether to shuffle the dataset.
    """
    def __init__(self, annotation_path, image_dir, batch_size=8, shuffle=False):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_dataloader(self):
        """
        Get a DataLoader for the LVIS dataset.

        Returns:
        DataLoader: A DataLoader for the LVIS dataset with the specified batch size and shuffle settings.
        """
        return DataLoader(LvisDataset(annotation_path=self.annotation_path,
                            image_dir=self.image_dir), 
                            batch_size=self.batch_size, 
                            shuffle=self.shuffle,
                            collate_fn=custom_collate_fn)
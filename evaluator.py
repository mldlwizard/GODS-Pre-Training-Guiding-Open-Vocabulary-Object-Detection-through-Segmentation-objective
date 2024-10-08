"""
Author: Yalala Mohit
Date: 12/10/2023
Course: CS 7180: Advanced Perception, Northeastern University
Description: Contains the Evaluation script for the LVIS dataset. It loads the LVIS dataset, and generates predictions and later, evaluates to generate the LVIS metrics.
"""

import json
import os
import numpy as np
import torch
from dataloader import LvisDataLoader
from lvis import LVIS, LVISEval
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from transformers import Owlv2ForObjectDetection, Owlv2Processor


class LVISCategoryLookup:
    """
    A lookup utility to map category names to IDs using the LVIS dataset.

    Args:
    annotation_path (str): Path to the LVIS annotation file.

    Attributes:
    category_map (dict): A dictionary mapping category names to IDs.
    """
    def __init__(self, annotation_path):
        self.category_map = self._create_category_map(annotation_path)

    def _create_category_map(self, annotation_path):
        """
        Create a category mapping from the LVIS dataset.

        Args:
        annotation_path (str): Path to the LVIS annotation file.

        Returns:
        dict: A mapping of category names to IDs.
        """
        # Load LVIS dataset
        lvis = LVIS(annotation_path)

        # Get all categories
        categories = lvis.load_cats(lvis.get_cat_ids())

        # Create a map from category names to IDs
        return {category['name']: category['id'] for category in categories}

    def get_category_id(self, category_name):
        """
        Get the category ID for a given category name.

        Args:
        category_name (str): The name of the category.

        Returns:
        int: The ID of the category or None if not found.
        """
        # Return the category ID from the map
        return self.category_map.get(category_name, None)

class LVISEvaluationPipeline:
    """
    Pipeline for evaluating a model using the LVIS dataset.

    Args:
    dataloader (DataLoader): The DataLoader for the dataset.
    model_name (str): Name of the model to be used.
    result_path (str): Path to save the evaluation results.
    annotation_path (str): Path to the LVIS annotation file.
    image_dir (str): Directory containing the images.
    device (str): The device to use for computation.

    Attributes:
    processor (Owlv2Processor): Processor for the OwlV2 model.
    model (Owlv2ForObjectDetection): The OwlV2 model for object detection.
    result_path (str): Path to store the results.
    image_dir (str): Directory of images.
    device (str): Computation device.
    dataloader (DataLoader): DataLoader for the dataset.
    annotation_path (str): Annotation path for LVIS dataset.
    category_lookup (LVISCategoryLookup): Utility for category ID lookup.
    """
    def __init__(self, dataloader, model_name, result_path, annotation_path, image_dir, device='cuda'):
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device)
        self.result_path = result_path
        self.image_dir = image_dir
        self.device = device
        self.dataloader = dataloader
        self.annotation_path = annotation_path
        self.category_lookup =  LVISCategoryLookup(annotation_path)

    def generate_predictions(self):
        """
        Generate predictions using the model and save them to a file.
        """
        self.model.eval()
        results = []

        with torch.no_grad():
            for image_ids, image_dims, images, texts, mask_combined in tqdm(self.dataloader, desc="Generating predictions"):
                batch_results = self.process_batch(images, texts, image_ids, image_dims)
                results.extend(batch_results)

        with open(self.result_path, 'w') as f:
            json.dump(results, f)

    def process_batch(self, images, texts, image_ids, image_dims):
        """
        Process a batch of images and texts to generate predictions.

        Args:
        images (Tensor): Batch of images.
        texts (list): Batch of texts associated with each image.
        image_ids (list): List of image IDs.
        image_dims (list): List of image dimensions.

        Returns:
        list: A list of batch results.
        """
        batch_results = []
        target_sizes = torch.tensor(image_dims, dtype=torch.float32).to(self.device)
        # Prepare the entire batch for processing
        texts = self.remove_pad_efficient(texts)
        inputs = self.processor(text=texts, 
                                images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(outputs=outputs, 
                                                               target_sizes=target_sizes, 
                                                               threshold=0.2)
        
        for idx in range(len(image_ids)):
            image_id = image_ids[idx]
            image = images[idx]
            text = texts[idx]
            top_boundingboxes, top_scores, top_labels = self.filter_top_scores(results[idx]['boxes'],
                                                                              results[idx]['scores'],
                                                                              results[idx]['labels'],
                                                                              300)
            for inner_idx in range(len(top_labels)):
                if top_labels[inner_idx].item() < len(text):
                    score = top_scores[inner_idx].item()
                    cat_id = self.category_lookup.get_category_id(text[top_labels[inner_idx].item()])
                    bounding_box = top_boundingboxes[inner_idx].detach().cpu().numpy().tolist()

                    result = {
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": bounding_box,  # Placeholder
                            "score": score
                        }
                    batch_results.append(result)
                
        return batch_results

    def remove_pad_efficient(self, lists):
        """
        Efficiently remove the '<PAD>' tokens from the list of lists.

        Args:
        lists (list of list): The lists containing '<PAD>' tokens.

        Returns:
        list: Updated lists with '<PAD>' tokens removed.
        """
        updated_lists = []
        for sublist in lists:
            if sublist[-1] != '<PAD>':
                updated_lists.append(sublist)
            else:
                first_pad_index = sublist.index('<PAD>')
                updated_lists.append(sublist[:first_pad_index])
        return updated_lists

    def filter_top_scores(self, bounding_boxes, scores, labels, max_num):
        """
        Filter bounding boxes, scores, and labels to keep only the top 'max_num' entries based on scores.

        Parameters:
        bounding_boxes (Tensor): The bounding boxes tensor.
        scores (Tensor): The scores tensor.
        labels (Tensor): The labels tensor.
        max_num (int): The maximum number of top scoring entries to keep.

        Returns:
        Tuple[Tensor, Tensor, Tensor]: Filtered bounding boxes, scores, and labels.
        """
        if (max_num < len(labels)):
            # Get the indices of the top 'max_num' scores
            scores, indices = scores.topk(max_num)

            # Filter bounding boxes and labels using the top indices
            bounding_boxes = bounding_boxes[indices]
            labels = labels[indices]
        return bounding_boxes, scores, labels

    def evaluate(self):
        """
        Run the evaluation pipeline.
        """
        # Uncomment if there are no generated predictions.
        self.generate_predictions()
        lvis_eval = LVISEval(self.annotation_path, self.result_path, "bbox")
        lvis_eval.run()
        lvis_eval.print_results()

# Parameters
MODEL_NAME = "google/owlv2-base-patch16-finetuned"
# RESULT_PATH = "/home/mohit.y/Owl-Vit_Segmentation/Results/lvis_v1_val_results.json"
RESULT_PATH = "/home/mohit.y/Owl-Vit_Segmentation/Results/lvis_v1_val_results_lvis.json"
IMAGE_DIR = "/work/socialmedia/pytorch/data/val2017/val2017/"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ANNOTATION_PATH = "/work/socialmedia/pytorch/data/lvis_v1_val.json/lvis_v1_val.json"
BATCH_SIZE = 12
VAL_DATALOADER = LvisDataLoader(ANNOTATION_PATH, IMAGE_DIR, BATCH_SIZE).get_dataloader()
# Run Evaluation
pipeline = LVISEvaluationPipeline(VAL_DATALOADER, MODEL_NAME, RESULT_PATH, ANNOTATION_PATH, IMAGE_DIR, DEVICE)
pipeline.evaluate()

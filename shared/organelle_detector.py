import dataclasses
from dataclasses import dataclass
from pathlib import Path

import hyperspy.api as hs
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset

# Define the class map
FULL_CLASS_MAP = {
    "Mitochondria": 1,
    "Golgi body": 2,
    "Lysosome": 3,
    "Secondary Lysosome": 4,
    "ER": 5,
    "Autophagosome": 6,
    "MVB": 7,
    "Vacuole": 8,
    "ILV": 9,
    "Cell": 100,
    "Unsure": 1000,
}

BINARY_CLASS_MAP = {
    "Organelle": 1,
}


class TEMImageDataset(Dataset):
    """
    Build a Torch-compatible dataset.

    Parameters
    ----------
    csv_path : str or Path
        Path to a CSV-file containing image paths and bounding boxes, exported from QuPath
    binarize_classes : bool
        If True, will convert all classes to "Organelle". By default, False.
    """

    def __init__(self, csv_path, binarize_classes=False):

        csv_path = Path(csv_path)

        # Store the data as a DataFrame
        self.df = pd.read_csv(csv_path)

        # Get the list of unique images
        self.images = list(self.df["Image path"].unique())

        # Convert class names to an integer id
        if binarize_classes:
            self.class_map = BINARY_CLASS_MAP
            self.df["Class id"] = 1
        else:
            self.class_map = FULL_CLASS_MAP
            self.df["Class id"] = self.df["Class"].map(self.class_map)

    def __len__(self):
        """Return number of images."""
        return len(self.images)

    def __getitem__(self, idx):
        # image, target

        image_path = self.images[idx]

        # Get the subset of data that match the image path
        annotations = self.df[self.df["Image path"] == image_path]

        # Get the annotations
        labels = annotations["Class id"].values
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        boxes = annotations[["XMin", "YMin", "XMax", "YMax"]].values
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
        }

        # Read the image
        s = hs.load(image_path)
        image = s.data

        img_tensor = torchvision.transforms.functional.to_tensor(image)
        return img_tensor, target


@dataclass
class TrainingConfig:
    """
    Data class for training settings.
    """

    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    num_epochs: int = 15

    checkpoint_path = None
    num_classes: int = 2
    batch_size: int = 2

    shuffle: bool = True
    anchor_sizes: tuple = (
        (32,),
        (64,),
        (128,),
        (320,),
        (768,),
    )
    anchor_aspect_ratios: tuple = (0.5, 1.0, 2.0)
    augment_images = False


def build_model(num_classes):

    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )
    return model


def train(num_classes=2, **kwargs):

    # Generate the TrainingConfig
    config = generate_training_config(**kwargs)

    # Configure CUDA if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the model
    model = build_model(num_classes)
    model.to(device)


def generate_training_config(**kwargs):
    """
    Create a TrainingConfig dataclass.
    """

    # Initialize a TrainingConfig object
    config = TrainingConfig()

    # Validate the inputs
    valid_fields = {f.name for f in dataclasses.fields(TrainingConfig)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

    # Update the TrainingConfig object
    config = dataclasses.replace(config, **filtered_kwargs)

    return config


def clean_data(root_data_dir, base_output_dir):

    # This function should parse the directory tree and extract:
    # CSV-files containing bounding boxes - renaming them with the mouse number
    # Export images to PNG, also renaming them with the mouse number

    root_data_dir = Path(root_data_dir)

    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    qpproj_path_list = list(root_data_dir.rglob("*.qpproj"))

    if not qpproj_path_list:
        raise FileNotFoundError(f"No QuPath projects found at {root_data_dir}")

    df_list = []
    for qpproj_path in qpproj_path_list:
        # Get the project parent folder
        project_folder = qpproj_path.parent

        # Get the Mouse number and the cell number from the project folder name
        mouse_id = (project_folder.name).split("_")[1]
        cell_id = (project_folder.name).split("_")[2]

        # Get the data from the export folder
        export_path = project_folder / "export"
        bbox_csv_list = list(export_path.glob("*_bounding_boxes.csv"))

        if not bbox_csv_list:
            raise FileNotFoundError(f"No CSV-files found at {export_path}")

        for bbox_csv in bbox_csv_list:
            # Read in the data to a DataFrame
            df = pd.read_csv(bbox_csv)

            # Add the mouse number and cell number columns
            df["Mouse ID"] = mouse_id
            df["Cell ID"] = cell_id

            # Add the image name
            image_name = (bbox_csv.stem).split("_")[0]
            df["Image"] = image_name

            # Find the image and record its path
            image_path = (project_folder.parent) / f"{image_name}.dm4"
            if image_path.is_file():
                df["Image path"] = image_path
            else:
                raise FileNotFoundError(f"Could not find image at path {image_path}")

            df_list.append(df)

    merged_df = pd.concat(df_list)

    # Sort the rows by cell ID, then image
    sorted_df = merged_df.sort_values(by=["Cell ID", "Image"])

    # Organize the columns
    front_cols = ["Mouse ID", "Cell ID", "Image"]

    remaining_cols = [col for col in sorted_df.columns if col not in front_cols]

    sorted_df = sorted_df[front_cols + remaining_cols]

    # Write CSV-file
    output_file = base_output_dir / f"mouse{mouse_id}.csv"
    sorted_df.to_csv(output_file, index=False)

    print(f"Output file written to: {output_file}")
    return output_file

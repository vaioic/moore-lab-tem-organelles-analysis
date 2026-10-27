import dataclasses
import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import hyperspy.api as hs
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

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
    "Cell": 10,
    "Unsure": 11,
}

BINARY_CLASS_MAP = {
    "Organelle": 1,
}


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
    binarize_classes: bool = False
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
    augment_images: bool = False
    save_every_n_epochs: int = 5

    def save(self, json_path):
        """Saves the current configuration to a JSON file."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w") as f:
            json.dump(asdict(self), f, indent=4)
        print(f"Configuration saved to JSON: {json_path}")

    @classmethod
    def load(cls, json_path):
        """Loads a TrainingConfig instance from an existing JSON file safely."""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        # Filter out fields that might no longer exist in the dataclass
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Convert JSON lists back into tuples for PyTorch compatibility
        if "anchor_sizes" in filtered_data:
            filtered_data["anchor_sizes"] = tuple(
                tuple(item) for item in filtered_data["anchor_sizes"]
            )
        if "anchor_aspect_ratios" in filtered_data:
            filtered_data["anchor_aspect_ratios"] = tuple(
                filtered_data["anchor_aspect_ratios"]
            )

        return cls(**filtered_data)


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

        # Get the number of cells
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

        # Read the image (expecting a dm4)
        s = hs.load(image_path)
        image = s.data

        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        img_tensor = torchvision.transforms.functional.to_tensor(
            image.astype("float32")
        )
        return img_tensor, target

    @property
    def num_classes(self) -> int:
        return len(self.class_map) + 1


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


def train(
    training_data_path,
    base_output_folder,
    model_name=None,
    validation_data_path=None,
    **kwargs,
):

    # Validate the base output folder
    base_output_folder = Path(base_output_folder)

    # Generate a base filename for the run
    strDateTime = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_name is None:
        base_run_name = strDateTime
    else:
        base_run_name = strDateTime + model_name

    # Create a sub-directory to log run statistics
    log_dir = base_output_folder / "runs" / base_run_name
    log_dir.mkdir(parents=True, exist_ok=False)

    # Initialize the TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    # Create the checkpoints directory
    checkpoints_dir = base_output_folder / "checkpoints" / base_run_name
    checkpoints_dir.mkdir(parents=True, exist_ok=False)

    # Generate and save the TrainingConfig
    config = generate_training_config(**kwargs)
    config.save(checkpoints_dir / "config.json")

    # Set up the datasets
    training_dataset = TEMImageDataset(
        training_data_path, binarize_classes=config.binarize_classes
    )
    # print(f"Training dataset length: {len(training_dataset)}")
    training_data_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=lambda x: tuple(zip(*x)),
    )
    # print(f"Training data loader length: {len(training_data_loader)}")
    num_classes = training_dataset.num_classes

    if validation_data_path:
        validation_dataset = TEMImageDataset(
            validation_data_path, binarize_classes=config.binarize_classes
        )
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            collate_fn=lambda x: tuple(zip(*x)),
        )

    # Configure CUDA if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the model
    model = build_model(num_classes)
    model.to(device)

    # Set up the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    # ---Training loop---
    model.train()
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        for images, targets in training_data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        training_loss = epoch_loss / len(training_data_loader)
        print(f"Epoch [{epoch + 1}/{config.num_epochs}] | Loss: {training_loss:.4f}")

        # Log metrics to tensor board
        writer.add_scalar("Loss/Train", training_loss, epoch + 1)

        # -- Validation loop (if validation data is set)
        if validation_data_path:
            val_loss = 0
            with torch.no_grad():
                for images, targets in validation_data_loader:
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    val_loss_dict = model(images, targets)
                    val_losses = sum(loss for loss in val_loss_dict.values())
                    val_loss += val_losses.item()

            validation_loss = (
                val_loss / len(validation_data_loader)
                if len(validation_data_loader) > 0
                else 0.0
            )

            print(
                f"Epoch [{epoch + 1}/{config.num_epochs}] | Validation Loss: {validation_loss:.4f}"
            )

            writer.add_scalar("Loss/Validation", validation_loss, epoch + 1)

        # Save checkpoints
        if (epoch + 1) % config.save_every_n_epochs == 0 or (
            epoch + 1
        ) == config.num_epochs:
            ckpt_path = checkpoints_dir / f"follicle_detector_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"-> Checkpoint saved: {ckpt_path}")

    # Write hyperparameters to log after a successful run
    final_metrics = {
        "hparam/train_loss": training_loss,
    }
    if validation_data_path:
        final_metrics["hparam/val_loss"] = validation_loss

    hparams_dict = asdict(config)
    hparams_dict["anchor_sizes"] = str(config.anchor_sizes)
    hparams_dict["anchor_aspect_ratios"] = str(config.anchor_aspect_ratios)

    writer.add_hparams(hparams_dict, final_metrics, run_name=".")
    writer.flush()
    writer.close()

    # Save the final model
    final_model_dir = base_output_folder / "model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = final_model_dir / (base_run_name + ".pth")

    torch.save(model.state_dict(), final_model_path)
    print("-------------------------------------------------------------")
    print(f"Training complete. Final weights saved to {final_model_path}")

    return final_model_path


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

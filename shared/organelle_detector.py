import dataclasses
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

import cv2
import hyperspy.api as hs
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
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

    Attributes
    ----------
    learning_rate : float, optional
        Optimizer learning rate. Default is 0.005.
    momentum : float, optional
        SGD momentum factor. Default is 0.9.
    weight_decay : float, optional
        L2 weight decay penalty. Default is 0.0005.
    num_epochs : int, optional
        Total number of training epochs. Default is 15.
    checkpoint_path : str or Path, optional
        Path to a checkpoint file to resume training from. Default is None.
    binarize_classes : bool, optional
        If True, collapses classes to a binary foreground/background mapping.
        Default is False.
    batch_size : int, optional
        Number of images per batch. Default is 2.
    shuffle : bool, optional
        Whether to shuffle training data each epoch. Default is True.
    anchor_sizes : tuple of tuple of int, optional
        Anchor box sizes for the RPN. Default is ((32,), (64,), (128,), (320,), (768,)).
    anchor_aspect_ratios : tuple of float, optional
        Anchor aspect ratios for the RPN. Default is (0.5, 1.0, 2.0).
    augment_images : bool, optional
        Whether to apply data augmentations. Default is False.
    save_every_n_epochs : int, optional
        Frequency of checkpoint saving. Default is 5.
    class_map : dict, optional
        Mapping dictionary linking class names to integer IDs.
        Automatically populated if empty based on `binarize_classes`.
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

    class_map: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.class_map:
            if self.binarize_classes:
                self.class_map = BINARY_CLASS_MAP
            else:
                self.class_map = FULL_CLASS_MAP

    @property
    def num_classes(self) -> int:
        """Total number of classes including background"""
        return len(self.class_map) + 1

    def save(self, json_path):
        """Saves the current configuration to a JSON file."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w") as f:
            json.dump(asdict(self), f, indent=4)
        print(f"Configuration saved to JSON: {json_path}")

    @classmethod
    def load(cls, json_path):
        """Loads a TrainingConfig instance from an existing JSON file"""
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

    def from_args(**kwargs):
        """
        Create a TrainingConfig dataclass from passed arguments
        """

        # Initialize a TrainingConfig object
        config = TrainingConfig()

        # Validate the inputs
        valid_fields = {f.name for f in dataclasses.fields(TrainingConfig)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        return dataclasses.replace(config, **filtered_kwargs)


class TEMImageDataset(Dataset):
    """
    Build a Torch-compatible dataset.

    Parameters
    ----------
    csv_path : str or Path
        Path to a CSV-file containing image paths and bounding boxes, exported from QuPath
    config : TrainingConfig
        Configuration parameters for the dataset (primarily binarization of classes and
        the class map)

    Attributes
    ----------
    df : pandas.DataFrame
        Annotations imported from QuPath containing the image path, bounding box
        coordinates, and classification.
    images : list of str
        Unique image paths extracted from the dataset.
    class_map : dict
        Dictionary that maps the class names to integer IDs.
    """

    def __init__(self, csv_path, config: TrainingConfig):

        csv_path = Path(csv_path)

        # Store the data as a DataFrame
        self.df = pd.read_csv(csv_path)

        # Get the number of cells
        self.images = list(self.df["Image path"].unique())

        # Define the class map
        if config.binarize_classes:
            self.df["Class id"] = 1
        else:
            self.df["Class id"] = self.df["Class"].map(config.class_map)

    def __len__(self):
        """Return number of images."""
        return len(self.images)

    def __getitem__(self, idx):
        # The output should match what is needed for pytorch DataLoader.
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
        image = read_image(image_path)
        img_tensor = image_to_tensor(image)

        return img_tensor, target


def read_image(image_path):
    """
    Read in an image.

    Will use different readers depending on filetype.

    Parameters
    ----------
    image_path : str or Path
        Path to image file.

    Raises
    ------
    FileNotFoundError
        Image file was not found at the provided path.
    ValueError
        Image file contains a currently unsupported file extension.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Could not find file {image_path}.")

    ext = image_path.suffix.lower()

    match ext:
        case ".dm4":
            s = hs.load(image_path)
            image = s.data
        case ".tiff" | ".tif":
            import tifffile

            image = tifffile.imread(image_path)
        case ".png" | ".jpg" | ".jpeg":
            image = np.array(Image.open(image_path))
        case _:
            raise ValueError(f"Unsupported file extension: {ext}")

    # Normalize the image
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)


def image_to_tensor(image):
    """
    Convert image to torch tensor.

    Parameters
    ----------
    image : ndarray
        Image data to convert

    Returns
    -------
    img_tensor : torch.tensor
        Float 32 Torch tensor.
    """

    img_tensor = torchvision.transforms.functional.to_tensor(image.astype("float32"))

    return img_tensor


def build_rcnn_model(num_classes):
    """
    Build the Faster-RCNN model.

    Future uses could expand this to include other models.

    Parameters
    ----------
    num_classes : int
        Number of classes that you want to classify.

    Returns
    -------
    nodel : FasterRCNN
        Faster-RCNN model.
    """

    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )
    return model


def train_rcnn(
    training_data_path,
    base_output_folder,
    validation_data_path=None,
    model_name=None,
    **kwargs,
):
    """
    Train a Faster-RCNN model using the specified configuration settings.

    The function generates the following outputs:
        - runs/<unique_fn>/          : TensorBoard event logs (loss curves, metrics)
        - checkpoints/<unique_fn>/   : Saved training epochs and final model weights
        - config.json                : Run configuration settings and hyperparameters
        - model/<unique_fn>.pth      : Final trained model weights

    Note: <unique_fn> is a filename generated by using the date and time and the
    model_name parameter if set.

    Parameters
    ----------
    training_data_path : str or Path
        Path to the training annotation CSV-file.
    base_output_folder : str or Path
        Path to the base output folder.
    validation_data_path : str or Path, optional
        If provided, will use the file provided to run validation during the training loop. By default, None.
    model_name : str, optional
        If provided, appends the name to the model files. By default, None.
    **kwargs : dict, optional
        Arbitrary keyword arguments to override default parameters in
        the `TrainingConfig` dataclass (e.g., `learning_rate`, `batch_size`,
        `num_epochs`).

    Returns
    -------
    Path
        Path to the trained model.
    """

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
    config = TrainingConfig.from_args(**kwargs)
    config.save(checkpoints_dir / (base_run_name + "_config.json"))

    # Set up the datasets
    training_dataset = TEMImageDataset(training_data_path, config)
    # print(f"Training dataset length: {len(training_dataset)}")
    training_data_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=lambda x: tuple(zip(*x)),
    )
    # print(f"Training data loader length: {len(training_data_loader)}")

    if validation_data_path:
        validation_dataset = TEMImageDataset(validation_data_path, config)
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            collate_fn=lambda x: tuple(zip(*x)),
        )

    # Configure CUDA if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the model
    num_classes = config.num_classes
    model = build_rcnn_model(num_classes)
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
            ckpt_path = checkpoints_dir / f"{base_run_name}_rcnn_epoch_{epoch + 1}.pth"
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

    # Also save the configuration in the completed model folder
    config.save(final_model_dir / (base_run_name + "_config.json"))

    print("-------------------------------------------------------------")
    print(f"Training complete. Final weights saved to {final_model_path}")

    return final_model_path


# ------------Prediction code------------------


@dataclass
class PredictionConfig:
    rcnn_model_path: str = None
    rcnn_confidence_thresh: float = 0.5
    rcnn_save_boxes: bool = True

    _rcnn_config_path: str = None

    use_segment_anything: bool = True
    sam_checkpoint_path: str = "./sam_checkpoints/sam_vit_h_4b8939.pth"
    sam_model_name: str = "vit_h"

    def save(self, json_path):
        """Saves the current configuration to a JSON file."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self)
        config_dict["rcnn_config_path"] = config_dict.pop("_rcnn_config_path")

        with open(json_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        print(f"Configuration saved to JSON: {json_path}")

    @classmethod
    def load(cls, json_path):
        """Loads a TrainingConfig instance from an existing JSON file"""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        # Filter out fields that might no longer exist in the dataclass
        valid_fields = {f.name for f in fields(cls)}

        filtered_data = {}
        for k, v in data.items():
            if k == "rcnn_model_path":
                filtered_data["_rcnn_model_path"] = v
            elif k in valid_fields:
                filtered_data[k] = v

        return cls(**filtered_data)

    def from_args(**kwargs):
        """
        Create a PredictionConfig dataclass from passed arguments
        """

        # Initialize a TrainingConfig object
        config = PredictionConfig()

        # Validate the inputs
        valid_fields = {f.name for f in dataclasses.fields(PredictionConfig)}

        filtered_kwargs = {}
        for k, v in kwargs.items():
            if k == "rcnn_model_path":
                filtered_kwargs["_rcnn_model_path"] = v
            elif k in valid_fields:
                filtered_kwargs[k] = v

        return dataclasses.replace(config, **filtered_kwargs)

    @property
    def rcnn_config_path(self):

        if self._rcnn_config_path is not None:
            return self._rcnn_config_path

        # Try to obtain the path automatically
        config_path = (
            self.rcnn_model_path.parent / f"{self.rcnn_model_path.stem}_config.json"
        )

        if config_path.exists():
            return config_path
        else:
            # Fallback - try to find the file recursively
            p = list(
                self.rcnn_model_path.parent.rglob(
                    f"{self.rcnn_model_path.stem}_config.json"
                )
            )

            if p:
                return p[0]
            else:
                raise FileNotFoundError(
                    f"Configuration file for {self.rcnn_model_path.name} was not found."
                )


def segment_organelles(images, output_dir, **kwargs):

    # Parse the images input argument
    if not isinstance(images, list):
        images = [
            images,
        ]

    image_path_list = []
    for p in images:
        p = Path(p)

        if p.is_file():
            image_path_list.append(p)
        elif p.is_dir():
            # Find images
            dm4_files = list(p.rglob("*.dm4"))

            if dm4_files:
                image_path_list.extend(dm4_files)
            else:
                raise FileNotFoundError(f"No .dm4 files found in {p}")

    # Validate the output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the PredictionConfig
    pred_config = PredictionConfig.from_args(**kwargs)

    # Predict bounding boxes
    bbox_results = predict_bounding_boxes(
        pred_config.rcnn_model_path,
        pred_config.rcnn_config_path,
        image_path_list,
        output_dir=output_dir,
        confidence_thresh=pred_config.rcnn_confidence_thresh,
        save_geojson=pred_config.rcnn_save_boxes,
    )

    # Predict segmentation
    segment_sam2(bbox_results, output_dir, pred_config)


def predict_bounding_boxes(
    model_path,
    config_path,
    image_paths,
    output_dir,
    confidence_thresh=0.5,
    save_geojson=True,
):

    # Validate the inputs
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"File not found at path: {model_path}")

    config_path = Path(config_path)

    image_paths = [Path(p) for p in image_paths]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up the prediction model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the TrainingConfig
    config = TrainingConfig.load(config_path)

    # Invert the class map dict
    idx_to_class = {v: k for k, v in config.class_map.items()}

    # Build the model
    model = build_rcnn_model(config.num_classes)

    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for img_path in image_paths:
            img_tensor = image_to_tensor(img_path)
            img_tensor = img_tensor.unsqueeze(0).to(device)

            predictions = model(img_tensor)[0]
            scores = predictions["scores"].cpu().numpy()
            keep = scores >= confidence_thresh

            boxes = predictions["boxes"].cpu().numpy()[keep]
            labels = predictions["labels"].cpu().numpy()[keep]
            filtered_scores = scores[keep]

            # Save data in raw format for SAM2 prediction
            raw_results = {
                "image_path": str(img_path),
                "boxes": boxes.tolist(),
                "labels": labels.tolist(),
                "scores": filtered_scores.tolist(),
            }
            results.append(raw_results)

            output_fn = img_path.stem
            json_file_path = output_dir / f"{output_fn}_bbox.json"
            with open(json_file_path, "w") as f:
                json.dump(raw_results, f, indent=2)
            print(f"Bounding box saved to JSON file: {json_file_path}")

            if save_geojson:
                all_features = []
                for box, label, score in zip(boxes, labels, filtered_scores):
                    xmin, ymin, xmax, ymax = box
                    class_name = idx_to_class.get(str(label), f"Class_{label}")

                    polygon_coords = [
                        [float(xmin), float(ymin)],
                        [float(xmax), float(ymin)],
                        [float(xmax), float(ymax)],
                        [float(xmin), float(ymax)],
                        [float(xmin), float(ymin)],
                    ]

                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [polygon_coords],
                        },
                        "properties": {
                            "object_type": "annotation",
                            "classification": class_name,
                            "confidence": float(score),
                        },
                    }
                    all_features.append(feature)

                # Construct the final GeoJSON FeatureCollection
                geojson_collection = {
                    "type": "FeatureCollection",
                    "features": all_features,
                }

                geojson_file_path = output_dir / f"{output_fn}_bbox_geojson.json"

                with open(geojson_file_path, "w") as f:
                    json.dump(geojson_collection, f, indent=2)

                print(f"Bounding box saved to GeoJSON: {geojson_file_path}")

    return results


def segment_sam2(bbox, output_dir, config: PredictionConfig):

    from segment_anything import SamPredictor, sam_model_registry

    # Build the SAM model
    sam = sam_model_registry[config.sam_model_name](
        checkpoint=config.sam_checkpoint_path
    )

    predictor = SamPredictor(sam)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if not isinstance(bbox, list):
        bbox = [bbox]

    sam2_results = []
    for b in bbox:
        if isinstance(b, dict):
            rcnn_results = b
        elif isinstance(b, (str, Path)):
            with open(b, "r") as f:
                rcnn_results = json.load(f)
        else:
            raise ValueError(
                f"Expected input to be a results dict or a path to a CSV-file. Instead it is a {type(b).__name__}"
            )

        img_path = Path(rcnn_results["image_path"])
        # Reshape boxes into (N, 4)
        boxes = np.array(rcnn_results["boxes"], dtype=np.float32)

        s = hs.load(img_path)
        image_np = s.data

        image_np = (image_np - image_np.min()) / (
            image_np.max() - image_np.min() + 1e-8
        )
        image_np = (image_np * 255).astype(np.uint8)

        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)

        predictor.set_image(image_np)

        segmentation_masks = []
        segmentation_scores = []
        for box in boxes:
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )

            segmentation_masks.append(masks)
            segmentation_scores.append(scores)

        # Convert to GeoJSON QuPath object
        features = []
        obj_idx = 0
        for idx, mask in enumerate(segmentation_masks):
            mask = masks[idx].squeeze()

            label = b.label[idx]

            feature = mask_to_geojson_feature(mask, label, scores[idx], obj_idx)
            if feature:
                features.append(feature)
                obj_idx += 1

        geojson_data = {"type": "FeatureCollection", "features": features}

        # Save the objects
        output_geojson_path = output_dir / "test_annotations.geojson"
        with open(output_geojson_path, "w") as f:
            json.dump(geojson_data, f, indent=2)

        print(f"GeoJSON exported to: {output_geojson_path}")

        # Create the results dict
        sam2_results.append(
            {
                "image_path": rcnn_results.im,
                "masks": segmentation_masks,
                "scores": segmentation_scores,
            }
        )

    return sam2_results


def mask_to_geojson_feature(mask, label, score, object_id):

    binary_mask = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    coords = approx_contour.squeeze(axis=1).tolist()

    if len(coords) < 3:
        return None

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    feature = {
        "type": "Feature",
        "id": f"detection_{object_id}",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {
            "objectType": "annotation",
            "classification": {"name": str(label)},
            "confidence": score,
        },
    }
    return feature


# def download_sam2_weights(checkpoint_name: str = "sam2.1_hiera_large.pt"):

#     # Check if weight already exists
#     sam2_checkpoints_dir = Path("../sam2_checkpoints")
#     sam2_checkpoints_dir.mkdir(parents=True, exist_ok=True)

#     chkpt_file = sam2_checkpoints_dir / checkpoint_name

#     if chkpt_file.exists():
#         print(f"Checkpoint {chkpt_file} exists.")
#         return chkpt_file

#     base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
#     url = f"{base_url}{checkpoint_name}"

#     print(f"Checkpoint not found at {chkpt_file}.")
#     print(f"Downloading {checkpoint_name} from Meta AI (this may take a minute)...")

#     try:
#         # A progress bar or basic download handler
#         def _progress(block_num, block_size, total_size):
#             downloaded = block_num * block_size
#             if total_size > 0:
#                 percent = min(int(downloaded * 100 / total_size), 100)
#                 print(
#                     f"\rDownloading: {percent}% ({downloaded / (1024 * 1024):.1f} MB)",
#                     end="",
#                 )

#         urllib.request.urlretrieve(url, chkpt_file, reporthook=_progress)
#         print("\nDownload complete!")
#     except Exception as e:
#         if chkpt_file.exists():
#             chkpt_file.unlink()  # Clean up partial file on failure
#         raise RuntimeError(f"Failed to download SAM 2 weights from {url}. Error: {e}")

#     return chkpt_file

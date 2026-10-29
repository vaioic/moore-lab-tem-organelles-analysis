from pathlib import Path

import pandas as pd


def clean_data(root_data_dir, base_output_dir):
    """
    Parse directory tree and combine annotation CSV-files.

    This helper function can be used to collect and merge the bounding-box information
    exported from QuPath. The function will validate that the image files exist in the
    specified location.

    The function assumes that the Qu{ath projects are saved in the directory
    structure:
        <root_folder>/Mouse_ID/Cell_ID/
          - images.dm4
          - qupath_<mouse_ID>_<cell_ID>/project.qpproj
          - qupath_<mouse_ID>_<cell_ID>/export/annotations.csv

    The final annotation CSV-file contains the following columns:
      - Mouse ID
      - Cell ID
      - Image (Image name)
      - Class
      - XMin
      - YMin
      - XMax
      - YMax
      - Image path

    Parameters
    ----------
    root_data_dir : str or Path
        Path to root data directory to parse.
    base_output_dir : str or Path
        Path to output directory for clean files.

    Returns
    -------
    output_file_path : str or Path
        Path to the output file

    Raises
    ------
    FileNotFoundError
        No .qpproj file was found along the path specified.
    FileNotFoundError
        No .csv files were found along the path specified.
    FileNotFoundError
        No images were found along the path specified.
    """

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

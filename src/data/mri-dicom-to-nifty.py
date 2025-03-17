"""
https://github.com/LIA-DiTella/PET-IA/blob/main/Full-DBs/mri/mri-dicom-to-nifty.py

Author: Hugo Alberto Massaroli
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import dicom2nifti
import argparse
import datetime
import json


def identify_scan_folders(base_path):
    """
    Identify and categorize ADNI scan folders

    Parameters:
        base_path (str): Base path containing ADNI folders
    Returns:
        dict: Dictionary of scan types and their paths
    """
    scan_types = {"mprage": [], "localizer": [], "pd_t2": [], "b1_cal": []}

    for folder in Path(base_path).iterdir():
        if folder.is_dir():
            folder_name = folder.name.lower()
            if "mp-rage" in folder_name:
                scan_types["mprage"].append(folder)
            if "mprage" in folder_name:
                scan_types["mprage"].append(folder)
            elif "localizer" in folder_name:
                scan_types["localizer"].append(folder)
            elif "pd_t2" in folder_name or "fse" in folder_name:
                scan_types["pd_t2"].append(folder)
            elif "b1" in folder_name and "cal" in folder_name:
                scan_types["b1_cal"].append(folder)

    return scan_types


def check_mprage_quality(mprage_folders):
    """
    Determine which MP-RAGE scan to use

    Parameters:
        mprage_folders (list): List of MP-RAGE folder paths
    Returns:
        Path: Path to the best MP-RAGE scan
    """
    if not mprage_folders:
        raise ValueError("No MP-RAGE folders found")

    # If there's only one MP-RAGE, use it
    if len(mprage_folders) == 1:
        return mprage_folders[0]

    # If there's a repeat, check both and use the better one
    # First, check if one is marked as 'repeat'
    for folder in mprage_folders:
        if "repeat" in folder.name.lower():
            return folder

    # If no repeat marking, use the first one by default
    # You might want to add more sophisticated quality checks here
    return mprage_folders[0]


def convert_dicoms_to_nifti(dicom_folder, output_dir):
    """
    Convert DICOM series to NIfTI format

    Parameters:
        dicom_folder (Path): Path to DICOM folder
        output_dir (Path): Output directory for NIfTI file
    Returns:
        Path: Path to converted NIfTI file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert entire DICOM series to NIfTI
    try:
        dicom2nifti.convert_directory(
            str(dicom_folder), str(output_dir), compression=True, reorient=True
        )
    except Exception as e:
        print(f"Error converting DICOMs: {e}")
        raise

    # Find the created NIfTI file (should be only one)
    nifti_files = list(output_dir.glob("*.nii.gz"))
    if not nifti_files:
        raise FileNotFoundError("No NIfTI file was created")

    return nifti_files[0]


def verify_nifti_quality(nifti_path):
    """
    Perform basic quality checks on the converted NIfTI

    Parameters:
        nifti_path (Path): Path to NIfTI file
    Returns:
        bool: Whether the file passes quality checks
    """
    try:
        img = nib.load(str(nifti_path))
        data = img.get_fdata()

        # Basic quality checks
        if data.size == 0:
            return False
        if np.all(data == 0):
            return False
        if np.any(np.isnan(data)):
            return False

        # Check dimensions (typical MP-RAGE should be ~256x256x160 or similar)
        if any(dim < 100 for dim in data.shape):
            print("Warning: Unusual image dimensions")
            return False

        return True
    except Exception as e:
        print(f"Error checking NIfTI quality: {e}")
        return False


def organize_scan(base_path, output_dir):
    """
    Main function to organize and convert ADNI scans

    Parameters:
        base_path (str): Path to ADNI scan folders
        output_dir (str): Output directory for converted files
    Returns:
        dict: Paths to converted files
    """
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identify all scan folders
    scan_types = identify_scan_folders(base_path)

    # Select the best MP-RAGE scan
    mprage_folder = check_mprage_quality(scan_types["mprage"])

    # Convert MP-RAGE to NIfTI
    nifti_output_dir = output_dir / "nifti"
    nifti_file = convert_dicoms_to_nifti(mprage_folder, nifti_output_dir)

    # Verify the conversion
    if not verify_nifti_quality(nifti_file):
        raise ValueError("Converted NIfTI failed quality checks")

    # Create organized structure
    organized_files = {
        "mprage": nifti_file,
        "original_folder": mprage_folder,
        "conversion_date": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Save organization info
    info_file = output_dir / "scan_info.json"
    with open(info_file, "w") as f:
        json.dump(
            {
                "mprage_source": str(mprage_folder),
                "nifti_output": str(nifti_file),
                "conversion_date": organized_files["conversion_date"],
            },
            f,
            indent=2,
        )

    return organized_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("base_path")  # pathto adni folders
    parser.add_argument("output_dir")  # path to output
    args = parser.parse_args()

    try:
        organized_files = organize_scan(args.base_path, args.output_dir)
        print(
            f"Successfully converted files. MP-RAGE NIfTI located at: {organized_files['mprage']}"
        )
    except Exception as e:
        print(f"Error processing scans: {e}")

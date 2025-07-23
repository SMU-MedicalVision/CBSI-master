import os
import glob
import argparse
import numpy as np
import nibabel as nib


def normalize_modality(data, brain_mask, scale, shift):
    """
    Normalize single modality MRI data using histogram-based scaling.

    Args:
        data (ndarray): Input 3D MRI volume
        brain_mask (ndarray): Binary brain mask (1=brain, 0=background)
        scale (float): Scaling factor for normalization
        shift (float): Shift value for normalization

    Returns:
        ndarray: Normalized volume with same shape as input
    """
    # Convert to 8-bit unsigned integer [0,255]
    # byte_data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
    byte_data = (data - data.min()) / (data.max() - data.min()) * 255

    # Only consider non-zero voxels within brain mask
    masked_data = byte_data[(brain_mask > 0) & (byte_data > 0)]

    # Calculate histogram and find peak intensity
    hist, bins = np.histogram(masked_data, bins=255, range=(0, 255))
    peak_value = bins[np.argmax(hist)]

    # Center data around peak and calculate standard deviation
    # adjusted = byte_data.astype(np.float32) - peak_value
    adjusted = byte_data - peak_value
    squared = np.square(adjusted) * brain_mask
    sigma = np.sqrt(np.sum(squared) / np.sum(brain_mask))

    # Apply z-score normalization with custom scaling
    normalized = (adjusted / sigma) * scale + shift
    return normalized * brain_mask


def Crop_padding_volume(brain_mask_data, crop_size):
    XX, YY, ZZ = np.where(brain_mask_data > 0)
    min_X, max_X = np.min(XX), np.max(XX)
    min_Y, max_Y = np.min(YY), np.max(YY)
    min_Z, max_Z = np.min(ZZ), np.max(ZZ)
    size_Z = max_Z-min_Z + 1
    center_X = (min_X + max_X) // 2
    center_Y = (min_Y + max_Y) // 2
    if max_X- min_X + 1 > crop_size:
        min_X = center_X - crop_size // 2
        max_X = center_X + crop_size // 2
        frame_size_X = crop_size // 2
    else:
        frame_size_X = (max_X - min_X + 1)//2
        min_X = center_X - frame_size_X
        max_X = center_X + frame_size_X
    if min_X<0:
        min_X = 0
        max_X = frame_size_X * 2
    if max_X>brain_mask_data.shape[0]:
        max_X = brain_mask_data.shape[0]
        min_X = max_X - frame_size_X * 2

    if max_Y - min_Y + 1 > crop_size:
        min_Y = center_Y - crop_size // 2
        max_Y = center_Y + crop_size // 2
        frame_size_Y = crop_size // 2
    else:
        frame_size_Y = (max_Y - min_Y + 1)//2
        min_Y = center_Y - frame_size_Y
        max_Y = center_Y + frame_size_Y
    if min_Y<0:
        min_Y = 0
        max_Y = frame_size_Y * 2
    if max_Y>brain_mask_data.shape[1]:
        max_Y = brain_mask_data.shape[1]
        min_Y = max_Y - frame_size_Y * 2
    return frame_size_X, frame_size_Y, size_Z, min_X, max_X, min_Y, max_Y, min_Z, max_Z+1


def process_subject(input_dir, output_dir, crop_size=424):
    """
    Process MRI data for a single subject.

    Args:
        input_dir (str): Path to subject's raw data directory
        output_dir (str): Path to save processed data
    """
    os.makedirs(output_dir, exist_ok=True)

    # Locate required MRI files
    files = {
        'ROI': glob.glob(os.path.join(input_dir, 'ROI.nii*')),
        'T2F': glob.glob(os.path.join(input_dir, 'T2F.nii*')),
        'T1C': glob.glob(os.path.join(input_dir, 'T1C.nii*')),
        'T1': glob.glob(os.path.join(input_dir, 'T1.nii*')),
        'Brain_mask': glob.glob(os.path.join(input_dir, 'Brain_mask.nii*'))
    }

    # Validate file existence (except Brain_mask which is optional)
    del_T1C = False
    for mod, paths in files.items():
        if not paths:
            if mod == 'Brain_mask':
                print(f"Missing {mod} file, Automatically generate")
            elif mod == 'T1C':
                # 字典files去掉'T1C'
                del_T1C = True
            else:
                raise FileNotFoundError(f"Missing {mod} file in {input_dir}")
    # Handle Brain_mask - copy if exists, otherwise create new
    if files['Brain_mask']:
        # Load and save using nibabel instead of shutil.copy
        brain_mask = nib.load(files['Brain_mask'][0])
        brain_mask_data = brain_mask.get_fdata()
        mask_affine = brain_mask.affine
    else:
        # Create new mask from first available modality if none exists
        img = nib.load(files['T1'][0])  # Using T1 as reference
        data = img.get_fdata()
        data[data<0] = 0
        mask_affine = img.affine
        # Create binary brain mask (threshold > 0.01)

        brain_mask_data = np.where(data > 0.01, 1, 0).astype(np.float32)

    frame_size_X, frame_size_Y, size_Z, min_X, max_X, min_Y, max_Y, min_Z, max_Z = Crop_padding_volume(brain_mask_data, crop_size)



    # Process each MRI modality with modality-specific parameters
    modality_params = {
        'T2F': (30, 75),  # (scale, shift)
        'T1C': (31, 99),
        'T1': (31, 99),
        'ROI': (None, None),
        'Brain_mask': (None, None),
    }
    if del_T1C:
        del modality_params['T1C']

    for mod, (scale, shift) in modality_params.items():

        if 'mask' in mod.lower():
            data = brain_mask_data
            affine = mask_affine
        else:
            # Load image data
            img = nib.load(files[mod][0])
            data = img.get_fdata()
            affine = img.affine
        if 'roi' not in mod.lower() and 'mask' not in mod.lower() :
            # Normalize and clip intensities
            data = normalize_modality(data, brain_mask_data, scale, shift)
            data = np.clip(data, 0, 255)
            data = data * brain_mask_data

        crop_pad_data = np.zeros((crop_size, crop_size, size_Z))
        crop_pad_data[crop_size//2-frame_size_X:crop_size//2+frame_size_X,
                      crop_size//2-frame_size_Y:crop_size//2+frame_size_Y, :] = data[min_X:max_X, min_Y:max_Y, min_Z:max_Z]
        assert crop_pad_data.shape[0] == crop_size and crop_pad_data.shape[1] == crop_size
        # Save processed volume
        output_img = nib.Nifti1Image(crop_pad_data, affine)
        nib.save(output_img, os.path.join(output_dir, f'{mod}.nii.gz'))

        # Print intensity range for QA
        print(f"{mod} Shape {crop_pad_data.shape} || intensity range: [{np.min(data):.1f}, "f"{np.max(data):.1f}]")


def main(override=False):
    """Main pipeline execution."""
    # Configure paths
    datset_list = ['Train', 'Val', 'Test']
    ET_label_list = ['ET-1', 'ET-0']
    required_files = ['T2F.nii.gz', 'T1C.nii.gz', 'T1.nii.gz', 'ROI.nii.gz', 'Brain_mask.nii.gz']

    for Dataset in datset_list:
        for ET_label in ET_label_list:
            if 'test' not in Dataset.lower():
                dataset_dir = f"./Glioma_DATA/RAW_DATA/{Dataset}/{ET_label}/"
                output_base = f"./Glioma_DATA/Preprocessing_DATA/{Dataset}/{ET_label}/"
            else:
                dataset_dir = f"./Glioma_DATA/RAW_DATA/{Dataset}/"
                output_base = f"./Glioma_DATA/Preprocessing_DATA/{Dataset}/"

            subjects = [  # Get all subject directories
                d for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d))
            ]

            for subject in subjects:
                input_dir = os.path.join(dataset_dir, subject)
                output_dir = os.path.join(output_base, subject)

                if all(os.path.exists(os.path.join(output_dir, f)) for f in required_files) and not override:
                    print(f"{Dataset} Subject {subject} already processed - skipping")
                    continue

                print(f"\nProcessing {Dataset} {subject}")
                try:
                    process_subject(input_dir, output_dir)
                    print(f"Successfully processed {subject}")
                except Exception as e:
                    print(f"!!! ERROR processing {subject}: {str(e)}")
                    # Only remove directory if we created it
                    if os.path.exists(output_dir) and not os.listdir(output_dir):
                        os.rmdir(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI Data Processing Pipeline")
    parser.add_argument('--override', action='store_true', help='Override existing processed files')
    opt = parser.parse_args()
    main(override=opt.override)
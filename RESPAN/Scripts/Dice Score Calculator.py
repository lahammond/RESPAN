import os
import numpy as np
import tifffile
import csv
#from collections import defaultdict


def calculate_dice_score(image1, image2, intensity):
    mask1 = image1 == intensity
    mask2 = image2 == intensity
    intersection = np.logical_and(mask1, mask2)
    return (2. * intersection.sum()) / (mask1.sum() + mask2.sum())


def process_image_pair(image1_path, image2_path, num_intensities):
    image1 = tifffile.imread(image1_path)
    image2 = tifffile.imread(image2_path)

    if image1.shape != image2.shape:
        raise ValueError(f"Images {image1_path} and {image2_path} must have the same dimensions")

    dice_scores = []
    for intensity in range(num_intensities):
        score = calculate_dice_score(image1, image2, intensity)
        dice_scores.append(score)

    return dice_scores


def process_folders(folder1, folder2, num_intensities):
    all_scores = []
    image_names = []

    # Get list of files in folder1
    files = [f for f in os.listdir(folder1) if f.endswith('.tif')]

    for file in files:
        image1_path = os.path.join(folder1, file)
        image2_path = os.path.join(folder2, file)

        if not os.path.exists(image2_path):
            print(f"Warning: Matching image not found for {file}")
            continue

        try:
            #print a counter to keep track of the progress
            print(f"Processing file {file} of {len(files)} files")
            scores = process_image_pair(image1_path, image2_path, num_intensities)
            all_scores.append(scores)
            image_names.append(file)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

    return image_names, all_scores


def generate_report(image_names, all_scores, num_intensities, output_folder, ModelID):
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{ModelID}_dice_scores_report.csv")

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = ["Image Name"] + [f"Intensity {i}" for i in range(num_intensities)] + ["Mean"]
        writer.writerow(header)

        # Write individual image scores
        for name, scores in zip(image_names, all_scores):
            row = [name] + [f"{score:.4f}" for score in scores] + [f"{np.mean(scores):.4f}"]
            writer.writerow(row)

        # Write mean scores across all images
        mean_scores = np.mean(all_scores, axis=0)
        writer.writerow([])  # Empty row for readability
        writer.writerow(
            ["Mean Across All Images"] + [f"{score:.4f}" for score in mean_scores] + [f"{np.mean(mean_scores):.4f}"])

        # Write overall mean
        overall_mean = np.mean(mean_scores)
        writer.writerow([])  # Empty row for readability
        writer.writerow(["Overall Mean", f"{overall_mean:.4f}"])

    print(f"Report saved to {output_file}")


####

ModelID = 'Dataset442_2024_10_01'
gt_labels = r"D:\nnUnet\raw\Dataset442_iso03umNeuronImmune_40xW_V2\labelsTr"
generated_labels = r"D:\nnUnet\results\Dataset442_iso03umNeuronImmune_40xW_V2\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_all\validation"

csv_output_folder = r"D:\nnUnet\results"


num_intensities = 6  # This will process intensities 0, 1, 2, 3, 4, 5

# Run the analysis
image_names, all_scores = process_folders(gt_labels, generated_labels, num_intensities)

# Generate and print the report
generate_report(image_names, all_scores, num_intensities, csv_output_folder, ModelID)
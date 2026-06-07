import os
import argparse
import csv
from PIL import Image

from vendi_score import image_utils
import warnings

def load_images_from_folder(folder_path, num_images_per_prompt):
    image_list = []
    filename_list = []

    # list of img files with the associated extension
    extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]

    # make the list of images
    img_files = [f for f in sorted(os.listdir(folder_path)) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(tuple(extensions))]
    
    for filename in img_files:
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                image_list.append(img.convert("RGB"))
                filename_list.append(filename)
        except Exception as e:
            print(f"Error occurs during handling {filename}: {e}")

    image_list = [image_list[i:i+num_images_per_prompt] for i in range(0, len(image_list), num_images_per_prompt)]
    filename_list = [filename_list[i:i+num_images_per_prompt] for i in range(0, len(filename_list), num_images_per_prompt)]
    return image_list, filename_list


def main():
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', type=str, default='./')
    parser.add_argument("--num_prompts", type=int, default=1000)
    parser.add_argument('--num_images_per_prompt', type=int, default=5, help='number of samples sharing the same condition')
    parser.add_argument('--f_type', type=str, default='inception', choices=['pixel', 'inception', 'sscd'])
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path (default: metrics/<basedir>_vendi.csv)")
    
    args = parser.parse_args()

    num_images_per_prompt = args.num_images_per_prompt
    eval_dir = args.eval_dir
    
    image_list, grouped_filenames = load_images_from_folder(eval_dir, num_images_per_prompt)
        
        
    if args.f_type == 'pixel':
        vs_mss_list = [image_utils.pixel_vs_mss(imgs) for imgs in image_list]
    elif args.f_type == 'inception':
        vs_mss_list = [image_utils.inception_vs_mss(imgs, device="cuda") for imgs in image_list]
    elif args.f_type == 'sscd':
        vs_mss_list = [image_utils.sscd_vs_mss(imgs, device="cuda") for imgs in image_list]
    else:
        raise ValueError(f"Invalid f_type: {args.f_type}")
        
    vs_list = [vs_mss[0] for vs_mss in vs_mss_list]
    mean_vs = sum(vs_list) / len(vs_list) 
    std_vs = sum([(vs - mean_vs) ** 2 for vs in vs_list]) / len(vs_list)
    
    mss_list = [vs_mss[1] for vs_mss in vs_mss_list]
    mean_mss = sum(mss_list) / len(mss_list)
    std_mss = sum([(mss - mean_mss) ** 2 for mss in mss_list]) / len(mss_list)

    if args.output_csv:
        output_csv = args.output_csv
    else:
        project_root = os.path.dirname(os.path.abspath(__file__))
        metrics_dir = os.path.join(project_root, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        basedir = os.path.basename(os.path.normpath(eval_dir))
        output_csv = os.path.join(metrics_dir, f"{basedir}_vendi.csv")

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["group_index", "filenames", "VendiScore", "MeanPairwiseSimilarity"])
        for idx, vs_mss in enumerate(vs_mss_list):
            writer.writerow([idx, ";".join(grouped_filenames[idx]), f"{vs_mss[0]:.6f}", f"{vs_mss[1]:.6f}"])
        writer.writerow([])
        writer.writerow(["metric", "mean", "std"])
        writer.writerow(["VendiScore", f"{mean_vs:.6f}", f"{std_vs:.6f}"])
        writer.writerow(["MeanPairwiseSimilarity", f"{mean_mss:.6f}", f"{std_mss:.6f}"])

    print(f"[Done] Saved metrics csv to {output_csv}")

if __name__ == "__main__":
    main()

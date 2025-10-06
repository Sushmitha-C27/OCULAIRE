import numpy as np, os, json, cv2
from tqdm import tqdm

def npz_to_png(npz_path, output_dir="/content/OCULAIRE/data"):
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)
    volume = data['volume'] if 'volume' in data else data[data.files[0]]
    case_id = os.path.splitext(os.path.basename(npz_path))[0]
    case_folder = os.path.join(output_dir, "volumes", case_id)
    os.makedirs(case_folder, exist_ok=True)

    for i in tqdm(range(volume.shape[0]), desc=f"Converting {case_id}"):
        slice_img = volume[i]
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        slice_img = (slice_img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(case_folder, f"slice_{i:04d}.png"), slice_img)

    meta = {
        "case_id": case_id,
        "num_slices": int(volume.shape[0]),
        "height": int(volume.shape[1]),
        "width": int(volume.shape[2])
    }
    meta_dir = "/content/OCULAIRE/meta"
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, f"{case_id}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Converted {case_id}: {meta['num_slices']} slices saved at {case_folder}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    npz_to_png(args.input)

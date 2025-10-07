import numpy as np, os, json, cv2
from tqdm import tqdm

def npz_to_png(npz_path, modality="BScan", base_output="/content/OCULAIRE/data/volumes"):
    data = np.load(npz_path, allow_pickle=True)
    volume = data["volume"] if "volume" in data else data[data.files[0]]
    case_id = os.path.splitext(os.path.basename(npz_path))[0]

    # Handle both 2D and 3D cases
    if volume.ndim == 3:
        num_slices, height, width = volume.shape
    elif volume.ndim == 2:
        num_slices, height, width = 1, *volume.shape
        volume = volume[np.newaxis, :, :]  # add slice dimension
    else:
        raise ValueError(f"Unexpected shape {volume.shape} in {case_id}")

    # Create folder structure
    case_folder = os.path.join(base_output, modality, case_id)
    os.makedirs(case_folder, exist_ok=True)

    # Save slices as PNG
    for i in tqdm(range(num_slices), desc=f"Converting {case_id}"):
        slice_img = volume[i]
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        slice_img = (slice_img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(case_folder, f"slice_{i:04d}.png"), slice_img)

    # Save metadata
    meta = {
        "case_id": case_id,
        "modality": modality,
        "num_slices": int(num_slices),
        "height": int(height),
        "width": int(width)
    }

    meta_dir = "/content/OCULAIRE/data/meta"
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, f"{modality}_{case_id}.json"), "w") as f:
        import json; json.dump(meta, f, indent=2)

    print(f"✅ Converted {case_id} ({modality}): {num_slices} slices saved at {case_folder}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--modality", default="BScan", help="BScan or RNFLT")
    args = parser.parse_args()
    npz_to_png(args.input, args.modality)

import glob, os, subprocess

BSCAN_RAW = "/content/OCULAIRE/data/raw/BScan"
RNFLT_RAW = "/content/OCULAIRE/data/raw/RNFLT"
ADAPTER   = "/content/OCULAIRE/src/harvard_npz_adapter.py"

# ---- CONFIG ----
convert_bscan = True       # set False if you only want RNFLT
convert_rnflt = True       # set False if you only want BScan
limit_per_modality = 3     # how many cases to convert each run (None = all)
# ----------------

def convert_npz(modality, src_folder):
    npz_files = sorted(glob.glob(f"{src_folder}/*.npz"))
    if limit_per_modality:
        npz_files = npz_files[:limit_per_modality]

    for f in npz_files:
        case_id = os.path.splitext(os.path.basename(f))[0]
        dest_folder = f"/content/OCULAIRE/data/volumes/{modality}/{case_id}"
        if os.path.exists(dest_folder):
            print(f"⚠️ Skipping {case_id} ({modality}) — already converted.")
            continue
        print(f"🚀 Converting {case_id} ({modality}) ...")
        subprocess.run(["python", ADAPTER, "--input", f, "--modality", modality], check=True)

if convert_bscan:
    convert_npz("BScan", BSCAN_RAW)

if convert_rnflt:
    convert_npz("RNFLT", RNFLT_RAW)

print("✅ Batch conversion complete.")

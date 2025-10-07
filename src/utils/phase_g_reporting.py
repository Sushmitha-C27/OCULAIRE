
import os, csv, matplotlib.pyplot as plt, numpy as np, pandas as pd
from fpdf import FPDF

# ===============================================================
# 1.  Append case-level severity to CSV
# ===============================================================
def append_severity_record(case_id, severity, metrics, label,
                           csv_path="/content/OCULAIRE/results/analysis/severity_summary.csv"):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header = ["case_id","label","mean","std","min","max","severity_percent"]
    row = [case_id, label,
           metrics["mean"], metrics["std"], metrics["min"], metrics["max"], severity]
    if not os.path.exists(csv_path):
        with open(csv_path,"w",newline="") as f:
            writer=csv.writer(f); writer.writerow(header)
    with open(csv_path,"a",newline="") as f:
        writer=csv.writer(f); writer.writerow(row)
    print(f"✅ Severity record saved for {case_id}")

# ===============================================================
# 2.  Generate PDF Report
# ===============================================================
def generate_pdf_report(case_id, rnflt_map, diff_map, risk_map,
                        severity, metrics, label,
                        output_dir="/content/OCULAIRE/results/reports"):
    os.makedirs(output_dir, exist_ok=True)
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Helvetica","B",16)
    pdf.cell(0,10,"OCULAIRE RNFLT Report", ln=True, align='C')

    pdf.set_font("Helvetica","",12)
    pdf.cell(0,10,f"Case ID: {case_id}", ln=True)
    pdf.cell(0,10,f"Predicted Label: {label}", ln=True)
    pdf.cell(0,10,f"Severity: {severity:.2f}%", ln=True)
    pdf.cell(0,10,f"Mean RNFLT: {metrics['mean']:.2f} µm", ln=True)

    # --- Save temporary images for embedding ---
    temp_dir="/tmp/oculaire_pdf"
    os.makedirs(temp_dir,exist_ok=True)
    plt.imsave(f"{temp_dir}/rnflt.png", rnflt_map, cmap="turbo")
    plt.imsave(f"{temp_dir}/diff.png", diff_map, cmap="bwr", vmin=-20,vmax=20)
    plt.imsave(f"{temp_dir}/risk.png", risk_map, cmap="hot")

    # --- Embed images ---
    pdf.image(f"{temp_dir}/rnflt.png", x=20, y=70, w=60)
    pdf.image(f"{temp_dir}/diff.png", x=80, y=70, w=60)
    pdf.image(f"{temp_dir}/risk.png", x=140, y=70, w=60)

    pdf.set_y(140)
    pdf.set_font("Helvetica","I",10)
    pdf.multi_cell(0,8,"Left: RNFLT Map  |  Center: Difference Map  |  Right: Risk Map")

    out_path=os.path.join(output_dir,f"{case_id}_report.pdf")
    pdf.output(out_path)
    print(f"📄 PDF report saved → {out_path}")
    return out_path

# ===============================================================
# 3.  Batch severity summary visualization
# ===============================================================
def plot_severity_distribution(csv_path="/content/OCULAIRE/results/analysis/severity_summary.csv"):
    if not os.path.exists(csv_path):
        print("⚠️ No severity summary yet."); return
    df=pd.read_csv(csv_path)
    plt.figure(figsize=(6,4))
    plt.hist(df["severity_percent"], bins=15)
    plt.xlabel("Severity % (< -10 µm area)")
    plt.ylabel("Case Count")
    plt.title("OCULAIRE Severity Distribution")
    plt.show()

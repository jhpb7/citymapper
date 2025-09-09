import sys, os, pathlib, fitz  # PyMuPDF
from tqdm import tqdm


def pdf_to_pngs(input_dir, output_dir, dpi=200, recursive=True):
    input_dir = pathlib.Path(input_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = list(input_dir.rglob("*.pdf") if recursive else input_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {input_dir}")
        return

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for pdf_path in tqdm(pdf_paths, desc="Converting PDFs"):
        # Mirror the input tree under output_dir
        rel = pdf_path.relative_to(input_dir)
        stem = rel.with_suffix("")  # keep subfolders, drop .pdf
        out_base = output_dir / stem
        out_base.parent.mkdir(parents=True, exist_ok=True)

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"\n[WARN] Skipping {pdf_path} ({e})")
            continue

        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(
                matrix=mat, alpha=False
            )  # alpha=True if you want transparency
            out_file = out_base.parent / f"{out_base.stem}_p{page_index+1:03}.png"
            pix.save(out_file)
        doc.close()


if __name__ == "__main__":
    # Usage: python pdfs_to_pngs.py <input_folder> [output_folder] [dpi]
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_png.py <input_folder> [output_folder] [dpi]")
        sys.exit(1)
    input_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) >= 3 else "png_output"
    dpi = int(sys.argv[3]) if len(sys.argv) >= 4 else 200
    pdf_to_pngs(input_folder, output_folder, dpi=dpi)

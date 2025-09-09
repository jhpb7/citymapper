# requirements:
#   pip install pymupdf

import sys
import fitz  # PyMuPDF


def to_rgb255(rgb01):
    return tuple(int(round(max(0, min(1, x)) * 255)) for x in rgb01[:3])


def to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def extract_vector_hex(pdf_path):
    doc = fitz.open(pdf_path)
    colors = set()

    for page in doc:
        for d in page.get_drawings():
            if d.get("fill"):
                colors.add(to_hex(to_rgb255(d["fill"])))
            if d.get("color"):  # stroke
                colors.add(to_hex(to_rgb255(d["color"])))

    doc.close()
    return sorted(colors)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_vector_colors.py <file.pdf>")
        sys.exit(1)

    hex_colors = extract_vector_hex(sys.argv[1])
    for c in hex_colors:
        print(c)

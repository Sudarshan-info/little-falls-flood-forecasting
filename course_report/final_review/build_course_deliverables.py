"""Rebuild the report and poster from the published course sources."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from markdown import markdown


HERE = Path(__file__).resolve().parent
REPORT = HERE / "academic_report_final_review"


def build_report(chrome):
    body = markdown(REPORT.with_suffix(".md").read_text(encoding="utf-8"),
                    extensions=["tables", "extra"])
    body = re.sub(r'(<img src="figures/[^>]+/>)\s*<p><em>(Figure [^<]+)</em></p>',
                  r'<figure>\1<figcaption>\2</figcaption></figure>', body)
    body = re.sub(r'(?<!["\w])https://[^\s<]+',
                  lambda match: f'<a href="{match[0]}">Source</a>', body)
    html = REPORT.with_suffix(".html")
    html.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>One-hour River-Stage Forecasting at Little Falls</title>'
        '<link rel="stylesheet" href="style.css">'
        '<style>figure { break-inside: avoid; margin: .10in 0 .14in; }'
        'figcaption { text-align: center; color: #43535d; font-size: 8.5pt; }'
        'a { text-decoration: underline; }</style>'
        f'<body>{body}</body></html>', encoding="utf-8")
    with TemporaryDirectory(prefix="littlefalls-course-chrome-") as profile:
        subprocess.run([str(chrome), "--headless", "--disable-gpu", "--no-first-run",
                        "--no-pdf-header-footer", f"--user-data-dir={profile}",
                        f"--print-to-pdf={REPORT.with_suffix('.pdf')}", html.as_uri()], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrome", type=Path, required=True, help="Path to Chrome executable")
    options = parser.parse_args()
    build_report(options.chrome)
    subprocess.run([sys.executable, "-B", str(HERE / "poster_source_final_review.py")], check=True)

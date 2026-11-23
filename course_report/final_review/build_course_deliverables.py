"""Rebuild the report and poster from the published course sources."""
import argparse
from html import escape
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from markdown import markdown


HERE = Path(__file__).resolve().parent
REPORT = HERE / "academic_report_submission_final"


def build_report(chrome):
    body = markdown(REPORT.with_suffix(".md").read_text(encoding="utf-8"),
                    extensions=["tables", "extra"])
    body = re.sub(r'<p>(<img src="figures/[^>]+/>)</p>\s*<p><em>(Figure [^<]+)</em></p>',
                  r'<figure>\1<figcaption>\2</figcaption></figure>', body)
    with TemporaryDirectory(prefix="littlefalls-course-") as temporary:
        work = Path(temporary)
        html = work / "report.html"
        html.write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<title>One-hour River-Stage Forecasting at Little Falls</title>'
            f'<base href="{escape(HERE.as_uri() + "/")}">'
            '<link rel="stylesheet" href="style.css">'
            f'<body>{body}</body></html>', encoding="utf-8")
        subprocess.run([str(chrome), "--headless", "--disable-gpu", "--no-first-run",
                        "--allow-file-access-from-files", "--no-pdf-header-footer",
                        f"--user-data-dir={work / 'profile'}",
                        f"--print-to-pdf={REPORT.with_suffix('.pdf')}", html.as_uri()], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrome", type=Path, required=True, help="Path to Chrome executable")
    parser.add_argument("--poster", action="store_true", help="Also rebuild the poster")
    options = parser.parse_args()
    build_report(options.chrome)
    if options.poster:
        subprocess.run([sys.executable, "-B", str(HERE / "poster_source_final_review.py")],
                       check=True)

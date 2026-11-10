"""Keyword scan of paper full texts (pdftotext output) for the baselines survey.

Usage: python keyword_scan.py <dir_with_txt_files>
Counts whole-word, case-insensitive hits of baseline names in each text file.
The texts themselves (copyrighted) stay in the session scratchpad; only counts are logged.
Deterministic; no randomness.
"""
import pathlib
import re
import sys

KEYS = {
    "NOMP": r"\bNOMP\b|Newtoni[sz]ed",
    "OMP(on-grid)": r"\b(?:SW-|S)?OMP\b",
    "atomic/ANM": r"atomic norm|\bANM\b|\bAST\b|Beurling|CBLasso",
    "MUSIC": r"\bMUSIC\b",
    "ESPRIT": r"\bESPRIT\b",
    "SBL/VALSE": r"\bSBL\b|sparse Bayesian|\bVALSE\b",
    "ML/RIMAX": r"\bMLE\b|\bDML\b|maximum.likelihood|\bRIMAX\b|\bSAGE\b",
    "CRB": r"\bCRL?B\b|Cram[ée]r",
}

def main(d):
    files = sorted(pathlib.Path(d).glob("*.txt"))
    print("file," + ",".join(KEYS))
    for f in files:
        t = f.read_text(encoding="utf-8", errors="ignore")
        row = [str(len(re.findall(p, t, flags=re.I))) for p in KEYS.values()]
        print(f.stem + "," + ",".join(row))

if __name__ == "__main__":
    main(sys.argv[1])

"""Simulate the gene-network data sets with GeneNetWeaver 3.1.2b.

    python simulate.py JAR [NETWORK DATASET] [--out DIR]

JAR is the file gnw-3.1.2b.jar (README.md has its address and SHA-256). Java
must be on the path. Without NETWORK and DATASET all 20 data sets of
manifest.csv are made, about half a minute each. A data set is written to
DIR (default: the folder of this file) as net<NETWORK>_g<DATASET>.npy with
the shape (2000, 41, 20): observations, time steps (time 0, 25, ..., 1000)
and series (mRNA of gene 1 to 10, then protein of gene 1 to 10). The SHA-256
of the file is compared with manifest.csv.

The network model and the default settings come out of the jar. One
simulation makes the 2000 observations of a data set: stochastic
differential equations with the noise coefficient 0.05, mRNA and protein
modeled, no measurement noise, and the same perturbation of three genes
(perturbation_net<NETWORK>.tsv) in the first half of every observation. The
random seed of a data set is in manifest.csv.
"""
import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OBSERVATIONS = 2000
# The settings that differ from the defaults of the tool, besides the random seed.
SETTINGS = {"ssKnockouts": 0, "ssKnockdowns": 0, "ssMultifactorial": 0, "ssDualKnockouts": 0, "simulateODE": 0,
            "addMicroarrayNoise": 0, "normalizeAfterAddingNoise": 0, "numTimeSeries": OBSERVATIONS, "dt": 25,
            "loadPerturbations": 1}


def read_series(path) -> np.ndarray:
    """Read a time-series file of the tool: blocks of rows (time, gene 1, ..., gene 10), one block per observation."""
    with open(path) as f:
        blocks = f.read().split("\n", 1)[1].strip().split("\n\n")
    return np.array([[[float(value) for value in row.split("\t")[1:]] for row in block.strip().split("\n")] for block in blocks])


def simulate(jar, network: int, seed: int, path) -> None:
    """Run one simulation of a network and save the array (OBSERVATIONS, 41, 20) at path."""
    name = f"insilico_size10_{network}"
    with tempfile.TemporaryDirectory() as work, zipfile.ZipFile(jar) as archive:
        with open(os.path.join(work, f"input_{name}.xml"), "wb") as f:
            f.write(archive.read(f"ch/epfl/lis/networks/dream4/{name}.xml"))
        values = {**SETTINGS, "randomSeed": seed}
        lines = archive.read("ch/epfl/lis/gnw/rsc/settings.txt").decode().splitlines()
        for i, line in enumerate(lines):
            key = re.match(r"\s*(\w+)\s*=", line)
            if key and key.group(1) in values:
                lines[i] = f"{key.group(1)} = {values[key.group(1)]}"
        with open(os.path.join(work, "settings.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        # The tool reads the perturbation of every observation from this file: the same row OBSERVATIONS times.
        with open(os.path.join(HERE, f"perturbation_net{network}.tsv")) as f:
            header, row = f.read().splitlines()
        with open(os.path.join(work, f"input_{name}_dream4_timeseries_perturbations.tsv"), "w") as f:
            f.write(header + "\n" + (row + "\n") * OBSERVATIONS)
        subprocess.run(["java", "-Xmx5g", f"-Duser.home={work}", "-jar", os.path.abspath(jar), "--simulate",
                        "-c", os.path.join(work, "settings.txt"), "--input-net", os.path.join(work, f"input_{name}.xml"),
                        "--output-path", work + os.sep], cwd=work, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mrna = read_series(os.path.join(work, f"input_{name}_dream4_timeseries.tsv"))
        protein = read_series(os.path.join(work, f"input_{name}_proteins_dream4_timeseries.tsv"))
    X = np.concatenate([mrna, protein], axis=2)
    if X.shape != (OBSERVATIONS, 41, 20):
        raise RuntimeError(f"the simulation gave an array of shape {X.shape}")
    np.save(path, X)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the gene-network data sets with GeneNetWeaver 3.1.2b.")
    parser.add_argument("jar", help="the file gnw-3.1.2b.jar")
    parser.add_argument("network", nargs="?", type=int, help="3 or 5 (default: every data set of manifest.csv)")
    parser.add_argument("dataset", nargs="?", type=int, help="0 to 9")
    parser.add_argument("--out", default=HERE, help="output folder (default: the folder of this file)")
    args = parser.parse_args()
    with open(os.path.join(HERE, "manifest.csv"), newline="") as f:
        rows = [row for row in csv.DictReader(f)
                if args.network is None or (int(row["network"]), int(row["dataset"])) == (args.network, args.dataset)]
    different = 0
    for row in rows:
        path = os.path.join(args.out, row["file"])
        simulate(args.jar, int(row["network"]), int(row["seed"]), path)
        with open(path, "rb") as f:
            equal = hashlib.sha256(f.read()).hexdigest() == row["sha256"]
        different += not equal
        print(f"{path}: {'SHA-256 equal to manifest.csv' if equal else 'SHA-256 DIFFERS from manifest.csv'}", flush=True)
    sys.exit(1 if different or not rows else 0)


if __name__ == "__main__":
    main()

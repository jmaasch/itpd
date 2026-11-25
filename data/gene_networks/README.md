# Gene-network data

Data of the two tables on simulated gene regulatory networks (`configs/gene_networks.yaml`, `configs/gene_networks_indirect.yaml`) and of the gene-network cells of `configs/extended_bic.yaml`.

## Files

- `truth/net<k>_edges.csv`: the edges of network `k` (3 or 5) from one time step to the next. Edge types are `protein_to_mRNA` (regulatory edge, with its sign), `mRNA_to_protein` (translation), `self_mRNA` and `self_protein` (self edges).
- `truth/net<k>_indirect.csv`: the indirect pairs of network `k`. An indirect pair is two series without an edge, of which the first changes the second within one time step through a third series (column `two_step_via`). The two ratio columns give the size of that change relative to the change of the target through its self edge, in the model without noise, with the perturbation (before time 500) and without it. A pair is listed when one of the two ratios is 0.05 or more.
- `perturbation_net<k>.tsv`: the perturbation of three genes in every observation of network `k` during the first half of the time course (time 0 to 500).
- `manifest.csv`: the 20 data sets (10 per network) with their random seed and the SHA-256 of the simulated array.
- `prefix/net<k>_g0_M200.npy`: the first 200 observations of data set 0 of network `k`, with shape (200, 41, 20) for observations, time steps (0, 25, ..., 1000) and series (mRNA of gene 1 to 10, then protein of gene 1 to 10).
- `simulate.py`: the script that builds the full arrays `net<k>_g<g>.npy` of shape (2000, 41, 20).

## How the data were simulated

- The simulator is GeneNetWeaver 3.1.2b (MIT license; see `NOTICE.md` at the repository root). The networks are its DREAM4 10-gene models `insilico_size10_3` and `insilico_size10_5`.
- One simulation produces 2000 observations using stochastic differential equations with noise coefficient 0.05, modeled mRNA and protein, and no measurement noise. It runs for 41 time steps from time 0 to 1000 with the first-half perturbation and the seed in `manifest.csv`; `simulate.py` lists non-default settings.
- The experiments drop the first time step (it is the same in every observation), take the first M observations, and center every series at every time step over those M observations.

## What runs without a simulation

- Runs with 200 observations of data set 0 work directly from the repository root:

```bash
python -m itpd.experiments run configs/gene_networks.yaml --out out/gene_networks --graphs 0 --M 200
```

- Expected means of the two gene-network configs average over 10 data sets. With data set 0 alone, `table --compare` prints data set 0 means and counts them as "with another number of graphs".

## Simulating the other data

- Get `gnw-3.1.2b.jar` from `https://github.com/tschaffter/genenetweaver` at commit `c5310349f5d5723306585c2bb62aedbdeb70db46` (SHA-256 `b44ce5fcc95d1efc882d0da8a2b0428df4b175fbe4f0a836b4e66e120c08be5b`).
- Run all 20 data sets from the repository root with Java on the path (about half a minute per data set, 262 MB total):

```bash
python data/gene_networks/simulate.py gnw-3.1.2b.jar
```

- Run one data set (network 3, data set 1):

```bash
python data/gene_networks/simulate.py gnw-3.1.2b.jar 3 1
```

- `simulate.py` checks the SHA-256 of every array against `manifest.csv`. Equal arrays were obtained with OpenJDK 1.8; other Java versions were not tried.

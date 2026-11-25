# Gene-network data

Data of the two tables on simulated gene regulatory networks (`configs/gene_networks.yaml`, `configs/gene_networks_indirect.yaml`) and of the gene-network cells of `configs/extended_bic.yaml`.

## Files

- `truth/net<k>_edges.csv`: the edges of network `k` (3 or 5) from one time step to the next, with their type: `protein_to_mRNA` (regulatory edge, with its sign), `mRNA_to_protein` (translation), `self_mRNA` and `self_protein` (self edges).
- `truth/net<k>_indirect.csv`: the indirect pairs of network `k`: two series without an edge, of which the first changes the second within one time step through a third series (column `two_step_via`). The two ratio columns give the size of that change relative to the change of the target through its self edge, in the model without noise, with the perturbation (before time 500) and without it. A pair is listed when one of the two ratios is 0.05 or more.
- `perturbation_net<k>.tsv`: the perturbation of three genes that every observation of network `k` has in the first half of its time course (time 0 to 500).
- `manifest.csv`: the 20 data sets (10 per network) with their random seed and the SHA-256 of the simulated array.
- `prefix/net<k>_g0_M200.npy`: the first 200 observations of data set 0 of network `k`. Shape (200, 41, 20): observations, time steps (time 0, 25, ..., 1000) and series (mRNA of gene 1 to 10, then protein of gene 1 to 10).
- `simulate.py`: makes the full arrays `net<k>_g<g>.npy`, shape (2000, 41, 20).

## How the data were simulated

- Tool: GeneNetWeaver 3.1.2b (MIT license; see `NOTICE.md` at the root of the repository). The networks are its models `insilico_size10_3` and `insilico_size10_5` (DREAM4 challenge, 10 genes).
- One simulation gives the 2000 observations of a data set: stochastic differential equations with the noise coefficient 0.05, mRNA and protein modeled, no measurement noise, 41 time steps from time 0 to 1000, the perturbation above in the first half, and the random seed of `manifest.csv`. `simulate.py` lists the settings that differ from the defaults of the tool.
- The experiments drop the first time step (it is the same in every observation), take the first M observations, and center every series at every time step over these M observations.

## What runs without a simulation

- The runs with 200 observations of data set 0: `python -m itpd.experiments run configs/gene_networks.yaml --graphs 0 --M 200 --out DIR`.

## Simulating the other data

- Get the file `gnw-3.1.2b.jar` of the repository `https://github.com/tschaffter/genenetweaver` at the commit `c5310349f5d5723306585c2bb62aedbdeb70db46`. Its SHA-256 is `b44ce5fcc95d1efc882d0da8a2b0428df4b175fbe4f0a836b4e66e120c08be5b`.
- Run `python data/gene_networks/simulate.py gnw-3.1.2b.jar` (Java on the path; about half a minute per data set; 262 MB for the 20 arrays). `python data/gene_networks/simulate.py gnw-3.1.2b.jar 3 1` makes one data set.
- The script compares the SHA-256 of every array with `manifest.csv`. Equal arrays were obtained with OpenJDK 1.8; other Java versions were not tried.

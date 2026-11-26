# Third-party notices

This repository is under the GNU General Public License, version 3 (`LICENSE`). It bundles no third-party program. Two parts derive from third-party work under the MIT License.

## DF-PC

- `src/itpd/external/dfpc.py` contains the function `search`, adapted from `PCStable.run` in `cddd/algorithms.py` of the DF-PC repository (`https://github.com/snu-causality-lab/DF-PC`, commit `bfe18f43992e58ed70a96821bc3506ce426bc0f0`).
- License: MIT. Copyright (c) 2026 DF-PC contributors. The full license text is at the top of `src/itpd/external/dfpc.py`.
- DF-PC itself is not bundled: the script imports it from a clone that the user provides.

## GeneNetWeaver

- The arrays in `data/gene_networks/prefix/` were simulated with GeneNetWeaver 3.1.2b from two network models that are part of that tool (`insilico_size10_3`, `insilico_size10_5`). The tables in `data/gene_networks/truth/` and the two perturbation files were derived from these models and from runs of the tool.
- License of GeneNetWeaver: MIT. Copyright (c) 2008-2012 Thomas Schaffter & Daniel Marbach.
- The tool and its model files are not bundled: `data/gene_networks/simulate.py` reads them from the jar file that the user provides.

## MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Tools that the scripts of `src/itpd/external/` call

None is bundled. Each is installed by the user from the files in `requirements/`: tigramite (GPL-3.0-or-later), causal-learn (MIT), causalnex (Apache-2.0), andrey-core (Apache-2.0).

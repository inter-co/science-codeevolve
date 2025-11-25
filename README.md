# CodeEvolve
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![preprint](https://img.shields.io/badge/preprint-arxiv.2510.14150-red)](https://arxiv.org/abs/2510.14150)

CodeEvolve is an open-source evolutionary coding agent, designed to iteratively improve an initial codebase against a set of user-defined metrics. This project was originally created as an attempt to replicate the results of [AlphaEvolve](https://arxiv.org/abs/2506.13131), a closed-source coding agent announced by Google DeepMind in 2025.

Our primary goal is to implement a transparent, reproducible, and community-driven framework for LLM-driven algorithm discovery. We evaluate CodeEvolve on the same mathematical benchmarks as AlphaEvolve. CodeEvolve has surpassed the reported state-of-the-art performance on 4 of its 13 problems (see the jupyter notebook with our [results](notebooks/results.ipynb)). We are actively working to improve our method on the remaining benchmarks.

## Overview

<img src='assets/codeevolve_diagram.png' align="center" width=900 />

CodeEvolve is built upon an island-based genetic algorithm designed to maintain population diversity and increase throughput through parallel search efforts. Within this architecture, the evolutionary cycle is driven by two primary operators designed to balance the search process: 

1. **Depth exploitatiom**: This operator selects high-performing parent solutions ($S$) via rank-based selection and uses the LLM Ensemble to perform targeted, incremental refinements. Crucially, it provides the LLM with the parent's full lineage, including its $k$ closest ancestors, $A_k(S)$, to constrain the search space toward local optima refinement.
 
2. **Meta-Prompting Exploration**: This operator fosters diversity by instructing an auxiliary LLM (MetaPromptingLLM) to analyze the current solution $S$ and its original prompt $P(S)$, generating an enriched new prompt $P'$. This new prompt is then used by the LLMEnsemble to generate a potentially novel solution $S'$. This step intentionally excludes the ancestral context, encouraging the model to explore distinct pathways.

Both operators are complemented by an **Inspiration-based Crossover** mechanism, which avoids traditional syntactic splicing by providing the LLMEnsemble with sampled high-performing solutions as additional context. This encourages the LLM to synthesize new solutions by semantically combining successful logic or patterns from multiple parents. 

## Usage
To setup the proper conda environment, run the following:
```bash
conda env create -f environment.yml
conda activate codeevolve
```
The command-line version of codeevolve is implemented in ```src/codeevolve/cli.py```, and ```scripts/run.sh``` contains a bash script for running codeevolve on a given benchmark. The most important variables to be defined in this file are the ```API_KEY, API_BASE``` environment variables for connecting with an LLM provider.

More comprehensive tutorials will be released soon.

## Next steps

We are actively developing CodeEvolve to be a more powerful and robust framework. Our immediate roadmap is focused on incorporating more sophisticated evolutionary mechanisms mentioned in our future work:

1. **Dynamic Exploration/Exploitation**: Currently, the choice between exploration (meta-prompting) and exploitation (depth) is set by a fixed probability (exploration_rate). A major planned feature is to implement a more dynamic scheduling, potentially using Reinforcement Learning methods, to adapt this trade-off based on the state of the evolutionary search.

We plan on working on performance improvements, e.g. faster sampling, asynchronous islands algorithm, etc. We also intend on implementing more benchmark problems to test CodeEvolve.
## Project background and inspirations

This project was initiated as an effort to replicate and build upon the work presented in Google DeepMind's AlphaEvolve whitepaper. The closed-source nature of that project  presented a barrier to community-driven progress. Our goal is to provide a transparent, open-source framework that implements the high-level concepts of LLM-driven evolution, allowing for reproducible research and collective innovation.

During the initial stages, we drew inspiration from other open-source efforts like [OpenEvolve](https://github.com/codelion/openevolve), which validated the community's interest in this domain. We are grateful to the contributors of such projects for paving the way and demonstrating the viability of open-source research in this field.

## Contributing

We are not accepting pull requests at this time, as we are still actively developing and changing most of the features from CodeEvolve. We plan on accepting pull requests soon. However, you can contribute by reporting issues or suggesting features through the creation of a [GitHub issue](https://github.com/inter-co/science-codeevolve/issues).

## Citation

```bibtex
@article{assumpção2025codeevolveopensourceevolutionary,
      title={CodeEvolve: An open source evolutionary coding agent for algorithm discovery and optimization}, 
      author={Henrique Assumpção and Diego Ferreira and Leandro Campos and Fabricio Murai},
      year={2025},
      eprint={2510.14150},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.14150}, 
}
```

## License and Disclaimer

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0.

**This is not an official Inter product.**
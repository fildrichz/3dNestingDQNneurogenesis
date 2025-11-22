# Citation Guide for Diploma Thesis

This guide contains recent (2023-2025) citations organized by topic for your diploma thesis on 3D Nesting with DQN and Neurogenesis.

## Files Created

1. **thesis_references.bib** - BibTeX bibliography with 25+ recent citations
2. **thesis_latex_examples.tex** - Example LaTeX text showing how to use the citations
3. **CITATION_GUIDE.md** - This guide

## Most Important Recent Papers (2024-2025)

### 3D Bin Packing with DRL

1. **DeepPack3D (2025)** - Open-source Python package with DQN implementation
   - Citation key: `deeppack3d2025`
   - Great for methodology comparison and benchmarking
   - Published in SoftwareX (March 2025)

2. **One4Many-StablePacker (Oct 2024)** - Generalizable DRL framework
   - Citation key: `one4many2024`
   - Excellent for discussing generalization capabilities
   - arXiv preprint

3. **GOPT (Sep 2024)** - Transformer-based DRL
   - Citation key: `gopt2024`
   - Good for modern architecture discussions
   - arXiv preprint

4. **Multimodal DRL (2024)** - Handles 100+ boxes
   - Citation key: `multimodal_drl2024`
   - Perfect for large-scale problem discussion
   - Published in Knowledge-Based Systems

### DQN Specifically

5. **DQN for Irregular Items (2023)** - DQN avoiding expert rules
   - Citation key: `dqn_irregular2023`
   - Essential for your DQN methodology section
   - Published in Applied Intelligence (Springer)

6. **Concurrent 3D Packing (2024)** - Online packing with limited look-ahead
   - Citation key: `drl_concurrent2024`
   - Good for online problem discussion
   - Published in International Journal of Production Research

### Neurogenesis

7. **Neurogenesis Deep Learning (2017)** - Foundational paper
   - Citation key: `neurogenesis_dl2017`
   - MUST cite for neurogenesis background
   - IEEE IJCNN conference

8. **Neuromorphic Algorithms (2025)** - Latest review
   - Citation key: `neuromorphic2025`
   - Excellent for current state of neuromorphic computing
   - Published in Frontiers in Neuroscience (2025)

9. **Neuromorphic Commercial Success (2025)** - Industry perspective
   - Citation key: `neuromorphic_commercial2025`
   - Great for practical applications and future work
   - Nature Communications (2025)

### 2D Irregular Nesting

10. **Nest Smarter (2025)** - Vision-based DRL for 2D irregular packing
    - Citation key: `nest_smarter2025`
    - 97% computation time improvement, 11% material utilization improvement
    - Journal of Intelligent Manufacturing (2025)

11. **Fidelity-Adaptive Evolution (2024)** - GA with skyline strategies
    - Citation key: `faeo2024`
    - Journal of Intelligent Manufacturing

12. **GA-LP Hybrid (2023)** - Genetic algorithm + linear programming
    - Citation key: `ga_lp2023`
    - Applied Sciences (MDPI)

## How to Use in Your Thesis

### In LaTeX Document

```latex
\documentclass{article}
\usepackage{cite}

\begin{document}

Your text here with citations~\cite{deeppack3d2025,one4many2024}.

\bibliographystyle{ieeetr}
\bibliography{thesis_references}

\end{document}
```

### Compilation Commands

```bash
pdflatex your_thesis.tex
bibtex your_thesis
pdflatex your_thesis.tex
pdflatex your_thesis.tex
```

## Citation Examples by Context

### When discussing DQN methodology:
```latex
Deep Q-Networks have been successfully applied to 3D bin packing problems,
demonstrating the ability to learn effective packing strategies without
relying on handcrafted expert rules~\cite{dqn_irregular2023}.
```

### When discussing recent DRL approaches:
```latex
Recent transformer-based architectures have achieved state-of-the-art
results in 3D packing optimization~\cite{transformer_packing2023,gopt2024}.
```

### When discussing neurogenesis:
```latex
Neurogenesis deep learning enables networks to accommodate new classes
by adding neurons to deep layers, inspired by adult neurogenesis in the
hippocampus~\cite{neurogenesis_dl2017}.
```

### When comparing multiple methods:
```latex
Various approaches have been proposed, including DQN-based
methods~\cite{dqn_irregular2023}, transformer architectures~\cite{gopt2024},
and multimodal frameworks~\cite{multimodal_drl2024}.
```

## Papers by Year (for impressive recency)

### 2025 (Most Recent!)
- `deeppack3d2025` - DeepPack3D Python package
- `nest_smarter2025` - Vision-based DRL for 2D irregular packing
- `neuromorphic2025` - Neuromorphic algorithms review
- `neuromorphic_commercial2025` - Neuromorphic commercial success
- `ga_image2025` - GA with image processing

### 2024
- `one4many2024` - Generalizable DRL framework
- `gopt2024` - Transformer-based DRL
- `multimodal_drl2024` - Multimodal DRL for 3D packing
- `dmrl_bpp2024` - Dynamic multimodal DRL
- `idnna2024` - Improved neural network for irregular items
- `brain_inspired_ai2024` - Brain-inspired AI
- `faeo2024` - Fidelity-adaptive evolution
- `irregular_flaw2024` - Nesting with flaw avoidance
- `gan_ga_packing2024` - GAN-GA synergy
- `cnn_heightmap2024` - CNN height map representation
- `ml_hierarchical2024` - Hierarchical DRL
- `drl_concurrent2024` - Online concurrent 3D packing

### 2023
- `dqn_irregular2023` - DQN for irregular items (IMPORTANT!)
- `boxstacker2023` - BoxStacker framework
- `transformer_packing2023` - Transformer with PPO
- `ga_lp2023` - GA-LP hybrid

## Tips for Your Thesis

1. **State of the Art Section**: Use 2-3 paragraphs per major paper as shown in `thesis_latex_examples.tex`

2. **Balance Citations**: Mix foundational papers (like neurogenesis_dl2017) with cutting-edge 2024-2025 papers

3. **Show Progression**: Demonstrate how methods evolved from 2023 → 2024 → 2025

4. **Your Contribution**: After describing existing methods, clearly state what YOU are adding (as noted in your konzultace notes)

5. **Czech Thesis Requirements**: Make sure to check if your school requires specific citation style (IEEE, APA, etc.)

## Quick Start

1. Copy `thesis_references.bib` to your thesis directory
2. Reference it in your main .tex file: `\bibliography{thesis_references}`
3. Use citation keys like: `\cite{deeppack3d2025}`
4. Compile with bibtex

## Need More Papers?

If you need papers on specific topics:
- Genetic algorithms for 3D packing
- Specific neural network architectures
- Benchmark datasets
- Evaluation metrics

Let me know and I can search for more recent publications!

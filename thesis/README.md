# Master Thesis: Neurogenesis of Neural Networks for 3D Nesting

**Author:** Filip Špidla
**Supervisor:** Ing. Mgr. Ladislava Smítková Janků, Ph.D.
**University:** Czech Technical University in Prague
**Faculty:** Faculty of Electrical Engineering
**Field:** Artificial Intelligence
**Date:** August 2025

## Structure

This thesis follows the supervisor's recommended outline:

1. **Introduction** - Context, motivation, problem delimitation, research method
2. **Problem Analysis** - Formal definition of 3D bin packing, constraints, neurogenesis fundamentals
3. **State of the Art** - Survey of 3D nesting methods and evolutionary neural architecture search
4. **Proposed Method** - Description of the neurogenesis-based approach
5. **Implementation** - Tools and technical decisions
6. **Experiments and Results** - Evaluation, comparison with baselines, performance analysis
7. **Conclusions** - Summary, limitations, future work

## Changes from Original PDF

### Removed Content
- **Traveling Salesman Problem (TSP)** - Section 2.1.3 and Chapter 3.3 have been completely removed as they are not part of the implemented work
- All TSP-related discussions in the introduction and problem analysis

### Expanded Content
- Chapter 3 (State of the Art) now includes detailed summaries of each cited paper (1-2 paragraphs each)
- Clear classification of nesting methods by information model, space representation, and candidate generation
- Identified research gap that this thesis addresses

### Restructured Content
- Introduction now clearly delimits what the thesis covers and what it does not cover
- Thesis structure explicitly laid out at the end of Chapter 1
- Bibliography consolidated and properly formatted

## Building the Thesis

To compile the thesis:

```bash
cd thesis
pdflatex thesis.tex
bibtex thesis
pdflatex thesis.tex
pdflatex thesis.tex
```

Or use:
```bash
latexmk -pdf thesis.tex
```

## TODO for Completion

The following chapters contain structured placeholders that need to be filled in:

- **Chapter 4 (Proposed Method)**: Describe the actual implementation details of your neurogenesis system
- **Chapter 5 (Implementation)**: Document the code structure, libraries, datasets, and computational resources
- **Chapter 6 (Experiments)**: Add experimental results, graphs, tables, and performance analysis
- **Chapter 7 (Conclusions)**: Summarize findings, acknowledge limitations, and propose future work

## Notes from Supervisor Consultation (July 23, 2025)

Key requirements:
- Each paper in state of the art should have 1-2 paragraphs
- Parameters for comparing methods (online/offline, success rate, etc.)
- Clearly separate what you did from what you didn't do
- Focus on 3D nesting methods, not TSP
- Experiments should include graphs, valid result ranges, and baseline comparisons

## Files

- `thesis.tex` - Main thesis document
- `chapters/` - Individual chapter files
- `references.bib` - Bibliography in BibTeX format
- `figures/` - Figures and diagrams (to be added)
- `tables/` - Additional tables (to be added)

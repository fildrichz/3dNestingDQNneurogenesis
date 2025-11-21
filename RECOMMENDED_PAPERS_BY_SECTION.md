# Recommended Research Papers by State-of-the-Art Section

This document provides specific paper recommendations for each section, subsection, and paragraph of the State-of-the-Art chapter (Chapter 3) based on the current thesis outline.

---

## Section 3.1: Nesting Methods

### 3.1.1 Classification Dimensions

#### Information model (offline vs. online) - Line 16-17
**Current citation:** ARCO22

**Additional recommended papers:**

1. **Comparing online algorithms for bin packing problems**
   - **Authors:** Coffman Jr., E. G., Csirik, J., Galambos, G., Martello, S., Vigo, D.
   - **Published:** Journal of Scheduling (2010)
   - **DOI:** 10.1007/s10951-009-0129-5
   - **Why relevant:** Comprehensive comparison of online vs offline bin packing approaches; discusses irrevocable decisions in online settings

2. **Online Bin Packing with Known T**
   - **Authors:** Various
   - **Published:** arXiv:2112.03200 (2021)
   - **Why relevant:** Modern analysis of online bin packing with partial information

3. **Fully Dynamic Bin Packing Revisited**
   - **Authors:** Ivković, Z., & Lloyd, E. L.
   - **Why relevant:** Discusses dynamic arrival scenarios and re-packing windows mentioned in the text

#### Space representation (rasterization vs. geometry-first) - Line 19-20
**Current citation:** ARCO22

**Well covered by ARCO22.** Optional additional citation:

4. **Voxel-based representation for bin packing**
   - Consider citing specific papers on voxel vs geometric representations if you want more depth

#### Candidate placements ("interesting points") - Line 22-23
**Current citation:** ARCO22

**Well covered.** Detailed discussion follows in Section 3.1.3 with extensive citations.

#### Neural scoring and learned proposals - Line 25-26
**Current citation:** ARCO22

**Additional recommended papers:**

5. **Heuristics Integrated Deep Reinforcement Learning for Online 3D Bin Packing**
   - **Published:** IEEE Transactions (2023)
   - **Link:** https://ieeexplore.ieee.org/document/10018146/
   - **Why relevant:** Directly addresses neural re-ranking of EP/EMS candidates; validates approach of neural components replacing hand-crafted tie-breakers

### 3.1.2 Solution Approaches

All three paragraphs (Constructive heuristics, Local improvement, Metaheuristics) cite ARCO22, which is appropriate. These are well-covered general overviews.

### 3.1.3 Placement Heuristics: Extreme Points and Maximal Empty Spaces

**Currently has excellent citations:** CPP08, CPP12, KI04, LC97, FB10, GR13, PSD+16

**This subsection is very well cited.** No additional papers needed.

### 3.1.4 Key Papers in 3D Nesting

**Currently has excellent citations:** ARCO22, dNAJ21, GTMP22, GOGL16, BTC23, AMSAVP22, GPAAVP23, KEK21

**This subsection is comprehensive.** No additional papers needed.

---

## Section 3.2: Machine Learning and Neural Networks in Bin Packing

### 3.2.1 Supervised Learning Approaches

**Current citations:** MBK07, HLZ17

**Additional recommended papers:**

6. **Deep learning for grasp planning in bin-picking (depth images)**
   - **Title:** "Depth Image–Based Deep Learning of Grasp Planning for Textureless Planar-Faced Objects in Vision-Guided Robotic Bin-Picking"
   - **Authors:** Liang, H., et al.
   - **Published:** Sensors 2020, 20(3), 706
   - **Link:** https://www.mdpi.com/1424-8220/20/3/706
   - **Why relevant:** Uses CNNs with depth images (similar to heightmaps) for spatial reasoning; demonstrates supervised learning for placement decisions

### 3.2.2 Reinforcement Learning for Combinatorial Optimization

**Current citations:** ZTL+21, LFU+18, HZZ+20, ZXL+22

**Additional recommended papers:**

7. **BoxStacker: Deep Reinforcement Learning for 3D Bin Packing Problem in Virtual Environment**
   - **Authors:** Nam, S., et al.
   - **Published:** Sensors 2023, 23(15), 6928
   - **Link:** https://www.mdpi.com/1424-8220/23/15/6928
   - **Why relevant:** Recent (2023) DRL approach to 3D bin packing; addresses real-time sequential packing

8. **DeepPack3D: A Python package for online 3D bin packing optimization**
   - **Authors:** Various
   - **Published:** Software Impacts (December 2024)
   - **Link:** https://www.sciencedirect.com/science/article/pii/S2665963824001209
   - **Why relevant:** Very recent (2024); integrates DRL with constructive heuristics; provides benchmarking foundation

9. **Online 3D Bin Packing with Constrained Deep Reinforcement Learning**
   - **Authors:** Zhao, H., et al.
   - **Published:** AAAI 2021
   - **Link:** https://cdn.aaai.org/ojs/16155/16155-13-19649-1-2-20210518.pdf
   - **Why relevant:** Formulates 3D-BPP as constrained MDP (CMDP); directly relevant to constraint handling discussion

10. **GOPT: Generalizable Online 3D Bin Packing via Transformer-based Deep Reinforcement Learning**
    - **Published:** arXiv:2409.05344 (2024)
    - **Why relevant:** Recent transformer-based approach; validates EMS-based placement generation with attention mechanisms

11. **Neural Combinatorial Optimization with Reinforcement Learning**
    - **Authors:** Bello, I., Pham, H., Le, Q. V., Norouzi, M., & Bengio, S.
    - **Published:** ICLR 2017
    - **arXiv:** 1611.09940
    - **Why relevant:** Foundational paper demonstrating RL for NP-hard combinatorial problems; validates neural approaches for discrete optimization

12. **A Dynamic Multi-modal deep Reinforcement Learning framework for 3D Bin Packing Problem**
    - **Published:** Knowledge-Based Systems (2024)
    - **Link:** https://dl.acm.org/doi/10.1016/j.knosys.2024.111990
    - **Why relevant:** Recent 2024 work on multi-modal DRL for 3D bin packing

13. **Solving Offline 3D Bin Packing Problem with Large-sized Bin via Two-stage Deep Reinforcement Learning**
    - **Published:** AAMAS 2024
    - **Link:** https://dl.acm.org/doi/abs/10.5555/3635637.3663232
    - **Why relevant:** Recent two-stage approach achieving 0.3%-1.7% improvement in utilization

### 3.2.3 Neural Architecture Components

#### State Encoders - Line 122-124
**Current citations:** None (general mention)

**Recommended papers:**

14. **Jointly-Trained State-Action Embedding for Efficient Reinforcement Learning**
    - **Authors:** Various
    - **Published:** CIKM 2021
    - **Link:** https://dl.acm.org/doi/10.1145/3459637.3482357
    - **Why relevant:** Addresses large state/action spaces by jointly learning embeddings; discusses MLP-based state encoding

15. **Graph-based State Representation for Deep Reinforcement Learning**
    - **arXiv:** 2004.13965 (2020)
    - **Why relevant:** Compares embedding methods vs matrix representations; shows MLPs are effective for vector-based state encoding

16. **A Survey of State Representation Learning for Deep Reinforcement Learning**
    - **arXiv:** 2506.17518
    - **Why relevant:** Comprehensive survey of state encoding techniques including MLPs for fixed-dimensional embeddings

#### Spatial Feature Extractors - Line 125-127
**Current citations:** HLZ17

**This is well-cited.** The HLZ17 paper is the key reference for heightmap CNNs.

#### Action Encoders - Line 128-130
**Current citations:** None (general mention)

**Recommended papers:**

17. **Matrix Encoding Networks for Neural Combinatorial Optimization**
    - **arXiv:** 2106.11113 (2021)
    - **Link:** https://arxiv.org/abs/2106.11113
    - **Why relevant:** Specifically addresses action encoding in combinatorial optimization; introduces matrix-based action representations

18. **Neural Combinatorial Optimization: A Tutorial**
    - **Published:** European Journal of Operational Research (2025)
    - **Link:** https://www.sciencedirect.com/science/article/pii/S0305054825001303
    - **Why relevant:** Recent comprehensive tutorial on action space design and encoding strategies for combinatorial problems

#### Attention Mechanisms - Line 131-132
**Current citations:** Points to Section 3.5

**Well-handled by forward reference.** No additional citations needed here.

---

## Section 3.3: Deep Reinforcement Learning

### 3.3.1 Q-Learning and Value-Based Methods

**Current citations:** Sutton1998, WD92

**Well-cited.** These are the foundational references.

#### Application to 3D Bin Packing paragraph - Line 147-149
**Current citations:** None

**Recommended papers:**

19. **Reinforcement learning for combinatorial optimization: A survey**
    - **Authors:** Mazyavkina, N., et al.
    - **Published:** Computers & Operations Research (2021)
    - **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0305054821001660
    - **Why relevant:** Surveys sparse reward challenges in combinatorial optimization; discusses state representation and action space design

20. **Credit Assignment in Sparse Rewards - Rewards Prediction-Based**
    - **Authors:** Ding, Y., et al.
    - **Published:** IEEE Access (2019)
    - **Link:** https://ieeexplore.ieee.org/document/8809762/
    - **Why relevant:** Directly addresses sparse binary rewards and credit assignment problem mentioned in the paragraph

### 3.3.2 Deep Q-Networks (DQN)

**Current citations:** MKS+13, MKS+15

**Excellent foundational citations.** No additional papers needed.

### 3.3.3 Double DQN

**Current citations:** vHGS16

**Perfect citation.** No additional papers needed.

### 3.3.4 N-Step Returns and Sparse Reward Environments

**Current citations:** SB18, HMM+18 (Rainbow), NHR99

**Excellent citations.** Well-covered section.

---

## Section 3.4: Spatial Feature Extraction

### 3.4.1 Heightmap Representations

**Current citations:** HLZ17

**Additional recommended papers:**

21. **Affordance-Based Grasping Point Detection Using Graph Convolutional Networks for Industrial Bin-Picking**
    - **Authors:** Various
    - **Published:** Sensors 2021
    - **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7865998/
    - **Why relevant:** Uses depth-based heightmap representations for spatial reasoning in bin-picking; demonstrates CNN effectiveness on spatial features

22. **Development of robotic bin picking platform with cluttered objects using CNN**
    - **Published:** Journal of Manufacturing Systems (2022)
    - **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0278612522000826
    - **Why relevant:** Uses CNNs with depth images for spatial feature extraction in cluttered environments

### 3.4.2 Graph Neural Networks for Relational Reasoning

**Current citations:** CCBT21 (Cappart et al.)

**Additional recommended papers:**

23. **Graph Neural Networks: A Review of Methods and Applications**
    - **Authors:** Zhou, J., et al.
    - **arXiv:** 1812.08434
    - **Why relevant:** Comprehensive GNN survey; covers message passing and aggregation schemes mentioned in the text

24. **SpatialSim: Recognizing Spatial Configurations of Objects With Graph Neural Networks**
    - **Published:** Frontiers in AI (2022)
    - **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC8826049/
    - **Why relevant:** Demonstrates GNNs for spatial configuration recognition; shows fully-connected GNNs perform well for spatial reasoning

25. **DepWiGNN: A Depth-wise Graph Neural Network for Multi-hop Spatial Reasoning**
    - **arXiv:** 2310.12557
    - **Why relevant:** Addresses multi-hop spatial reasoning with GNNs; aggregates information over depth dimension; relevant for understanding item relationships in packing

26. **Combinatorial Optimization and Reasoning with Graph Neural Networks**
    - **Authors:** Cappart, Q., et al.
    - **Published:** JMLR 2023, Vol 24
    - **Link:** https://jmlr.org/papers/volume24/21-0449/21-0449.pdf
    - **Why relevant:** Comprehensive paper on GNNs for combinatorial optimization; discusses graph representations for discrete problems

---

## Section 3.5: Attention Mechanisms

### 3.5.1 Transformer Architecture

**Current citations:** VSP+17 (Vaswani et al.)

**Perfect foundational citation.** No additional papers needed.

### 3.5.2 Set Transformer for Permutation Invariance

**Current citations:** LHS+19 (Lee et al.)

**Additional recommended papers:**

27. **Deep Sets**
    - **Authors:** Zaheer, M., Kottur, S., Ravanbhakhsh, S., et al.
    - **Published:** NeurIPS 2017
    - **arXiv:** 1703.06114
    - **Why relevant:** Foundational work on permutation-invariant neural networks; provides theoretical foundation that Set Transformers build upon; should be cited when discussing permutation invariance

28. **On permutation-invariant neural networks: A survey**
    - **arXiv:** 2403.17410 (2024)
    - **Why relevant:** Recent comprehensive survey on permutation-invariant architectures; provides broader context for Set Transformers

---

## Section 3.6: Evolutionary Neural Architecture Search

### 3.6.1 Overview of ENAS

**Current citations:** EMH19, WSS+23, LSX+20, RMS+17, BB24, TK01

**This is very well-cited.** Excellent coverage of ENAS literature.

**Optional additional papers:**

29. **Neuroevolution of Augmenting Topologies (NEAT)**
    - **Authors:** Stanley, K. O., & Miikkulainen, R.
    - **Published:** Evolutionary Computation, 10(2):99-127 (2002)
    - **Link:** https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
    - **Why relevant:** Foundational neuroevolution paper; if you want to cite the historical origin of evolving neural topologies (you already have this in RESEARCH_PAPERS.md)

30. **NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm**
    - **Authors:** Lu, Z., et al.
    - **Published:** GECCO 2019
    - **arXiv:** 1810.03522
    - **Why relevant:** Multi-objective GA for NAS; relevant to your fitness function design (already in RESEARCH_PAPERS.md)

### 3.6.2 Gap in the Literature

**No citations needed** (identifies research gap)

---

## Additional Recommended Papers for Related Topics

### Pointer Networks (mentioned in 3.2.2, line 114)

31. **Pointer Networks**
    - **Authors:** Vinyals, O., Fortunato, M., & Jaitly, N.
    - **Published:** NeurIPS 2015
    - **arXiv:** 1506.03134
    - **Link:** https://proceedings.neurips.cc/paper_files/paper/2015/file/29921001f2f04bd3baee84a12e98098f-Paper.pdf
    - **Why relevant:** Cited in line 114 (TAP-Net combines pointer networks with attention); foundational paper for using attention as a pointer mechanism

### Transfer Learning (mentioned in 3.2.2, line 114-115)

32. **Transfer Reinforcement Learning for Combinatorial Optimization Problems**
    - **Published:** Algorithms 2024, 17(2), 87
    - **Link:** https://www.mdpi.com/1999-4893/17/2/87
    - **Why relevant:** Directly addresses transfer learning in RL for combinatorial optimization; relevant to discussion of training on small instances and fine-tuning on larger ones

33. **Leveraging Transfer Learning in Deep Reinforcement Learning for Solving Combinatorial Optimization Problems**
    - **Published:** IEEE Xplore (2024)
    - **Link:** https://ieeexplore.ieee.org/abstract/document/10766597/
    - **Why relevant:** Recent work on transfer learning for CO problems; discusses generalization challenges

34. **Improving Generalization of Neural Combinatorial Optimization via Test-Time Projection Learning**
    - **arXiv:** 2506.02392
    - **Why relevant:** Addresses generalization across problem sizes; relevant to TAP-Net's transfer learning discussion

### Constraint Handling in RL (relevant to sparse rewards and constraints discussion)

35. **A survey of constraint formulations in safe reinforcement learning**
    - **Published:** IJCAI 2024
    - **Link:** https://dl.acm.org/doi/10.24963/ijcai.2024/913
    - **Why relevant:** Comprehensive survey on constraint handling in RL; relevant for complex constraints (weight, fragility, load-bearing) mentioned throughout

36. **Policy Learning with Constraints in Model-free Reinforcement Learning: A Survey**
    - **Published:** IJCAI 2021
    - **Link:** https://www.ijcai.org/proceedings/2021/0614.pdf
    - **Why relevant:** Surveys constraint handling approaches in RL; directly relevant to constrained bin packing

---

## Summary by Priority

### High Priority Additions (fill clear gaps):

1. **Section 3.2.3 State Encoders** → Add papers #14, #15, or #16
2. **Section 3.2.3 Action Encoders** → Add papers #17 or #18
3. **Section 3.2.2 RL for Bin Packing** → Add papers #7, #8, #9, or #10 (recent 2023-2024 works)
4. **Section 3.4.2 GNNs** → Add papers #23, #24, or #26 (strengthen GNN discussion)
5. **Section 3.5.2 Set Transformers** → Add paper #27 (Deep Sets - foundational work)
6. **Section 3.3.1 Application to 3D Bin Packing** → Add paper #19 or #20

### Medium Priority Additions (strengthen existing sections):

7. **Section 3.4.1 Heightmaps** → Add paper #21 or #22 (additional CNN heightmap examples)
8. **Pointer Networks reference (line 114)** → Add paper #31
9. **Transfer Learning reference (line 114-115)** → Add paper #32, #33, or #34

### Low Priority (optional, already well-cited):

10. Any additional papers for already well-cited sections

---

## How to Use This Document

For each section of your state-of-the-art chapter:

1. **Review current citations** - Check if the section already has sufficient coverage
2. **Identify gaps** - Look for paragraphs with generic statements but no citations
3. **Add relevant papers** - Use the recommendations above to add 1-3 papers per gap
4. **Maintain balance** - Don't over-cite; 2-3 papers per key point is usually sufficient
5. **Prioritize recent work** - For rapidly evolving areas (DRL for bin packing), prefer 2020-2024 papers
6. **Keep foundational papers** - For established concepts (DQN, transformers), cite the original papers

---

## Citation Format Notes

When adding these papers to your thesis:

- Use consistent citation keys (e.g., `\cite{Vinyals15}` for Vinyals et al. 2015)
- Add full bibliographic information to your `.bib` file
- For arXiv papers, prefer published versions if available
- Include DOI or URL for accessibility
- Check your thesis guidelines for citation format requirements

---

**Document Created:** 2025-11-21
**Last Updated:** 2025-11-21
**Total Recommended Papers:** 36 unique papers
**High Priority Additions:** 6 sections identified


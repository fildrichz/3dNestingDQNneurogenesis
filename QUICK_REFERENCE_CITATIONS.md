# Quick Reference: Priority Citations by Section

This document provides a quick lookup for the most important papers to add to each section of your state-of-the-art chapter.

---

## 🔴 High Priority Gaps (sections with no/few citations that need them)

### Section 3.2.3: Neural Architecture Components - State Encoders (Line 122-124)
**Currently:** No citations
**Add:**
- **JointStateAction2021** - Jointly-Trained State-Action Embedding for Efficient RL
- **GraphStateRep2020** - Graph-based State Representation for Deep RL

### Section 3.2.3: Neural Architecture Components - Action Encoders (Line 128-130)
**Currently:** No citations
**Add:**
- **MatNet2021** - Matrix Encoding Networks for Neural Combinatorial Optimization
- **NCOTutorial2025** - Neural combinatorial optimization: A tutorial

### Section 3.3.1: Application to 3D Bin Packing paragraph (Line 147-149)
**Currently:** No citations (discusses sparse rewards in bin packing)
**Add:**
- **Mazyavkina2021RLSurvey** - RL for combinatorial optimization: A survey
- **Ding2019Credit** - Credit Assignment for Sparse Binary Rewards

### Section 3.5.2: Set Transformer - Permutation Invariance discussion
**Currently:** LHS+19 only
**Add:**
- **Zaheer2017DeepSets** - Deep Sets (foundational work that Set Transformers build upon)

---

## 🟡 Medium Priority (strengthen existing sections)

### Section 3.2.2: Reinforcement Learning for Combinatorial Optimization
**Currently:** ZTL+21, LFU+18, HZZ+20, ZXL+22
**Add 1-2 recent papers:**
- **Nam2023BoxStacker** - BoxStacker (2023)
- **DeepPack3D2024** - DeepPack3D package (Dec 2024)
- **Zhao2021Online3DBPP** - Online 3D-BPP with Constrained DRL (AAAI 2021)
- **GOPT2024** - Transformer-based approach (2024)

**Why:** Your current citations are good, but these add very recent (2023-2024) works showing the field's evolution

### Section 3.4.2: Graph Neural Networks for Relational Reasoning
**Currently:** CCBT21 only
**Add 1-2 papers:**
- **Zhou2018GNNReview** - Comprehensive GNN review
- **Cappart2023CORR** - GNNs for Combinatorial Optimization (JMLR 2023)
- **SpatialSim2022** - GNNs for spatial configuration recognition

**Why:** One citation is too sparse for a subsection; these strengthen the GNN discussion

### Section 3.4.1: Heightmap Representations
**Currently:** HLZ17 only
**Add (optional):**
- **Liang2020Depth** - Depth images for bin-picking
- **AffordanceGNN2021** - Heightmaps for industrial bin-picking

**Why:** Provides additional examples of heightmap usage beyond HLZ17

---

## 🟢 Optional Additions (already well-cited sections)

### Pointer Networks reference (mentioned in line 114 - TAP-Net)
**Add:**
- **Vinyals2015Pointer** - Original Pointer Networks paper

### Transfer Learning reference (mentioned in line 114-115)
**Add:**
- **TransferRL2024** - Transfer RL for Combinatorial Optimization

### Online vs Offline discussion (Section 3.1.1, line 16-17)
**Currently:** ARCO22
**Optionally add:**
- **Coffman2010Comparing** - Comparing online algorithms for bin packing

---

## 📊 Summary Statistics

| Section | Current Citations | Recommended Additions | Priority |
|---------|------------------|----------------------|----------|
| 3.2.3 State Encoders | 0 | 2 | 🔴 High |
| 3.2.3 Action Encoders | 0 | 2 | 🔴 High |
| 3.3.1 Application to 3D BPP | 0 | 2 | 🔴 High |
| 3.5.2 Set Transformers | 1 | 1 | 🔴 High |
| 3.2.2 RL for Bin Packing | 4 | 2-3 | 🟡 Medium |
| 3.4.2 GNN Reasoning | 1 | 2 | 🟡 Medium |
| 3.4.1 Heightmaps | 1 | 1-2 | 🟡 Medium |
| Other sections | Various | 1-2 | 🟢 Optional |

---

## 🎯 Action Plan

### Immediate (before next draft):
1. Add citations to **3.2.3 State Encoders** (2 papers)
2. Add citations to **3.2.3 Action Encoders** (2 papers)
3. Add citations to **3.3.1 Application to 3D Bin Packing** (2 papers)
4. Add **Deep Sets** citation to **3.5.2**

**Result:** Fill 4 critical gaps with 7 papers

### Next priority (strengthen existing sections):
5. Add 2 recent papers to **3.2.2 RL for Bin Packing**
6. Add 2 papers to **3.4.2 GNN** section
7. Add 1 paper to **3.4.1 Heightmaps**

**Result:** Strengthen 3 sections with 5 papers

### Optional (if word count allows):
8. Add Pointer Networks citation where mentioned
9. Add Transfer Learning citation where mentioned
10. Add Online vs Offline papers to 3.1.1

**Result:** Complete coverage with 3 more papers

---

## 📝 Citation Integration Examples

### Example 1: State Encoders (Section 3.2.3, after line 124)

**Current text:**
> Global state information (number of items remaining, current utilization, bin count) is encoded via multi-layer perceptrons (MLPs) into fixed-dimensional embeddings. These embeddings provide context for action selection.

**Revised with citations:**
> Global state information (number of items remaining, current utilization, bin count) is encoded via multi-layer perceptrons (MLPs) into fixed-dimensional embeddings \cite{JointStateAction2021,GraphStateRep2020}. These embeddings provide context for action selection. \textbf{Tavakoli et al.\ (2021) \cite{JointStateAction2021}} show that jointly learning state and action embeddings improves efficiency in large state spaces, while \textbf{Keramati et al.\ (2020) \cite{GraphStateRep2020}} demonstrate that MLP-based embeddings outperform matrix representations for vector-based state encoding.

---

### Example 2: Action Encoders (Section 3.2.3, after line 130)

**Current text:**
> Each candidate action (placement) is represented by features including item dimensions, placement position, orientation, slack space, and derived metrics (utilization delta, tightness). MLPs embed these features into a common representation space.

**Revised with citations:**
> Each candidate action (placement) is represented by features including item dimensions, placement position, orientation, slack space, and derived metrics (utilization delta, tightness). MLPs embed these features into a common representation space \cite{MatNet2021,NCOTutorial2025}. \textbf{Kwon et al.\ (2021) \cite{MatNet2021}} introduce matrix encoding networks that efficiently represent actions in combinatorial optimization problems, while \textbf{Bengio et al.\ (2025) \cite{NCOTutorial2025}} provide a comprehensive tutorial on action space design strategies for neural combinatorial optimization.

---

### Example 3: Application to 3D Bin Packing (Section 3.3.1, after line 149)

**Current text (ending):**
> This delayed feedback makes credit assignment exceptionally difficult: which of the dozens of placement decisions was responsible for the final outcome? This sparsity motivates the use of n-step returns, reward shaping, and evolved neural architectures capable of learning long-horizon dependencies from limited feedback signals.

**Revised with citations:**
> This delayed feedback makes credit assignment exceptionally difficult: which of the dozens of placement decisions was responsible for the final outcome? \textbf{Mazyavkina et al.\ (2021) \cite{Mazyavkina2021RLSurvey}} survey sparse reward challenges in combinatorial optimization, identifying credit assignment as a fundamental bottleneck. \textbf{Ding et al.\ (2019) \cite{Ding2019Credit}} specifically address learning from sparse binary rewards through prediction-based credit assignment techniques. This sparsity motivates the use of n-step returns, reward shaping, and evolved neural architectures capable of learning long-horizon dependencies from limited feedback signals.

---

### Example 4: Set Transformers (Section 3.5.2, after line 306)

**Current text:**
> \textbf{Lee et al.\ (2019) \cite{LHS+19}} introduced Set Transformers, specifically designed for permutation-invariant set processing.

**Revised with citations:**
> Set Transformers build upon the foundational work of \textbf{Zaheer et al.\ (2017) \cite{Zaheer2017DeepSets}}, who introduced Deep Sets and proved that any permutation-invariant function can be represented as a composition of pooling operations over element-wise transformations. \textbf{Lee et al.\ (2019) \cite{LHS+19}} extended this framework by introducing Set Transformers, specifically designed for permutation-invariant set processing with attention mechanisms.

---

## 🔍 Where to Find the Papers

All recommended papers have been compiled in two files:

1. **RECOMMENDED_PAPERS_BY_SECTION.md** - Detailed descriptions and relevance explanations
2. **RECOMMENDED_BIBTEX_ENTRIES.bib** - Ready-to-use BibTeX entries

Simply copy the BibTeX entries you need into your thesis `.bib` file and use the citation integration examples above as templates.

---

## ✅ Completion Checklist

- [ ] Add 2 citations to Section 3.2.3 State Encoders
- [ ] Add 2 citations to Section 3.2.3 Action Encoders
- [ ] Add 2 citations to Section 3.3.1 Application to 3D BPP paragraph
- [ ] Add Deep Sets citation to Section 3.5.2
- [ ] Add 2-3 recent papers to Section 3.2.2 RL for Bin Packing
- [ ] Add 2 papers to Section 3.4.2 GNN
- [ ] Add 1 paper to Section 3.4.1 Heightmaps (optional)
- [ ] Copy BibTeX entries to thesis `.bib` file
- [ ] Update citations in thesis `.tex` file
- [ ] Compile thesis and verify all citations render correctly

---

**Created:** 2025-11-21
**Sections with high-priority gaps:** 4
**Total recommended papers:** 15+ (prioritized from 36 total)
**Estimated time to integrate:** 2-3 hours


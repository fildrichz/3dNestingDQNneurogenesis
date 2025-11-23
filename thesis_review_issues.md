# Thesis Review: Issues Found

## Executive Summary

This document outlines critical issues found in the Neural Architecture Search and Neurogenesis sections of the thesis, including fabricated references, incorrect citations, wrong publication years, and typographical errors.

---

## 1. CRITICAL: Fabricated/Unverifiable References

### 1.1 Booysen & Bosman (2024) - FABRICATED
**Location:** `neurogenesis_references.bib:54-64`, cited in `neurogenesis_section.tex:228,264`

**Issue:** This reference appears to be completely fabricated.

```bibtex
@article{Booysen2024,
  author    = {Marius Booysen and Peter A. N. Bosman},
  title     = {Multi-Objective Neural Architecture Search with Pareto Front Approximation},
  journal   = {IEEE Transactions on Evolutionary Computation},
  volume    = {28},
  number    = {1},
  pages     = {45--59},
  year      = {2024},
  doi       = {10.1109/TEVC.2023.1234567},
```

**Evidence:**
- Web search found NO results for these authors with this title
- DOI `10.1109/TEVC.2023.1234567` appears to be a placeholder (sequential pattern "1234567")
- No publication matches these authors and topic in 2024

**Action Required:** Remove this reference entirely or replace with actual multi-objective NAS paper.

**Suggested Replacement:** Lu et al., "NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm" (2019) or similar verified papers on multi-objective NAS.

---

### 1.2 Xu et al. (2024) "Learning to Pack" - UNVERIFIABLE
**Location:** `neurogenesis_references.bib:115-121`

**Issue:** Cannot verify this paper exists.

```bibtex
@inproceedings{Xu2024,
  author    = {Jingwei Xu and Yuan Gao and Nianzu Yang and Haoyuan Hu and Jianye Hao},
  title     = {Learning to Pack: A Data-Driven Approach for 3D Bin Packing},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2024},
```

**Evidence:**
- Web search found NO results for this paper
- No AAAI 2024 paper with this title or author combination

**Action Required:** Verify this reference or remove it. Check AAAI 2024 proceedings directly.

---

### 1.3 Wang et al. (2023) - UNVERIFIABLE
**Location:** `neurogenesis_references.bib:123-129`

```bibtex
@article{Wang2023,
  author    = {Yunhao Wang and Jingyi Fan and Zhipeng Lü and Bo Huang},
  title     = {Deep Reinforcement Learning for 3D Bin Packing with Complex Constraints},
  journal   = {IEEE Transactions on Automation Science and Engineering},
  year      = {2023},
```

**Evidence:**
- Web search found NO results
- Cannot verify publication exists

**Action Required:** Verify or remove.

---

### 1.4 Li et al. (2023) - UNVERIFIABLE
**Location:** `neurogenesis_references.bib:93-99`

```bibtex
@article{Li2023,
  author    = {Zhuoyi Li and Pengfei Zhang and Hui-Ling Zhen and Mingxuan Yuan and Zhipeng Lü},
  title     = {Neural Architecture Search for Combinatorial Optimization: A Survey},
  journal   = {IEEE Transactions on Neural Networks and Learning Systems},
  year      = {2023},
```

**Evidence:**
- Web search found NO results for this specific paper
- May be confusion with "Zhuoyi Lin" (different surname) who works on combinatorial optimization
- No survey with this exact title found in IEEE TNNLS

**Action Required:** Verify or remove. Possibly replace with verified survey like "Solving Combinatorial Optimization Problems with Deep Neural Network: A Survey" (Wang et al., 2024).

---

### 1.5 Zhang et al. (2021) - UNVERIFIABLE
**Location:** `neurogenesis_references.bib:101-110`

```bibtex
@inproceedings{Zhang2021,
  author    = {Haoran Zhang and Mingxuan Yuan and Jia Zeng and Qingfu Zhang and Han Zhao},
  title     = {Learning to Solve Combinatorial Optimization Problems with Graph Neural Architecture Search},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {35},
  number    = {12},
  pages     = {10627--10635},
  year      = {2021},
```

**Evidence:**
- Web search found NO results for this paper at AAAI 2021

**Action Required:** Verify or remove.

---

## 2. CRITICAL: Incorrect Citation Attribution

### 2.1 BLX-α Crossover Misattributed
**Location:** `neurogenesis_section.tex:112-122`, `thesis_nas_section_expansion.md:342-365`

**Issue:** BLX-α crossover is attributed to Takahashi & Kita (2001), but BLX-α was actually invented by **Eshelman & Schaffer (1993)**.

**Current (INCORRECT) text:**
```latex
\paragraph{BLX-$\alpha$ Crossover}
Following \citet{Takahashi2001}, for real-valued genes we employ BLX-$\alpha$ crossover...
```

**Facts:**
- **BLX-α was introduced by:** L.J. Eshelman and J.D. Schaffer (1993) in "Real-Coded Genetic Algorithms and Interval-Schemata", *Foundations of Genetic Algorithms 2*, pp. 187-202.
- **Takahashi & Kita (2001)** did NOT invent BLX-α. Their paper "A Crossover Operator Using Independent Component Analysis for Real-Coded Genetic Algorithms" **combines** BLX-α with ICA to improve performance on non-separable functions.

**Evidence:**
- [R-project documentation](https://search.r-project.org/CRAN/refmans/adana/html/blxa.html) confirms Eshelman & Schaffer (1993) as origin
- [Takahashi & Kita paper on IEEE](https://ieeexplore.ieee.org/document/934452/) clearly states: "The blend crossover (BLX-α) proposed by L.J. Eshelman and J.D. Schaffer shows a good searching ability for separable fitness functions."

**Action Required:**
1. Add correct reference: Eshelman & Schaffer (1993)
2. Change citation in text from `\citet{Takahashi2001}` to `\citet{Eshelman1993}`
3. Optionally mention Takahashi & Kita's ICA extension separately

---

## 3. Wrong Publication Year

### 3.1 White et al. - Should be 2021, not 2023
**Location:** `neurogenesis_references.bib:80-88`

**Issue:** Paper published at NeurIPS **2021**, not NeurIPS 2023.

**Current (INCORRECT):**
```bibtex
@article{White2023,
  author    = {Colin White and Arber Zela and Robin Ru and Yang Liu and Frank Hutter},
  title     = {How Powerful are Performance Predictors in Neural Architecture Search?},
  journal   = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {36},
  pages     = {28924--28955},
  year      = {2023},
```

**Correct Information:**
- **Year:** 2021 (not 2023)
- **Volume:** 34 (not 36)
- **arXiv:** https://arxiv.org/abs/2104.01177 (submitted April 2021)
- **NeurIPS Proceedings:** https://proceedings.neurips.cc/paper/2021/hash/ef575e8837d065a1683c022d2077d342-Abstract.html

**Evidence:**
- Paper first appeared on arXiv April 2, 2021
- Presented at NeurIPS 2021
- All search results confirm 2021, none mention 2023

**Action Required:**
1. Change `year = {2023}` to `year = {2021}`
2. Change `volume = {36}` to `volume = {34}`
3. Update citation key from `White2023` to `White2021` (and update all references in text)
4. Verify page numbers

---

## 4. Typographical Errors

### 4.1 "evolutionary evolution" - Redundant/Awkward Phrasing
**Location:** `neurogenesis_section.tex:4`

**Current:**
```latex
...refers to the evolutionary evolution of neural network architectures...
```

**Issue:** "evolutionary evolution" is redundant.

**Suggested Fix:**
```latex
...refers to the evolution of neural network architectures...
```
OR
```latex
...refers to the evolutionary development of neural network architectures...
```

---

### 4.2 "over fitting" - Should be one word
**Location:** `thesis_nas_section_expansion.md:20`

**Current:**
```
...to avoid over fitting and excessive computational cost.
```

**Issue:** "overfitting" is one word in machine learning terminology.

**Suggested Fix:**
```
...to avoid overfitting and excessive computational cost.
```

---

## 5. Other Issues to Review

### 5.1 Verify BLX-α Formula
**Location:** `neurogenesis_section.tex:114-119`

The BLX-α formula given is:
```latex
c_{\min} = \min(p_1, p_2), \quad c_{\max} = \max(p_1, p_2), \quad r = c_{\max} - c_{\min}
\end{equation}
\begin{equation}
\text{Offspring} \sim \text{Uniform}(c_{\min} - \alpha r, c_{\max} + \alpha r)
```

**Note:** This formula appears correct for BLX-α, but should be verified against Eshelman & Schaffer (1993) once proper citation is added.

### 5.2 Verify Adaptive Mutation Rate Formula
**Location:** `neurogenesis_section.tex:143-145`

```latex
p_{\text{mutation}}(t) = p_{\text{init}} \times \exp(-\lambda t)
```

**Note:** This exponential decay formula is attributed to "Ding et al." but should be verified from the actual paper. The specific form and whether the max() clipping in equations 220-222 comes from Ding should also be verified.

### 5.3 Real et al. (2017) Claims Need Verification
**Location:** `neurogenesis_section.tex:260-261`, `thesis_nas_section_expansion.md:306-312`

**Claims about Real et al.:**
- "mutation-only evolution with aging"
- "Binary tournament: Sample 2 individuals, keep fitter, mutate, replace worse"

**Note:** Could not access PDF to verify these specific algorithmic details. Should verify against actual paper.

---

## Summary Statistics

- **Fabricated/Unverifiable References:** 5 (Booysen2024, Xu2024, Wang2023, Li2023, Zhang2021)
- **Incorrect Attributions:** 1 (BLX-α to Takahashi instead of Eshelman)
- **Wrong Publication Years:** 1 (White et al. 2021 cited as 2023)
- **Typographical Errors:** 2 (evolutionary evolution, over fitting)
- **Needs Verification:** 3 (BLX-α formula, mutation formula, Real et al. details)

---

## Recommended Actions

### Immediate Priority (Critical):
1. ✅ **Remove or replace Booysen2024** - completely fabricated with fake DOI
2. ✅ **Fix BLX-α attribution** - cite Eshelman & Schaffer (1993), not Takahashi (2001)
3. ✅ **Correct White et al. year** - 2021, not 2023
4. ✅ **Verify/remove unverifiable references** - Xu2024, Wang2023, Li2023, Zhang2021

### Secondary Priority:
5. Fix typographical errors (evolutionary evolution, over fitting)
6. Add missing Eshelman & Schaffer (1993) reference to bibliography
7. Verify formulas against original papers if possible

### Verification Needed:
8. Access Ding et al. (2010) to verify mutation rate decay formula
9. Access Real et al. (2017) to verify mutation-only evolution claims
10. Check if AAAI/IEEE papers can be found with different author names or titles

---

## Sources

- [BLX-α Documentation (R-project)](https://search.r-project.org/CRAN/refmans/adana/html/blxa.html)
- [Takahashi & Kita IEEE Paper](https://ieeexplore.ieee.org/document/934452/)
- [White et al. arXiv](https://arxiv.org/abs/2104.01177)
- [White et al. NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/ef575e8837d065a1683c022d2077d342-Abstract.html)
- [Real et al. arXiv](https://arxiv.org/abs/1703.01041)
- [Real et al. ICML 2017](http://proceedings.mlr.press/v70/real17a/)
- [Ding et al. ResearchGate](https://www.researchgate.net/publication/220147127_Using_Genetic_Algorithms_to_Optimize_Artificial_Neural_Networks)

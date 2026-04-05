# Unsupervised Structural Anomaly Detection in Financial Networks via Stigmergic Mesh

## Paper Outline — Draft

### Abstract
We present the first application of Ambient Structure Discovery (ASD), a
stigmergic mesh architecture grounded in Adaptive Resonance Theory, to the
detection of structural anomalies in financial networks without labeled training
data. Through controlled experiments on the Elliptic++ Bitcoin dataset (203K
transactions, 822K wallet addresses), we demonstrate that complement-coded
continuous structural vectors enable the mesh to self-organize into meaningful
financial behavior categories. The mesh identifies exchange and mining
operations as structural anomalies (2.2x enrichment, p < 0.00002) in a network
where illicit transaction patterns dominate the topological norm. We compare
three signal construction strategies — text-based vocabulary, discretized
structural terms, and complement-coded continuous coefficients — and show that
only the continuous approach produces genuine category differentiation in the
Fuzzy ART framework.

### 1. Introduction
- The detection problem: financial crime concealed through structural topology
- First-order vs second-order ignorance in financial networks
- Limitations of supervised approaches: label scarcity, label staleness, adversarial adaptation
- ASD: unsupervised structural anomaly detection via self-organizing mesh
- Contributions: (1) first financial application of ASD, (2) signal construction
  methodology for Fuzzy ART on financial data, (3) empirical evidence that
  complement coding is necessary for structural differentiation

### 2. Background
- Adaptive Resonance Theory and Fuzzy ART
- Stigmergy: coordination through environment modification
- Ambient Structure Discovery: eigenvalue collapse, linguistic compression, action decoupling
- Prior work on Bitcoin AML: GNN approaches, Elliptic benchmarks, Chainalysis heuristics
- The complement coding requirement for continuous-valued financial data

### 3. Methodology

#### 3.1 Signal Construction
- The identity-vs-topology problem: why raw text saturates the mesh
- Seven structural coefficients: fragmentation, relay score, feature variance,
  neighborhood divergence, feature extremity, fan ratio, neighborhood heterogeneity
- Complement coding: I = (a, 1-a), 14-dimensional input vectors
- Wallet-level coefficients: activity asymmetry, volume concentration, temporal
  regularity, counterparty diversity, fee behavior, lifetime intensity, repeat ratio

#### 3.2 Mesh Configuration
- Familiarity function reweighting: embedding_similarity = 0.60 (structural primary)
- Worker capacity, vigilance parameters, gap detection threshold
- No labeled data used in signal construction or mesh routing
- Labels used ONLY for post-hoc evaluation

#### 3.3 Experimental Design
- A/B comparison: text-based vs complement-coded on same 203K transactions
- Transaction-level vs wallet-level analysis
- Evaluation metrics: Cohen's d, anomaly enrichment (binomial test),
  decile analysis, Mann-Whitney U

### 4. Results

#### 4.1 Text-Based Baseline (Negative Control)
- 203K transactions, 60 workers, 4 hours
- All workers converge to identical 30-term vocabulary
- Cohen's d = -0.29, 1 anomaly
- Conclusion: discrete vocabulary saturates Fuzzy ART categories

#### 4.2 Complement-Coded Transaction-Level
- Same data, structural vectors, reweighted familiarity
- 59 anomalies (59x baseline), lower familiarity scores
- Licit transactions 2.2x enriched in anomalies (p = 1.29e-05)
- Cohen's d = -0.21 (illicit more familiar = dominant pattern)
- Workers differentiate by structural shape in continuous space

#### 4.3 Wallet-Level Analysis
- 50K sample: directional signal (zero illicit in top deciles)
- Stratified sample (14K illicit, 30K licit): [PENDING]
- Coefficient distribution shift: volume_concentration diagnostic

#### 4.4 Interpretation: When "Normal" Is Illicit
- Bitcoin's structural norm is illicit-compatible activity
- Exchanges and miners are the true structural anomalies
- The mesh correctly identifies this without being told
- Implications for anomaly-based detection in adversarial environments

### 5. Discussion
- ASD detects structural categories, not "good" vs "bad"
- The interpretation framework must be domain-aware
- Complement coding is not optional — it is necessary for Fuzzy ART
- Comparison with supervised approaches: different question, different strengths
- Stochastic resonance as a potential amplifier for sub-threshold signals

### 6. Limitations
- Elliptic features are partially anonymized
- 203K transactions is modest; full blockchain would test scalability
- Mesh routing time grows with worker count (O(workers * terms) per signal)
- The wallet-level experiment needs more statistical power
- Corporate ownership domain not yet tested with continuous vectors

### 7. Future Work
- Corporate beneficial ownership with proper structural coefficients
  (D_L, F_o, N_P, J_S, C_B, V_C, A_D)
- Full Bitcoin blockchain temporal analysis (detect before sanctions)
- Stochastic resonance integration for severity inversion mitigation
- Comparison with GNN baselines on identical feature sets
- Real-time streaming deployment for compliance monitoring

### 8. Conclusion
- ASD self-organizes into meaningful structural categories from financial data
- Signal construction determines detection capability
- Complement-coded continuous vectors are necessary and sufficient for Fuzzy ART
- The unsupervised approach discovers structural truths that supervised methods
  cannot: in Bitcoin, illicit is the norm

---

## Data and Code Availability

All code: https://github.com/jmcentire/stigmergy (src/stigmergy/ownership/)
Elliptic++ dataset: https://github.com/git-disl/EllipticPlusPlus
ASD patent: USPTO Provisional Application #63/981,369

## Key Figures Needed
1. Worker vocabulary convergence: text-based (all identical) vs structural (differentiated)
2. Anomaly enrichment by class: bar chart of licit/illicit/unknown in anomaly set vs population
3. Score distributions: illicit vs licit density plots for each experiment
4. Decile analysis: illicit rate by familiarity score decile
5. Coefficient distributions: illicit vs licit for each structural coefficient
6. Mesh self-organization timeline: worker count and merge/fork events over signal ingestion

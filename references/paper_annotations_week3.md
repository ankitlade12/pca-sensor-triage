# Paper Annotations — Week 3
## Final 5 Papers (Total: 33 papers)

---

### Paper 29: Downs & Vogel — Original TEP Paper
- **Title:** "A plant-wide industrial process control problem"
- **Authors:** James J. Downs, Ernest F. Vogel (Eastman Chemical Company)
- **Year:** 1993 | **Journal:** Computers & Chemical Engineering, 17(3), 245-255
- **Citations:** 3000+
- **Summary:** Defines the TEP benchmark: reactor/separator/recycle with 2 exothermic reactions, 12 manipulated variables, 41 measured variables, 20 programmed disturbances. FORTRAN simulation available.
- **Cite for:** Experiments §4.1 — canonical reference for TEP dataset.

### Paper 30: Compressive Sensing for IoT Sensor Networks
- **Title:** "Application of Compressive Sensing Techniques in Distributed Sensor Networks" (survey)
- **Year:** 2017 | **Source:** arXiv 1709.10401
- **Summary:** Reviews compressive sensing (CS) for sensor bandwidth reduction. CS exploits signal sparsity to sample below Nyquist rate. Requires sparse signal representation + random projection matrices + reconstruction algorithms (L1 minimization, OMP).
- **Gap for us:** CS requires signal sparsity assumption + expensive reconstruction. Our PCA-triage doesn't require sparsity, doesn't discard data (reduces rate, doesn't skip), and has no reconstruction cost at the receiver.
- **Cite for:** Related Work §2.3 — alternative bandwidth reduction approach. Differentiate clearly.

### Paper 31: Ross et al. — Incremental Learning for Eigenbasis
- **Title:** "Incremental Learning for Robust Visual Tracking"
- **Authors:** David A. Ross, Jongwoo Lim, Ruei-Sung Lin, Ming-Hsuan Yang
- **Year:** 2008 | **Journal:** Int. J. Computer Vision, 77(1-3), 125-141
- **Summary:** Foundational paper behind sklearn's IncrementalPCA. Sequential SVD updates for online eigenbasis learning. O(batch_size × d²) per update. Constant memory O(batch_size × d).
- **Cite for:** Method §3 — algorithmic basis for our IncrementalPCA backbone.

### Paper 32: Rieth et al. — Extended TEP Dataset (Harvard Dataverse)
- **Title:** "Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation"
- **Authors:** Christoph A. Rieth, Ben D. Amsel, Randy Tran, Maia B. Cook
- **Year:** 2017 | **Source:** Harvard Dataverse, doi:10.7910/DVN/6C3JR1
- **Summary:** Extended TEP simulation with 500 training + 960 testing runs per fault type. Fault-free + 20 fault types. RData format. This is the exact dataset we use.
- **Cite for:** Experiments §4.1 — our data source.

### Paper 33: Levy & Lindenbaum — Sequential Karhunen-Loeve
- **Title:** "Sequential Karhunen-Loeve Basis Extraction and its Application to Images"
- **Authors:** A. Levy, M. Lindenbaum
- **Year:** 2000 | **Journal:** IEEE Trans. Image Processing, 9(8), 1371-1374
- **Summary:** Original sequential KL transform that IncrementalPCA extends. Updates eigenbasis from sequential data blocks without reprocessing entire dataset.
- **Cite for:** Method §3 — theoretical foundation for incremental eigenbasis updates.

---

## Final Bibliography Count: 33 papers (target: 30-35 ✓)

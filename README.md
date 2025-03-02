# HACSurv: A Hierarchical Copula-based Approach for Survival Analysis with Dependent Competing Risks

**PyTorch implementation for the paper:** [HACSurv: A Hierarchical Copula-based Approach for Survival Analysis with Dependent Competing Risks](https://arxiv.org/abs/2410.15180) (AISTATS 2025).

## 1. Framingham Dataset

### Step 1: Learning the Hierarchical Archimedean Copula (HAC)
- **Notebook:** `HACSurv_FRAMINGHAM.ipynb`
  - Learn copulas for pairs (0,1), (0,2), and (1,2).
  - Select the **strongest copula** as the *Inner copula*.
  - Select the **weakest copula** as the *Outer copula*.

### Step 2: Conducting Survival Analysis Experiments
- **Reproducing HACSurv:**  
  Run `New_HACSurv_FramingHam_bached.py` after setting the paths to the inner and outer copula parameter files.

- **Reproducing HACSurv (Symmetry):**  
  Set the appropriate paths in the script and run `New_SymHACSurv_FramingHam_bached.py`.

- **Reproducing HACSurv (Independent):**  
  In the optimizer configuration, comment out the following line:
  ```python
  {"params": model.phi.parameters(), "lr": 5e-4}
## 2. Synthetic Dataset

### Use Pre-trained HAC Checkpoint
- Run the following command:
  ```bash
  python HACSurv_competing_syn.py
  ```

### Learn HAC Parameters Yourself
1. **Step 1:** Run `Synthetic_competing_learn_step1_copula.py` to capture the copula between events and censoring.
2. **Step 2:** Use `competing_syn_get_HAC.ipynb` to:
   - Select two strongly dependent copulas (typically 01 and 23) as the inner copula.
   - Choose the weakest copula as the outer copula.
   - (This notebook also visualizes the learned HAC structure.)
3. **Step 3:** Update the HAC checkpoint path in `HACSurv_competing_syn.py` and run the survival analysis experiments.

- To reproduce **HACSurv (Symmetry)** and **HACSurv (Independent)** results, run:
  ```bash
  python Symmetry_HACSurv_competing_syn.py
  ```
  - **For HACSurv (Independent):** In the optimizer configuration, comment out the line:
    ```python
    {"params": model.phi.parameters(), "lr": 5e-4}
    ```


## 3. MIMIC-III Dataset

- **Direct Experimentation:**  
  Run the following command to use our pre-trained HAC checkpoint:
  
  ```bash
  python New_HACSurv_MIMIC.py
  ```

- **Learning HAC Structure:**  
  i. Run `MIMIC_copula.py` to learn the copula between events and censoring.  
  ii. Since events 1, 3, and 5 (related to respiratory diseases) show strong dependency, an inner copula is learned to capture their dependency. Use `MIMIC_learn_HAC.ipynb` to learn the copula for events 1, 3, and 5 (copula 135) and to learn the inner copula 135 and 24.  
  iii. Experiment: Update the copula checkpoint path in `New_HACSurv_MIMIC.py` and perform the survival analysis experiments.
  
- **Reproducing HACSurv (Symmetry) and HACSurv (Independent) Results:**  
  Run the following command:
  
  ```bash
  python New_HACSurv_Sym__MIMIC.py
  ```

---

## 4. Specifying a Known Copula for Survival Analysis

Some practitioners may wish to specify a known copula for survival analysis. Using our HACSurv framework, this is achievable.  
First, you can use the copula toolbox to sample from your specified copula. Then, use MLE to learn the copula network from these samples. Finally, freeze the HAC network to perform survival analysis.

If you have any questions or are interested in collaboration, feel free to contact:  
xliu@seu.edu.cn

---


## Citation

If you find this work useful, please consider citing our paper:
```bibtex
@inproceedings{
liu2025hacsurv,
title={{HACS}urv: A Hierarchical Copula-based Approach for Survival Analysis with Dependent Competing Risks},
author={Xin Liu and Weijia Zhang and Min-Ling Zhang},
booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
year={2025},
url={https://openreview.net/forum?id=R2Xaxv4ADz}
}
```


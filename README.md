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

- **Direct Experimentation:** Run the following command to use our pre-trained HAC checkpoint:
  ```bash
  python New_HACSurv_MIMIC.py
  ```
- **Learning HAC Structure:**
  1. Run `MIMIC_copula.py`.
  2. Since events 1, 3, and 5 (related to respiratory diseases) show strong dependency, an inner copula is learned to capture their dependency.
     - *Note: Code for this part is still being organized and will be released later.*

## Citation

If you find this work useful, please consider citing our paper:
```bibtex
@article{hac_surv,
  title={HACSurv: A Hierarchical Copula-based Approach for Survival Analysis with Dependent Competing Risks},
  author={...},
  journal={...},
  year={2025}
}
```


# Methodological Decisions Log

Every non-obvious decision in the pipeline is documented here with its rationale.
This file is the source of truth for "why did we do X instead of Y" questions.

Format: decision → rationale → thesis implication (where relevant).

---

## DATA

### D-01 · Filter `choice_type == "single"` only

**Decision:** Drop all MedMCQA examples where `choice_type != "single"` before any split.

**Rationale:** Multi-answer questions (e.g., "A and C", "All of the above") are incompatible with the restricted softmax over {A, B, C, D}. The model is expected to place probability mass on exactly one token; a question with two correct answers has an ambiguous ground-truth label that would corrupt both the correctness label `y` and the probe training signal.

**Effect:** 182,822 → 120,765 examples (66.1%). Lower than expected (~80%); documented in thesis Dataset section.

---

### D-02 · Stratify probe/train split by `subject_name`

**Decision:** Use `stratify=subjects` (21 medical categories) when splitting off the 16K probe_set from train_ft.

**Rationale:** MedMCQA is heavily imbalanced across subjects (Medicine 9%, Skin 1.1%). Without stratification, the probe_set could under-represent rare subjects, making the correctness probe unreliable for those domains. Stratification ensures probe and train_ft have the same subject mix, so probe AUROC is not artificially inflated by subject-level confounding.

---

### D-03 · Deduplication: exact match only, not TF-IDF cosine

**Decision:** Remove from probe_set any question that appears verbatim in train_ft (after lowercasing and stripping punctuation). TF-IDF cosine similarity was considered but rejected as the deduplication criterion.

**Rationale:** We removed 2,693 probe-set questions that appeared verbatim in the fine-tuning set. TF-IDF cosine similarity was considered but rejected: at the median prompt length of 53 tokens, template phrases such as "Which of the following is..." produce spurious high-similarity scores between semantically distinct questions. At threshold 0.90, TF-IDF removed 36.7% of probe_set — mostly false positives sharing a question template, not shared content. Exact-string matching is the only criterion that unambiguously identifies contamination at the hidden-state level: two questions with the same text will produce nearly identical hidden states regardless of fine-tuning, inflating probe AUROC.

**Thesis implication:** Report as: *"2,693 verbatim duplicates between train_ft and probe_set were identified and removed from probe_set (exact string match after normalisation). The source dataset (MedMCQA) was scraped from overlapping question banks, making such duplicates expected."*

**Effect:** probe_set 16,000 → 13,307; routing_train 12,000 → 9,307; iso_cal unchanged at 4,000.

---

### D-04 · Answer distribution imbalance is a Limitation, not a filter

**Decision:** Do not re-balance the A/B/C/D answer distribution. Report it as a limitation.

**Rationale:** EDA shows A=31.5%, B=27.5%, C=23.2%, D=17.8% (max deviation ±7.2pp from uniform). Re-balancing by downsampling would reduce the training set size and introduce a selection bias not present in real-world medical MCQ deployment. The imbalance is a property of the source exam banks, not a dataset construction error. It should be acknowledged in the thesis: the model may develop a systematic bias against option D, which could affect the routing policy's uncertainty signals.

---

### D-05 · Cross-dataset contamination: OOD sets are clean

**Finding:** Zero exact-match overlaps between MedMCQA train (120,765 questions) and MedQA-USMLE test (1,273) or MMLU medical test (945). Two near-duplicates with MMLU at cosine ≥ 0.85 were false positives (short generic phrases matching by template). The OOD evaluation sets are uncontaminated.

**Thesis implication:** The three-tier OOD evaluation (in-dist → near-OOD → far-OOD) is methodologically valid. State explicitly in the Evaluation section.

---

## MODEL & TRAINING

### D-06 · LoRA FP16 primary; QLoRA 4-bit fallback only

**Decision:** Fine-tune in full FP16 precision with LoRA adapters. Use QLoRA 4-bit NF4 only if VRAM < 24 GB, and run the KL ablation (Section 9.4) in that case.

**Rationale:** Restricted entropy and logit gap are computed directly from model logits. 4-bit NF4 quantisation compresses the dynamic range of weights, which can systematically distort logit distributions — corrupting the uncertainty signals the routing policy depends on. FP16 eliminates this confound. If QLoRA is used, the KL divergence ablation (mean KL between 4-bit and FP16 logits on 200 val examples) quantifies the distortion; if mean KL > 0.10 nats, it must be discussed in Limitations.

---

### D-07 · max_seq_len = 512 tokens

**Decision:** Truncate all prompt+answer sequences to 512 tokens during fine-tuning.

**Rationale:** EDA on 5,000 training examples (Mistral tokenizer) shows p50=53 tokens, p95=102, p99=149, max=314. Zero examples exceed 512 tokens. The limit is safe with a factor-of-1.6 margin above p99.

---

### D-08 · Loss only on the answer token (DataCollatorForCompletionOnlyLM)

**Decision:** Use `response_template="Answer:"` so cross-entropy loss is computed only on the single answer token (A/B/C/D), not on the full prompt.

**Rationale:** The model already knows how to generate text — we are teaching it medical MCQ answering, not language modelling. Computing loss on the prompt would dilute the gradient signal and slow convergence. The answer token is the only token where our restricted softmax operates; that is what we want to calibrate.

---

## UNCERTAINTY & ROUTING

### D-09 · Restricted entropy over {A,B,C,D} only

**Decision:** Compute entropy as `H = −Σ p_i log(p_i + 1e-10)` where `p = softmax(logits[ids_ABCD])`, not over the full vocabulary.

**Rationale:** Full-vocabulary entropy is dominated by the mass on unrelated tokens (punctuation, common words). Restricting to the four answer tokens gives a direct measure of the model's uncertainty about the correct option. The `1e-10` epsilon prevents `-inf` when `p_i = 0` (which can occur after softmax over exactly 4 logits).

**Verified:** ids_ABCD = [1098, 1133, 1102, 1152] for Mistral-7B-Instruct-v0.3. All single-token. `' A'` and `'A'` produce identical token IDs — safe to use either.

---

### D-10 · 5-fold cross-fitting for probe scores on probe_set

**Decision:** Use out-of-fold (OOF) predictions when generating probe scores for probe_set examples. The final probe (trained on the full probe_set) is used only for val/test inference.

**Rationale:** If the probe is trained on the full probe_set and evaluated on the same probe_set, its predicted scores are optimistic — the probe has seen the labels. Cross-fitting ensures that the probe score for example `i` was generated by a model that never saw `i` during training. This is required for the routing LR to receive unbiased probe scores as input features.

---

### D-11 · Conformal quantile: `method="higher"` (not default interpolation)

**Decision:** Always use `np.quantile(..., method="higher")` when computing the conformal threshold `q_hat`.

**Rationale:** Split conformal prediction requires the empirical quantile at level `⌈(1−α)(n+1)⌉/n`. The "higher" method returns the smallest value in the calibration set that is ≥ the target quantile — this is the mathematically correct choice for the finite-sample coverage guarantee. Default interpolation (linear) can return a value between two calibration scores, which breaks the formal guarantee. This is a common implementation mistake in CP papers.

---

### D-12 · AUGRC as primary selective prediction metric

**Decision:** Report AUGRC (Area Under Generalized Risk-Coverage curve) as the primary metric for selective prediction quality. AURC is secondary.

**Rationale:** AUGRC (Kläser et al., NeurIPS 2024, arXiv:2407.01032) measures expected risk of undetected failures across all coverage thresholds, giving equal weight to each coverage level. Standard AURC weights high-coverage regimes more heavily, which can obscure poor performance at low escalation rates — precisely the regime where our routing policy operates in deployment. AUGRC is the more appropriate metric for a system that must control risk at a specific operating point.

---

### D-13 · Isotonic regression calibration (vanilla sklearn); ROC-isotonic as ablation

**Decision:** Use `IsotonicRegression(out_of_bounds="clip")` from sklearn as the primary calibrator. ROC-regularised isotonic regression (Dimitriadis et al. 2023, `regcal`) is evaluated in the ablation table.

**Rationale:** Vanilla isotonic regression is universally available, well-understood, and monotone — it preserves the ordering of routing LR scores, which is required for the coverage guarantee to hold. ROC-isotonic optimises calibration jointly with discrimination, potentially improving Brier score at the cost of interpretability. If Brier score improvement > 0.01 in the ablation, switch primary to ROC-isotonic and document.

---

## EVALUATION

### D-14 · Conformal coverage guarantee holds only on in-distribution data

**Decision:** Label the in-dist (MedMCQA test) results as "CP guarantee valid" and near-OOD / far-OOD results as "CP guarantee not valid". Do not re-calibrate for OOD sets.

**Rationale:** Split CP guarantees marginal coverage only when the test distribution matches the calibration distribution (exchangeability). MedQA-USMLE and MMLU are drawn from different exam banks — the exchangeability assumption fails. We deliberately do not re-calibrate on OOD data (that would require labels at test time). The OOD results are reported as empirical observations of coverage degradation under distribution shift — which is itself the main empirical contribution of the thesis.

---

### D-15 · Bootstrap CI: 1,000 resamples, seed=42, reported as [2.5%, 97.5%]

**Decision:** All confidence intervals use 1,000 bootstrap resamples with `np.random.default_rng(seed=42)`, reporting the 2.5th and 97.5th percentiles.

**Rationale:** 1,000 resamples gives stable CI estimates for the sample sizes involved (945–6,150 test examples). Percentile bootstrap (not BCa) is used for simplicity and interpretability; BCa would be more accurate for skewed statistics but the difference is negligible at these sample sizes. Seed is fixed for reproducibility.

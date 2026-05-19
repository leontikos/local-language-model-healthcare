# Roadmap — Safe Enough to Answer Locally?
# Risk-Controlled Escalation for On-Device Medical Language Models

---

## CZĘŚĆ I — KOMPLETNA SPECYFIKACJA PIPELINE'U

---

### OVERVIEW

**Cel systemu:** Zbudować routing policy dla edge-deployed medical LLM (≥8GB VRAM), który z formalną conformal coverage guarantee decyduje, czy odpowiedzieć lokalnie czy eskalować do eksperta. System działa wyłącznie na 4-opcyjnym MCQ.

**Stack:**
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Fine-tuning: LoRA FP16 (primary) / QLoRA 4-bit (fallback)
- Uncertainty: restricted entropy + logit gap (conditional) + hidden-state probe + p(True)
- Routing: Logistic Regression + ROC-isotonic calibration
- Coverage: Split Conformal Prediction (primary) + Mondrian CP (ablation)
- Metrics: AUROC + AUGRC (primary), AURC + Brier + ECE (secondary)

---

## SEKCJA 1 — DANE

### 1.1 Datasety

| Dataset | HuggingFace ID | Rola | Rozmiar | Licencja |
|---|---|---|---|---|
| MedMCQA | `openlifescienceai/medmcqa` | Train / Probe / Cal / In-dist eval | 182,822 train / 4,183 val / 6,150 test | Apache 2.0 |
| MedQA-USMLE | `GBaker/MedQA-USMLE-4-options` | Near-OOD eval | 1,273 test | CC-BY 4.0 |
| MMLU medical | `cais/mmlu` (5 kategorii) | Far-OOD eval | 945 test | MIT |

**MMLU — dokładnie 5 kategorii (college_biology wycięte — zbyt ogólna):**

| Kategoria | Test |
|---|---|
| clinical_knowledge | 265 |
| professional_medicine | 272 |
| college_medicine | 173 |
| medical_genetics | 100 |
| anatomy | 135 |
| **Razem** | **945** |

### 1.2 Schematy kolumn

**MedMCQA:**
```
id          : string (UUID)
question    : string
opa, opb, opc, opd : string   # opcje A, B, C, D
cop         : int (ClassLabel) # ZWERYFIKOWAĆ EMPIRYCZNIE: 0=A czy 1=A?
choice_type : string           # "single" lub "multi" (pole zawodne)
exp         : string (nullable, ~50% brakuje) — NIEUŻYWANE w pipeline
subject_name: string (21 kategorii medycznych)
topic_name  : string (nullable)
```

**MedQA-USMLE:**
```
question    : string
options     : dict {"A": str, "B": str, "C": str, "D": str}
answer      : string (tekst poprawnej odpowiedzi)
answer_idx  : string  # "A", "B", "C" lub "D" — już litera, bez konwersji
meta_info   : string  # "step1" lub "step2&3"
metamap_phrases : list[str]  — NIEUŻYWANE
```

**MMLU:**
```
question    : string
choices     : list[str]  # choices[0]=A, [1]=B, [2]=C, [3]=D
answer      : string     # "A", "B", "C" lub "D"
subject     : string
```

### 1.3 Unified Prompt Template

Identyczny dla wszystkich trzech datasetów:

```
Question: {question}
A) {option_A}
B) {option_B}
C) {option_C}
D) {option_D}
Answer:
```

**Normalizacja:**
```python
def normalize_example(example, source):
    if source == "medmcqa":
        opts = {
            "A": example["opa"], "B": example["opb"],
            "C": example["opc"], "D": example["opd"]
        }
        # cop encoding MUSI być zweryfikowany (patrz BLOCKER 1)
        cop_map = {0: "A", 1: "B", 2: "C", 3: "D"}  # zakładamy 0-indexed
        label = cop_map[example["cop"]]

    elif source == "medqa":
        opts = example["options"]       # już dict A/B/C/D
        label = example["answer_idx"]  # już litera

    elif source == "mmlu":
        letters = ["A", "B", "C", "D"]
        opts = {l: example["choices"][i] for i, l in enumerate(letters)}
        label = example["answer"]       # już litera

    return opts, label


def format_prompt(question, opts, include_answer=False, answer=None):
    prompt = (
        f"Question: {question}\n"
        f"A) {opts['A']}\n"
        f"B) {opts['B']}\n"
        f"C) {opts['C']}\n"
        f"D) {opts['D']}\n"
        f"Answer:"
    )
    if include_answer:
        prompt += f" {answer}"
    return prompt
```

### 1.4 Data splits i flow

```
MedMCQA raw train (~182,822)
    │
    ▼ filter choice_type == "single"
    │ + ręczna weryfikacja 100 przykładów (error rate < 5%)
    │
    ▼ (120,765 próbek — choice_type='single' = 66.1% z 182,822; mniej niż zakładane ~80%)
    │
    ▼ stratified split po subject_name, seed=42
    │  → nominalne: train_ft 104K + probe 16K
    │  → po exact-match dedup (2,693 duplikatów między train_ft i probe usunięte z probe):
    │
    ├── train_ft  (104,765) ─────────────────── FINE-TUNING ONLY
    └── probe_set  (13,307, po dedup)            (model NIE widzi probe_set podczas FT)
              │
              ▼ dalszy podział (seed=42)
              ├── routing_train  (9,307) ── routing LR training
              └── iso_cal        (4,000) ── isotonic regression calibration

    Uwaga: nominalne rozmiary (128K / 16K / 12K) to cele przed dedup.
    Faktyczne rozmiary po uruchomieniu 01_prepare.py: patrz data/splits/meta.json.

MedMCQA val  (4,183) ─── WYŁĄCZNIE conformal calibration (q_hat)
MedMCQA test (6,150) ─── in-distribution evaluation
MedQA test   (1,273) ─── near-OOD evaluation
MMLU test      (945) ─── far-OOD evaluation
```

**Zasada: żaden split nie jest używany w więcej niż jednej fazie.**

```python
import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Krok 1: wczytaj i filtruj
train_raw = datasets.load_dataset("openlifescienceai/medmcqa", split="train")
train_single = train_raw.filter(lambda x: x["choice_type"] == "single")

# Krok 2: wyodrębnij probe_set PRZED fine-tuningiem
subjects = [ex["subject_name"] for ex in train_single]
all_idx  = list(range(len(train_single)))

train_idx, probe_idx = train_test_split(
    all_idx, test_size=16000,
    stratify=subjects, random_state=42
)
# Uwaga: po exact-match dedup (2,693 duplikatów) faktyczne rozmiary:
# train_ft=104,765  probe=13,307  (patrz data/splits/meta.json)

train_ft  = train_single.select(train_idx)   # 104,765 po dedup
probe_set = train_single.select(probe_idx)   # 13,307 po dedup

# Krok 3: podziel probe_set na routing_train i iso_cal
probe_y = [int(normalize_example(ex, "medmcqa")[1] is not None)
           for ex in probe_set]  # placeholder, prawdziwe y po ekstrakcji

routing_idx, iso_idx = train_test_split(
    list(range(len(probe_set))), test_size=4000,
    random_state=42
)
# routing_train = probe_set.select(routing_idx)  # 9,307
# iso_cal       = probe_set.select(iso_idx)      # 4,000
```

---

## SEKCJA 2 — FINE-TUNING

### 2.1 Metoda

**Primary (≥24GB VRAM):** LoRA FP16
**Fallback (16GB VRAM):** QLoRA 4-bit NF4 + obowiązkowa ablacja KL (patrz Sekcja 8.4)

**Uzasadnienie LoRA FP16:** restricted_entropy i logit_gap są liczone bezpośrednio z logitów modelu. 4-bit NF4 kwantyzacja kompresuje dynamiczny zakres wag co może systematycznie zniekształcać logity — corrupting uncertainty signals. FP16 eliminuje ten confound.

### 2.2 Config

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,              # 2×r — standard scaling
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,      # effective batch = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    optim="adamw_torch",        # A100: szybszy niż paged_adamw (nie potrzebujemy page offload)
    dataloader_num_workers=4,
    max_grad_norm=1.0,
    report_to="wandb",                  # opcjonalnie
    seed=42,
)

# Loss tylko na tokenie odpowiedzi
collator = DataCollatorForCompletionOnlyLM(
    response_template="Answer:",
    tokenizer=tokenizer
)
```

**QLoRA fallback:**
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto"
)
# Reszta identyczna z LoRA config powyżej
# PLUS: uruchom ablację KL (Sekcja 8.4) po fine-tuningu
```

### 2.3 Formatowanie sekwencji treningowych

```python
def format_train_example(example):
    opts, label = normalize_example(example, "medmcqa")
    text = format_prompt(example["question"], opts,
                         include_answer=True, answer=label)
    return {"text": text}

train_dataset = train_ft.map(format_train_example, remove_columns=train_ft.column_names)
```

**Max sequence length:** 512 tokenów (MedMCQA pytania ~13 tokenów avg, opcje ~50 łącznie — 512 z dużym zapasem)

---

## SEKCJA 3 — FEATURE EXTRACTION

Feature extraction uruchamiana na: `probe_set` (16K), `val` (4183), `test_medmcqa` (6150), `test_medqa` (1273), `test_mmlu` (945).

### 3.1 Weryfikacja tokenów A/B/C/D (BLOCKER 2 — patrz Roadmap)

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# Po "Answer: " model generuje token z poprzedzającą spacją ('▁A' w SentencePiece)
for letter in ['A', 'B', 'C', 'D']:
    with_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
    no_space   = tokenizer.encode(letter, add_special_tokens=False)
    print(f"' {letter}' → token IDs: {with_space}")
    print(f"'{letter}'  → token IDs: {no_space}")

# OCZEKIWANE: każda litera = dokładnie 1 token
# Jeśli multi-token → restricted entropy wymaga innego podejścia
ids_ABCD = [tokenizer.encode(f" {l}", add_special_tokens=False)[0]
            for l in ['A', 'B', 'C', 'D']]
print(f"ids_ABCD = {ids_ABCD}")
```

### 3.2 Layer sweep (uruchom na 500 próbkach przed pełną ekstrakcją)

```python
num_layers = model.config.num_hidden_layers  # 32 dla Mistral-7B

sweep_layers = [
    num_layers // 4,        # L=8
    num_layers // 2,        # L=16
    3 * num_layers // 4,    # L=24
    num_layers - 2,         # L=30
    num_layers - 1          # L=31
]

# Dla każdej warstwy: fit LogisticRegression na 400 próbkach,
# eval AUROC na 100 próbkach → wybierz best_layer
# Raportuj wyniki jako ablation table w papierze
```

### 3.3 Pełna ekstrakcja

```python
import torch
import torch.nn.functional as F
import numpy as np

def extract_features(model, tokenizer, example, source, ids_ABCD, best_layer):
    opts, true_label = normalize_example(example, source)

    # Prompt BEZ tokenu odpowiedzi
    prompt = format_prompt(example["question"], opts)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    last_input_pos = input_ids.shape[1] - 1  # TBG: ostatni token inputu

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    logits_last = outputs.logits[0, -1, :]   # [vocab_size]

    # --- Sygnał 1: Restricted Entropy (H) ---
    logits_ABCD = logits_last[ids_ABCD]       # [4]
    p = F.softmax(logits_ABCD, dim=-1)        # [4], sums to 1
    H = -(p * torch.log(p + 1e-10)).sum().item()   # ∈ [0, log4 ≈ 1.386]

    # --- Sygnał 2: Logit Gap ---
    sorted_p = p.sort(descending=True).values
    gap = (sorted_p[0] - sorted_p[1]).item()  # ∈ [0, 1]

    # --- Predicted answer + correctness label ---
    pred_idx    = p.argmax().item()
    pred_letter = "ABCD"[pred_idx]
    y = int(pred_letter == true_label)

    # --- Sygnał 3: Hidden State (dla probe) ---
    h = outputs.hidden_states[best_layer][0, last_input_pos, :]  # [4096]
    h = h.cpu().float().numpy()

    # --- Sygnał 4: p(True) ---
    p_true = extract_p_true(model, tokenizer, prompt, pred_letter)

    return {
        "H": H,
        "gap": gap,
        "h": h,
        "p_true": p_true,
        "y": y,
        "pred": pred_letter,
        "true": true_label,
        "subject": example.get("subject_name", "unknown"),
    }
```

### 3.4 p(True) ekstrakcja (BLOCKER 3 — zweryfikuj przed pełną ekstrakcją)

```python
# Ustaw IDs raz globalnie
yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[-1]
no_id  = tokenizer.encode(" No",  add_special_tokens=False)[-1]

def extract_p_true(model, tokenizer, original_prompt, pred_answer):
    """
    Pyta model: "Is my answer correct? (Yes/No)"
    Zwraca P(Yes) / (P(Yes) + P(No)) ∈ [0, 1]
    """
    p_true_prompt = (
        original_prompt +
        f" {pred_answer}\n"
        f"Is this answer correct? (Yes/No):"
    )
    input_ids = tokenizer.encode(
        p_true_prompt, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]  # [vocab_size]

    p_yn = F.softmax(logits[[yes_id, no_id]], dim=-1)
    return p_yn[0].item()   # P(Yes)

# WERYFIKACJA przed pełną ekstrakcją (30 przykładów):
# Sprawdź: mean(p_yes + p_no) — jeśli < 0.3 → p(True) jest bezużyteczne → wyklucz
# Jeśli ≥ 0.3 → sygnał działa, kontynuuj
```

**Przechowywanie:** `~28,500 próbek × 4096 dim × 4 bytes = ~467MB` — bez problemu w RAM.

---

## SEKCJA 4 — CORRECTNESS PROBE (5-fold cross-fitting)

```python
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 03_extract.py zapisuje dwa pliki na split:
#   data/features/probe_features.npz  — skalary: H, gap, p_true, y, pred, true, subject
#   data/features/probe_hidden.npy    — hidden states [N, 4096], float32  (osobny plik — szybszy load)
# NIE ma klucza "h" w .npz — hidden states są WYŁĄCZNIE w _hidden.npy

FEATURES_DIR = Path("data/features")
SPLITS_DIR   = Path("data/splits")

d        = np.load(FEATURES_DIR / "probe_features.npz", allow_pickle=False)
H_hidden = np.load(FEATURES_DIR / "probe_hidden.npy")   # [13307, 4096], float32
y_probe  = d["y"].astype(np.int32)                       # [13307]
# Dostępne też: d["H"], d["gap"], d["p_true"], d["pred"], d["true"], d["subject"]

n_probe = len(y_probe)  # 13307 po dedup
probe_scores_oof = np.zeros(n_probe)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(H_hidden, y_probe)):
    probe = LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
        cv=3,
        penalty="l2",
        scoring="roc_auc",
        max_iter=1000,
        n_jobs=-1
    )
    probe.fit(H_hidden[train_idx], y_probe[train_idx])
    probe_scores_oof[val_idx] = probe.predict_proba(
        H_hidden[val_idx]
    )[:, 1]

    auroc = roc_auc_score(y_probe[val_idx], probe_scores_oof[val_idx])
    print(f"Fold {fold+1}/5: AUROC = {auroc:.4f}")

print(f"Mean OOF AUROC = {roc_auc_score(y_probe, probe_scores_oof):.4f}")

# Final probe na pełnych 13,307 (dla ekstrakcji probe_scores na val/test)
final_probe = LogisticRegressionCV(
    Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
    cv=3, penalty="l2", scoring="roc_auc", max_iter=1000
)
final_probe.fit(H_hidden, y_probe)
```

**Ablacja probe architecture (raportuj jako Table):**
```python
# MLP probe: 4096 → 64 → 1 z dropout=0.2
# Jeśli AUROC_MLP - AUROC_LR < 0.02 → używaj LR (interpretowalny)
# Jeśli różnica > 0.02 → używaj MLP i uzasadnij w papierze
```

---

## SEKCJA 5 — ANALIZA KORELACJI SYGNAŁÓW

```python
import pandas as pd

# d i probe_scores_oof z Sekcji 4 (probe_features.npz + 5-fold OOF)
features_df = pd.DataFrame({
    "H":      d["H"].astype(float),
    "gap":    d["gap"].astype(float),
    "probe":  probe_scores_oof,
    "p_true": d["p_true"].astype(float),
})

corr_matrix = features_df.corr(method="pearson")
print("\nPearson correlation matrix:")
print(corr_matrix.round(3))

# DECYZJA (automatyczna):
r_H_gap = abs(corr_matrix.loc["H", "gap"])
if r_H_gap > 0.85:
    print(f"\nr(H, gap) = {r_H_gap:.3f} > 0.85 → DROP gap")
    routing_feature_names = ["H", "probe", "p_true"]
else:
    print(f"\nr(H, gap) = {r_H_gap:.3f} ≤ 0.85 → KEEP gap")
    routing_feature_names = ["H", "gap", "probe", "p_true"]

# Raportuj macierz jako Table w papierze (Section 4.x)
```

---

## SEKCJA 6 — ROUTING POLICY

### 6.1 Budowanie feature matrix

```python
def build_feature_matrix(indices, feature_names, features_list, probe_scores):
    cols = []
    for name in feature_names:
        if name == "probe":
            cols.append(probe_scores[indices])
        else:
            vals = np.array([features_list[i][name] for i in indices])
            cols.append(vals)
    return np.column_stack(cols)
```

### 6.2 Routing Logistic Regression (na 12K)

```python
X_routing = build_feature_matrix(
    routing_idx, routing_feature_names, probe_features, probe_scores_oof
)
y_routing = y_probe[routing_idx]

routing_lr = LogisticRegressionCV(
    Cs=[0.01, 0.1, 1.0, 10.0],
    cv=3, penalty="l2",
    scoring="roc_auc", max_iter=1000
)
routing_lr.fit(X_routing, y_routing)

# Wypisz nauczone wagi — wchodzą do papieru
print("\nRouting LR weights:")
for name, w in zip(routing_feature_names, routing_lr.coef_[0]):
    print(f"  {name:12s}: {w:+.4f}")
print(f"  intercept   : {routing_lr.intercept_[0]:+.4f}")
```

### 6.3 Kalibracja — ROC-isotonic regression (na 4K)

```python
from sklearn.isotonic import IsotonicRegression

X_iso = build_feature_matrix(
    iso_idx, routing_feature_names, probe_features, probe_scores_oof
)
y_iso = y_probe[iso_idx]

lr_proba_iso = routing_lr.predict_proba(X_iso)[:, 1]

# Primary: vanilla isotonic (sklearn, zawsze dostępny)
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(lr_proba_iso, y_iso)

# Ablacja: ROC-regularized isotonic (Dimitriadis et al. 2023)
# pip install regcal
try:
    from regcal import ROCIsotonicRegression
    calibrator_roc = ROCIsotonicRegression()
    calibrator_roc.fit(lr_proba_iso, y_iso)
except ImportError:
    print("regcal not available, using vanilla isotonic")
    calibrator_roc = calibrator

# Porównaj Brier score obu w ablacji (Table 2)


def get_routing_score(feature_matrix, routing_lr, calibrator):
    """Zwraca P(model is correct | features) ∈ [0, 1]"""
    lr_proba = routing_lr.predict_proba(feature_matrix)[:, 1]
    return calibrator.transform(lr_proba)
```

---

## SEKCJA 7 — CONFORMAL CALIBRATION

### 7.1 Ekstrakcja scores na val set (4183 próbki)

```python
# val_features = extract_all_features(val_set, "medmcqa", ...)
# Analogicznie jak probe_features, używając final_probe

X_val = build_feature_matrix(
    list(range(len(val_features))),
    routing_feature_names,
    val_features,
    final_probe.predict_proba(
        np.stack([ex["h"] for ex in val_features])
    )[:, 1]
)

routing_scores_val = get_routing_score(X_val, routing_lr, calibrator)
y_val = np.array([ex["y"] for ex in val_features])
```

### 7.2 Threshold q_hat

```python
def compute_conformal_threshold(routing_scores, alpha):
    """
    Split CP: q_hat = ceiling quantile of non-conformity scores
    Non-conformity score: s(x) = 1 - routing_score(x)
    Gwarancja: P(routing_score(X_test) >= 1 - q_hat) >= 1 - alpha
    """
    n = len(routing_scores)
    s_cal = 1.0 - routing_scores
    q_hat = np.quantile(
        s_cal,
        np.ceil((1 - alpha) * (n + 1)) / n,
        method="higher"     # wymagane dla formalnej gwarancji
    )
    return q_hat

# Primary alpha = 0.10 (cel: 90% coverage)
alphas = [0.05, 0.10, 0.15, 0.20]
thresholds = {a: compute_conformal_threshold(routing_scores_val, a) for a in alphas}

alpha_primary = 0.10
q_hat = thresholds[alpha_primary]
print(f"q_hat @ alpha={alpha_primary}: {q_hat:.4f}")

# Bootstrap stability
def bootstrap_q_hat(routing_scores, alpha, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    boot_q = [
        compute_conformal_threshold(
            rng.choice(routing_scores, size=len(routing_scores), replace=True),
            alpha
        )
        for _ in range(n_boot)
    ]
    return np.percentile(boot_q, [2.5, 97.5])

ci = bootstrap_q_hat(routing_scores_val, alpha_primary)
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
# Oczekiwane: ±0.0046 (±0.46pp)
```

### 7.3 Decision rule

```python
def route(routing_score, q_hat):
    """
    Returns: 'local' jeśli model odpowiada sam
             'escalate' jeśli przekazuje do eksperta
    """
    return "local" if routing_score >= (1.0 - q_hat) else "escalate"
```

### 7.4 Mondrian CP (ablacja — 5 clinical domains)

```python
DOMAIN_MAP = {
    # Clinical
    "Surgery": "clinical", "Medicine": "clinical",
    "Obstetrics & Gynecology": "clinical", "Pediatrics": "clinical",
    "Anesthesia": "clinical", "Radiology": "clinical",
    "Ophthalmology": "clinical", "ENT": "clinical",
    "Orthopedics": "clinical", "Dental": "clinical",
    # Basic science
    "Pathology": "basic_science", "Microbiology": "basic_science",
    "Forensic Medicine": "basic_science",
    # Pharmacology
    "Pharmacology": "pharmacology", "Biochemistry": "pharmacology",
    # Anatomy/Physiology
    "Anatomy": "anatomy_physiology", "Physiology": "anatomy_physiology",
    # Other
    "Psychiatry": "other", "Skin": "other",
    "Preventive & Social Medicine": "other", "General Medicine": "other",
}

mondrian_thresholds = {}
for domain in set(DOMAIN_MAP.values()):
    mask = np.array([
        DOMAIN_MAP.get(ex["subject"], "other") == domain
        for ex in val_features
    ])
    n_domain = mask.sum()
    if n_domain < 50:
        print(f"WARNING: domain '{domain}' has only {n_domain} cal samples")
    scores_domain = routing_scores_val[mask]
    mondrian_thresholds[domain] = compute_conformal_threshold(
        scores_domain, alpha_primary
    )
    print(f"{domain}: n={n_domain}, q_hat={mondrian_thresholds[domain]:.4f}")
```

---

## SEKCJA 8 — EWALUACJA

### 8.1 Implementacja metryk

```python
from sklearn.metrics import roc_auc_score, brier_score_loss
import numpy as np


def compute_augrc(routing_scores, y_true):
    """
    AUGRC: Area Under Generalized Risk-Coverage curve (NeurIPS 2024, arXiv:2407.01032)
    Mierzy expected risk of undetected failures across all coverage thresholds.
    Lower is better. Range: [0, 0.5]
    """
    thresholds = np.sort(np.unique(routing_scores))[::-1]
    risks, coverages = [], []
    for t in thresholds:
        mask = routing_scores >= t
        if mask.sum() == 0:
            continue
        coverage = mask.mean()
        risk = (1 - y_true[mask]).mean()
        risks.append(risk)
        coverages.append(coverage)
    if len(coverages) < 2:
        return float("nan")
    return float(np.trapz(risks[::-1], coverages[::-1]))


def compute_aurc(routing_scores, y_true):
    """
    AURC: Area Under Risk-Coverage curve (secondary metric)
    """
    order = np.argsort(routing_scores)[::-1]
    y_sorted = y_true[order]
    n = len(y_true)
    risks, coverages = [], []
    for k in range(1, n + 1):
        risks.append(1 - y_sorted[:k].mean())
        coverages.append(k / n)
    return float(np.trapz(risks, coverages))


def compute_ece(probs, labels, n_bins=15, strategy="quantile"):
    """ECE z equal-mass bins (quantile strategy)"""
    if strategy == "quantile":
        bin_edges = np.quantile(probs, np.linspace(0, 1, n_bins + 1))
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def evaluate_routing(routing_scores, y_true, q_hat, alpha, subjects=None):
    decisions = routing_scores >= (1.0 - q_hat)  # True = answer locally
    n = len(y_true)
    n_local = decisions.sum()

    y_local = y_true[decisions]

    results = {
        # Discrimination
        "AUROC":  roc_auc_score(y_true, routing_scores),
        # Selective prediction
        "AUGRC":  compute_augrc(routing_scores, y_true),
        "AURC":   compute_aurc(routing_scores, y_true),
        # Calibration
        "Brier":  brier_score_loss(y_true, routing_scores),
        "ECE":    compute_ece(routing_scores, y_true, n_bins=15),
        # Coverage and efficiency
        "empirical_coverage": float(y_local.mean()) if n_local > 0 else 0.0,
        "escalation_rate":    float(1.0 - n_local / n),
        "n_local":            int(n_local),
        # Accuracy
        "standalone_acc":     float(y_true.mean()),
        "system_acc_oracle":  float(
            (y_true[decisions].sum() + (~decisions).sum()) / n
        ),  # oracle: escalated always correct — upper bound
    }
    return results


def bootstrap_ci(metric_fn, routing_scores, y_true, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            boot_vals.append(metric_fn(routing_scores[idx], y_true[idx]))
        except Exception:
            continue
    return np.percentile(boot_vals, [2.5, 97.5])
```

### 8.2 Trzy-tierowa ewaluacja

```python
eval_configs = [
    ("in-dist",  test_medmcqa_features, "medmcqa", q_hat, True),   # CP valid
    ("near-OOD", test_medqa_features,   "medqa",   q_hat, False),  # CP not guaranteed
    ("far-OOD",  test_mmlu_features,    "mmlu",    q_hat, False),  # CP not guaranteed
]

all_results = {}
for name, features, source, threshold, cp_valid in eval_configs:
    # Ekstrakcja probe scores dla tego datasetu
    H_test = np.stack([ex["h"] for ex in features])
    probe_scores_test = final_probe.predict_proba(H_test)[:, 1]

    X_test = build_feature_matrix(
        list(range(len(features))),
        routing_feature_names,
        features,
        probe_scores_test
    )

    scores = get_routing_score(X_test, routing_lr, calibrator)
    y_test = np.array([ex["y"] for ex in features])

    res = evaluate_routing(scores, y_test, threshold, alpha_primary)

    # Bootstrap CIs dla kluczowych metryk
    for metric, fn in [
        ("AUROC", lambda s, y: roc_auc_score(y, s)),
        ("AUGRC", compute_augrc),
        ("Brier", lambda s, y: brier_score_loss(y, s)),
    ]:
        ci = bootstrap_ci(fn, scores, y_test)
        res[f"{metric}_CI_low"]  = ci[0]
        res[f"{metric}_CI_high"] = ci[1]

    res["cp_guarantee_valid"] = cp_valid
    all_results[name] = res

# Drukuj tabelę wyników
print("\n{'='*70}")
print(f"{'Metric':<20} {'in-dist':>12} {'near-OOD':>12} {'far-OOD':>12}")
print("="*70)
for metric in ["standalone_acc", "AUROC", "AUGRC", "AURC",
               "Brier", "ECE", "empirical_coverage", "escalation_rate",
               "system_acc_oracle"]:
    vals = [all_results[k].get(metric, float("nan")) for k in all_results]
    print(f"{metric:<20}" + "".join(f"{v:>12.4f}" for v in vals))
```

### 8.3 Cost model dla wyboru alpha (sekcja Discussion)

```python
# Uzasadnienie dlaczego alpha = 0.10 nie jest arbitralne
def compute_utility(routing_scores, y_true, q_hat, alpha, c=0.3):
    """
    U(alpha) = Acc_local * (1 - Esc_rate) - c * Esc_rate
    c = relatywny koszt eskalacji (latency, privacy, API cost)
    """
    decisions = routing_scores >= (1.0 - q_hat)
    acc_local = y_true[decisions].mean() if decisions.sum() > 0 else 0.0
    esc_rate  = 1.0 - decisions.mean()
    return acc_local * (1 - esc_rate) - c * esc_rate

# Pokaż utility dla różnych alpha i różnych c
import matplotlib.pyplot as plt

for c in [0.1, 0.3, 0.5]:
    utilities = []
    for a in alphas:
        qh = thresholds[a]
        u = compute_utility(routing_scores_val, y_val, qh, a, c=c)
        utilities.append(u)
    plt.plot(alphas, utilities, label=f"c={c}")

plt.xlabel("alpha (escalation threshold)")
plt.ylabel("Utility")
plt.legend()
plt.title("Utility vs alpha for different escalation costs")
plt.savefig("figures/utility_alpha.pdf")
```

---

## SEKCJA 9 — ABLACJE

### 9.1 Signal ablation (Table 1 w papierze)

```python
signal_subsets = {
    "H only":             ["H"],
    "probe only":         ["probe"],
    "p_true only":        ["p_true"],
    "H + probe":          ["H", "probe"],
    "H + probe + p_true": ["H", "probe", "p_true"],
    "all signals":        routing_feature_names,
}

signal_ablation = {}
for name, feature_subset in signal_subsets.items():
    X_sub_train = build_feature_matrix(
        routing_idx, feature_subset, probe_features, probe_scores_oof
    )
    lr_sub = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0], cv=3,
        penalty="l2", scoring="roc_auc"
    )
    lr_sub.fit(X_sub_train, y_routing)

    X_sub_val = build_feature_matrix(
        list(range(len(val_features))), feature_subset,
        val_features,
        final_probe.predict_proba(np.stack([ex["h"] for ex in val_features]))[:, 1]
    )
    scores_sub = lr_sub.predict_proba(X_sub_val)[:, 1]
    signal_ablation[name] = {
        "AUROC": roc_auc_score(y_val, scores_sub),
        "AUGRC": compute_augrc(scores_sub, y_val),
        "Brier": brier_score_loss(y_val, scores_sub),
    }
    print(f"{name:<25}: AUROC={signal_ablation[name]['AUROC']:.4f}, "
          f"AUGRC={signal_ablation[name]['AUGRC']:.4f}")
```

### 9.2 Calibration ablation (Table 2)

```python
# Porównaj: uncalibrated LR | Platt | isotonic vanilla | ROC-isotonic
from sklearn.calibration import CalibratedClassifierCV

calibration_ablation = {}
for calib_name, scoring_fn in [
    ("uncalibrated", lambda: routing_lr.predict_proba(X_iso)[:, 1]),
    ("platt",  None),
    ("isotonic_vanilla", None),
    ("roc_isotonic", None),
]:
    ...  # compute Brier i ECE na val (12K/4K split)
    # Raportuj w Table 2
```

### 9.3 CP variant ablation (Table 3)

```python
cp_ablation = {
    "split_CP_global":  {},  # jeden q_hat dla wszystkich
    "mondrian_CP_5dom": {},  # oddzielne q_hat per domain
}
# Raportuj: empirical coverage, AUGRC, escalation_rate
# Dla Mondrian: coverage PER DOMAIN
```

### 9.4 QLoRA KL ablation (tylko jeśli używany QLoRA fallback)

```python
# Na 200 losowych próbkach z val setu
kl_divs = []
for ex_idx in sample_200_idx:
    ex = val_features[ex_idx]
    p_fp16 = ...   # restricted probs z modelu FP16
    p_4bit = ...   # restricted probs z modelu 4-bit
    # KL(4-bit || FP16)
    kl = F.kl_div(
        torch.log(p_4bit + 1e-10),
        p_fp16,
        reduction="sum"
    ).item()
    kl_divs.append(kl)

mean_kl = np.mean(kl_divs)
print(f"Mean KL(4-bit || FP16) = {mean_kl:.4f} nats")
# Jeśli < 0.05: kwantyzacja nie jest confoundem
# Jeśli > 0.10: wymaga dyskusji w Limitations
```

### 9.5 Layer sweep ablation (Table 4)

```python
layer_auroc = {}
for layer_idx in sweep_layers:
    H_layer = np.stack([
        extract_hidden_at_layer(model, ex, layer_idx)
        for ex in probe_features[:500]
    ])
    y_500 = y_probe[:500]
    lr_layer = LogisticRegression(C=1.0, penalty="l2", max_iter=500)
    # 5-fold na 500 próbkach
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(lr_layer, H_layer, y_500,
                             cv=5, scoring="roc_auc")
    layer_auroc[layer_idx] = scores.mean()
    print(f"Layer {layer_idx}: AUROC = {scores.mean():.4f} ± {scores.std():.4f}")

best_layer = max(layer_auroc, key=layer_auroc.get)
print(f"\nBest layer: {best_layer}")
```

---

## SEKCJA 10 — OCZEKIWANE WYNIKI (reference punkty)

| Metryka | In-dist (MedMCQA) | Near-OOD (MedQA) | Far-OOD (MMLU) |
|---|---|---|---|
| Standalone accuracy | ~65-70% | ~55-65% | ~60-70% |
| AUROC routing | ~0.70-0.80 | ~0.65-0.75 | ~0.60-0.70 |
| Empirical coverage @ α=0.10 | ~0.90 (guaranteed) | ~0.85-0.90 | ~0.80-0.88 |
| Escalation rate @ α=0.10 | ~15-25% | — | — |
| AUGRC | niskie (main result) | wyższe | najwyższe |

**Kluczowe odkrycie:** degradacja empirical coverage pod distribution shift (in-dist → near-OOD → far-OOD) bez rekalibracji. Żaden paper nie mierzy tego systematycznie dla medical routing.

---

## SEKCJA 11 — WYMAGANIA SPRZĘTOWE I CZASOWE

| Faza | GPU | Czas (est.) |
|---|---|---|
| Weryfikacje (3 blokery) | CPU / dowolne | ~2 godziny |
| Fine-tuning FP16 | A100 40GB | ~1-2 dni |
| Fine-tuning FP16 | RTX 3090 | ~5-7 dni |
| Fine-tuning QLoRA 4-bit | RTX 3090 | ~2-3 dni |
| Feature extraction (28K próbek) | RTX 3090 | ~4-8 godzin |
| Probe + routing (sklearn) | CPU | ~1-2 godziny |
| Conformal + ewaluacja | CPU | ~30 minut |
| Ablacje | CPU + GPU | ~1 dzień |

**Rekomendacja:** Google Colab Pro+ (A100 40GB) dla fine-tuningu, reszta na CPU/mniejszym GPU.

---

## CZĘŚĆ II — ROADMAP IMPLEMENTACJI

---

## KROK 0 — SETUP ŚRODOWISKA ✅ DONE

```bash
pip install -e ".[dev]"    # instaluje pakiet thesis + wszystkie zależności
# lub: make setup
```

**Struktura projektu (finalna, zgodna z best practices):**
```
diploma-thesis/
├── CLAUDE.md            # instrukcje dla Claude Code (każda sesja)
├── roadmap.md           # ten plik — żywy dokument, aktualizuj na bieżąco
├── pyproject.toml       # definicja pakietu + zależności
├── Makefile             # make setup / verify / finetune / evaluate itd.
├── .gitignore
│
├── configs/             # YAML configs (OmegaConf)
│   ├── main.yaml
│   ├── model/
│   │   ├── lora_fp16.yaml
│   │   └── qlora_4bit.yaml
│   └── data/
│       └── datasets.yaml
│
├── scripts/             # entry points — cienkie wrappery, logika w src/thesis/
│   ├── 00_verify.py     # KROK 1: 3 krytyczne weryfikacje
│   ├── 01_prepare.py    # KROK 2: data preparation
│   ├── 02_finetune.py   # KROK 3: fine-tuning
│   ├── 03_extract.py    # KROK 4: feature extraction + layer sweep
│   ├── 04_probe.py      # KROK 5: probe training (5-fold CV)
│   ├── 05_routing.py    # KROK 6: routing LR + isotonic calibration
│   ├── 06_conformal.py  # KROK 7: conformal calibration (q_hat)
│   ├── 07_evaluate.py   # KROK 8: 3-tier evaluation
│   └── 08_ablations.py  # KROK 9: ablation tables
│
├── src/thesis/          # instalowany pakiet — cała logika tutaj
│   ├── data/            # dataset loading, preprocessing, normalize_example()
│   ├── models/          # model wrappers, extract_features()
│   ├── routing/         # probe.py, routing_lr.py, calibration.py, conformal.py
│   └── utils/           # metrics.py (AUGRC/AURC/ECE/bootstrap), plotting.py
│
├── data/
│   ├── raw/             # oryginalne datasety — NIE commitować
│   ├── processed/       # NIE commitować
│   ├── splits/          # JSON z indeksami train/probe/val (commitować — małe)
│   └── features/        # .npz feature cache — NIE commitować (~500MB)
│
├── checkpoints/         # LoRA adaptery — NIE commitować
├── results/             # JSON z wynikami ewaluacji — commitować
├── figures/             # wykresy do tezy — commitować
└── notebooks/           # EDA only
```

---

## KROK 1 — TRZY KRYTYCZNE WERYFIKACJE ⚠️

**Muszą być zrobione PRZED jakimkolwiek innym krokiem.**

### BLOCKER 1: cop encoding w MedMCQA

```python
# src/00_verify.py — uruchom jako pierwsze

import datasets

print("=" * 60)
print("BLOCKER 1: Weryfikacja cop encoding w MedMCQA")
print("=" * 60)

ds = datasets.load_dataset("openlifescienceai/medmcqa", split="train")
sample = ds.shuffle(seed=0).select(range(20))

for i, ex in enumerate(sample):
    cop = ex["cop"]
    options = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
    question_preview = ex["question"][:60]

    print(f"\n[{i+1}] Question: {question_preview}")
    print(f"     cop={cop}")
    print(f"     options[cop]  = '{options[cop]}'")
    print(f"     Ręcznie sprawdź czy to POPRAWNA odpowiedź")

# Po sprawdzeniu 20 przykładów:
# Jeśli cop=0 → opcja A: plik src/config.py: COP_OFFSET = 0
# Jeśli cop=1 → opcja A: plik src/config.py: COP_OFFSET = 1
```

### BLOCKER 2: Tokenizacja A/B/C/D w Mistral

```python
print("\n" + "=" * 60)
print("BLOCKER 2: Weryfikacja tokenizacji A/B/C/D")
print("=" * 60)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

ids_ABCD = []
for letter in ['A', 'B', 'C', 'D']:
    with_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
    no_space   = tokenizer.encode(letter, add_special_tokens=False)

    print(f"\n' {letter}' → token IDs: {with_space} "
          f"(n_tokens={len(with_space)})")
    print(f"'{letter}'  → token IDs: {no_space} "
          f"(n_tokens={len(no_space)})")

    if len(with_space) == 1:
        ids_ABCD.append(with_space[0])
        print(f"  ✓ Single token, ID={with_space[0]}")
    else:
        print(f"  ⚠ MULTI-TOKEN — restricted entropy wymaga modyfikacji!")

print(f"\nids_ABCD = {ids_ABCD}")
assert len(ids_ABCD) == 4, "Nie wszystkie litery tokenizują jako single token!"
print("✓ Wszystkie litery są single-token. ids_ABCD gotowe.")

# Zapisz do config
# IDS_ABCD = ids_ABCD
```

### BLOCKER 3: p(True) na fine-tuned modelu

```python
print("\n" + "=" * 60)
print("BLOCKER 3: Weryfikacja p(True) na fine-tuned modelu")
print("=" * 60)

# URUCHOM PO FINE-TUNINGU (nie przed)
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./checkpoints/final")
model.eval()

yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[-1]
no_id  = tokenizer.encode(" No",  add_special_tokens=False)[-1]

val_ds = datasets.load_dataset("openlifescienceai/medmcqa", split="validation")
sample_30 = val_ds.shuffle(seed=1).select(range(30))

yes_no_masses = []
for ex in sample_30:
    opts, label = normalize_example(ex, "medmcqa")
    prompt = format_prompt(ex["question"], opts)

    # Wygeneruj predykcję
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, ids_ABCD]
    pred = "ABCD"[logits.argmax().item()]

    # p(True) prompt
    p_true_prompt = prompt + f" {pred}\nIs this answer correct? (Yes/No):"
    pt_ids = tokenizer.encode(p_true_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        pt_logits = model(pt_ids).logits[0, -1, :]

    p_yn = F.softmax(pt_logits[[yes_id, no_id]], dim=-1)
    mass = p_yn.sum().item()
    yes_no_masses.append(mass)
    print(f"p(Yes)={p_yn[0]:.3f}, p(No)={p_yn[1]:.3f}, "
          f"mass={mass:.3f}, pred={pred}, true={label}")

mean_mass = sum(yes_no_masses) / len(yes_no_masses)
print(f"\nMean Yes/No mass = {mean_mass:.3f}")
if mean_mass >= 0.3:
    print("✓ p(True) działa — włącz do pipeline'u")
else:
    print("⚠ p(True) zdegradowane — wyklucz z routing features")
    # routing_feature_names = [x for x in routing_feature_names if x != "p_true"]
```

---

## KROK 2 — DATA PREPARATION

**Po weryfikacjach.** Deterministyczny, seed=42 wszędzie.

```python
# src/01_prepare.py

# 1. Wczytaj i filtruj
# 2. Stratified split → train_ft (128K) + probe_set (16K)
# 3. Zapisz indeksy do data/splits/ (JSON) — reproducibility
# 4. Weryfikacja: losowa próbka 100 z probe_set, ręczne sprawdzenie choice_type
# 5. Wczytaj MedQA test i MMLU (5 kategorii)
# 6. Aplikuj unified prompt template do wszystkich
```

---

## KROK 3 — FINE-TUNING

```python
# src/02_finetune.py

# 1. Wczytaj Mistral-7B-Instruct-v0.3 (FP16 lub QLoRA)
# 2. Aplikuj LoRA config (r=16, all attention+MLP)
# 3. Trenuj na train_ft (~128K), 3 epochs
# 4. Zapisz best checkpoint (val loss lub val accuracy)
# 5. Jeśli QLoRA: uruchom ablację KL (Sekcja 9.4) po treningu
```

---

## KROK 4 — FEATURE EXTRACTION

```python
# src/03_extract.py

# PRZED pełną ekstrakcją:
# 1. Layer sweep na 500 próbkach → wybierz best_layer
# 2. Wyniki sweep zapisz do results/layer_sweep.json

# Pełna ekstrakcja (wszystkie datasety):
# 3. probe_set (16K) → probe_features.npz
# 4. val (4183)      → val_features.npz
# 5. test_medmcqa (6150) → test_medmcqa_features.npz
# 6. test_medqa (1273)   → test_medqa_features.npz
# 7. test_mmlu (945)     → test_mmlu_features.npz

# Każdy .npz zawiera: H, gap, h [4096], p_true, y, pred, true, subject
# Całkowity rozmiar: ~500MB
```

---

## KROK 5 — PROBE TRAINING

```python
# src/04_probe.py

# 1. Wczytaj probe_features.npz
# 2. 5-fold cross-fitting (StratifiedKFold, seed=42)
#    → probe_scores_oof [13307]
# 3. Final probe na pełnych 13,307
#    → zapisz jako checkpoints/final_probe.pkl
# 4. Ablacja: MLP probe vs LR → raportuj AUROC obu
# 5. Analiza korelacji sygnałów → decyzja o gap
#    → zapisz routing_feature_names do config
```

---

## KROK 6 — ROUTING POLICY

```python
# src/05_routing.py

# WAŻNE: routing_train_idx.json i iso_cal_idx.json zawierają GLOBALNE indeksy
# (do train_single z MedMCQA), ale probe_features.npz jest indeksowane LOKALNIE
# (wiersz 0 = pierwszy element probe_idx, wiersz 1 = drugi, itd.).
# Wymagane mapowanie global → local przed użyciem:
#
#   probe_idx         = json.loads((splits_dir / "probe_idx.json").read_text())
#   routing_global    = json.loads((splits_dir / "routing_train_idx.json").read_text())
#   iso_global        = json.loads((splits_dir / "iso_cal_idx.json").read_text())
#   g2l = {g: i for i, g in enumerate(probe_idx)}
#   routing_local = [g2l[g] for g in routing_global]   # indeksy do probe_features.npz
#   iso_local     = [g2l[g] for g in iso_global]

# 1. Wczytaj probe_features.npz + probe_scores_oof z 04_probe.py
# 2. Zamień global → local (patrz wyżej); sprawdź n_routing=9307, n_iso=4000
# 3. Trenuj routing LR na routing_local
# 4. Kalibruj: isotonic regression na iso_local
#    → zapisz routing_lr.pkl + calibrator.pkl
# 5. Wypisz nauczone wagi → do papieru
# 6. Ablacja kalibracji: uncalibrated | Platt | isotonic | ROC-isotonic
```

---

## KROK 7 — CONFORMAL CALIBRATION

```python
# src/06_conformal.py

# 1. Wczytaj val_features.npz
# 2. Compute routing scores na val (4183)
# 3. Compute q_hat dla alpha ∈ {0.05, 0.10, 0.15, 0.20}
# 4. Bootstrap stability CI dla każdego q_hat
# 5. Mondrian CP: compute q_hat per domain (5 domains)
# 6. Zapisz thresholds do results/thresholds.json
```

---

## KROK 8 — EWALUACJA

```python
# src/07_evaluate.py

# Dla każdego z 3 test setów:
# 1. Compute routing scores
# 2. Apply decision rule (route @ q_hat)
# 3. Compute wszystkie metryki:
#    AUROC, AUGRC, AURC, Brier, ECE, empirical_coverage,
#    escalation_rate, standalone_acc, system_acc_oracle
# 4. Bootstrap 95% CI dla AUROC, AUGRC, Brier (1000 bootstraps)
# 5. Plot: coverage-risk curve, reliability diagram, utility curve
# 6. Zapisz do results/evaluation.json
```

---

## KROK 9 — ABLACJE

```python
# src/08_ablations.py

# Table 1: Signal ablation (6 subsets)
# Table 2: Calibration ablation (4 metody)
# Table 3: CP variant (split CP vs Mondrian CP)
# Table 4: Layer sweep results
# Table 5: OOD coverage degradation (główna tabela)
# (Opcjonalnie) Table 6: QLoRA KL ablation
```

---

## PODSUMOWANIE — KOLEJNOŚĆ WYKONANIA

```
[✅] KROK 0   Setup: struktura projektu, CLAUDE.md, pyproject.toml, Makefile, configs/
              scripts/00_verify.py gotowy

[✅] KROK 1a  BLOCKER 1: cop encoding — 0-indexed {0:'A',1:'B',2:'C',3:'D'}
             cop_map zapisany w scripts/config.json

[✅] KROK 1b  BLOCKER 2: tokenizacja A/B/C/D — wszystkie single token
             ids_ABCD = [1098, 1133, 1102, 1152]  (A, B, C, D)
             Uwaga: ' A' i 'A' dają ten sam ID — bezpieczne

[✅] KROK 2   Data preparation                  DONE
             → python scripts/01_prepare.py
             train_ft=104 765 (nienaruszony), probe=13 307, routing_train=9 307, iso_cal=4 000
             Deduplikacja: 2 693 exact duplicates usunięte z probe_set (znormalizowany string match)
             Uzasadnienie: TF-IDF cosine zbyt agresywny przy krótkich MCQ — łapie template similarity
             Uwaga: choice_type='single' = 66.1% (120 765/182 822) — poniżej oczekiwanych ~80%
             Uwaga: 2 693 exact duplicates między MedMCQA train_ft i probe (źródło: scraping tych samych stron)
             Splity zapisane w data/splits/ (seed=42, stratified by subject_name, dedup=exact_match)

[ ] KROK 3   Fine-tuning                       (1-2 dni, GPU PROFESORA ≥24GB)
             → python scripts/02_finetune.py
             LoRA (bf16), r=16, 3 epoki max, early stopping patience=3 @ eval_loss
             eval_steps=650, save_steps=650, bf16=True, adamw_torch
             Monitoring: W&B (student widzi postęp bez SSH)
             Output: checkpoints/final/ (~150MB LoRA adapter)

[ ] KROK 1c  BLOCKER 3: p(True) weryfikacja   (~1h, GPU PROFESORA, zaraz po fine-tuningu)
             → python scripts/00_verify.py --blocker 3
             ↓ jeśli mean Yes/No mass ≥ 0.3 → włącz p_true; jeśli nie → wyklucz z routing features

[ ] KROK 4   Feature extraction                (~4-8h, GPU PROFESORA ≥8GB)
             → python scripts/03_extract.py
             Ekstrahuje: H, gap, p_true, y, pred + hidden states [4096] dla:
               probe_set (13 307), val (4 183), test_medmcqa (6 150),
               test_medqa (1 273), test_mmlu (945)  — łącznie ~28 500 próbek
             Output: data/features/*.npz (~500MB łącznie)
             !! PROFESOR ODSYŁA: checkpoints/final/ + data/features/*.npz

[ ] KROK 5   Probe training (5-fold CV)        (~1-2h, CPU LOKALNIE)
             → python scripts/04_probe.py

[ ] KROK 6   Routing policy + calibration      (~30 min, CPU LOKALNIE)
             → python scripts/05_routing.py

[ ] KROK 7   Conformal calibration             (~10 min, CPU LOKALNIE)
             → python scripts/06_conformal.py

[ ] KROK 8   Evaluation (3 tiers)              (~30 min, CPU LOKALNIE)
             → make evaluate

[ ] KROK 9   Ablations                         (~kilka godzin, GPU PÓŹNIEJ — osobna sesja)
             → python scripts/08_ablations.py
             Wymaga GPU dla: layer sweep (03_extract na 5 warstwach × 500 próbek),
             QLoRA KL ablation (jeśli używano QLoRA)
             Reszta ablacji (signal subset, calibration, CP variant): CPU

[ ]          Pisanie diploma.md / thesis
```

---

## PODZIAŁ ZASOBÓW — CO GDZIE URUCHAMIAMY

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GPU PROFESORA (sesja 1, ~2-3 dni łącznie)                              │
│  Wymagania: ≥24 GB VRAM (KROK 3), ≥8 GB VRAM (KROK 4), SSH opcjonalny │
├─────────────────────────────────────────────────────────────────────────┤
│  KROK 3   Fine-tuning           1-2 dni   scripts/02_finetune.py        │
│  KROK 1c  BLOCKER 3 (p(True))  ~1h       scripts/00_verify.py --b 3    │
│  KROK 4   Feature extraction   4-8h      scripts/03_extract.py         │
│                                                                          │
│  Profesor odsyła:                                                        │
│    checkpoints/final/     (LoRA adapter, ~150MB)                        │
│    data/features/*.npz    (hidden states, ~500MB)                       │
│    wandb run link         (logi treningu)                                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  CPU LOKALNIE (po otrzymaniu plików od profesora, ~2-4h)                │
├─────────────────────────────────────────────────────────────────────────┤
│  KROK 5   Probe training        1-2h     scripts/04_probe.py            │
│  KROK 6   Routing + calibration ~30 min  scripts/05_routing.py          │
│  KROK 7   Conformal calibration ~10 min  scripts/06_conformal.py        │
│  KROK 8   Evaluation (3 tiers)  ~30 min  make evaluate                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  GPU PÓŹNIEJ (sesja 2, osobna — opcjonalna/na żądanie)                  │
├─────────────────────────────────────────────────────────────────────────┤
│  KROK 9   Layer sweep ablation  ~2-4h    scripts/08_ablations.py        │
│           QLoRA KL ablation     ~1h      (tylko jeśli używano QLoRA)    │
│           Signal/CP ablations   CPU      (reszta ablacji bez GPU)       │
└─────────────────────────────────────────────────────────────────────────┘
```

## EARLY STOPPING — FINALNA KONFIGURACJA (po deep research × 3 rundy)

```
Metryka:       eval_loss   (NIE accuracy — SFTTrainer GitHub #1222: EarlyStoppingCallback
                            nie może używać custom compute_metrics z SFTTrainer)
eval_steps:    650         (~10 ewaluacji per epoka; 104K/16=6.5K kroków/epoka)
save_steps:    650         (MUSI = eval_steps gdy load_best_model_at_end=True)
patience:      3            (eval EVENTS, nie epoki — przy eval_steps=650:
                            patience=3 = 1950 kroków ≈ 0.30 epoki; właściwa granularność)
greater_is_better: False   (minimalizujemy loss)
bf16:          True        (A100: bfloat16 stabilniejszy niż fp16; fp16=False jawnie)
fp16:          False       (jawnie wyłącz — unika konfliktów z bf16)
upper_bound:   3 epoki     (Raschka dot. ogólnego instruction-tuning; medical MCQ = 3+)

Uzasadnienie eval_loss zamiast accuracy:
  - Med42 (najbliższy precedens: medical LLM MCQ LoRA) → używa eval_loss
  - Badanie Feb 2026: accuracy-based ES wypadało GORZEJ niż loss-based na benchmarkach
  - Przy masked completion (loss tylko na tokenie A/B/C/D) eval_loss ≈ task objective
  - SFTTrainer automatycznie liczy mean_token_accuracy w logach (W&B) — widoczne diagnostycznie

Uwaga: mean_token_accuracy jest logowane automatycznie przez SFTTrainer (widoczne w W&B)
ale NIE może być użyte jako trigger dla EarlyStoppingCallback bez custom callback.
```

---

*Wersja pipeline: finalna. Zatwierdzona po 3-rundowym review metodologicznym.*
*Wszystkie decyzje są udokumentowane z uzasadnieniem powyżej.*

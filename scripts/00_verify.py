"""
KROK 1 — TRZY KRYTYCZNE WERYFIKACJE
====================================
Uruchom ten plik jako PIERWSZE przed jakimkolwiek innym krokiem.
Blocker 3 (p_true) uruchamiaj DOPIERO po fine-tuningu (KROK 3).

Użycie:
    python 00_verify.py --blocker 1    # cop encoding (uruchom teraz)
    python 00_verify.py --blocker 2    # tokenizacja ABCD (uruchom teraz)
    python 00_verify.py --blocker 3    # p(True) (uruchom po fine-tuningu)
    python 00_verify.py --blocker all  # wszystkie naraz (po fine-tuningu)
"""

import argparse
import json
import os
import sys

# ─────────────────────────────────────────────
# BLOCKER 1 — cop encoding w MedMCQA
# ─────────────────────────────────────────────

def verify_cop_encoding():
    print("\n" + "=" * 60)
    print("BLOCKER 1: Weryfikacja cop encoding w MedMCQA")
    print("=" * 60)
    print("Sprawdzamy czy cop=0 → opcja A (0-indexed)")
    print("czy cop=1 → opcja A (1-indexed)\n")

    try:
        import datasets
    except ImportError:
        print("ERROR: zainstaluj 'datasets': pip install datasets")
        sys.exit(1)

    ds = datasets.load_dataset("openlifescienceai/medmcqa", split="train")
    sample = ds.shuffle(seed=0).select(range(20))

    print("Wypisuję 20 losowych przykładów. Sprawdź ręcznie czy options[cop]")
    print("jest poprawną odpowiedzią dla każdego pytania.\n")

    for i, ex in enumerate(sample):
        cop = ex["cop"]
        options = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        question_preview = ex["question"][:70]

        print(f"[{i+1:02d}] Q: {question_preview}")
        print(f"      cop={cop} → options[cop] = '{options[cop]}'")
        print(f"      Opcje: A={ex['opa'][:30]} | B={ex['opb'][:30]}")
        print(f"             C={ex['opc'][:30]} | D={ex['opd'][:30]}")
        if ex.get("exp"):
            print(f"      Wyjaśnienie: {ex['exp'][:80]}")
        print()

    print("─" * 60)
    print("Po sprawdzeniu powyższych — wpisz wynik:")
    encoding = input("  cop=0 oznacza opcję A? (t/n): ").strip().lower()

    if encoding == "t":
        cop_offset = 0
        cop_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        print("✓ Zapisuję: cop encoding = 0-indexed (0=A, 1=B, 2=C, 3=D)")
    else:
        cop_offset = 1
        cop_map = {1: "A", 2: "B", 3: "C", 4: "D"}
        print("✓ Zapisuję: cop encoding = 1-indexed (1=A, 2=B, 3=C, 4=D)")

    config = load_config()
    config["cop_offset"] = cop_offset
    config["cop_map"] = {str(k): v for k, v in cop_map.items()}
    config["blocker_1_done"] = True
    save_config(config)

    print(f"✓ BLOCKER 1 DONE — config zapisany do src/config.json")
    return True


# ─────────────────────────────────────────────
# BLOCKER 2 — tokenizacja A/B/C/D w Mistral
# ─────────────────────────────────────────────

def verify_tokenization():
    print("\n" + "=" * 60)
    print("BLOCKER 2: Weryfikacja tokenizacji A/B/C/D w Mistral")
    print("=" * 60)
    print("Po 'Answer: ' model generuje token z poprzedzającą spacją.")
    print("Sprawdzamy czy ' A', ' B', ' C', ' D' = dokładnie 1 token każdy.\n")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: zainstaluj 'transformers': pip install transformers")
        sys.exit(1)

    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"Wczytuję tokenizer: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    ids_ABCD = []
    all_single = True

    for letter in ["A", "B", "C", "D"]:
        with_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        no_space   = tokenizer.encode(letter,        add_special_tokens=False)
        token_str  = tokenizer.decode(with_space)

        print(f"  ' {letter}' → token IDs: {with_space}  (n={len(with_space)})  decoded='{token_str}'")
        print(f"  '{letter}'  → token IDs: {no_space}   (n={len(no_space)})")

        if len(with_space) == 1:
            ids_ABCD.append(with_space[0])
            print(f"  ✓ Single token\n")
        else:
            print(f"  ⚠ MULTI-TOKEN — restricted entropy wymaga modyfikacji!\n")
            all_single = False

    if all_single:
        print(f"ids_ABCD = {ids_ABCD}")
        print("✓ Wszystkie litery są single-token. Restricted entropy działa bez zmian.")
    else:
        print("⚠ Niektóre litery są multi-token.")
        print("  Konieczna modyfikacja: użyj greedy decode zamiast restricted softmax.")
        print("  Zaktualizuj roadmap.md sekcję 3.3 przed kontynuowaniem.")
        # Nie zapisuj jako done — wymaga zmiany planu
        return False

    # Weryfikacja kontekstu "Answer:"
    test_prompt = "Question: What is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6\nAnswer:"
    test_ids = tokenizer.encode(test_prompt, add_special_tokens=False)
    print(f"\nTest prompt tokenizuje się na {len(test_ids)} tokenów.")
    print(f"Ostatni token przed generacją: '{tokenizer.decode([test_ids[-1]])}'")

    config = load_config()
    config["ids_ABCD"] = ids_ABCD
    config["blocker_2_done"] = True
    save_config(config)

    print(f"\n✓ BLOCKER 2 DONE — ids_ABCD={ids_ABCD} zapisane do src/config.json")
    return True


# ─────────────────────────────────────────────
# BLOCKER 3 — p(True) na fine-tuned modelu
# ─────────────────────────────────────────────

def verify_p_true(checkpoint_path=None):
    print("\n" + "=" * 60)
    print("BLOCKER 3: Weryfikacja p(True) na fine-tuned modelu")
    print("=" * 60)
    print("Sprawdzamy czy fine-tuned model daje sensowne P(Yes)/P(No).")
    print("Mean(P(Yes) + P(No)) powinno być ≥ 0.30\n")

    config = load_config()
    if not config.get("blocker_2_done"):
        print("ERROR: Najpierw uruchom BLOCKER 2.")
        sys.exit(1)

    ids_ABCD = config["ids_ABCD"]

    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import datasets
    except ImportError as e:
        print(f"ERROR: brakuje biblioteki: {e}")
        sys.exit(1)

    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

    if checkpoint_path is None:
        checkpoint_path = "./checkpoints/final"
        if not os.path.exists(checkpoint_path):
            candidates = [
                d for d in os.listdir("./checkpoints")
                if os.path.isdir(f"./checkpoints/{d}")
            ] if os.path.exists("./checkpoints") else []
            if candidates:
                checkpoint_path = f"./checkpoints/{sorted(candidates)[-1]}"
                print(f"Używam checkpointa: {checkpoint_path}")
            else:
                print("ERROR: Brak checkpointów. Uruchom najpierw fine-tuning (KROK 3).")
                sys.exit(1)

    print(f"Wczytuję model z: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[-1]
    no_id  = tokenizer.encode(" No",  add_special_tokens=False)[-1]
    print(f"yes_id={yes_id}, no_id={no_id}")

    cop_map = {int(k): v for k, v in config["cop_map"].items()}

    val_ds = datasets.load_dataset("openlifescienceai/medmcqa", split="validation")
    sample_30 = val_ds.shuffle(seed=1).select(range(30))

    yes_no_masses = []
    p_true_vals   = []
    correctness   = []

    print("\nTestuję 30 przykładów z val set...\n")
    for i, ex in enumerate(sample_30):
        opts = {"A": ex["opa"], "B": ex["opb"],
                "C": ex["opc"], "D": ex["opd"]}
        true_label = cop_map[ex["cop"]]

        prompt = (
            f"Question: {ex['question']}\n"
            f"A) {opts['A']}\nB) {opts['B']}\n"
            f"C) {opts['C']}\nD) {opts['D']}\n"
            f"Answer:"
        )

        # Predykcja
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, ids_ABCD]
        p_abcd = F.softmax(logits, dim=-1)
        pred   = "ABCD"[p_abcd.argmax().item()]
        y      = int(pred == true_label)

        # p(True)
        pt_prompt = (
            prompt + f" {pred}\n"
            f"Is this answer correct? (Yes/No):"
        )
        pt_ids = tokenizer.encode(pt_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            pt_logits = model(pt_ids).logits[0, -1, :]

        p_yn   = F.softmax(pt_logits[[yes_id, no_id]], dim=-1)
        p_yes  = p_yn[0].item()
        mass   = p_yn.sum().item()

        yes_no_masses.append(mass)
        p_true_vals.append(p_yes)
        correctness.append(y)

        status = "✓" if y else "✗"
        print(f"[{i+1:02d}] pred={pred} true={true_label} {status} | "
              f"p(Yes)={p_yes:.3f} p(No)={p_yn[1].item():.3f} mass={mass:.3f}")

    mean_mass   = sum(yes_no_masses) / len(yes_no_masses)
    mean_p_true = sum(p_true_vals) / len(p_true_vals)
    accuracy    = sum(correctness) / len(correctness)

    print(f"\n{'─'*60}")
    print(f"Model accuracy na 30 próbkach: {accuracy:.2%}")
    print(f"Mean Yes/No mass:              {mean_mass:.3f}")
    print(f"Mean p(True):                  {mean_p_true:.3f}")

    if mean_mass >= 0.30:
        verdict = "INCLUDE"
        print(f"\n✓ p(True) działa (mass={mean_mass:.3f} ≥ 0.30)")
        print("  → WŁĄCZ p_true do routing features")
    else:
        verdict = "EXCLUDE"
        print(f"\n⚠ p(True) zdegradowane (mass={mean_mass:.3f} < 0.30)")
        print("  → WYKLUCZ p_true z routing features")
        print("  → routing_feature_names = ['H', 'probe'] lub ['H', 'gap', 'probe']")

    config["p_true_verdict"]  = verdict
    config["p_true_mean_mass"] = mean_mass
    config["blocker_3_done"]  = True
    save_config(config)

    print(f"\n✓ BLOCKER 3 DONE — verdict={verdict} zapisany do src/config.json")
    return verdict == "INCLUDE"


# ─────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

def print_config_status():
    config = load_config()
    print("\n" + "=" * 60)
    print("STATUS WERYFIKACJI")
    print("=" * 60)
    b1 = "✓ DONE" if config.get("blocker_1_done") else "✗ TODO"
    b2 = "✓ DONE" if config.get("blocker_2_done") else "✗ TODO"
    b3 = "✓ DONE" if config.get("blocker_3_done") else "✗ TODO (wymaga fine-tuningu)"
    print(f"  BLOCKER 1 — cop encoding:     {b1}")
    if config.get("cop_map"):
        print(f"              cop_map = {config['cop_map']}")
    print(f"  BLOCKER 2 — tokenizacja ABCD: {b2}")
    if config.get("ids_ABCD"):
        print(f"              ids_ABCD = {config['ids_ABCD']}")
    print(f"  BLOCKER 3 — p(True) test:     {b3}")
    if config.get("p_true_verdict"):
        print(f"              verdict = {config['p_true_verdict']} "
              f"(mass={config.get('p_true_mean_mass', 0):.3f})")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline verification checks")
    parser.add_argument(
        "--blocker", choices=["1", "2", "3", "all", "status"],
        default="status",
        help="Który blocker uruchomić"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Ścieżka do checkpointa (dla blokera 3)"
    )
    args = parser.parse_args()

    if args.blocker == "status":
        print_config_status()

    elif args.blocker == "1":
        verify_cop_encoding()
        print_config_status()

    elif args.blocker == "2":
        verify_tokenization()
        print_config_status()

    elif args.blocker == "3":
        verify_p_true(args.checkpoint)
        print_config_status()

    elif args.blocker == "all":
        verify_cop_encoding()
        verify_tokenization()
        print("\nBLOCKER 3 wymaga fine-tuned modelu.")
        print("Uruchom po KROK 3: python 00_verify.py --blocker 3")
        print_config_status()

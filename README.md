### **INSTRUCTION FINE-TUNING A DECODER-ONLY LLM (GPT-2 Medium) — FROM DATA TO EVALUATION**

**Introduction**
Instruction fine-tuning (SFT) adapts a pretrained language model to *follow task instructions* reliably. In this project, a GPT-2–style decoder-only LLM is fine-tuned on a compact instruction dataset (≈1.1k pairs) using a clean, reproducible pipeline: dataset preparation → Alpaca-style prompt formatting → tokenization & dynamic padding with label masking → supervised fine-tuning → inference → lightweight “LLM-as-a-judge” evaluation. The goal is to produce visibly better, more on-task responses with minimal compute, in a way that’s easy for reviewers to scan and rerun.

---

**Objective**
Build an end-to-end SFT workflow that:

* Converts generic instruction/response JSON into a robust **Alpaca-style** format (Instruction, optional Input, Response).
* Tokenizes and **dynamically pads** batches while masking non-training tokens via `ignore_index=-100`.
* Fine-tunes **GPT-2 Medium (355M)** to improve instruction-following behavior.
* Generates responses deterministically and stores predictions for the test split.
* Scores outputs with a **lightweight judge** (Ollama Llama-3) on a 0–100 scale for quick comparative quality checks.

---

**IMPLEMENTATION:**
Open the training notebook in Colab: **[Open Colab Notebook](https://github.com/Aishwarya-chen11/Fine-tuned-LLM-Classification-Model/blob/main/Fine_tuned_LLM_classification_model.ipynb)**

**Tools and Technologies Used**

* **Python** for data prep, training, and evaluation.
* **PyTorch** for model execution, optimization, and training loops.
* **tiktoken (GPT-2 BPE)** for fast tokenization (`vocab_size=50,257`).
* **urllib / ssl / json** for data download & parsing.
* **tqdm** for progress bars during batch inference.
* (Optional) **Ollama** local API (e.g., `llama3`) to score model outputs deterministically.

---

**Model & Training Setup**

* **Backbone:** GPT-2 **Medium (355M)** — `emb_dim=1024`, `n_layers=24`, `n_heads=16`, context length **1024**, `qkv_bias=True`, `drop_rate=0.0`.
  *Rationale:* the 124M model is often too capacity-limited for visibly strong SFT; 355M is still light enough for single-GPU/CPU experimentation yet responds much better to instruction tuning.
* **Initialization:** Load official pretrained GPT-2 weights, then **SFT from scratch** (no LoRA here) with cross-entropy loss.
* **Optimizer:** `AdamW(lr=5e-5, weight_decay=0.1)` (simple, strong default).
* **Batching:** `batch_size=8`, `num_workers=0`, device auto-select (`cuda` if available; MPS/CPU fallback).
* **Epochs:** 1 (kept short for demonstration). The loop supports easy increases.
* **Eval cadence:** periodic lightweight evaluation with `eval_iter=5` batches to keep feedback snappy.

---

**Dataset**

* **Source size:** \~**1,100** instruction–response pairs (JSON, \~204 KB).

* **Fields:** `instruction`, `input` (may be empty), `output` (target).

* **Formatting:** Converted to **Alpaca-style** prompts:

  ```
  Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  {instruction}

  ### Input:
  {input}  # omitted if empty

  ### Response:
  {output}
  ```

  This standardization reliably separates instruction/context from the expected response.

* **Splits:** `85%` **train**, `10%` **test**, `5%` **validation** (seeded order).
  Persisting predictions and using fixed seeds ensure repeatability.

---

**Tokenization, Dynamic Padding, and Label Masking**

* **Tokenizer:** GPT-2 BPE (tiktoken) with **`<|endoftext|>`** used as both **pad**/*eos* token (ID **50256**).
* **Pre-tokenization:**
  A custom `InstructionDataset` encodes `Instruction + (optional) Input + Response` once at initialization for speed and reproducibility.
* **Custom collate function** (key SFT logic):

  * **Dynamic right-padding to the batch max length + 1** then shift to form language-model targets (next-token prediction).
  * **Label masking:**

    * All **pad tokens** are set to **`ignore_index=-100`** so they don’t contribute to loss.
    * (For SFT correctness) **Only the response portion** is trained: tokens *preceding* `"### Response:"` are set to `-100`. This teaches the model to produce the answer given the prompt, rather than wasting capacity learning to *copy* the prompt text.
  * **Length control:** optional `allowed_max_length` (default `1024`) safely truncates long examples to the model’s context window.
* **Result:** compact, variable-length batches that are compute-efficient and *loss-correct* for SFT.

---

**Training & Evaluation Loop**

* **Loss:** token-level cross-entropy over **response tokens only** (thanks to masking), standard next-token objective on a decoder-only LLM.
* **Monitoring:**

  * Periodic **train/val loss** snapshots over a few batches for quick signal.
  * **Sample generations** printed at epoch end to visually inspect instruction adherence.
* **Stability:** `torch.manual_seed(123)` for deterministic data ordering and text decoding; deterministic eval in the judge (temperature 0, fixed seed).

---

**Inference & Artifacts**

* **Deterministic generation** with EOS stopping on `<|endoftext|>`; `max_new_tokens=256` by default for test-time responses.
* **Post-processing:** strip the leading `"### Response:"` in the decoded string to leave clean answers.
* **Saved outputs:**

  * `instruction-data-with-response.json` (test set + `model_response`)
  * Model checkpoint: **`gpt2-medium355M-sft.pth`** (name auto-derived from the model choice)
  * You can later `load_state_dict` to reuse without retraining.

---

**LLM-as-a-Judge Scoring (Optional but Useful)**

* A **local judge** (e.g., **Ollama** `llama3`) is prompted:
  “Given the input `{formatted_instruction+input}` and correct output `{gold}`, score the model response `{pred}` on a scale 0-100 (best=100). Respond with the integer only.”
* **Deterministic options** (temperature=0, fixed seed) minimize score variance.
* **Result:** quick **quality proxy** without building a reference-based metric; in the included run, the finetuned model achieves an **average score > 50** on the held-out set (exact numbers depend on environment and judge).

---

**Results (Qualitative Summary)**

* Before SFT, the pretrained model often rambles, ignores structure, or under-answers.
* After **one short epoch**:

  * Responses are **more on-task**, concise, and aligned with the instruction format.
  * The judge scores climb to a **usable baseline** for a 355M model trained on a small dataset.
* Expect further gains with longer training, larger batch sizes, or richer data.

---

**Implementation Notes & Design Choices**

* **Why mask only the response?**
  It aligns with SFT’s goal: condition on the prompt, *learn the answer*. Training on the prompt adds loss on tokens the model shouldn’t have to generate.
* **Why GPT-2 Medium?**
  Strikes a practical balance of **capacity vs. cost**; noticeably more compliant than 124M while still runnable on modest hardware.
* **Why simple AdamW (no scheduler)?**
  Keeps the didactic loop simple; schedulers (e.g., cosine/warmup) can be added later.
* **Why Alpaca format?**
  Ubiquitous and robust; keeps prompts consistent across tasks and models.

---

**How to Run (Colab/Local Friendly)**

1. **Install deps:** `torch`, `tiktoken`, `tqdm`.
2. **Run the notebook top-to-bottom:**

   * Downloads the **\~1.1k** JSON dataset.
   * Formats to Alpaca style.
   * Builds dataset/dataloaders with **dynamic padding + label masking**.
   * Loads **GPT-2 Medium** weights and runs **1 epoch** of SFT.
   * Generates test responses and saves **JSON** + **checkpoint**.
3. *(Optional)* **Start Ollama** locally and run the judge cells to obtain 0–100 scores.

> Tip: If you’re CPU-bound, keep `batch_size=8`, shorten `max_new_tokens`, or train for 1–2 epochs. On a single modern GPU, you can scale epochs and eval frequency comfortably.

---

**Libraries Used**

* Core: **torch**, **torch.utils.data**, **json**, **urllib.request**, **ssl**
* Tokenization: **tiktoken** (GPT-2 BPE)
* Utilities: **tqdm**, **re** (for filename sanitization)

---

**Actionable Extensions & Next Steps**

1. **Training Depth**

   * Increase **epochs**, use **cosine LR with warmup**, or **gradient accumulation** for larger effective batch sizes.
   * Switch on **dropout** in the transformer blocks and tune **weight\_decay**.

2. **Parameter-Efficient Fine-Tuning**

   * Add **LoRA/QLoRA** adapters to train on consumer GPUs with larger backbones.
   * Compare full SFT vs. PEFT in throughput and quality.

3. **Data Improvements**

   * Enlarge the instruction set; diversify domains and styles.
   * Add **self-consistency** (multiple sampled responses) for pseudo-labels.

4. **Evaluation**

   * Complement LLM-judge with **reference metrics** (BLEU/ROUGE for suitable tasks) or **MT-Bench-style** prompts.
   * Aggregate judge scores over multiple deterministic seeds.

5. **Serving & Packaging**

   * Export to **TorchScript/ONNX**, wrap an **inference API** (FastAPI/Flask), and add basic **safety filters**.

---

**Why this project is portfolio-worthy**

* Shows **correct SFT mechanics**: Alpaca formatting, **dynamic padding**, **label masking**, next-token loss over **response tokens only**.
* Demonstrates **pragmatic model selection** (355M GPT-2) and **clean training scaffolding** that runs on modest hardware.
* Includes **deterministic generation & scoring**, reproducible seeds, and saved artifacts for easy verification.
* Easy to **extend** to LoRA, bigger datasets, better schedulers, or richer evaluation — a solid foundation for real instruction-following work.

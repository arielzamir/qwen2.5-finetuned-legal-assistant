# ⚖️ Legal Assistant - Qwen2.5 Fine-Tuned with LoRA

This project demonstrates how to fine-tune **[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)** using **LoRA (Low-Rank Adaptation)** on the **CUAD (Contract Understanding Atticus Dataset)** for legal question answering.  
The fine-tuned model is designed to assist with **contract analysis and legal Q&A**, making it easier to extract key clauses and obligations from legal documents.

---

## 🚀 Project Overview

- **Base Model:** Qwen2.5-1.5B-Instruct
- **Fine-Tuning Method:** LoRA (efficient parameter-efficient fine-tuning)
- **Dataset:** [CUAD](https://huggingface.co/datasets/chenghao/cuad_qa) – Legal contract Q&A
- **Frameworks:** Hugging Face `transformers`, `trl`, `peft`, `datasets`
- **Experiment Tracking:** Weights & Biases (wandb)
- **Platform:** Google Colab with GPU (T4)

---

## 📂 Repository Contents

- `legal_assistant_qwen_finetuned.ipynb` → Google Colab notebook with all training steps
- `README.md` → Project documentation (this file)

---

## ⚙️ Training Workflow

1. **Install dependencies** → Hugging Face ecosystem (`transformers`, `trl`, `peft`, `datasets`) + `bitsandbytes` for memory-efficient training.
2. **Dataset preparation** → Convert CUAD dataset into chat format with `system`, `user`, and `assistant` roles.
3. **Load base model** → Qwen2.5-1.5B-Instruct + tokenizer.
4. **Apply LoRA** → Fine-tune only small adapter layers for efficiency.
5. **Train with SFTTrainer** → Handles tokenization, batching, training loop, logging.
6. **Log with wandb** → Track metrics, loss, learning rate schedules.
7. **Save & Push** → Push trained adapter/model to Hugging Face Hub.

---

## 🔎 Inference Example (Quick Start)

You can try the fine-tuned model directly with 🤗 Transformers:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="ArielZamir23/legal-assistant-qwen2_5-1_5b-lora")

question = "What is the termination clause in this contract?"
output = generator([{"role": "user", "content": question}], max_new_tokens=256)

print(output[0]["generated_text"])
```

---

## Sample Output:

This Agreement shall terminate automatically at any time upon expiration of 12 months from the date it was executed or if the Parties do not enter into an agreement to continue providing Services pursuant to this Agreement within [***] after such expiration (the "Initial Term").

---

## 📊 Results

- The model successfully generates **contract-specific answers**
- Handles **legal terminology** and structures better than the base model
- Efficient training with **LoRA on Colab T4 GPUs**

---

## ✅ Intended Use

Contract analysis → Extract key clauses like termination, exclusivity, confidentiality.

Legal Q&A → Answer natural language questions about contract obligations.

Educational tool → Demonstrating domain adaptation of LLMs.

---

## ⚠️ Limitations

Not a substitute for legal advice.

Answers are dataset-dependent and may not generalize perfectly.

Sensitive to prompt phrasing.

---

## 📚 References

- [Qwen2.5 Models](https://huggingface.co/Qwen)
- [CUAD Dataset](https://huggingface.co/datasets/chenghao/cuad_qa)
- [Hugging Face TRL](https://huggingface.co/docs/trl/index)
- [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685)

---

## 👤 Author

**Ariel Zamir**

- GitHub: [@arielzamir](https://github.com/arielzamir)
- Hugging Face: [ArielZamir23](https://huggingface.co/ArielZamir23)

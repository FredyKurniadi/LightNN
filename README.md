# ModelCompress: A Practical Pipeline for Neural Network Compression

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Not state-of-the-art. Not production-grade. But a complete, educational proof-of-concept showing how model compression works from start to finish — and why it matters.**

---

## 🤔 Why This Exists

Modern AI faces a paradox: we build increasingly powerful models, then deploy them on resource-constrained devices. Cloud inference carries ongoing costs. Edge deployment demands efficiency. And the environmental footprint of large-scale inference is non-trivial.

**Model compression bridges this gap** — but most educational resources stop after explaining knowledge distillation in theory. Few show the complete journey from a trained model to a deployment-ready compressed artifact.

This repository demonstrates that journey. It uses the California Housing dataset not because the problem is hard, but because the *pipeline* is what matters. Swap the architecture, change the task — the compression principles remain the same.

---

## 🧠 The Pipeline: Four Stages of Compression

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. TRAIN   │ ──▶ │  2. PRUNE    │ ──▶ │  3. DISTILL   │ ──▶ │  4. QUANTIZE │
│   Teacher   │     │  (Neuron-by- │     │  (Knowledge  │     │  (Precision  │
│   Model     │     │   Neuron)    │     │   Transfer)  │     │   Reduction) │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                     │
                                                                     ▼
                                                              ┌──────────────┐
                                                              │  5. EXPORT   │
                                                              │  (TFLite,    │
                                                              │   ONNX, C)   │
                                                              └──────────────┘
```

### Stage 1: Train a Large Teacher
Build a deliberately oversized network (encoder-bottleneck-decoder architecture) and train it to convergence. This is the "expert" we'll compress.

### Stage 2: Neuron Importance Analysis & Structured Pruning
- **Not random guessing.** Each neuron in every layer is individually removed and the model's performance impact is measured.
- Neurons causing minimal degradation are identified as redundant.
- Entire neurons are removed (structured pruning), producing a dense, GPU-friendly smaller network — **no sparse matrix overhead.**
- The reduced architecture reveals which layers truly need depth.

### Stage 3: Knowledge Distillation with Custom Loss
A smaller student model is designed based on insights from Stage 2:
- **Fewer layers** than the teacher
- **Narrower width** per layer (except the bottleneck)
- **Same input/output dimensions**

The student trains using a three-component loss function:

| Component | Purpose |
|-----------|---------|
| **Feature Loss (α)** | Match intermediate representations (the *thought process*) |
| **Distillation Loss (β)** | Match the teacher's final predictions (the *dark knowledge*) |
| **Student Loss (γ)** | Stay grounded in real labels (the *ground truth*) |

Projection layers bridge dimension mismatches, enabling feature-level alignment between differently-sized architectures.

### Stage 4: Quantization (Precision Reduction)
The student model is trained with simulated 8-bit quantization using straight-through estimation. This reduces memory footprint by **4× per weight** with minimal accuracy loss.

### Stage 5: Deployment Export
The compressed model is exported to multiple formats:
- **ONNX** — cross-platform inference
- **TensorFlow Lite** — mobile & edge deployment
- **Pure NumPy** — zero-framework deployment
- **C header file** — bare-metal embedded systems

---

## 📊 Results Snapshot

| Model | Parameters | Memory | MSE | R² |
|-------|-----------|--------|-----|-----|
| Teacher (Original) | 21,089 | 84 KB | 0.218 | 0.79 |
| Student (Compressed + Quantized) | 4,079 | **4 KB** | 0.240 | 0.77 |

**~5× parameter reduction, ~20× memory reduction, with only 0.02 MSE loss.**

---

## 🗂️ Repository Structure

```
├── model_compression.ipynb    # Complete pipeline (4 cells)
│   ├── Cell 1: Setup & Data
│   ├── Cell 2: Train Teacher
│   ├── Cell 3: Neuron Analysis & Pruning
│   └── Cell 4: Distillation + Quantization + Export
├── teacher_model.pth          # Trained teacher (PyTorch)
├── student_model_quantized.pth # Final compressed student
├── model_weights.npz          # NumPy-compatible weights
├── model_weights.h            # C header for embedded deployment
└── README.md
```

---

## 🔧 How This Extends to Other Architectures

This pipeline is intentionally architecture-agnostic. The principles apply equally to:

- **CNNs**: Prune convolutional filters by L1-norm, distill with feature maps
- **RNNs/LSTMs**: Prune hidden units, distill with hidden state alignment
- **Transformers**: Prune attention heads, distill with attention-map matching
- **Any feedforward network**: Directly applicable as-is

The code is structured as modular components you can swap into your own projects.

---

## 🌍 Why Compression Matters

- **Hardware Accessibility**: Run capable models on phones, microcontrollers, and edge devices
- **Long-Term Cost**: Cloud inference costs compound over millions of requests
- **Environmental Impact**: Smaller models consume less energy per inference
- **Privacy**: On-device inference means data never leaves the user's device

---

## ⚠️ Important Caveats

This is a **proof-of-concept**, not a production library:
- The California Housing dataset is simple — real gains are larger on complex tasks
- Single-neuron pruning is computationally intensive for very large models
- Quantization-aware training here uses simulated quantization; real INT8 deployment requires framework-specific tooling

This code is meant to be **studied, adapted, and built upon** — not dropped into production.

---

## 🎓 Why I Built This

I'm an undergraduate student exploring the intersection of machine learning and systems engineering. This project emerged from hands-on curiosity:

> *"How do you actually make a model small enough to run on a phone? And can I do the whole thing myself, end-to-end?"*

If you're hiring interns or junior ML engineers — this is the kind of work I do. I care about the practical engineering behind AI, not just chasing benchmark scores.

---

## 📄 License

MIT — Use this code however you'd like. Attribution appreciated but not required.

---

**Built with curiosity, PyTorch, and a concern for AI's environmental footprint.**
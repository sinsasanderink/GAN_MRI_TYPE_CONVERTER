# 🧠 MRI T1-T2 Style Transfer using CycleGAN

![MRI T1↔T2 Conversion Example](conversion.png)


**Author:** Ursina Sanderink  
**Technology Stack:** TensorFlow, Python, Google Colab  
**Model Architecture:** CycleGAN with 114+ Million Parameters

---

## 🗂 Overview

This project implements a Cycle-Consistent Generative Adversarial Network (CycleGAN) for **bidirectional transformation** between **T1-weighted** and **T2-weighted** brain MRI scans — *without requiring paired training data*.

---

## 🏥 Clinical Motivation

- ⏱ **Reduced Scan Times**: Generate missing MRI sequences virtually
- 😌 **Patient Comfort**: Shorter procedures reduce discomfort and movement
- 💸 **Cost Efficiency**: Streamlined radiology workflows
- 🚑 **Emergency Imaging**: Faster decisions with limited sequences
- 🌍 **Global Health**: Extend diagnostic capabilities in low-resource settings

---

## 🧰 Technical Architecture

### 🧬 Model Components

- **Generator G (T1 → T2)** — 54.4M params  
- **Generator F (T2 → T1)** — 54.4M params  
- **Discriminator X (T1 domain)** — 2.8M params  
- **Discriminator Y (T2 domain)** — 2.8M params  
- **Total Parameters**: 114,344,708

### 🔧 Key Features

- 🧠 **U-Net Generators**: 8-level encoder-decoder with skip connections  
- 📦 **PatchGAN Discriminators**: 70×70 receptive field for realism  
- 🧼 **Instance Normalization**: Optimized for style transfer  
- 🔁 **Cycle Consistency (λ=10.0)**: Preserves anatomical structures  
- 🆔 **Identity Loss (λ=0.5)**: Maintains image characteristics

---

## 💡 Implementation Highlights

### 🧪 Medical Image Optimizations

- Preprocessing tailored for **16-bit medical data**
- Loss functions aware of **anatomical structure**
- Preservation of **quantitative voxel relationships**
- Medical-specific **data augmentation**

### ☁️ Google Colab Integration

- 🔍 Automatic **GPU detection & optimization**
- 📈 Memory scaling for **T4 GPUs**
- ☁️ One-click **dataset upload and extraction**
- 📊 Real-time training visualization

---

## ⚙️ Performance

### 💻 Training Specs

- **Hardware**: Google Colab (T4 GPU – 16GB VRAM)  
- **Memory Usage**: ~12GB  
- **Batch Size**: 1  
- **Speed**: ~45 sec/epoch  

### ⏱ Training Duration

| Epochs     | Time         | Use Case         |
|------------|--------------|------------------|
| 20         | 15–30 min    | Quick demo       |
| 100        | 2–4 hours    | Good quality     |
| 200        | 6–8 hours    | Production level |

### 🧠 Model Performance

- ⚡ Inference: <1 second/image  
- 📐 Input/Output: 256×256×1 grayscale  
- 🔁 Cycle Recons.: High structural fidelity  
- 🎨 Contrast Transfer: Realistic and accurate

---

## 🚀 Usage

### ☁️ Google Colab Setup

1. Upload `Complete_MRI_CycleGAN_Colab.ipynb`
2. Enable GPU → `Runtime > Change runtime type > GPU`
3. Run all cells
4. Upload your dataset when prompted
5. Watch training unfold with live graphs

### 📁 Dataset Structure

```
MRI_Dataset/
├── Tr1/TrainT1/     # T1-weighted images
└── Tr2/TrainT2/     # T2-weighted images
```

---

## 🔧 Key Hyperparameters

- 📉 Learning Rate: `2e-4`  
- 🧠 Optimizer: Adam (β₁ = 0.5)  
- 🔁 Cycle Loss Weight: 10.0  
- 🆔 Identity Loss Weight: 0.5  
- 🖼 Image Resolution: 256×256

---

## 💡 Technical Innovation

- 🧬 Instance norm customized for medical domains  
- ⚖️ Loss balancing for structure preservation  
- ☁️ Memory-efficient for free cloud GPUs  
- 📈 Interactive monitoring & reproducibility

---

## 🧪 Research Contributions

- 🔄 Unpaired medical image translation  
- ☁️ Full Google Colab support for accessibility  
- 🎓 Teaching framework for GAN learners  
- 📚 Reproducible and open-source medical AI research

---

## 📊 Results

### 📏 Quantitative Metrics

- 💾 Model Size: ~450MB (float32)  
- ⚙️ Parameters: 114M  
- ☁️ Free-tier Colab compatible  
- 📈 Stable training with consistent convergence

### 👁 Qualitative Assessment

| Category                | Rating            |
|------------------------|-------------------|
| Anatomical Preservation| ⭐️⭐️⭐️⭐️⭐️         |
| Contrast Accuracy      | ⭐️⭐️⭐️⭐️           |
| Artifacts              | Minimal           |
| Cycle Consistency      | High Fidelity     |

---

## 🏥 Clinical Validation

**Status**: 🚧 Research & educational use only

**Future Steps**:

- 🩻 Radiologist evaluation  
- 📊 Diagnostic performance studies  
- 🏥 Workflow integration  
- 🛡 Regulatory clearance

---

## 📁 File Structure

```
├── Complete_MRI_CycleGAN_Colab.ipynb  # Notebook for Colab
├── gan_architecture.py                # Model architecture
├── data_pipeline.py                   # Data preprocessing
├── train_cyclegan.py                  # Training script
├── demo_inference.py                  # Sample inference
└── README.md                          # This file
```

---

## 📦 Dependencies

```python
tensorflow>=2.8.0  
numpy>=1.21.0  
matplotlib>=3.5.0  
opencv-python>=4.5.0  
pillow>=8.3.0  
tqdm>=4.62.0  
```

---

## 🔖 Citation

If you use this implementation in your research:

```
@misc{sanderink2025mri_cyclegan,
  title={MRI T1-T2 Style Transfer using CycleGAN},
  author={Ursina Sanderink},
  year={2025},
  howpublished={\url{https://github.com/your-repo/mri-cyclegan}}
}
```

---

## 📄 License

MIT License — see `LICENSE` file for details.

---

## 🙏 Acknowledgments

- 🧠 Built on CycleGAN by **Zhu et al. (2017)**  
- 💡 Inspired by the medical imaging research community  
- ☁️ Optimized for **Google Colab** usage  
- 🎓 Designed as an **educational tool** for deep learning in healthcare

---

> ⚠️ **Disclaimer**: For research and education only. Not for clinical diagnosis or medical decision-making without proper validation.


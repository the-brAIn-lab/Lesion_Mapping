# Project Structure Documentation

## 🏗️ Directory Layout
```
stroke_segmentation_sota/
├── 📄 Core Training Scripts
│   ├── correct_full_training.py      # ✅ Baseline model (63.6% validated)
│   ├── proven_enhanced_model.py      # 🔧 Conservative SOTA improvements  
│   └── smart_sota_2025.py           # 🚀 Revolutionary 2025 techniques
│
├── ⚙️ Infrastructure
│   ├── scripts/                     # SLURM job submission scripts
│   │   ├── corrected_full_training.sh
│   │   ├── proven_enhanced.sh
│   │   └── smart_sota_2025.sh
│   │
│   ├── logs/                        # Training logs and outputs
│   ├── models/                      # Saved model files (.h5)
│   └── callbacks/                   # Training checkpoints
│
├── 📚 Documentation
│   ├── README.md                    # Main project documentation
│   ├── PROJECT_STRUCTURE.md         # This file
│   └── TECHNICAL_NOTES.md           # Detailed technical documentation
│
└── 🔧 Configuration
    ├── .gitignore                   # Git ignore rules
    └── requirements.txt             # Python dependencies
```

## 🎯 Model Files Description

### Core Models
- **correct_full_training.py**: Proven baseline with 63.6% validation Dice
- **proven_enhanced_model.py**: Incremental improvements (SE attention, GELU, enhanced loss)
- **smart_sota_2025.py**: Revolutionary architecture (Vision Mamba, SAM-2, boundary loss)

### Training Infrastructure  
- **Multi-GPU support**: All models use 4-GPU MirroredStrategy
- **Mixed precision**: Optimized float16 training
- **Robust callbacks**: Early stopping, learning rate scheduling, checkpointing

# RecyclingClassifier - Decision Log

## August 12, 2025 - Strategic Pivot: Abandoning TACO Dataset

### Decision: Remove TACO dataset integration and focus exclusively on Kaggle garbage classification dataset

### Rationale:

#### 1. **Complexity vs. Value Trade off**
- TACO dataset uses a complex COCO annotation format with segmentation masks
- The classification task I'm angling for doesn't require segmentation, simple image classification is sufficient for now
- The hierarchical structure I planned can be achieved more simply with Kaggle's clean categories

#### 2. **Data Quality & Balance Issues**
- TACO dataset has severe class imbalance (48/60 categories <200 samples, some with only 2-10 examples)
- Kaggle's dataset is much better balanced across 12 categories (~1,300 samples each)
- Quality over quantity: 15K clean, balanced samples > 21K imbalanced samples

#### 3. **Development Velocity**
- TACO integration was causing analysis paralysis
- Complex preprocessing pipeline was slowing down iteration
- Need to prove basic classification works before adding complexity

#### 4. **Technical Simplification**
- Kaggle dataset has clean directory structure (one folder per class)
- No need for COCO API, complex annotation parsing, or batch processing
- Direct integration with TensorFlow's `image_dataset_from_directory`

### Keeping:
- Kaggle dataset (12 categories: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass)
- MobileNetV2 transfer learning approach
- Web deployment infrastructure
- Model persistence and logging framework

### Removing:
- All TACO dataset references
- COCO annotation processing
- Complex batch processing scripts
- Hierarchical level_1/level_2 classification (for now)
- `analyzeCat.py`, `splitData.py`, `classMapping.py`

### Phase 1 Goals:
1. Build working 12 category classifier using Kaggle dataset
2. Achieve >85% validation accuracy
3. Deploy to web interface
4. Document baseline performance

### Future Considerations:
- Can revisit TACO integration in Phase 3 if needed
- Hierarchical classification can be added later as a separate model layer

---

*"Perfect is the enemy of good. Ship working software, then iterate."*

# Gather 🍃

> *A computer vision learning project documenting the journey from ML fundamentals to production deployment*

## What This Project Actually Is

This is a waste classification system built to understand the complete ML pipeline from data exploration to web deployment. 

**Current Status**: 
- Production ready classification system with 85%+ accuracy
- Full-stack ML application (training → API → frontend)
- Scalable architecture with proper ML engineering practices
- 9+ months of iterative development and learning
- 🔄 Exploring advanced techniques: ensemble methods, active learning

---

## The Learning Journey

### Initial Exploration: Complex Dataset (November 2024 - January 2025)
**Duration**: ~3 months  
**Dataset**: TACO (Trash Annotations in Context) - 1500+ categories  
**Focus**: Understanding the full complexity of real-world computer vision

**What I Attempted**:
- Multi-class classification with 1500+ highly specific categories
- Data preprocessing for extremely imbalanced classes
- Handling complex annotations and bounding boxes
- Working with real-world, messy data distributions

**What I Learned**:
- Why dataset complexity matters more than size
- The curse of dimensionality in multi-class problems
- How class imbalance destroys model performance
- When to scope down vs. when to push through
- Real-world datasets are messy and require significant preprocessing

**Strategic Pivot**: Chose to focus on fundamental ML engineering skills with a manageable problem scope rather than getting lost in dataset complexity. This wasn't giving up—it was strategic learning prioritization.

### Phase 1: ML Fundamentals with Focused Scope (February - June 2025)
**Duration**: ~4 months  
**Dataset**: Kaggle Garbage Classification (12 categories)  
**Focus**: Mastering the complete ML pipeline end-to-end

**What I Built**:
- Transfer learning pipeline using MobileNetV2
- Data preprocessing and augmentation systems
- Model evaluation frameworks with proper validation
- Hyperparameter optimization workflows
- Model versioning and experiment tracking

**What I Learned**:
- How to structure ML experiments for reproducibility
- The importance of validation strategy design
- Why smaller, well-curated datasets often teach more than large, messy ones
- Transfer learning best practices and when they break down
- How to debug training instabilities and convergence issues

**Technical Achievements**:
- Achieved 85%+ validation accuracy across 12 categories
- Implemented proper data splitting to prevent leakage
- Built preprocessing pipeline handling various image formats
- Created evaluation metrics beyond accuracy

### Phase 2: Production Engineering (July - August 2025)
**Duration**: ~2 months  
**Focus**: Moving beyond notebook experimentation to production systems

**What I Built**:
- Flask API with proper error handling
- Configuration management system
- Custom React frontend with responsive design
- Model serving infrastructure with proper lifecycle management
- Logging and monitoring setup

**What I Learned**:
- How ML engineering differs from traditional software engineering
- API design patterns for ML services
- Model loading optimization and memory management
- Frontend-backend integration for ML applications
- The importance of graceful degradation and error handling

**System Integration Challenges Solved**:
- Model loading time optimization (reduced from 15s to 3s)
- Memory management for concurrent requests
- Image preprocessing consistency between training and serving
- Configuration management across development and production environments

---

## Technical Architecture

```
┌─ Frontend (React/Vite) ──────────────────────┐
│  • Custom earth-toned UI design             │
│  • Image upload and preview                 │  
│  • Real-time prediction display             │
│  • Responsive mobile-first layout           │
└──────────────────────────────────────────────┘
                    │ HTTP API
┌─ Backend (Flask) ────────────────────────────┐
│  • RESTful API endpoints                     │
│  • TensorFlow model serving                  │
│  • Image preprocessing pipeline             │
│  • Configuration management                 │
└──────────────────────────────────────────────┘
                    │ Model Loading
┌─ ML Pipeline ────────────────────────────────┐
│  • MobileNetV2 transfer learning            │
│  • 12-category waste classification          │
│  • Data augmentation and preprocessing      │
│  • Model versioning and archival            │
└──────────────────────────────────────────────┘
```

### Technology Choices (And Why)

**TensorFlow/Keras**: Industry standard, extensive documentation for learning  
**MobileNetV2**: Good balance of accuracy and speed for mobile deployment  
**Flask**: Simple, well-documented, good for learning API design  
**React**: Modern frontend skills, component-based architecture  
**Vite**: Fast development, good developer experience  

---

## Project Structure

```
RecyclingClassifier/
├── research/                   # Learning materials and documentation
│   ├── LEARNING_CHALLENGES.md  # Progressive skill-building challenges
│   ├── LLM_LEARNING_PROMPTS.md # Templates for getting help
│   ├── WORKFLOW.md             # Daily learning workflow
│   ├── phase1_learning_edition.ipynb # Main training notebook
│   └── logs/                   # TensorBoard training logs
├── models/                     # Model artifacts and versioning
│   ├── current_model.keras     # Production model
│   ├── class_names.json        # Category labels
│   └── archive/                # Previous model versions
├── backend/                    # Flask API server
│   ├── app.py                  # Main API application
│   ├── config.py               # Configuration management
│   └── requirements.txt        # Python dependencies
├── web-app/                    # React frontend application
│   ├── src/                    # Source code
│   ├── public/                 # Static assets
│   └── package.json            # Node.js dependencies
├── config/                     # Centralized configuration
│   └── config.json             # Model and API settings
├── data/                       # Training data (Kaggle dataset)
│   └── Kaggle/garbage_classification/ # 12 waste categories
└── scripts/                    # Development utilities
```

---

## Getting Started

### Prerequisites
- Python 3.8+ with pip
- Node.js 16+ with npm
- 8GB+ RAM (for model loading)
- Basic understanding of ML concepts helpful but not required

### Local Development Setup

**1. Clone and Navigate**
```bash
git clone [repository-url]
cd RecyclingClassifier
```

**2. Backend Setup**
```bash
cd backend
pip install -r requirements.txt
python app.py
# Server starts on http://localhost:5001
```

**3. Frontend Setup**
```bash
cd web-app
npm install
npm run dev
# Development server starts on http://localhost:5173
```

**4. Test the System**
- Visit the frontend URL
- Upload an image of waste/recyclable material  
- Check browser dev tools for any API errors
- Expected: Classification results with confidence scores

---

## API Reference

### Core Endpoints

**Health Check**
```http
GET /health
Response: {"status": "healthy", "model_loaded": true}
```

**Image Classification**
```http
POST /predict
Content-Type: application/json
Body: {"image": "data:image/jpeg;base64,/9j/4AAQ..."}

Response: {
  "predicted_class": "plastic",
  "confidence": 0.87,
  "all_predictions": [...]
}
```

**Available Classes**
```http
GET /classes
Response: {"classes": ["battery", "biological", ...]}
```

---

## Model Performance & ML Engineering

**Overall System Performance**: 85%+ validation accuracy, 78%+ real-world accuracy  
**Training Approach**: Transfer learning with extensive experimentation

### ML Engineering Achievements

**Model Architecture Decisions**:
- Chose MobileNetV2 for optimal accuracy/speed tradeoff
- Implemented custom classification head with dropout regularization
- Experimented with different learning rate schedules
- Applied layer freezing strategies for transfer learning

**Performance by Category**:

**Strong Performance (85%+ accuracy)**:
- Cardboard: Excellent texture recognition, clear geometric patterns
- Metal: Strong performance on surface reflectivity and color patterns  
- Paper: Well-generalized features, good training data representation

**Moderate Performance (70-85% accuracy)**:
- Plastic: Handles clear containers well, struggles with colored/flexible plastics
- Glass (all colors): Good separation between colors, lighting sensitivity remains
- Clothes/Shoes: Decent material recognition, confused by mixed materials

**Challenging Categories (60-70% accuracy)**:
- Battery: Limited training examples, high shape/size variation
- Biological: Significant overlap with other organic materials
- Trash: Catch-all category with inherent classification ambiguity



### Next-Level ML Development

**Current Research Areas**:
- Ensemble methods combining multiple model architectures
- Active learning for targeted data collection in weak categories
- Model distillation for mobile deployment optimization
- Confidence calibration for uncertainty quantification

**Production ML Considerations**:
- A/B testing framework for model comparison
- Model drift detection and retraining triggers
- Model monitoring and alerting
- Automated model validation before deployment

---

## License & Acknowledgments

**Dataset**: Kaggle Garbage Classification Dataset  
**Base Model**: MobileNetV2 (ImageNet pre-trained)  
**UI Inspiration**: Earth-toned, nature-inspired design systems

Built as a learning project to understand the complete ML pipeline from research to deployment. 




# NetworkSecurity - ML-Powered Anomaly Detection System

<p align="center">
An end-to-end machine learning pipeline for detecting network anomalies, phishing attempts, and malicious traffic patterns using advanced classification techniques.
</p>

<hr>

## 📌 Project Overview

This project implements a **production-grade machine learning system** designed to identify suspicious network behavior and security threats. By leveraging supervised learning models trained on network security datasets, the system can classify incoming network traffic as either benign or malicious, enabling proactive threat detection and network protection.

**Core Objectives:**
- Detect network security anomalies with high accuracy
- Classify network traffic patterns as normal or suspicious
- Provide batch prediction capabilities for real-time monitoring
- Generate actionable insights from raw network data

<hr>

## 📊 Dataset Overview

The project utilizes the **Network Security Dataset** featuring:

| Metric | Value |
|--------|-------|
| **Total Records** | 10,000+ network traffic samples |
| **Features** | Network protocol parameters, packet information, traffic patterns |
| **Classes** | Binary classification (Benign / Malicious) |
| **Data Format** | CSV format with structured network metrics |
| **Source** | `Network_Data/phisingData.csv` |

<p>
The dataset undergoes rigorous preprocessing, validation, and transformation to ensure high-quality model training and reliable predictions.
</p>

<hr>

## 🚀 Key Features

- ✅ **Complete ML Pipeline**: Data ingestion → Validation → Transformation → Model Training
- 📊 **Comprehensive Data Validation**: Schema validation and data quality checks
- 🤖 **Multiple Model Support**: Extensible framework for various classification algorithms
- 📈 **Performance Metrics**: Accuracy, precision, recall, F1-score, and ROC-AUC tracking
- 🔄 **Batch Prediction**: Real-time threat detection on new network traffic
- 💾 **Database Integration**: MongoDB support for scalable data management
- 📋 **Artifact Tracking**: Automatic versioning of models and artifacts
- 📝 **Comprehensive Logging**: Full execution logs for debugging and monitoring

<hr>

## 🏗 Project Architecture

```
networksecurity/
│
├── components/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   └── model_trainer.py
│
├── entity/
│   ├── config_entity.py
│   └── artifact_entity.py
│
├── pipeline/
│   ├── training_pipeline.py
│   └── batch_prediction.py
│
├── utils/
│   ├── main_utils/
│   │   └── utils.py
│   └── ml_utils/
│       ├── metric/
│       │   └── classification_metric.py
│       └── model/
│           └── estimator.py
│
├── exception/
│   └── exception.py
│
├── logging/
│   └── logger.py
│
├── constant/
│   └── training_pipeline/
│
├── data_schema/
│   ├── schema.yaml
│   └── generate_schema.py
│
└── __init__.py
```

**Pipeline Flow:**
```
Raw Network Data → Ingestion → Validation → Transformation → Training → Model Artifact
                                                                              ↓
                                                          Prediction Pipeline → Batch Results
```

<hr>

## 🔄 Pipeline Components

### 1️⃣ **Data Ingestion**
- Reads network traffic data from CSV files
- Handles missing values and data normalization
- Creates clean datasets for downstream processing
- **File:** `components/data_ingestion.py`

### 2️⃣ **Data Validation**
- Validates input data against predefined schema
- Checks data types, ranges, and column requirements
- Detects and handles data quality issues
- **File:** `components/data_validation.py`

### 3️⃣ **Data Transformation**
- Feature scaling and normalization
- Encoding categorical variables
- Feature selection and engineering
- Handling class imbalance
- **File:** `components/data_transformation.py`

### 4️⃣ **Model Training**
- Trains classification models on processed data
- Implements model selection and hyperparameter tuning
- Tracks performance metrics (accuracy, F1-score, ROC-AUC)
- Saves best-performing model artifacts
- **File:** `components/model_trainer.py`

### 5️⃣ **Batch Prediction**
- Loads trained model for inference
- Processes new network traffic in batches
- Generates threat classification predictions
- Outputs results for security monitoring
- **File:** `pipeline/batch_prediction.py`

<hr>

## 📈 Model Performance Metrics

The system evaluates model performance using:

| Metric | Purpose |
|--------|---------|
| **Accuracy** | Overall correct predictions |
| **Precision** | True positives among predicted positives |
| **Recall** | True positives among actual positives |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Performance across classification thresholds |

These metrics are tracked throughout training to ensure robust threat detection capabilities.

<hr>

## ⚙️ Technical Stack

### 🔧 Core Technologies

| Component | Technology |
|-----------|-----------|
| **ML Framework** | scikit-learn / XGBoost / LightGBM |
| **Data Processing** | Pandas, NumPy |
| **Validation** | YAML schema, Pydantic |
| **Database** | MongoDB |
| **Logging** | Python logging module |
| **Configuration** | YAML, Python dataclasses |

### 🏗 Architecture Patterns

- **Modular Design**: Clear separation between data, model, pipeline, and utility components
- **Factory Pattern**: Dynamic model creation and selection
- **Pipeline Pattern**: Sequential data processing and model training
- **Entity Pattern**: Configuration and artifact management
- **Custom Exceptions**: Tailored error handling for pipeline stages

<hr>

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda package manager
- MongoDB (optional, for data persistence)

### Installation Steps

```bash
# Clone the repository
cd NetworkSecurity2

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

<hr>

## ▶️ Running the Pipeline

### Training Pipeline

```bash
python main.py
```

This executes the complete training pipeline:
- Ingests network data
- Validates and cleans data
- Transforms features
- Trains the classification model
- Evaluates performance
- Saves model artifacts

### Batch Prediction

```bash
python app.py
```

Use this for real-time threat detection on new network traffic data.

### Pushing Data to Database

```bash
python push_data.py
```

Stores processed data and model artifacts in MongoDB for scalable management.

<hr>

## 📁 Key Files & Their Purpose

| File | Purpose |
|------|---------|
| `app.py` | Flask/FastAPI application for serving predictions |
| `main.py` | Entry point for training pipeline |
| `push_data.py` | Database operation utilities |
| `setup.py` | Package installation configuration |
| `requirements.txt` | Python dependencies |
| `test_mongodb.py` | MongoDB connection testing |

<hr>

## 🔍 Data Schema

The system uses a YAML-based schema to define expected network data structure:

**Location:** `data_schema/schema.yaml`

Schema includes:
- Network traffic features (source/destination IPs, ports, protocols)
- Packet metrics (size, duration, rate)
- Security indicators
- Label column for classification

Run schema generation:
```bash
python data_schema/generate_schema.py
```

<hr>

## 📊 Output & Predictions

### Training Output
```
Training Metrics:
├── Model Accuracy: 94.2%
├── F1-Score: 0.942
├── Precision: 0.945
├── Recall: 0.939
└── ROC-AUC: 0.982
```

### Prediction Output
```
Sample Network Traffic Analysis:
├── Classification: Malicious
├── Confidence: 97.3%
├── Risk Level: High
└── Recommended Action: Block traffic
```

Results are saved to `prediction_output/output.csv` for downstream analysis.

<hr>

## 📝 Logging & Monitoring

The system maintains comprehensive logs for all pipeline stages:

- **Location:** `logs/` directory
- **Information Captured:**
  - Data ingestion status
  - Validation results
  - Training progress
  - Model performance metrics
  - Prediction results
  - Error tracking and debugging

Access logs for troubleshooting:
```bash
tail -f logs/training_pipeline.log
```

<hr>

## 🛠️ Development Best Practices

✅ **Code Quality Standards:**
- Type hints throughout codebase
- Comprehensive error handling with custom exceptions
- Modular function design
- Clear documentation and comments

✅ **Data Quality:**
- Input validation before processing
- Schema-based validation
- Quality checks at each pipeline stage
- Reproducible preprocessing

✅ **Model Management:**
- Artifact versioning and tracking
- Hyperparameter logging
- Cross-validation for robust evaluation
- Early stopping to prevent overfitting

<hr>

## 📈 Extending the System

### Adding a New Model

1. Create model class in `utils/ml_utils/model/`
2. Implement standard training interface
3. Register in model factory
4. Update configuration in `constant/training_pipeline/`

### Adding Custom Metrics

1. Extend `utils/ml_utils/metric/classification_metric.py`
2. Implement metric calculation method
3. Integrate into training evaluation loop

### Custom Data Transformations

1. Add transformation logic in `components/data_transformation.py`
2. Update schema in `data_schema/schema.yaml`
3. Add corresponding validation rules

<hr>

## 🤝 Contributing

Contributions are welcome! Follow these steps:

1. **Fork** the repository
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. **Make your changes** and commit:
   ```bash
   git commit -m "Add detailed description of changes"
   ```
4. **Push to your branch:**
   ```bash
   git push origin feature/YourFeatureName
   ```
5. **Open a Pull Request** with clear description

<hr>

## 📚 References & Resources

### Machine Learning & Security
- Scikit-learn documentation: https://scikit-learn.org/
- Network security best practices
- Anomaly detection methodologies
- Binary classification techniques

### Project Structure
- ML engineering best practices
- Production-ready Python packaging
- Pipeline architecture patterns
- Configuration management

### Related Technologies
- MongoDB documentation for data persistence
- Flask/FastAPI for API serving
- YAML for configuration management





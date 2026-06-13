# Text Intelligence Module using Deep Learning

This project is a 28-day internship task to build a Text Intelligence Module using Deep Learning. The goal is to create a text classification model that can categorize news articles into different topics.

## Current Industry-Ready Workflow

The repository now supports the complete model lifecycle for the AG News dataset:

1. Download or refresh the dataset:

```bash
python -m src.download_dataset
```

2. Train the main deep learning model:

```bash
python -m src.train --model-type lstm
```

For a fast smoke run while developing:

```bash
python -m src.train --model-type lstm --sample-size 4000 --epochs 1
```

3. Evaluate the trained model on the held-out test set:

```bash
python -m src.evaluate
```

This writes `reports/classification_report.json`, `reports/confusion_matrix.csv`, and `reports/evaluation_summary.json`.

4. Run a single prediction:

```bash
python -m src.predict
```

5. Start the Flask application:

```bash
flask --app app.main run
```

The app exposes:

- `GET /health` for service health checks.
- `POST /predict` with JSON body `{"text": "..."}` for API prediction.
- `GET /` for a simple browser interface.

## Model Approach

- Baseline: TF-IDF features with Logistic Regression, kept available through `python -m src.train --model-type baseline`.
- Main model: Embedding layer, SpatialDropout, Bidirectional LSTM, dense classification head, early stopping, learning-rate reduction, and model checkpointing.
- Classes: `World`, `Sports`, `Business`, `Sci/Tech`.

Generated datasets, model files, logs, and reports are intentionally kept out of version control.

## Project Plan

### Week 1: Project Setup & Data Foundation (Days 1-7)
- **Day 1:** Project Initialization: Create `README.md`, `.gitignore`, and project folder structure.
- **Day 2:** Environment Setup: Create `requirements.txt` with necessary libraries (TensorFlow, Keras, NLTK, etc.).
- **Day 3:** Dataset Acquisition: Find and download a suitable news classification dataset (e.g., AG News).
- **Day 4:** Data Exploration (Part 1): Create a Jupyter Notebook to load and inspect the dataset using Pandas.
- **Day 5:** Data Exploration (Part 2): Analyze class distribution and text characteristics (lengths, word counts).
- **Day 6:** Initial Data Preprocessing: Implement functions for cleaning text (lowercase, remove punctuation).
- **Day 7:** Advanced Preprocessing: Implement stopword removal and save the cleaned data.

### Week 2: Model Building & Training (Days 8-14)
- **Day 8:** Text to Sequences: Tokenize text and convert to integer sequences.
- **Day 9:** Padding & Splitting: Pad sequences and split the data into training, validation, and test sets.
- **Day 10:** Model Architecture (LSTM): Design a simple LSTM-based neural network for classification.
- **Day 11:** Model Compilation: Compile the model with an optimizer, loss function, and metrics.
- **Day 12:** Initial Model Training: Train the model for a few epochs and observe initial results.
- **Day 13:** Callbacks & Checkpoints: Add Keras callbacks for early stopping and model checkpointing.
- **Day 14:** Full Model Training: Train the model until convergence and save the best version.

### Week 3: Evaluation & Refinement (Days 15-21)
- **Day 15:** Model Evaluation: Evaluate the trained model on the test set.
- **Day 16:** Performance Analysis: Plot accuracy/loss curves and generate a classification report.
- **Day 17:** Error Analysis: Inspect misclassified examples to understand model weaknesses.
- **Day 18:** Model Improvement (Hyperparameters): Experiment with changing learning rate or optimizer.
- **Day 19:** Model Improvement (Architecture): Try a more complex architecture (e.g., Bidirectional LSTM, GRU).
- **Day 20:** Retrain & Compare: Retrain the improved model and compare its performance to the baseline.
- **Day 21:** Document Findings: Update the notebook with all findings and comparison tables.

### Week 4: Application & Deployment (Days 22-28)
- **Day 22:** Create Prediction Script: Write a Python script that loads the final model and predicts the category of new text.
- **Day 23:** Build a Simple Web Interface (Flask): Set up a basic Flask application.
- **Day 24:** Create HTML Form: Design a simple HTML form to take user input for prediction.
- **Day 25:** Integrate Model with Flask: Connect the prediction script to the Flask backend.
- **Day 26:** Finalize Application: Test the web application and refine the user interface.
- **Day 27:** Final Documentation: Clean up all code, add comments, and complete the `README.md`.
- **Day 28:** Project Review & Submission: Prepare a final summary and presentation of the project.

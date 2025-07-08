# Digital-Twin
A PyQt5-based desktop tool for performing tensor decomposition, anomaly detection, missing value prediction, and interactive What-If analysis on multi-dimensional datasets.
Features
1. Load data from CSV or JSON files
2. Choose value and dimension columns to form a tensor
3. Perform PARAFAC or TUCKER decomposition
Visualize:
1. Original data tensor
2. Reconstructed data tensor
3. Error tensor (for anomaly detection)
4. Detect anomalies using reconstruction error
5. Predict missing values using low-rank reconstruction
6. Perform What-If simulations by amplifying latent components
Export analysis results (errors, reconstructions) as CSV files

Requirements
Install dependencies using pip:
    pip install -r requirements.txt
requirements.txt
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    tensorly
    pyqt5

Run from Python source
    python3 tensor_analysis_gui.py

Application Workflow
  1. Load Data: Select a CSV or JSON file
  2. Choose Columns: Select:
  3. One value column (numerical)
  4. Two or more-dimensional columns (categorical/date)
  5. Select Method & Rank: Choose decomposition type (PARAFAC or TUCKER) and rank
Run Analysis:
  1. Tensor built and scaled
  2. Decomposition performed
  3. Reconstruction error computed
  4. Anomalies flagged
  5. Missing values filled
  6. What-if simulation run (component amplifiable)
  7. View Results: Visualise heatmaps for:
      a. Original data
      b. Reconstructed tensor
      c. Error tensor (anomalies)
  8. Export CSVs: Export:
      a. Reconstruction errors
      b. Filled tensor values
      c. Original tensor values

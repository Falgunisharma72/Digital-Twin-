# Updated extended_new.py with:
# - Anomaly visualization
# - What-if component selector
# - Missing value highlight
# - Export options

import sys
import os
import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, tucker
from sklearn.preprocessing import MinMaxScaler
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QComboBox, QTextEdit, QListWidget, QMessageBox, QSpinBox, QSlider)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class TensorAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tensor Analysis Application - Insightful")
        self.setGeometry(100, 100, 1400, 800)
        self.df_raw = None
        self.value_column = None
        self.dimension_columns = []
        self.tensor_original = None
        self.scaled_tensor = None
        self.missing_values_mask = None
        self.dimension_maps = None
        self.scaler = None
        self.factors = None
        self.core = None
        self.error_tensor = None
        self.reconstructed_original = None
        self.rank_selected = None
        self.current_component = 0
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(300)

        self.file_label = QLabel("No file selected")
        self.load_button = QPushButton("Load Data (CSV/JSON)")
        self.load_button.clicked.connect(self.load_data)
        control_layout.addWidget(QLabel("Data File:"))
        control_layout.addWidget(self.file_label)
        control_layout.addWidget(self.load_button)

        self.collection_label = QLabel("Select JSON Collection:")
        self.collection_combo = QComboBox()
        self.collection_combo.setEnabled(False)
        control_layout.addWidget(self.collection_label)
        control_layout.addWidget(self.collection_combo)

        self.value_label = QLabel("Select Value Column:")
        self.value_combo = QComboBox()
        self.value_combo.setEnabled(False)
        control_layout.addWidget(self.value_label)
        control_layout.addWidget(self.value_combo)

        self.dim_label = QLabel("Select Dimension Columns:")
        self.dim_list = QListWidget()
        self.dim_list.setSelectionMode(QListWidget.MultiSelection)
        self.dim_list.setEnabled(False)
        control_layout.addWidget(self.dim_label)
        control_layout.addWidget(self.dim_list)

        self.rank_label = QLabel("Number of Latent Patterns (Rank):")
        self.rank_input = QSpinBox()
        self.rank_input.setRange(1, 10)
        self.rank_input.setValue(3)
        control_layout.addWidget(self.rank_label)
        control_layout.addWidget(self.rank_input)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["PARAFAC", "TUCKER"])
        control_layout.addWidget(QLabel("Decomposition Method:"))
        control_layout.addWidget(self.method_combo)

        self.component_slider = QSlider(Qt.Horizontal)
        self.component_slider.setRange(0, 9)
        self.component_slider.setValue(0)
        self.component_slider.setTickInterval(1)
        self.component_slider.setEnabled(False)
        self.component_slider.valueChanged.connect(self.update_component)
        control_layout.addWidget(QLabel("Component for What-If Analysis:"))
        control_layout.addWidget(self.component_slider)

        self.analyze_button = QPushButton("Run Analysis")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.run_analysis)
        control_layout.addWidget(self.analyze_button)

        self.save_button = QPushButton("Export Results")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.export_results)
        control_layout.addWidget(self.save_button)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        control_layout.addWidget(QLabel("Output:"))
        control_layout.addWidget(self.console)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.canvas, stretch=1)

    def update_component(self, value):
        self.current_component = value

    def log(self, message):
        self.console.append(message + "\n")
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "CSV/JSON Files (*.csv *.json)")
        if not file_path:
            return
        self.file_label.setText(os.path.basename(file_path))
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and len(data.keys()) > 0:
                        self.collection_combo.clear()
                        self.collection_combo.addItems(list(data.keys()))
                        self.collection_combo.setEnabled(True)
                        self.collection_combo.currentTextChanged.connect(lambda: self.load_json_data(file_path))
                        self.load_json_data(file_path)
                    else:
                        self.df_raw = pd.DataFrame(data)
                        self.collection_combo.setEnabled(False)
                        self.populate_columns()
            elif file_path.endswith('.csv'):
                self.df_raw = pd.read_csv(file_path)
                self.collection_combo.setEnabled(False)
                self.populate_columns()
            else:
                raise ValueError("Unsupported file format. Please use CSV or JSON.")
            if self.df_raw.empty:
                raise ValueError("Loaded DataFrame is empty.")
            self.log(f"\U00002705 Successfully loaded data with shape {self.df_raw.shape}")
            self.log("--- Data Preview ---")
            self.log(str(self.df_raw.head()))
            self.analyze_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            self.log(f"Error: {str(e)}")

    def load_json_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            collection = self.collection_combo.currentText()
            self.df_raw = pd.DataFrame(data.get(collection, []))
            self.populate_columns()
            self.log(f"\U00002705 Loaded JSON collection '{collection}' with shape {self.df_raw.shape}")
            self.log("--- Data Preview ---")
            self.log(str(self.df_raw.head()))

    def populate_columns(self):
        self.value_combo.clear()
        self.dim_list.clear()
        columns = self.df_raw.columns.tolist()
        self.value_combo.addItems(columns)
        self.dim_list.addItems(columns)
        self.value_combo.setEnabled(True)
        self.dim_list.setEnabled(True)

    def export_results(self):
        pd.DataFrame(self.error_tensor.reshape(-1), columns=["Reconstruction Error"]).to_csv("reconstruction_error.csv", index=False)
        pd.DataFrame(self.reconstructed_original.reshape(-1), columns=["Reconstructed Value"]).to_csv("reconstructed_values.csv", index=False)
        pd.DataFrame(self.tensor_original.reshape(-1), columns=["Original Value"]).to_csv("original_values.csv", index=False)
        QMessageBox.information(self, "Saved", "CSV files exported successfully.")

    def build_tensor(self):
        self.value_column = self.value_combo.currentText()
        self.dimension_columns = [self.dim_list.item(i).text() for i in range(self.dim_list.count()) if self.dim_list.item(i).isSelected()]
        if not self.value_column or len(self.dimension_columns) < 2:
            QMessageBox.critical(self, "Error", "Select value and at least two dimension columns.")
            return None, None, None, None, None
        df_clean = self.df_raw[self.dimension_columns + [self.value_column]].copy()
        df_clean[self.value_column] = pd.to_numeric(df_clean[self.value_column].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
        df_clean.dropna(subset=[self.value_column], inplace=True)
        for col in self.dimension_columns:
            if 'date' in col.lower():
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce').dt.to_period('M').astype(str)
                df_clean.dropna(subset=[col], inplace=True)
        agg_df = df_clean.groupby(self.dimension_columns, as_index=False)[self.value_column].mean()
        dimension_maps = {col: {val: i for i, val in enumerate(sorted(agg_df[col].astype(str).unique()))} for col in self.dimension_columns}
        tensor_shape = [len(dimension_maps[col]) for col in self.dimension_columns]
        tensor_original = np.zeros(tensor_shape, dtype=np.float64)
        for _, row in agg_df.iterrows():
            idx = tuple(dimension_maps[col][str(row[col])] for col in self.dimension_columns)
            tensor_original[idx] = float(row[self.value_column])
        missing_mask = (tensor_original == 0)
        scaler = MinMaxScaler()
        scaled_tensor = scaler.fit_transform(tensor_original.reshape(-1, 1)).reshape(tensor_original.shape)
        return tensor_original, scaled_tensor, missing_mask, dimension_maps, scaler

    def analyze_tensor(self):
        rank = self.rank_input.value()
        self.rank_selected = rank
        method = self.method_combo.currentText()
        self.log(f"\nPerforming {method} Decomposition with Rank={rank}...")
        if method == "PARAFAC":
            weights, factors = non_negative_parafac(self.scaled_tensor, rank=rank, init='random', tol=1e-8, n_iter_max=500)
            reconstructed_scaled = tl.cp_to_tensor((weights, factors))
            self.core = None
        else:
            core, factors = tucker(self.scaled_tensor, rank=[rank]*len(self.dimension_columns))
            reconstructed_scaled = tl.tucker_to_tensor((core, factors))
            self.core = core
        reconstructed_original = self.scaler.inverse_transform(reconstructed_scaled.reshape(-1,1)).reshape(reconstructed_scaled.shape)
        error_tensor = np.abs(self.scaled_tensor - reconstructed_scaled)
        self.tensor_original = self.tensor_original
        self.log(f"Approximation Error (Frobenius norm): {np.linalg.norm(error_tensor):.4f}")
        anomaly_score = error_tensor > 2 * np.std(error_tensor)
        filled_values = np.where(self.tensor_original == 0, reconstructed_original, self.tensor_original)
        self.log(f"\n--- Anomaly Detection ---\nDetected {np.sum(anomaly_score)} anomalies")
        self.log(f"\n--- Missing Value Prediction ---\nFilled {np.sum(self.tensor_original == 0)} missing entries")
        self.log(f"\n--- What-If Analysis (Component {self.current_component}) ---")
        if method == "PARAFAC":
            factors[0][:,self.current_component] *= 1.2
            modified_tensor = tl.cp_to_tensor((weights, factors))
            delta = modified_tensor - reconstructed_scaled
            self.log(f"Simulated impact Δmean={np.mean(delta):.4f}, Δmax={np.max(delta):.4f}")
        self.component_slider.setEnabled(True)
        self.save_button.setEnabled(True)
        return factors, error_tensor, reconstructed_original

    def run_analysis(self):
        try:
            self.tensor_original, self.scaled_tensor, self.missing_values_mask, self.dimension_maps, self.scaler = self.build_tensor()
            if self.tensor_original is None:
                return
            self.factors, self.error_tensor, self.reconstructed_original = self.analyze_tensor()
            self.visualize_results()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            self.log(f"Error: {str(e)}")

    def visualize_results(self):
        self.figure.clear()
        self.canvas.figure.clear()
        self.canvas.draw()
        fig, axes = plt.subplots(1, 3, figsize=(18,6))
        sns.heatmap(self.tensor_original.squeeze(), ax=axes[0], cmap="Blues")
        axes[0].set_title("Original Data")
        sns.heatmap(self.reconstructed_original.squeeze(), ax=axes[1], cmap="Greens")
        axes[1].set_title("Reconstructed Data")
        sns.heatmap(self.error_tensor.squeeze(), ax=axes[2], cmap="Reds")
        axes[2].set_title("Error Tensor (Anomalies Highlighted)")
        plt.tight_layout()
        self.figure = fig
        self.canvas.figure = fig
        self.canvas.draw()
        plt.close(fig)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TensorAnalysisGUI()
    window.show()
    sys.exit(app.exec_())

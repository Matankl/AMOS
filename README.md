# Abdominal Multi-Organ Segmentation (AMOS) Project

## Authors
- **Matan Ziv**
- **Yohann Pardes** 

## Project Objective
Develop a machine learning pipeline for segmenting abdominal organs in CT/MRI scans using the AMOS dataset. The task involves predicting segmentation masks that accurately delimit multiple organs based on extracted image features.

---

## Dataset Overview
- **Source**: AMOS dataset containing CT scans from diverse patients.
- **Classes**: 16 segmentation classes, including spleen, liver, kidneys, pancreas, etc.
- **Data Format**: Volumetric CT scans (.nii.gz) preprocessed into 2D slices.
- **Preprocessing**:
  - Resolution standardized to **640×640 pixels**.
  - Dataset split into:
    - **70% Training** (240 scans)
    - **15% Validation** (60 scans)
    - **15% Testing** (60 scans)

---

## Evaluation Metrics
1. **Pixel-wise Accuracy**: Proportion of correctly classified pixels.
2. **Mean Intersection over Union (Mean IoU)**: Measures overlap between predicted and ground truth masks for each class, averaged across all classes.

---

## Model Development and Results

### 1. **Baseline Model**
- **Method**: Probabilistic classifier predicting the most frequent label at each pixel.
- **Results**:
  - **Accuracy**: 95.42%
  - **Mean IoU**: 5.96%
- **Analysis**: Strong bias towards background class; failed to identify organ pixels.

### 2. **Logistic Regression**
- **Approach**: Pixel-wise classification using intensity and spatial patches.
  - Tested 3×3 and 5×5 patches.
- **Results**:
  - Accuracy: ~85%
  - Mean IoU: ~6.6%
- **Analysis**: Unable to capture spatial relationships effectively.

### 3. **Fully Connected Neural Network**
- **Architecture**: Dense layers with input downsampled to 220×220.
- **Results**:
  - **Accuracy**: 50.5%
  - **Mean IoU**: 11.2%
- **Analysis**: Limited spatial awareness; over-sensitivity to pixel-wise variations.

### 4. **U-Net Architecture**
- **Features**:
  - Encoder-decoder structure with skip connections.
  - Weighted categorical cross-entropy loss for class imbalance.
  - Hyperparameter tuning via Optuna.
- **Results**:
  - **Accuracy**: 83.5%
  - **Mean IoU**: 52.6%
- **Analysis**: Significant improvement in segmentation performance, especially for underrepresented classes.

---

## Model Comparison

| Model                  | Accuracy | Mean IoU | Overall Performance |
|------------------------|----------|----------|---------------------|
| **Baseline**           | 95.42%   | 5.96%    | Poor                |
| **Logistic Regression**| 85.8%    | 6.63%    | Poor                |
| **Fully Connected NN** | 50.5%    | 11.2%    | Poor                |
| **U-Net**              | 83.5%    | 52.6%    | Good                |

**note! the U-net can provide better preformance with more training and larger data set**

---

## 3D Reconstruction
Using segmented CT slices, organs were reconstructed in 3D. This technique is commonly applied in clinical settings for pre-surgical planning (e.g., 3D heart models).

---

## Key Insights
- **Baseline and Logistic Regression**: Inadequate due to inability to capture spatial dependencies.
- **Fully Connected NN**: Struggled with spatial relationships and overreacted to pixel-wise variations.
- **U-Net**: Demonstrated superior performance through spatial and hierarchical feature extraction.

---

## Future Work
- Explore advanced architectures like Transformers for medical image segmentation.
- Incorporate more diverse datasets to improve generalizability.
- Investigate real-time segmentation for clinical applications.


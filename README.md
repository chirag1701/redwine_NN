# Wine Quality Prediction using Neural Networks

This project aims to predict the quality of red wine using neural networks implemented with both NumPy and TensorFlow. It includes comprehensive data analysis, visualization, and model training to understand and predict wine quality based on various chemical properties. By implementing the neural network using two different approaches, we gain insights into the strengths and challenges of each method.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data Analysis](#data-analysis)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Results and Comparison](#results-and-comparison)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

## Features
- Extensive data analysis and visualization of red wine properties
- Correlation analysis between different wine features
- Two neural network implementations:
  1. NumPy-based neural network built from scratch
  2. TensorFlow-based neural network using Keras API
- Hyperparameter tuning for both implementations
- Performance evaluation using multiple metrics:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R-squared (R²) score
- Comparative analysis of NumPy and TensorFlow implementations

## Requirements
- Python 3.7+
- pandas==1.3.3
- numpy==1.21.2
- matplotlib==3.4.3
- seaborn==0.11.2
- scikit-learn==0.24.2
- tensorflow==2.6.0

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/your-username/wine-quality-prediction.git
   cd wine-quality-prediction
   ```
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Ensure you have the `winequality-red.csv` dataset in the `data` directory.
2. Run the Jupyter notebooks to perform data analysis and train the models:
   - `notebooks/data_analysis.ipynb` for exploratory data analysis
   - `notebooks/numpy_implementation.ipynb` for the NumPy-based neural network
   - `notebooks/tensorflow_implementation.ipynb` for the TensorFlow-based neural network
3. The trained models will be saved in the `models` directory.
4. Visualizations and results will be saved in the `results` directory.

## Data Analysis
The project includes a comprehensive data analysis phase with various visualization techniques:
- Correlation matrix heatmap to identify relationships between features
- Pairplots of selected features to visualize pairwise relationships
- Histograms of wine properties to understand data distributions
- Box plots and violin plots to identify outliers and compare distributions
- Scatter plots of features vs. quality to identify potential predictors
- Feature importance analysis using random forests

These visualizations help in understanding the relationships between different wine properties and their impact on quality, guiding feature selection and preprocessing steps.

## Model Architecture

### NumPy Implementation
The NumPy-based neural network is built from scratch with the following architecture:
- Input layer: 11 neurons (one for each feature)
- Hidden layer 1: 64 neurons with ReLU activation
- Hidden layer 2: 32 neurons with ReLU activation
- Output layer: 1 neuron with linear activation

### TensorFlow Implementation
The TensorFlow model is built using the Keras API with the following architecture:
- Input layer: 11 neurons
- Hidden layer 1: 64 neurons with ReLU activation and dropout (0.2)
- Hidden layer 2: 32 neurons with ReLU activation and dropout (0.2)
- Output layer: 1 neuron with linear activation

## Model Training

### NumPy Implementation
The NumPy implementation includes the following features:
- Custom implementation of forward and backward propagation
- Mini-batch gradient descent optimization
- Learning rate scheduling with exponential decay
- L2 regularization to prevent overfitting
- Early stopping based on validation loss

Hyperparameters:
- Learning rate: 0.001 (initial)
- Batch size: 32
- Epochs: 1000 (with early stopping)
- L2 regularization strength: 0.01

### TensorFlow Implementation
The TensorFlow model is trained using:
- Adam optimizer
- Mean Squared Error (MSE) as the loss function
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping based on validation loss
- Model checkpointing to save the best model

Hyperparameters:
- Learning rate: 0.001 (initial)
- Batch size: 32
- Epochs: 1000 (with early stopping)
- Dropout rate: 0.2

## Results and Comparison
Both models' performances are evaluated using the following metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (R²) score

The training processes and model performances are visualized using:
- Loss curves (training and validation) over epochs
- MSE and MAE plots over epochs
- Predicted vs. Actual quality scatter plots
- Residual plots to assess model assumptions

A detailed comparison of the two implementations is provided, discussing:
- Training time and computational efficiency
- Ease of implementation and debugging
- Flexibility and extensibility
- Performance on the test set
- Overfitting tendencies

## Future Improvements
- Experiment with different neural network architectures (e.g., deeper networks, skip connections)
- Implement ensemble methods combining multiple models
- Explore feature engineering techniques to create more informative predictors
- Investigate the use of other optimization algorithms (e.g., RMSprop, Adagrad)
- Perform more extensive hyperparameter tuning using techniques like Bayesian optimization

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
Chirag Sindhwani - IIT BHU , Department of EE
Email: sindhwanichirag17@gmail.com
LinkedIn: [Chirag Sindhwani](https://www.linkedin.com/in/chirag-sindhwani-profile/)


---

Feel free to star ⭐ this repository if you find it helpful!

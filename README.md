# Stream Dataset

A comprehensive dataset suite for evaluating sequence modeling capabilities of neural networks, particularly focusing on memory, long-term dependencies, and temporal reasoning tasks.

## 🚀 Features

- **12 Diverse Tasks**: From simple memory tests to complex pattern recognition
- **Multiple Difficulty Levels**: Small, medium and large configurations. Advice : if you're building an architecture, you should begin with small.
- **Unified Interface**: Consistent API across all tasks with standardized evaluation metrics
- **Ready-to-Use**: Pre-configured datasets with train/validation/test splits
- **Flexible**: Support for both classification and regression tasks

## 📦 Installation

```bash
pip install stream-dataset
```

Or install from source:

```bash
git clone https://github.com/Naowak/stream-dataset.git
cd stream-dataset
pip install -e .
```

## 🎯 Quick Start

```python
import stream_dataset as sd

# Build a task
task_data = sd.build_task('simple_copy', difficulty='small', seed=0)

# Access the data
X_train = task_data['X_train']  # Training inputs
Y_train = task_data['Y_train']  # Training targets
T_train = task_data['T_train']  # Prediction timesteps

# Train your model (example with dummy predictions)
Y_pred = your_model.predict(X_train)

# Evaluate performance
score = sd.compute_score(
    Y=Y_train, 
    Y_hat=Y_pred, 
    prediction_timesteps=T_train,
    classification=task_data['classification']
)
print(f"Score: {score}")
```

## 📚 Available Tasks

### Memory Tests
### Postcasting Tasks
- **`discrete_postcasting`**: Copy discrete sequences after a delay  

    ![discrete_postcasting](https://raw.githubusercontent.com/Naowak/stream-dataset/main/images/discrete_postcasting.png)

- **`continuous_postcasting`**: Copy continuous sequences after a delay  

    ![continuous_postcasting](https://raw.githubusercontent.com/Naowak/stream-dataset/main/images/continuous_postcasting.png)

### Signal Processing
- **`sinus_forecasting`**: Predict frequency-modulated sinusoidal signals 

    ![sinus_forecasting](https://raw.githubusercontent.com/Naowak/stream-dataset/main/images/sinus_forecasting.png)

- **`chaotic_forecasting`**: Forecast Lorenz system dynamics 

    ![chaotic_forecasting](https://raw.githubusercontent.com/Naowak/stream-dataset/main/images/chaotic_forecasting.png)


### Long-term Dependencies
- **`discrete_pattern_completion`**: Complete masked repetitive patterns (discrete)  

    ![discrete_pattern_completion](https://raw.githubusercontent.com/Naowak/stream-dataset/main/images/discrete_pattern_completion.png)

- **`continuous_pattern_completion`**: Complete masked repetitive patterns (continuous) 

    ![continuous_pattern_completion](https://raw.githubusercontent.com/Naowak/stream-dataset/main/images/continuous_pattern_completion.png)

- **`simple_copy`**: Memorize and reproduce sequences after delay + trigger  

    ![simple_copy](https://raw.githubusercontent.com/Naowak/stream-dataset/main/images/simple_copy.png)

- **`selective_copy`**: Memorize only marked elements and reproduce them  

    ![selective_copy](https://raw.githubusercontent.com/Naowak/stream-dataset/main/images/selective_copy.png)


### Information Manipulation
- **`adding_problem`**: Add numbers at marked positions  

    ![adding_problem](https://raw.githubusercontent.com/Naowak/stream-dataset/main/images/adding_problem.png)

- **`sorting_problem`**: Sort sequences according to given positions 

    ![sorting_problem](https://raw.githubusercontent.com/Naowak/stream-dataset/main/images/sorting_problem.png)

- **`bracket_matching`**: Validate parentheses sequences

    ![bracket_matching](https://raw.githubusercontent.com/Naowak/stream-dataset/main/images/bracket_matching.png)

- **`cross_situation`**: Classify objects based on multiple attributes (color, shape, position)

    ![cross_situation](https://raw.githubusercontent.com/Naowak/stream-dataset/main/images/cross_situation.png)


## 🔧 Task Configuration

Each task supports three difficulty levels (the following numbers may vary in function of the task):

### Small 
- Reduced sequence lengths and sample counts
- Suitable for quick experiments and debugging
- Example: 100 training samples, sequences of ~50-100 timesteps

### Medium 
- Realistic problem sizes
- Suitable for thorough model evaluation
- Example: 1,000 training samples, sequences of ~100-200 timesteps

### Large 
- Big Data configurations
- Suitable for high-performance models
- Example: 10,000 training samples, sequences of ~200-500 timesteps

For the tasks that allow it, the difficulty is also adjusted by the dimensions of input & output.

```python
# Small configuration (fast)
task_small = sd.build_task('bracket_matching', difficulty='small', seed=0)

# Medium configuration (thorough)
task_medium = sd.build_task('bracket_matching', difficulty='medium', seed=0)

# Large configuration (comprehensive)
task_large = sd.build_task('bracket_matching', difficulty='large', seed=0)
```

## 📊 Data Format

All tasks return a standardized dictionary:

```python
{
    'X_train': np.ndarray,      # Training inputs [batch, time, features]
    'Y_train': np.ndarray,      # Training targets [batch, time, outputs]
    'T_train': np.ndarray,      # Training prediction timesteps [batch, n_predictions]
    'X_valid': np.ndarray,      # Validation inputs
    'Y_valid': np.ndarray,      # Validation targets  
    'T_valid': np.ndarray,      # Validation prediction timesteps
    'X_test': np.ndarray,       # Test inputs
    'Y_test': np.ndarray,       # Test targets
    'T_test': np.ndarray,       # Test prediction timesteps
    'classification': bool      # True for classification, False for regression : used to select the correspongind loss
}
```

## 🎨 Example: Complete Evaluation Pipeline

```python
import stream_dataset as sd
import numpy as np
from MyModel import MyModel

def evaluate_model_on_all_tasks(model, difficulty='small'):
    """Evaluate a model on all available tasks."""
    
    results = {}
    task_names = [
        'simple_copy', 'selective_copy', 'adding_problem',
        'discrete_postcasting', 'continuous_postcasting',
        'discrete_pattern_completion', 'continuous_pattern_completion',
        'bracket_matching', 'sorting_problem', 'cross_situation',
        'sinus_forecasting', 'chaotic_forecasting'
    ]
    
    for task_name in task_names:
        print(f"Evaluating on {task_name}...")
        
        # Load task
        task_data = sd.build_task(task_name, difficulty=difficulty)
        
        # Train model (simplified)
        model = MyModel(...)
        model.train(task_data['X_train'], task_data['Y_train'], task_data['T_train'])
        # If you want your model to learn all timesteps, including the ones that are not evaluated :
        # Comment the previous line and uncomment the following one
        # model.train(task_data['X_train'], task_data['Y_train'])

        # Predict on test set
        Y_pred = model.predict(task_data['X_test'])
        
        # Compute score
        score = sd.compute_score(
            Y=task_data['Y_test'],
            Y_hat=Y_pred,
            prediction_timesteps=task_data['T_test'],
            classification=task_data['classification']
        )
        
        results[task_name] = score
        print(f"  Score: {score:.4f}")
    
    return results

# Usage
# results = evaluate_model_on_all_tasks(your_model, difficulty='medium')
```

## 📈 Evaluation Metrics

- **Classification tasks**: Error rate (1 - accuracy)
- **Regression tasks**: Mean Squared Error (MSE)

Lower scores indicate better performance for both metrics.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use Stream Dataset in your research, please cite:

(paper in progress...)

```bibtex
@software{stream_dataset,
  title={Stream Dataset: Sequential Task Review to Evaluate Artificial Memory},
  author={Yannis Bendi-Ouis, Xavier Hinaut},
  year={2025},
  url={https://github.com/Naowak/stream-dataset}
}
```

## 🙏 Acknowledgments

- Inspired by classic sequence modeling datasets
- Built with NumPy and Hugging Face Datasets

## 📞 Support

- 🐛 Issues: [GitHub Issues](https://github.com/Naowak/stream-dataset/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Naowak/stream-dataset/discussions)

---

*Stream Dataset - Advancing sequence modeling evaluation, one task at a time.*

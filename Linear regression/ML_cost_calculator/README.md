# ML Cost Calculator 🧮

**A personal Python project demonstrating core machine learning concepts through practical implementation of loss functions.**  
Created to deepen my understanding of ML/AI fundamentals while building a reusable educational tool.

[![Python 3.13](https://img.shields.io/badge/Python-3.13+-blue?logo=python&logoColor=white)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)  

### Features ✨
- **Implemented Loss Functions**:
  - 📏 Mean Squared Error (MSE)
  - 📐 Mean Absolute Error (MAE)
  - 🔴 Binary Cross-Entropy
  - 🟣 Categorical Cross-Entropy  
- **Interactive Command Line Interface** 🖥️
- **Educational Focus** 📚 with clear mathematical implementation
- **Input Validation** 🛡️ for robust error handling
- **NumPy Powered** ⚡ for efficient computations

---

## Getting Started 🚀

1. **Clone & Install**:
   ```bash
   git clone https://github.com/JVBotelho/ML_cost_calculator.git
   cd ML_cost_calculator
   pip install -r requirements.txt
   ```

2. **Try the CLI**:
   ```bash
   $ python -m cost_functions.app
   > Choose function (1-4): 1
   > Enter true values: 2,4,6
   > Enter predicted values: 1.8,3.9,5.5
   Result: 0.0967 ✅
   ```

3. **Use in Python**:
   ```python
   from cost_functions import mean_absolute_error

   y_true = [2, 4, 6]
   y_pred = [1.8, 3.9, 5.5]
   print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")  # Output: MAE: 0.2000
   ```

---

## Project Purpose 💡
This project serves as:
- **Learning Journey** 🧭: Hands-on exploration of essential ML concepts
- **Concept Demonstration** 🎓: Concrete implementation of theoretical knowledge
- **Personal Reference** 📖: Reusable codebase for future projects
- **Skill Showcase** 💼: Evidence of practical Python/ML capabilities

---

## Development Notes 🔧
- **Python 3.13** compatible
- Requires NumPy for numerical operations
- Simple architecture for easy code review
- Focused on clean, documented implementations

---

## License 📄  
MIT License - See [LICENSE](LICENSE) for details.  

---

✨ **Built with curiosity and passion for machine learning!** ✨  
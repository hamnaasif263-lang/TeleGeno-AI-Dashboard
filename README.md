# TeleGeno AI Dashboard

> **Free and Open Source** Pharmacogenomic Analysis Dashboard

A Streamlit-based dashboard for pharmacogenomic (PGx) analysis, providing explainable medication guidance and emergency triage capabilities.

## ✨ Features

- **Pharmacogenomic Analysis**: Analyze SNP genotypes and their effects on drug metabolism
- **Multiple Input Methods**:
  - JSON SNP Input
  - VCF File Upload (with GT field extraction)
  - Demographics + Medical History (AI-powered risk prediction)
- **Emergency Triage Mode**: Quick assessment based on vitals and PGx predictions
- **Interactive Visualizations**: Charts and graphs for genotype distribution and effects
- **Medication Recommendations**: Explainable guidance based on CYP2C19 and APOE variants
- **Comprehensive Reporting**: Generate downloadable reports for patients

## 🆓 Free to Use in VS Code

This project is **completely free** and open source. You can:
- ✅ Use it in Visual Studio Code without any cost
- ✅ Modify and customize it for your needs
- ✅ Run it locally on your machine
- ✅ Deploy it for research or educational purposes

## 🚀 Getting Started with VS Code

### Prerequisites

- Python 3.8 or higher
- Visual Studio Code
- Git

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hamnaasif263-lang/TeleGeno-AI-Dashboard.git
   cd TeleGeno-AI-Dashboard
   ```

2. **Open in VS Code**:
   ```bash
   code .
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

The dashboard will open in your default browser at `http://localhost:8501`

## 🔧 VS Code Configuration

This project includes VS Code configuration files for an optimal development experience:

- **`.vscode/settings.json`**: Python environment and formatting settings
- **`.vscode/launch.json`**: Debug configurations for Streamlit
- **`.vscode/extensions.json`**: Recommended extensions

### Recommended VS Code Extensions

The following extensions are recommended (will be suggested automatically):
- **Python**: Python language support
- **Pylance**: Fast Python language server
- **Python Debugger**: Debugging support
- **autoDocstring**: Automatic docstring generation

## 📋 Usage

### Basic Workflow

1. **Select Input Method** in the sidebar:
   - JSON SNP Input: Paste SNP data in JSON format
   - Simulated VCF: Upload VCF files or use sample data
   - Demographics + History (AI): Fill out patient information for AI prediction

2. **Analyze Data**: Click the appropriate button to process the input

3. **Review Results**:
   - SNP genotype table
   - Visual insights (charts and graphs)
   - Medication recommendations

4. **Generate Reports**: Download comprehensive reports or CSV data

### Emergency Mode

Enable the "🚨 Enable Emergency Assessment" checkbox in the sidebar for quick triage:
- Capture patient vitals (or simulate)
- Get immediate risk assessment
- Receive actionable guidance

## 🧬 Supported Genes and Variants

- **CYP2C19** (rs4244285): Drug metabolism assessment
  - Normal, Intermediate, Poor metabolizer
  - Affects medications like clopidogrel

- **APOE** (rs429358): Risk assessment
  - Low Risk, Medium Risk, High Risk
  - Relevant for cardiovascular and cognitive health

## 📊 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests (if test suite is available)
pytest tests/
```

### Debugging in VS Code

1. Open the Debug panel (Ctrl+Shift+D / Cmd+Shift+D)
2. Select "Streamlit: Debug" from the dropdown
3. Press F5 to start debugging

## ⚠️ Important Notice

This dashboard is for **demonstration and research purposes only**. It is **not intended for clinical use** or medical decision-making. Always consult with qualified healthcare professionals for medical advice.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is free and open source software.

## 🔗 Links

- Repository: https://github.com/hamnaasif263-lang/TeleGeno-AI-Dashboard
- Issues: https://github.com/hamnaasif263-lang/TeleGeno-AI-Dashboard/issues

## 💻 System Requirements

- **OS**: Windows, macOS, or Linux
- **Python**: 3.8+
- **RAM**: 2GB minimum (4GB recommended)
- **Browser**: Chrome, Firefox, Safari, or Edge

## 📝 Notes

- The VCF parser extracts the GT (genotype) field from VCF files
- AI predictions are simulated based on demographics and medical history
- Emergency triage provides quick risk assessment before detailed PGx analysis

---

**Made with ❤️ for the genomics community**

# Quick Start Guide for VS Code

This guide will help you get the TeleGeno AI Dashboard running in VS Code in just a few minutes.

## 1. Open the Project in VS Code

```bash
code .
```

## 2. Install Recommended Extensions

When you open the project, VS Code will prompt you to install recommended extensions. Click **"Install All"** or install them manually:

- Python
- Pylance
- Python Debugger

## 3. Set Up Python Environment

### Option A: Automatic (Recommended)
Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and run:
```
Python: Create Environment
```
Choose "Venv" and select your Python interpreter.

### Option B: Manual
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Run the Application

### Method 1: Using Tasks (Easiest)
1. Press `Ctrl+Shift+B` (or `Cmd+Shift+B` on Mac)
2. Select "Run Streamlit App"
3. The app will open in your browser at http://localhost:8501

### Method 2: Using Debug Configuration
1. Go to the Debug panel (Ctrl+Shift+D / Cmd+Shift+D)
2. Select "Streamlit: Debug" from the dropdown
3. Press F5

### Method 3: Using Terminal
```bash
streamlit run app.py
```

## 5. Start Using the Dashboard

1. Enter a patient name in the sidebar
2. Choose an input method (JSON, VCF, or Demographics)
3. Analyze the data
4. View results and medication recommendations

## Common Tasks in VS Code

### Run the App
- **Keyboard**: `Ctrl+Shift+B` (Cmd+Shift+B on Mac)
- **Menu**: Terminal → Run Task → "Run Streamlit App"

### Debug the App
- **Keyboard**: F5
- **Menu**: Run → Start Debugging

### Install Dependencies
- **Task**: Ctrl+Shift+P → "Tasks: Run Task" → "Install Dependencies"
- **Terminal**: `pip install -r requirements.txt`

### Format Code
- **Task**: Ctrl+Shift+P → "Tasks: Run Task" → "Format Code with Black"
- **On Save**: Enabled by default in settings.json

## Tips

1. **Auto-save**: File → Auto Save (helps see changes immediately in Streamlit)
2. **Split View**: Right-click on app.py → "Split Editor Right" to view code and browser side-by-side
3. **Integrated Terminal**: Ctrl+` (backtick) to toggle terminal
4. **Git Integration**: Use the Source Control panel (Ctrl+Shift+G) for version control

## Troubleshooting

### Port Already in Use
If you see "Address already in use", kill the existing Streamlit process:
```bash
pkill -f streamlit
```

### Module Not Found
Make sure you've activated your virtual environment and installed dependencies:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Python Interpreter Not Found
1. Press Ctrl+Shift+P (Cmd+Shift+P on Mac)
2. Type "Python: Select Interpreter"
3. Choose the interpreter in `.venv/bin/python`

## Next Steps

- Read the [README.md](README.md) for detailed documentation
- Explore the code in `app.py`
- Try different input methods and features
- Customize the dashboard for your needs

---

**Happy Coding! 🎉**

Remember: This is **free and open source** software. Feel free to modify and adapt it to your needs!

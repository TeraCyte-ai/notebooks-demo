# TeraCyte Data Overview

Welcome to the TeraCyte Data Overview repository! This resource is designed for customers to explore, visualize, and analyze their TeraCyte single-cell imaging data using interactive notebooks.

---

## 🚀 Quick Start (Recommended: Google Colab)

The easiest way to use these notebooks is with **Google Colab**—no installation required!

### 1. Open in Google Colab

Click the badge below to launch the main notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TeraCyte-ai/notebooks-demo/blob/main/notebooks/teracyte_data_overview.ipynb)

Or, open the notebook manually in Colab:
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **GitHub** tab
3. Enter `TeraCyte-ai/data-overview` and select `notebooks/teracyte_data_overview.ipynb`

### 2. Follow the Notebook Instructions

1. **Update User Information**: In the notebook, look for cells marked with `# --- USER INPUT REQUIRED ---`. Enter your email and your sample's serial number as instructed.
2. **Select Experiment and Assay**: Set the experiment and assay IDs in the next cell to choose which data to analyze.
3. **Run Cells Step-by-Step**: Click the "Run" button (▶️) in the toolbar or press `Shift+Enter` to execute each cell in order. Follow the instructions in the notebook for data exploration, visualization, and analysis.
4. **Explore Data**: Use the interactive widgets to view metadata, images, and analysis results. You can download your data as CSV files for further use.

---

## 💻 Alternative: Run Locally

If you prefer to run the notebooks on your own computer:

1. **Clone the Repository**
	```bash
	git clone https://github.com/TeraCyte-ai/data-overview.git
	cd data-overview
	```
2. **Install Python (if needed)**
	- Make sure you have Python 3.8 or newer. [Download Python](https://www.python.org/downloads/)
3. **Set Up a Virtual Environment (Recommended)**
	```bash
	python -m venv venv
	source venv/bin/activate  # On Windows use: venv\Scripts\activate
	```
4. **Install Required Packages**
	```bash
	pip install -r requirements.txt
	```
5. **Launch Jupyter Notebook**
	```bash
	jupyter notebook
	```
	Then open `notebooks/teracyte_data_overview.ipynb` from the Jupyter interface.

---

## 💡 Tips

- If you are new to Jupyter Notebooks or Colab, see the [Jupyter Notebook Beginner Guide](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html) or [Google Colab Help](https://research.google.com/colaboratory/faq.html).
- For any issues or questions, please contact TeraCyte support.

---

## 📁 Repository Structure

- `notebooks/` — Example and analysis notebooks for customers
- `teracyte_notebooks_utils/` — Backend Python utilities (no need to modify)
- `requirements.txt` — List of required Python packages
- `assets/` — Images and logos used in notebooks

---

## 🛟 Support

If you need help or have questions, please reach out to your TeraCyte representative or email info@teracyte.ai.

Enjoy exploring your data!

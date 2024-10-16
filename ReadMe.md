# Market Basket Analysis with Apache Spark

This project performs market basket analysis on the **Online Retail Dataset** using Apache Spark’s FP-Growth algorithm. The goal is to identify frequent itemsets and association rules that help understand customer purchasing patterns.

## Features

- **Frequent Itemset Mining**: Identifies items often purchased together.
- **Association Rules**: Derives rules such as “if item A is purchased, item B is likely to be purchased.”
- **Visualization**: Top frequent itemsets are visualized using bar charts for insights.

## Dataset

The dataset used is the **Online Retail Dataset**, available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail). It contains transactions from a UK-based online retail store between 2010 and 2011.

## Requirements

Make sure you have the following dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```
# Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/Thin-Equation/SuperMarketBasketAnalysis.git
    cd SuperMarketBasketAnalysis
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure that **Apache Spark** is installed and properly configured on your local machine, or use a cloud-based environment like **Databricks**.

4. Download the **Online Retail Dataset** from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx) and place it in the project directory.

## Usage

To run the project, open your terminal and run the main script for Market Basket Analysis:

```bash
python BasketAnalysisProductRecommendation.py
```

## Output

The project identifies frequent itemsets and association rules using FP-Growth and visualizes the top 10 frequent itemsets in a bar chart.

import json
import os

def read_nb(path):
    with open(path, 'r') as f:
        return json.load(f)

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [source]
    }

combined_cells = []

# 1. Benchmark Viz
print("Processing benchmark_viz.ipynb...")
nb1 = read_nb("benchmark/benchmark_viz.ipynb")
combined_cells.append(create_markdown_cell("# Combined Analysis\n\n## 1. Benchmark Visualization"))
for cell in nb1['cells']:
    if cell['cell_type'] == 'code':
        # Fix paths
        # source is a list of strings
        new_source = []
        for line in cell['source']:
            # Naive replacement, but should work for the specific lines we saw
            new_line = line.replace('Path("results")', 'Path("benchmark/results")')
            new_source.append(new_line)
        cell['source'] = new_source
    combined_cells.append(cell)

# 2. NCU Viz
print("Processing ncu_viz.ipynb...")
nb2 = read_nb("profiling/ncu_viz.ipynb")
combined_cells.append(create_markdown_cell("## 2. NCU Profiling Visualization"))
for cell in nb2['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Replace BASE_PATH = Path(f"results/ncu/{TIMESTAMP}")
            # The original line might be: BASE_PATH = Path(f"results/ncu/{TIMESTAMP}")
            # We want: BASE_PATH = Path(f"profiling/results/ncu/{TIMESTAMP}")
            # We can simply replace "results/ncu/" with "profiling/results/ncu/"
            if "results/ncu/" in line:
                new_line = line.replace("results/ncu/", "profiling/results/ncu/")
            else:
                new_line = line
            new_source.append(new_line)
        cell['source'] = new_source
    combined_cells.append(cell)

# 3. NSYS Viz
print("Processing nsys_viz.ipynb...")
nb3 = read_nb("profiling/nsys_viz.ipynb")
combined_cells.append(create_markdown_cell("## 3. NSYS Profiling Visualization"))
for cell in nb3['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Replace BASE_PATH = Path(f"results/nsys/{TIMESTAMP}")
            if "results/nsys/" in line:
                new_line = line.replace("results/nsys/", "profiling/results/nsys/")
            else:
                new_line = line
            new_source.append(new_line)
        cell['source'] = new_source
    combined_cells.append(cell)

# Create combined notebook struct
combined_nb = {
    "cells": combined_cells,
    "metadata": nb1.get("metadata", {}),
    "nbformat": nb1.get("nbformat", 4),
    "nbformat_minor": nb1.get("nbformat_minor", 5)
}

with open("combined_analysis.ipynb", 'w') as f:
    json.dump(combined_nb, f, indent=1)

print("Done. Created combined_analysis.ipynb")

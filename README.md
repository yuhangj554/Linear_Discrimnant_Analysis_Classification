# LDA Classification Implementation

## Overview
This repository contains an implementation of **Linear Discriminant Analysis (LDA) Classification** in Python. LDA maximizes between class scatter and minimizes the within class scatter of the dataset on converted space, bringing dataset more separated among the classes the in reduced vector space. This preserves
the identity of each class upon dimension reduction, while also provides clearer boundaries upon classification
needs. Besides, since the reductions\conversions are all done by linear transformation, the discriminant vectors
 are all in linear forms. 
 The package provides tools for dimensionality reduction and supervised classification for multivariate and multi-class data. 

## Features
- Multi-class LDA implementation
- Dimensionality reduction to 1-D (Line):\
 -> Though LDA could reduce the dimension in to at most C - 1 dimensions (C is class count), 1-D dimension much easier and more concise for classifications.\
 -> Higher dimensions reduction is under development.
- Training and prediction functions
- Computation mostly done by python built-in packages and data structures, aimed for run-time efficiency and memory efficiency
- Two differnent classification methods provide: midpoints and z-scores

## Example Usage
```python
from LDA import LDA

# Example dataset
X = [[4, 2], [2, 4], [2, 3], [3, 6], [4, 4], [9, 10], [6, 8], [9, 5], [8, 7], [10, 8]]
Label = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]

# Optional: Redefine the categories involved
categories = [[0],[1,2]]

# Initialize and train LDA model
lda = LDA()
lda.train_discriminant_vector(X, y)

# Classify all data and generate classification rule
lda.classify_all()

# Print out the classification result
lda.Classification_Result()

```

## Benchmark & Performance
This implementation is optimized for efficiency and has been tested on real-world datasets. Performance comparisons against other classification techniques will be provided soon.



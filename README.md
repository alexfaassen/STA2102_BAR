# STA2102_BAR

Self-study of a Bayesian approach to time series forecasting by creating a Python class to facilitate a standard workflow.

## Overview

This repository is part of a course project for STA2102 and is aimed at exploring Bayesian Autoregressive (BAR) models. The goal is to design and implement a Python package that provides a consistent and user-friendly workflow for modeling univariate time series using a Bayesian framework.

The project serves as both:
- A self-study in Bayesian time series analysis (using `PyMC`), and
- A toolbox to simplify Autoregressive model building, inference, and prediction using Bayesian methods.

## Contents

- **Module:** `bar.py` - all of the code is included here.
- **Vignette:** `vignette.ipynb` - a guided example using synthetic AR(3) data.

## Reflection + A Note for the Professor

Hi Professor,

I just wanted to say that I really enjoyed the course! It challenged my understanding of familiar concepts while introducing me to new ideas, inspiring me to try my hand with `PyMC` for this Course Project. While I'm pretty proud of it as a first-time user of a probabalistic programming language for Bayesian inference, I'm sure there are inefficiencies and missed cases.

All the best,

Alex Faassen

## References

- [PyMC](https://www.pymc.io/) - Probabilistic programming in Python
- [ArviZ](https://www.arviz.org/) - Exploratory analysis of Bayesian models
- [Conducting Time Series Bayesian Analysis using PyMC](https://charlescopley.medium.com/conducting-time-series-bayesian-analysis-using-pymc-22269aeb208b) - Simple Bayesian approach to Time Series Analysis with PyMC
- [Bayesian Vector Autoregression in PyMC](https://www.pymc-labs.com/blog-posts/bayesian-vector-autoregression/) - Bayesian Vector Autoregression (BVAR) with PyMC
- [The Bayesians are Coming to Time Series](https://www.youtube.com/watch?v=P_RnURpkgdE) - Lecture on BVAR intuition
- [Machine Learning with 10 Data Points - Or an Intro to PyMC3](https://www.youtube.com/watch?v=SP-sAAYvGT8) - Video Tutorial on PyMC
- [Bayesian vector autoregressive models](https://kevinkotze.github.io/ts-9-bvar/) - Report on BVAR Theory
- [Bayesian Methods for Hackers](https://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/) - Book on Bayesian methods using PyMC, Chapter 2 useful for PyMC fundamentals.
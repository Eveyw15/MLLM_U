# Project status

## Topic

Multimodal privacy unlearning in MLLMs

## Fixed focus

Cross-modal residual leakage / backpath leakage

## What we are trying to show

1. baseline evaluation reveals leakage
2. lightweight selective suppression can reduce leakage
3. utility should not be completely broken

## Current code plan

- minimal_eval_unlok.py
- analyze_channels.py
- suppression.py
- eval_with_suppression.py

## Immediate next step

- read current notebook
- extract stable baseline evaluator
- then implement channel sensitivity analysis

## Important constraints

- honours timeline is tight
- prefer small runnable method
- avoid heavy retraining unless explicitly requested

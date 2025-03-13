# Agentic Ablation

A framework for automated code ablation studies using LLM agents. This project uses a multi-agent workflow to analyze, modify, and test code for ablation studies.

## Overview

The system uses multiple specialized agents in a workflow:
- **CodeGeneratorAgent**: Generates code modifications for ablation
- **ExecutorAgent**: Tests and validates the modified code
- **ReflectorAgent**: Analyzes errors and suggests improvements
- **AnalyzerAgent**: Provides final analysis of the ablation results

## Project Structure 
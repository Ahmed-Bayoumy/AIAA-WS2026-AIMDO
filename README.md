# 3rd AIAA-WS on on Multifidelity Methods for Design and Uncertainty Quantification: AI-Driven Multidisciplinary Design Optimization Session

This repository contains the materials for the **AI-Driven MDO** Session, focused on practical and research-oriented workflows for AI-MDO.

## Workshop Summary

The session introduces how AI/ML methods can accelerate multidisciplinary design optimization (MDO), especially in expensive blackbox settings. It combines:

- Core MDO concepts and challenges
- Uncertainty-aware AI-assisted MDO
- Agentic AI orchestration ideas for MDO pipelines
- Practical case studies and exercises (Sellar, Airfoil SU2+OMADS, SBJ/DMDO)
- Hands-on tool ecosystem for workshop exercises

## Session Flow

1. **Introduction to MDO**
- What MDO is and why it is hard
- Blackbox optimization challenges
- Coupled multidisciplinary complexity

2. **AI/ML in MDO**
- Surrogate-Based Optimization (SBO) workflow (XDSM)
- Gaussian Process recap and uncertainty for active learning
- Bayesian Neural Networks (BNN) for probabilistic prediction
- Physics-Informed Neural Networks (PINNs)
- Agentic AI orchestration concepts
- Bayesian Deep-Ensemble of NNs
- Concurrency voids in Parallel MDO (if time allows)

3. **Methodology and Validation**
- AI-assisted constrained MDO formulation
- Risk estimates and hybrid simulation/prediction policy
- Wing-design methodology and validation results

4. **Case Studies**
- **Sellar benchmark** with XDSM and validation figure
- - **Airfoil shape optimization (XFOIL + Pytorch + OMADS)**:
  - Objective: maximize lift (implemented as minimize `-CL/CD`)
  - Constraint: drag threshold (`|CD| - 0.006 <= 0`, `Max_Stress - 30MPa <= 0`)
  - Workflow: Geometry Parameterization -> Aerodynamics (XFOIL) -> Structural Analysis -> OMADS Search
- **Airfoil shape optimization (SU2 + Pytorch + OMADS)**:
  - Objective: maximize lift (implemented as minimize `-CL`)
  - Constraint: drag threshold (`|CD| - 0.006 <= 0`)
  - Workflow: Geometry Parameterization -> Meshing -> SU2 CFD -> Structural Analysis -> OMADS Search
- **Supersonic Business Jet (SBJ)** using distributed MDO/NHATC context

5. **Tools and Exercises**
- DMDO, OMADS, samplers, Pytorch, DeepXDE
- Exercise set centered on LLM-in-MDO, Sellar, and coupled SBJ workflows

6. **Future Directions**
- Near-term and long-term outlook for AI-enabled MDO
- Open challenges: scalability, extrapolation, certification, data scarcity, human factors

## Main Files

- `AIAA-WS2026-AIMDO-slides.pdf`: slides
- `AIAA_WS2026_SBJ_DMDO.ipynb`: Supersonic Business Jet NHATC Notebook
- `Exercises`: Python scripts 
    - `LLM_AIMDO`
    - `PINN`
    - `MDO/Sellar`
    - `MDO/UAAI`
    - `MDO/Wing_MDO`
    - `MDO/Wing_MDO_XFOIL`

## Installation Instructions

From the repository root:

```powershell
pip install omads dmdo samplersLib torch deepxde
```



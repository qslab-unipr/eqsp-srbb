# Exact Quantum State Preparation with the Standard Recursive Block Basis (EQSP-SRBB)

## Description
The Standard Recursive Block Basis (SRBB) has been used to implement diagonal Uniformly Controlled Gates (UCGs) within the traditional framework for Quantum State Preparation (QSP), which consists of a UCG ladder structure over the entire quantum register (*arXiv:2503.13647*). The corresponding QSP algorithm, named **QSP-SRBB**, works by means of a Quantum Neural Network (QNN) and is available on the GitHub repository *https://github.com/qslab-unipr/aquaman-sp*.

This repository documents the transition to the *exact* version of **QSP-SRBB**, thanks to the definition of the analytical parametric map that identifies the exact rotation angles for each UCG layer. This new exact QSP algorithm, based on the SRBB-decomposition and named **EQSP-SRBB**, represents a computational framework for traditional QSP without training limitations of the variational approach (local minima and barren plateaus). This work is linked to the paper submitted to RC26, awaiting review. 

## Requirements
The project has been implemented using *Python* 3.10 and *PennyLane* 0.40. The implementation also makes use of the *numpy* and *matplotlib* modules.

# py-OATS
py-**O**nsager **A**nalysis of **T**ransport in amorphous **S**olids

Author: Vir Karan

A python package to analyze correlated ionic transport in amorphous solids. This is based on the original repo by Kara Fong (@kdfong), linked here: https://github.com/kdfong/transport-coefficients-MSD.

This package provides functions to compute diffusive transport coefficients, as per the Onsager transport framework, for solids from MD simulations. Reading outputs from AIMD simulations (e.x. from VASP), and LAMMPS MD are currently supported. 
Since the framework is different from what is typically applied for electrolytes with solvated ions, refer to the SI of [(this paper)](https://arxiv.org/abs/2501.08560). 

Additionally, it is possible to compute transport across an amorphous solid-solid reaction interface using the computed transport coefficients. Refer to [(this paper)](https://arxiv.org/abs/2501.08560) for an example. 

If you found this useful, consider citing the following:
@inproceedings{Karan2025IonCE,
  title={Ion correlations explain kinetic selectivity in diffusion-limited solid state synthesis reactions},
  author={Vir Karan and Max C. Gallant and Yuxing Fei and Gerbrand Ceder and Kristin A Persson},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:275544572}
}

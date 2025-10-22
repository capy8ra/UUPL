# Towards Uncertainty Unification: A Case Study for Preference Learning [RSS 2025]
Arxiv: https://arxiv.org/abs/2503.19317v1

Authors: [Shaoting Peng](https://shaotingpeng.github.io/), [Haonan Chen](https://haonan16.github.io/), [Katie Driggs-Campbell](https://krdc.web.illinois.edu/)

Project page: https://sites.google.com/view/uupl-rss25/home
#

## Abstract
Learning human preferences is essential for human-robot interaction, as it enables robots to adapt their behaviors to align with human expectations and goals. However, the inherent uncertainties in both human behavior and robotic systems make preference learning a challenging task. While probabilistic robotics algorithms offer uncertainty quantification, the integration of human preference uncertainty remains underexplored. To bridge this gap, we introduce uncertainty unification and propose a novel framework, uncertainty-unified preference learning (UUPL), which enhances Gaussian Process (GP)-based preference learning by unifying human and robot uncertainties. Specifically, UUPL includes a human preference uncertainty model that improves GP posterior mean estimation, and an uncertainty-weighted Gaussian Mixture Model (GMM) that enhances GP predictive variance accuracy. Additionally, we design a user-specific calibration process to align uncertainty representations across users, ensuring consistency and reliability in the model performance. Comprehensive experiments and user studies demonstrate that UUPL achieves state-of-the-art performance in both prediction accuracy and user rating. An ablation study further validates the effectiveness of human uncertainty model and uncertainty-weighted GMM of UUPL.

#

## Usage
We provide code for the implmentation of our method and all three other baselines on simulation 2: Tabletop importance task. They can be easily modified to test other functions. `sim_UUPL.py` provides accuracy matrics and visualization for our method, and `sim_baseline1.ipynb`, `sim_baseline2.ipynb`, `sim_baseline3.py` provide results of other methods.

#

## BibTex
```
@INPROCEEDINGS{PengS-RSS-25, 
    AUTHOR    = {Shaoting Peng AND Haonan Chen AND Katherine Rose Driggs-Campbell}, 
    TITLE     = {Towards Uncertainty Unification: A Case Study for Preference Learning}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2025}, 
    ADDRESS   = {LosAngeles, CA, USA}, 
    MONTH     = {June}, 
    DOI       = {10.15607/RSS.2025.XXI.091} 
}
```

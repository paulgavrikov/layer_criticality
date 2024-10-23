# How Do Training Methods Influence the Utilization of Vision Models?
Paul Gavrikov, Shashank Agnihotri, Margret Keuper, and Janis Keuper

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

Presented at: NeurIPS 2024 IAI Workshop: Interpretable AI: Past, Present and Future

[ArXiv](https://arxiv.org/abs/2410.14470)<!-- | [Paper]() | [HQ Poster]() -->


Abstract: *Not all learnable parameters (e.g., weights) contribute equally to a neural network’s decision function. In fact, entire layers’ parameters can sometimes be reset to random values with little to no impact on the model’s decisions. We revisit earlier studies that examined how architecture and task complexity influence this phenomenon and ask: is this phenomenon also affected by how we train the model? We conducted experimental evaluations on a diverse set of ImageNet-1k classification models to explore this, keeping the architecture and training data constant but varying the training pipeline. Our findings reveal that the training method strongly influences which layers become critical to the decision function for a given task. For example, improved training regimes and self-supervised training increase the importance of early layers while significantly under-utilizing deeper layers. In contrast, methods such as adversarial training display an opposite trend. Our preliminary results extend previous findings, offering a more nuanced understanding of the inner mechanics of neural networks.*


[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

<p align="center">
  <img src="assets/teaser.png" width="80%" />
</p>

## Reproduce our results

Coming soon.


## Citation 

If you find our work useful in your research, please consider citing:

```
@inproceedings{
    gavrikov2024how,
    title={How Do Training Methods Influence the Utilization of Vision Models?},
    author={Paul Gavrikov and Shashank Agnihotri and Margret Keuper and Janis Keuper},
    booktitle={NeurIPS 2024 Workshop on Interpretable AI: Past, Present and Future},
    year={2024},
    url={https://openreview.net/forum?id=zJFvjdW9JS}
}
```

### Legal
This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

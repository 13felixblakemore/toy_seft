This project explores sparse fine-tuning for neural networks, comparing:

- Static sparse fine-tuning (fixed mask after pruning)
- SEFT-style dynamic sparsity (drop-and-grow during training)

The goal was to better understand how sparse connectivity evolves during training, and why many sparse methods do not translate into real computational efficiency. 
Modern pruning techniques can make neural networks sparse, reducing parameter count. However, sparse models often suffer a drops in accuracy, and rely on dense computation under the hood.

This project implements a simplified version of Sparsity Evolution Fine-Tuning (SEFT) that drops weights with the smallest magnitude, and grows weights with the largest gradients.


Sparse weights alone do not imply efficient computation.

Even with high sparsity:
- Parameters remain dense tensors
- Gradients are still fully computed
- Optimiser state remains dense

This highlights the gap between sparse models (logical) and sparse systems (computational).

To run: python seft.py

Add all scripts to run full unlearn and evaluation pipeline here.

The folders here follow a layered structure.

Layer 1:
- Indepedent: Unlearn one concept for baseline (Not Continual Unlearning)
- Sequential: Unlearn concepts sequentially by starting from the previously unlearned model for each unlearning request
- Simultaneous: Unlearn concepts restarting at the base model for each unlearning request

Layer 2:
- Object: Unlearn object domain (UnlearnCanvas)
- Style: Unlearn style domain (UnlearnCanvas)
- Celebrity: Unlearn celebrity domain

Layer 3:
- Base: No regularizers Applied
- {Insert Regularizer}: Regularizer used

Layer 4:
- {Unlearning Method}: Unlearning method used
# Neural Network From Scratch (NumPy only)

A single-hidden-layer neural network built entirely in NumPy — no PyTorch, no TensorFlow, no autograd. Forward pass, backpropagation, and the weight updates are all written out by hand. This isn't meant to be a good model. It's meant to prove I understand what's happening inside one before I rely on a framework to do it for me.

## The problem

Dataset: [Garment Worker Productivity](https://www.kaggle.com/datasets/adnananam/garments-worker-productivity) (Kaggle). Each row is a shift for a team in a garment factory, with features like team size, overtime, incentive pay, idle time, and number of style changes. The target is `actual_productivity`, a continuous score.

I turned this into binary classification: was the team's actual productivity at or above 0.75 (a reasonable "on target" threshold) or not.

## What the network does

- 12 input features, normalised to [0, 1] with min-max scaling — fitted on the training split only, then applied to validation and test, so no information leaks across splits.
- A 70/15/15 train/validation/test split, done with plain `numpy` (shuffle indices, slice) since the rest of the project avoids `scikit-learn` on principle.
- One hidden layer, 20 neurons, sigmoid activation. One output neuron, sigmoid, giving a probability.
- Backpropagation implemented manually — the gradient at the output layer, the error propagated back through the hidden layer, and both weight matrices updated with plain full-batch gradient descent (learning rate 0.002).
- Trained for 149,000 epochs, with training and validation loss (mean absolute error) both logged every 1,000 epochs.

## Results

**Training and validation loss over training:**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 6,000 | 0.4031 | 0.4193 |
| 50,000 | 0.2605 | 0.3007 |
| 100,000 | 0.1959 | 0.2435 |
| 149,000 | 0.1591 | 0.2212 |

**Test set performance (held out, never seen during training):**

| Metric | Value |
|---|---|
| Accuracy | 0.7556 |
| Precision | 0.8452 |
| Recall | 0.6961 |
| F1 | 0.7634 |

![Train vs. validation loss](attachment:loss_curve.png)

## What the curves actually show

Train and validation loss track each other closely early on (gap of ~0.016 at epoch 6,000), but the gap widens to ~0.062 by epoch 149,000 as train loss keeps falling faster than validation loss. That's a genuine, mild overfitting signal — the network is picking up some patterns specific to the training split rather than fully generalisable ones. It's not severe: validation loss never reverses and starts climbing in this range, it just decelerates relative to train loss. There's also a small non-monotonic wobble in training loss around epochs 63,000 and 113,000–115,000 (a brief uptick before resuming its decline) — a minor instability you'd expect from plain gradient descent at a fixed learning rate, and a natural candidate to fix with momentum or a decay schedule.

On the actual test set, the model favours precision over recall: it's right about 85% of the time when it predicts a team hit the productivity target, but misses roughly 30% of the teams that actually did. For a single hidden layer with no feature engineering beyond scaling, that's a reasonable, honest result — not a benchmark-beating model, but a working one, on data it never touched during training.

## Known limitations

- **Full-batch gradient descent, no mini-batching or momentum.** Fine for this dataset size (~1,200 rows), but the training wobble noted above is a direct symptom of this choice.
- **MAE loss, not cross-entropy.** Simpler to derive and code by hand, but cross-entropy is the theoretically correct loss for a classification problem and would likely train faster and more stably.
- **Fixed 0.75 threshold used twice** — once to binarise the continuous productivity score into a label, and again to threshold the predicted probability. Works, but the second threshold should ideally be tuned independently (e.g. via the ROC curve) rather than reused from the first.
- **`Quarter5` doesn't exist in the real data** — a harmless leftover in the mapping dictionary that maps to nothing.

## Your handwritten notes go here

*(Scan/photograph your derivation of the backpropagation equations — the chain rule steps from the output error back through the hidden layer, matching them to the `d_predicted_output` / `error_hidden_layer` / `d_hidden_layer` lines in the code. A short paragraph next to it explaining, in your own words, why the sigmoid derivative shows up the way it does in each line would tie the maths directly to the implementation.)*

## What I'd do differently next time

- Try mini-batch gradient descent with momentum, and compare convergence smoothness against the wobble seen here.
- Switch to binary cross-entropy loss.
- Widen the train/val loss gap analysis into an actual early-stopping rule, rather than just training for a fixed number of epochs and inspecting the curve afterward.
- Compare this from-scratch version against a two-line `scikit-learn`/`Keras` equivalent on the same split, to sanity-check the from-scratch version isn't quietly worse due to an implementation bug rather than just being a simpler model.

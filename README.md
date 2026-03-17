# GirlOrBoyNeuralNetwork

An early exploratory computer vision project I built around 2017.

At the time, I was trying to understand neural networks from the inside rather than only use ready-made frameworks. I wanted to build the whole pipeline myself, end to end: collect data, extract features, train a small neural network written from scratch, and see whether something non-trivial could emerge from it.

This project grew out of a very simple question:

**How much signal is there in facial landmark geometry alone?**

More specifically: if you take only 68 facial landmarks, normalize them, turn them into a compact handcrafted feature vector, and feed that into a very small feedforward neural network, can it learn a useful binary classification signal from noisy real-world social-network avatars?

For me, this was less about making a production-ready classifier and more about curiosity, intuition, and engineering. It was one of those early projects where the magic really hit me: I had a weak old CPU-only machine, no GPU, a self-written neural net, a live loss curve, and at some point the loss started going down.

That felt like alchemy.

---

## What this repository contains

This repository contains an end-to-end experimental pipeline:

- collecting social-network avatar photos for an exploratory experiment
- detecting faces and extracting 68 facial landmarks
- normalizing landmark geometry
- converting landmarks into a compact handcrafted feature vector
- training a small feedforward neural network written from scratch in Java
- saving/loading learned weights
- serving inference through a small Python/Flask app

Historically, the internal name of the project was **BoyOrGirl**. I keep the old repository name as part of the project's history, but today I would describe the task more carefully than I did in 2017.

---

## Core idea

Instead of feeding raw pixels into a large model, I used **facial landmark geometry**.

For each detected face:

1. `dlib` was used to detect a face and extract **68 landmark points**
2. the landmarks were normalized:
   - rotated to make the eyes horizontal
   - scaled using face width
3. for each of the 68 points, distances to two anchor landmarks were computed
4. this produced a **136-dimensional handcrafted feature vector**
5. that vector was fed into a small feedforward neural network

So the question was not "can a huge vision model solve this?" but rather:

**can a tiny geometry-based pipeline learn something meaningful from landmarks alone?**

---

## Dataset and cleaning

The data for this experiment came from **real-world social-network avatars collected at the time for an exploratory project**.

That made the task interesting, but also messy.

The labels were not manually authored ground truth about a person's biological sex or gender identity. In practice, the target was closer to a **binary profile label attached to a social-media avatar**, which is a much noisier and more limited thing.

A lot of label noise came from the fact that avatars often did not cleanly correspond to the account owner's profile label. For example, some avatars contained:

- couples
- children
- fictional characters
- pets
- cars
- other non-face or ambiguous images

Images for which the face detector / landmark extractor failed were discarded automatically, which implicitly filtered out many non-frontal or otherwise hard-to-process faces.

After that, the dataset was also cleaned manually to remove obvious mismatches and non-representative images. This part turned out to be surprisingly important and surprisingly revealing: a real dataset is never just "data" - it is also culture, habits, noise, edge cases, and human judgment.

One interesting lesson from this project was that the problem was at least as much **data-centered** as **model-centered**.

---

## Model

The classifier itself is intentionally small.

From the Java training code:

- input size: **136**
- architecture: **136 -> 10 -> 10 -> 1**
- implementation: **custom feedforward neural network written from scratch in Java**
- training setup:
  - mini-batch training
  - L2 regularization
  - saved weight snapshots
  - experiments with difficult / misclassified examples

This project was built on modest hardware - an old CPU-only machine, no GPU - which was part of the fun. I wanted to see how far I could get with a lightweight pipeline and very limited compute.

---

## Result

In my experiments, the model reached **about 92% accuracy on a 5,000-image test split**.

That result should be read in the spirit of an exploratory project, not as a polished benchmark.

There was no cross-validation, and the dataset itself was noisy and imperfect. Also, because the data came from real-world avatars, the task was shaped heavily by:

- image quality
- face size in the image
- degree of frontal pose
- landmark quality
- cultural/stylistic biases in how faces are presented online
- residual label noise even after cleaning

Interestingly, after manual cleaning, the score improved less than I initially expected. That was a useful lesson by itself: beyond a certain point, the bottleneck was not only obvious garbage in the dataset, but also the limits of the representation and the task formulation.

---

## Side experiment: can a human guess from landmarks alone?

As a small side experiment, I also built a simple interactive game that displayed only the normalized facial landmark dots and asked the user to guess the label.

This was partly for fun, and partly to build intuition.

My own expectation before training was that the neural net might reach something like **55-60%**. I was genuinely surprised when the model learned a much stronger signal than that from landmark geometry alone.

That contrast - a human struggling to infer much from landmark dots alone, while a tiny model still extracts usable structure - was one of the most memorable parts of the project for me.

---

## Reproducibility

One reason I still like this repository is that it is not just a vague memory - it is still reproducible.

The project can still be run, and the same pipeline still works. The dataset artifacts were preserved, and the Java side in particular has remained pleasantly stable over time.

So although this is an old project, it is not just a historical sketch: it is still a working experimental pipeline.

---

## Why I still think this project matters

I do not view this project as a claim of novelty or a grand scientific contribution.

What I still value in it is something simpler:

- I built a full CV pipeline myself
- I learned by implementing things directly
- I got a real feel for the "psychology" of neural networks
- I saw firsthand how much results depend on data quality and task framing
- I learned that even very small models can extract surprisingly rich structure from carefully chosen features

For me, this project was part engineering exercise, part research toy, part intuition-building tool.

---

## Limitations and ethical note

This is an old exploratory project, and I would frame the task more carefully today than I did in 2017.

A few important limitations:

- the labels were noisy proxy labels tied to social-network profiles and avatars, not clean ground truth
- the task should **not** be interpreted as inferring a person's gender identity
- real-world sex/gender-related categories are socially and ethically more complex than a binary avatar-label classification setup
- the dataset likely contains cultural, stylistic, and sampling biases
- this project was never intended for deployment in any real decision-making or demographic inference setting

I keep this repository as a technical and historical project: an example of early curiosity-driven computer vision work, not as a model for real-world classification of people.

---

## Acknowledgements

Special thanks to Diana for spending a huge amount of time helping with dataset cleaning and manual inspection.

Also, more broadly, this project belongs to a stage of my life where curiosity, experimentation, and the people around me mattered a lot.
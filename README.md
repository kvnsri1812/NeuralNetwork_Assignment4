# CS5720 – Home Assignment 4: Generative AI, Ethics and Fairness

## Student Information
* 
* Course: CS5720 – Neural Networks and Deep Learning
* Term: Summer 2025
* University: University of Central Missouri

---
# Name: Komatlapalli Venkata Naga Sri

# Student ID: 700773763


---

## 1. GAN Architecture – Adversarial Training

In a Generative Adversarial Network (GAN), two models compete in a feedback loop:

* Generator (G): Learns to generate fake data that resembles real data
* Discriminator (D): Learns to distinguish real data from fake data

As training progresses:

* The generator improves at producing realistic samples
* The discriminator becomes better at identifying fakes

This adversarial process pushes both models to improve. The generator tries to fool the discriminator, and the discriminator tries to avoid being fooled.

---

## 2. Ethics and AI Harm – Representational Harm

Example: Facial recognition systems sometimes perform poorly on darker-skinned women due to underrepresentation in training data. This leads to representational harm.

Mitigation strategies:

1. Use diverse datasets that represent all demographics fairly
2. Conduct regular bias audits and involve a diverse team in model development

---

## 3. Programming Task – GAN on MNIST

Framework: PyTorch
Dataset: MNIST (handwritten digits)
Visualization: matplotlib, torchvision

Generator architecture:

* Fully connected layers with ReLU and Tanh activations

Discriminator architecture:

* Fully connected layers with ReLU and Sigmoid

Training process:

* Trained for 100 epochs
* Used alternating updates for generator and discriminator
* Losses monitored using Binary Cross Entropy

Output images saved at:

* Epoch 0: generated\_epoch\_0.png
* Epoch 50: generated\_epoch\_50.png
* Epoch 100: generated\_epoch\_100.png

Loss plots were generated to show generator and discriminator performance over time.

---

## 4. Data Poisoning Simulation – Sentiment Classifier

A sentiment classifier was trained on movie review data. A data poisoning attack was simulated by flipping labels of reviews containing the phrase "UC Berkeley."

Before poisoning:

* Accuracy: 89 percent

After poisoning:

* Accuracy: 74 percent

The attack reduced the classifier's ability to correctly predict positive sentiment, particularly on the targeted phrases.

---

## 5. Legal and Ethical Concerns in Generative AI

Issue 1: Memorizing Private Data

* Legal concern: Violates privacy laws such as GDPR
* Ethical concern: Users may not have consented to their data being used

Issue 2: Generating Copyrighted Material

* Legal concern: May breach intellectual property rights
* Ethical concern: Disrespects authorship and creator rights

Conclusion: Generative AI should avoid training on private or copyrighted data to ensure ethical use and legal compliance.

---

## 6. Bias and Fairness Tools – Aequitas

Metric: False Negative Rate (FNR) Parity

What it measures:

* Ensures different demographic groups have similar false negative rates

Why it matters:

* In fields like healthcare or hiring, high false negatives mean missing opportunities or critical diagnoses

How models fail:

* Poor representation in training data
* Bias inherited from historical patterns



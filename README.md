# Variational Autoencoder (VAE) Implementation

This repository contains a personal implementation of the **Variational Autoencoder (VAE)**, a generative model introduced in the  paper *"Auto-Encoding Variational Bayes"* by Kingma and Welling (2014). The VAE combines deep learning with probabilistic modeling, enabling efficient inference and learning in models with intractable posterior distributions.

---
## 📜 **Background**

The Variational Autoencoder is a probabilistic model used to:
- Learn efficient representations of data in a continuous latent space.
- Generate new data points by sampling from the learned latent space.
- Solve inference tasks like denoising, inpainting, or anomaly detection.

The key innovation of the VAE lies in:
1. **The Reparameterization Trick:** A method to reparameterize latent variable distributions to make gradient-based optimization feasible.
2. **The Evidence Lower Bound (ELBO):** A tractable objective function that approximates the intractable log-likelihood of data.

---

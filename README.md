# Geophysical-Waveform-Inversion
Estimate subsurface properties — velocity maps — from on seismic waveform data 

{Source: https://www.kaggle.com/competitions/waveform-inversion}

In this code, estimate subsurface properties— velocity maps—from on seismic waveform data. Known as Full Waveform Inversion (FWI), this process could lead to more accurate and efficient seismic analysis, making it more useful in a variety of fields.

Imagine a doctor analyzing an ultrasound scan—not just to see a blurry outline but to achieve a clearer, more detailed image crucial for an accurate diagnosis. That's the challenge geophysicists face when imaging the Earth's hidden structures. Beneath the surface lie vital resources, potential hazards, and clues to our planet's history—all requiring sharper, more precise subsurface imaging to be fully understood and effectively utilized.

Full Waveform Inversion (FWI) is the key to unlocking these secrets. This powerful technique, crucial for energy exploration, carbon storage, medical ultrasound, and advanced material testing, aims to build a detailed picture of the subsurface by analyzing the entire shape of seismic waves. But current methods are hindered by a noisy reality.

Traditional physics-based approaches are accurate, but incredibly slow and prone to errors when the signal is weak from noisy data. Pure machine learning solutions are faster, but require vast amounts of labeled data and often fail to generalize to new, unfamiliar "signal".

All data (>200 GB) for the code is presented in the source competition page above. The repository contains only the main program code and the trained weights of the model.

Briefly about the code:

- Import dependencies and configure the environment

Libraries are connected: tensorflow, keras, numpy, pandas, matplotlib, cv2, etc.

- Data loading and preprocessing

Reading images and their masks.

Resizing, normalization.

Creating data generators or a tf.data pipeline to feed into the model.

- Building a U-Net model

Encoder (downsampling): Conv2D + BatchNorm + Activation + MaxPooling blocks.

Decoder (upsampling): Conv2DTranspose or UpSampling + skip connections with corresponding encoder levels.

Output layer (1 channel) with sigmoid.

- Compilation and training

Loss function: cross entropy.

Metrics: Accuracy.

Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.

- Inference and submission formation

Mask prediction based on test data.

Post-processing of masks (thresholding, morphology).

Creating a submission file (csv) with the required format for Kaggle.

- Visualization of the results

Showing some images with ground-truth and predicted masks.

Loss/metric charts by epoch.

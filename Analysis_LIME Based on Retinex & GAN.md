# Low-light Image Enhancement Algorithm Based on Retinex and Generative Adversarial Network
## Intro
##### 1. Zero-DCE is non-reference, i.e. is independent of paired and unpaired training data, thus avoid the risk of overfitting.
    1. Non-reference is achieved by a set of specially designed non-reference loss functions.
    2. The loss function consider multiple factors of light enhancement, including spatial consistency loss, exposure control loss, color constancy loss, and illumination smoothness loss.
##### 2. The author designs image-specific curve that is able to approximate pixel-wise and higher-order curves by iteratively applying itself.
    * High-order curves are pixel-wise adjustment on the dynamic range of input to obtain an enhanced image as these curves maintain the range of enhanced image and remain contrast of neighboring pixels.
    * Curves are differentiable so that we can learn adjustable parameters of curves through a CNN.
##### 3. Zero-DCE supersedes state-of-art performance both qualitatively and quantitatively, e.g. CNN-based method and EnlightenGAN.
##### 4. Zero-DCE is capable of improving high-level visual tasks, e.g. face detection, and it is lightweight.

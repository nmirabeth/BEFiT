# BEFiT

This is the official repository of "One Embedding to Predict Them All: Visible and Thermal Universal Face Representations for Soft Biometric  Estimation via Vision Transformers" accepted at IEEE Biometrics Workshop at CVPR 2024.

The Bidirectional Encoder Face representation from image Transformers (BEFiT) is a model that leverages the multi-attention transformer mechanisms to capture local and global features and produce a multi-purpose face embedding.

[transformers_fusion_pipeline.pdf](https://github.com/nmirabeth/BEFiT/files/14918909/transformers_fusion_pipeline.pdf)

The unique face embedding produced by BEFiT enables the estimation of different demographics without having to re-train the model for each soft-biometric trait.
Our model is based on the BEiT [1] implementation. We train BEFiT for FR on more than 200K RGB face images from the CelebA [2] dataset. After that, BEFiT is fine-tuned with the thermal faces from the TUFTS [3] database to perform FR in thermal spectra. We will refer to the different versions of BEFiT as BEFiT-V and BEFiT-T depending on the spectra in which they work.

# Library requirements

'Python = 3.9.12'

'tensorflow = 2.4.1'

'huggingface-hub = 0.17.3'

'jupyter = 1.0.0'

'matplotlib = 3.5.1'

'numpy = 1.22.3'

'pandas = 1.5.3'

'Pillow 9.0.1'

'scikit-learn = 1.3.0'

# Citing

-- Coming soon --

# Acknowledgement

This work has been partially supported by the European CHIST-ERA program (grant agreement CHIST-ERA-19-XAI-011).

# References
[1] Bao, H., Dong, L., Piao, S., & Wei, F. (2021). Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254.

[2] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face attributes in the wild. In Proceedings of the IEEE international conference on computer vision (pp. 3730-3738).

[3] Panetta, K., Wan, Q., Agaian, S., Rajeev, S., Kamath, S., Rajendran, R., ... & Yuan, X. (2018). A comprehensive database for benchmarking imaging systems. IEEE transactions on pattern analysis and machine intelligence, 42(3), 509-520.

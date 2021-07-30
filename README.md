# SI-CEERI-ML-DL
The Repository has the accumulation of the work done during the summer internship 2021 at Central Electronics Engineering Research Institute. The Research Internship was guided by scientist Dr. Suriya Prakash J at CSIR - CEERI, Chennai. The work comprise of Machine Learning and Deep Learning Techniques for texture classification.
The main aim of the internship was to build models to successfully extract the texture features from the images using deep-learning and traditional machine laerning techniques.
The database used were OuTex, KTH-Tips-2a, FMD and VisTex.

### Abstract:
Texture Classification is the problem of distinguishing between textures, a classic problem in pattern recognition.Local binary patterns (LBP) are considered among the most computationally efficient high-performance texture features. We used the novel techniques of Complete Local Binary Pattern(CLBP) and Median Robust Extended Local Binary Pattern(MRELBP) to overcome the disadvantages of LBP. CLBP improves the accuracy by combining the different operators namely sign, magnitude and center gray scale. MRELBP improves on CLBP further using multiple resolutions and median filtering and hence gives even better results.Above techniques use manually extracted features and need to learn what features are important for classification. We implemented Deep learning models like InceptionV3 ,Bilinear CNN’s which represent an image as a pooled outer product of features derived from two CNNs and capture localized feature interactions. However, bilinear features are high dimensional,which makes them impractical for subsequent analysis. Hence making use of compact representations for the features extracted using BCNN we implemented Compact BCNN .The datasets OuTex, KTH-Tips-2a & FMD were used in evaluating the same.

[Link](https://docs.google.com/document/d/1bIjKIv3eOZIGPFHHXYagqRVD54xtD-YErzpn5DDnTEk/edit?usp=sharing) for the report.



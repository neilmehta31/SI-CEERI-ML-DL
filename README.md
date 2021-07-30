# SI-CEERI-ML-DL
The Repository has the accumulation of the work done during the Practice School -I Summer Internship 2021 at Central Electronics Engineering Research Institute(CSIR-CEERI). The Research Internship was guided by scientist Dr. Suriya Prakash J at CSIR - CEERI, Chennai. The work comprise of Machine Learning and Deep Learning Techniques for texture classification.
The main aim of the internship was to build models to successfully extract the texture features from the images using deep-learning and traditional machine laerning techniques.
The database used were OuTex, KTH-Tips-2a, FMD and VisTex. Full [folder](https://drive.google.com/drive/folders/1yhV62kd3IKXu2qbxZtD2hzgvUs-XDFV0?usp=sharing) for the work

### Abstract:
Texture Classification is the problem of distinguishing between textures, a classic problem in pattern recognition.Local binary patterns (LBP) are considered among the most computationally efficient high-performance texture features. We used the novel techniques of Complete Local Binary Pattern(CLBP) and Median Robust Extended Local Binary Pattern(MRELBP) to overcome the disadvantages of LBP. CLBP improves the accuracy by combining the different operators namely sign, magnitude and center gray scale. MRELBP improves on CLBP further using multiple resolutions and median filtering and hence gives even better results.Above techniques use manually extracted features and need to learn what features are important for classification. We implemented Deep learning models like InceptionV3 ,Bilinear CNN’s which represent an image as a pooled outer product of features derived from two CNNs and capture localized feature interactions. However, bilinear features are high dimensional,which makes them impractical for subsequent analysis. Hence making use of compact representations for the features extracted using BCNN we implemented Compact BCNN .The datasets OuTex, KTH-Tips-2a & FMD were used in evaluating the same.

[Link](https://docs.google.com/document/d/1bIjKIv3eOZIGPFHHXYagqRVD54xtD-YErzpn5DDnTEk/edit?usp=sharing) for the report.

# Datasets :
- [KTH-Tips-2a](https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2.pdf)
- [OuTex](https://static.aminer.org/pdf/PDF/000/348/186/outex_new_framework_for_empirical_evaluation_of_texture_analysis_algorithms.pdf)
- [FMD](https://people.csail.mit.edu/celiu/CVPR2010/FMD/)
- [VisTex](https://vismod.media.mit.edu/vismod/imagery/VisionTexture/vistex.html)
> If unable to find any of the database ping me!
> 
# Results:
### Complete Local Binary Pattern (CLBP) 
We used both the methods, CLBP and MRELBP, on two given datasets - KTH-Tips-2a and OuTex. We used different machine learning classifiers such as Support Vector Machine classifier, KNearestNeighbour classifier, Naive Bayesian classifier, Logistic Regression classifier, etc. The operations were performed on different numbers of training and testing images to analyse the accuracy on increasing the dataset images.Table 1 gives the accuracy of texture classification techniques on different datasets with different classifiers:


<table>
  <tr>
   <td rowspan="2" >
Method
   </td>
   <td colspan="4" >OuTex-TC-10
   </td>
   <td colspan="4" >KTH-Tips-2a
   </td>
  </tr>
  <tr>
   <td>KNN
<p>
(n_neighbours = 1)
   </td>
   <td>SVM
   </td>
   <td>Naive Bayes
<p>
(multinom-
<p>
ial)
   </td>
   <td>Logistic Regression
   </td>
   <td>KNN
<p>
(n_neighbours = 1)
   </td>
   <td>SVM
   </td>
   <td>Naive Bayes
<p>
(multinom-
<p>
ial)
   </td>
   <td>Logistic Regression
   </td>
  </tr>
  <tr>
   <td>CLBP_S<sup>riu2</sup>
   </td>
   <td>91.66
   </td>
   <td>88.54
   </td>
   <td>85.41
   </td>
   <td>92.70
   </td>
   <td>92.30
   </td>
   <td>83.71
   </td>
   <td>54.75
   </td>
   <td>79.18
   </td>
  </tr>
  <tr>
   <td>CLBP_M<sup>riu2</sup>
   </td>
   <td>94.79
   </td>
   <td>90.625
   </td>
   <td>78.13
   </td>
   <td>88.54
   </td>
   <td>90.95
   </td>
   <td>87.33
   </td>
   <td>58.82
   </td>
   <td>75.56
   </td>
  </tr>
  <tr>
   <td>CLBP_M/C
   </td>
   <td>92.70
   </td>
   <td>97.91
   </td>
   <td>89.58
   </td>
   <td>97.91
   </td>
   <td>96.38
   </td>
   <td>95.92
   </td>
   <td>68.77
   </td>
   <td>89.59
   </td>
  </tr>
  <tr>
   <td>CLBP_S_M/C
   </td>
   <td>98.95
   </td>
   <td><strong>100.00</strong>
   </td>
   <td>96.86
   </td>
   <td><strong>100.00</strong>
   </td>
   <td>97.73
   </td>
   <td>97.28
   </td>
   <td>72.39
   </td>
   <td>91.85
   </td>
  </tr>
  <tr>
   <td>CLBP_S/M
   </td>
   <td><strong>100.00</strong>
   </td>
   <td><strong>100.00</strong>
   </td>
   <td>95.83
   </td>
   <td><strong>100.00</strong>
   </td>
   <td>95.92
   </td>
   <td>95.92
   </td>
   <td>75.56
   </td>
   <td>97.28
   </td>
  </tr>
  <tr>
   <td>CLBP_S/M/C
   </td>
   <td><strong>100.00</strong>
   </td>
   <td><strong>100.00</strong>
   </td>
   <td>98.83
   </td>
   <td><strong>100.00</strong>
   </td>
   <td><strong>99.09</strong>
   </td>
   <td>95.92
   </td>
   <td>75.56
   </td>
   <td>97.28
   </td>
  </tr>
</table>


**Table 1.**  The following table gives the results with different machine learning techniques of two of the datasets namely, OuTex-TC-10 and KTH-Tips-2a, on the novel technique of Complete local binary pattern.
- For OuTex, **480 **images consisting of all the **24 classes** were taken with test: train ratio as 0.2. Hence, there were 96 and 384 test and train images respectively.
- For KTH-Tips-2a,  **1100 **images were taken consisting of the sample_a of each of the **11 classes** with test: train ratio as 0.2. Hence, there were 220 and 880 test and train images respectively.
- We also compared the **2 classifiers** i.e., **KNearestNeighbours **and **Support Vector Machine Classifier** on the same data with **220 images** containing 20 images from each of the **24 classes** of KTH-Tips-2a and found out that KNN performs better than SVM Classifier with accuracy as **88.88%** and **84.44%** respectively.
    
  
### Median Robust Extended Local Binary Pattern (MRELBP) 

<table>
  <tr>
   <td rowspan="2" >
Method
   </td>
   <td colspan="8" >OuTex-TC-10
   </td>
   <td colspan="8" >KTH-Tips-2a
   </td>
  </tr>
  <tr>
   <td>KNN
<p>
(n_neighbours = 1)
   </td>
   <td>KNN
<p>
(n_neighbours = 3)
   </td>
   <td>KNN
<p>
(n_neighbours = 5)
   </td>
   <td>KNN
<p>
(n_neighbours = 7)
   </td>
   <td>KNN
<p>
(n_neighbours = 9)
   </td>
   <td>SVM
   </td>
   <td>Naive Bayes
<p>
(multinom-
<p>
ial)
   </td>
   <td>Logistic Regression
   </td>
   <td>KNN
<p>
(n_neighbours = 1)
   </td>
   <td>KNN
<p>
(n_neighbours = 3)
   </td>
   <td>KNN
<p>
(n_neighbours = 5)
   </td>
   <td>KNN
<p>
(n_neighbours = 7)
   </td>
   <td>KNN
<p>
(n_neighbours = 9)
   </td>
   <td>SVM
   </td>
   <td>Naive Bayes
<p>
(multinom-
<p>
ial)
   </td>
   <td>Logistic Regression
   </td>
  </tr>
  <tr>
   <td>MRELBP<sup>riu2</sup>
   </td>
   <td>88.13
   </td>
   <td>89.58
   </td>
   <td>88.13
   </td>
   <td><strong>100.00</strong>
   </td>
   <td><strong>100.00</strong>
   </td>
   <td>70.83
   </td>
   <td>30.83
   </td>
   <td>71.67
   </td>
   <td>76.05
   </td>
   <td>74.78
   </td>
   <td>72.27
   </td>
   <td>71.00
   </td>
   <td>69.74
   </td>
   <td>58.82
   </td>
   <td>30.67
   </td>
   <td>56.72
   </td>
  </tr>
</table>


**Table 2.**  The following table gives the results with different machine learning techniques of the datasets namely,            OuTex-TC-10 and KTH-Tips-2a, on the novel technique of Median Robust Extended local binary pattern.

- We ran the MRELBP technique on the whole dataset of OuTex-TC10 having **3960 images** with training images being **3480** and tested on **480** images spread over **24 different classes**. For comparison reasons, we ran the **OuTex** dataset over different KNearestNeighbours classifiers differing in the parameter of number of neighbours. We saw that as we **increase** the number of neighbours, we tend to see a **rise in the accuracy** because of the reason that the features are closely packed and dense. The accuracy reaches **100%** for neighbours 7 and 9. According to the comparison, we saw that **KNN performs the best** followed by Logistic Regression, Support Vector Machine and Naive Bayes respectively.

- For the dataset of **KTH-Tips-2a** which has **4752 images** having trained on **3564 images** and tested on **1188 images**, we found that KNN classifier proved to be the best among all the four classifiers. Here as we **increase** the number of neighbours, we see a **decrease in the accuracy** of the technique. This is due to the fact that the features extracted from the dataset are sparsely distributed and hence it is difficult to correctly classify. Again as in the case of OuTex, we see that KNN outperformed the other three techniques.

### Deep Learning Techniques

<table>
  <tr>
   <td rowspan="2" >
Method
   </td>
   <td colspan="2" >OuTex-TC-10
   </td>
   <td colspan="6" >KTH-Tips-2a
   </td>
   <td colspan="2" >FMD
   </td>
  </tr>
  <tr>
   <td>w/o fine tuning
   </td>
   <td>Fine tuning
   </td>
   <td>Only Sample_a 
<p>
<strong>(</strong>without fine tuning<strong>)</strong>
   </td>
   <td>Only Sample_a 
<p>
<strong>(</strong>fine tuning<strong>)</strong>
   </td>
   <td>without fine tuning**
   </td>
   <td>Fine tuning**
   </td>
   <td>Test train split on whole dataset
<p>
<strong>(</strong>without fine tuning<strong>)</strong>
   </td>
   <td>Test train split on whole dataset
<p>
<strong>(</strong>fine tuning<strong>)</strong>
   </td>
   <td>without fine tuning
   </td>
   <td>Fine tuning
   </td>
  </tr>
  <tr>
   <td>InceptionV3 - GoogLeNet
   </td>
   <td>88.93
   </td>
   <td><strong>98.43</strong>
   </td>
   <td>86.54
   </td>
   <td>77.73
   </td>
   <td>56.64
   </td>
   <td>61.36
   </td>
   <td>75.27
   </td>
   <td><strong>91.75</strong>
   </td>
   <td>23.99
   </td>
   <td>17.00
   </td>
  </tr>
  <tr>
   <td>B-CNN_m+d
   </td>
   <td><strong>100.00</strong>
   </td>
   <td><strong>100.00</strong>
   </td>
   <td><strong>100.00</strong>
   </td>
   <td><strong>100.00</strong>
   </td>
   <td>78.45
   </td>
   <td>80.05
   </td>
   <td><strong>99.78</strong>
   </td>
   <td><strong>100.00</strong>
   </td>
   <td><strong>94.00</strong>
   </td>
   <td><strong>77.49</strong>
   </td>
  </tr>
  <tr>
   <td>B-CNN_d+d
   </td>
   <td><strong>100.00</strong>
   </td>
   <td><strong>100.00</strong>
   </td>
   <td><strong>99.54</strong>
   </td>
   <td><strong>100.00</strong>
   </td>
   <td>76.17
   </td>
   <td>78.28
   </td>
   <td><strong>99.34</strong>
   </td>
   <td><strong>100.00</strong>
   </td>
   <td><strong>94.99</strong>
   </td>
   <td>72.50
   </td>
  </tr>
  <tr>
   <td>C-BCNN
   </td>
   <td><strong>100.00</strong>
   </td>
   <td><strong>99.79</strong>
   </td>
   <td><strong>99.15</strong>
   </td>
   <td><strong>100.00</strong>
   </td>
   <td>71.46
   </td>
   <td>80.89
   </td>
   <td><strong>98.69</strong>
   </td>
   <td><strong>99.24</strong>
   </td>
   <td>67.50
   </td>
   <td>67.00
   </td>
  </tr>
</table>


**Table 3.** he following table gives the results with different machine learning techniques of the datasets namely, OuTex-TC-10, KTH-Tips-2a and FMD on the different deep learning techniques.

** KTHTips trained on Sample a,c,d and tested on b 3200 train 1100 test


 - We ran the all the deep learning techniques, namely InceptionV3 - GoogLeNet, Bilinear-CNN_m+d, Bilinear-CNN_d+d and compact Bilinear-CNN on the whole dataset of OuTex-TC10 having **3960 images** with training images being **3480** and tested on **480** images spread over **24 different classes**. We calculated the results using initially without any fine tuning and then by fine tuning a few top layers to increase the trainable parameters in order to train the large dataset efficiently. As we can see, OuTex dataset gives **100% accuracy** on many of the novel deep learning techniques. This is due to the similar types of images in the dataset.


 - For the dataset of **KTH-Tips-2a** which has **4752 images** having trained on **3564 images** and tested on **1188 images**. Sample a,c and d was used to train the model and the sample_b was used to test the model. Comparing the results of KTH-Tips-2a to OuTex, we notice that it has a slightly lower accuracy which is due to the fact that the dataset has quite **varied images** which in turn makes it difficult for the DL models to distinguish between two images that easily. The case when we used the whole dataset with test train split, we see a stark difference in the accuracy as now the trained model consists of some of the images of sample_b.


- FMD dataset performs poorly on InceptionV3(GoogLeNet) but shows a good accuracy on BCNN and Compact-BCNN.


## Conclusion

We have compared 2 state of the art texture descriptors CLBP and MRELBP on Outex and KTH- TIPS. For CLBP in particular we demonstrated that the sign component is more important than the magnitude component in preserving the local difference information,which can explain why the CLBP_S (i.e., conventional LBP) features are more effective than the CLBP_M features. Finally, by fusing the CLBP_S, CLBP_M, and CLBP_C codes, all of which are in binary string format, either in a joint or in a hybrid way the accuracies improved further. MRELBP outperforms recent state of the art LBP type descriptors in noise free situations and demonstrates striking robustness to image noise including Gaussian white noise, Gaussian blur, Salt-and-Pepper and pixel corruption. It has attractive properties of strong discriminativeness, gray scale and rotation invariance, no need for pretraining, no tuning of parameters,and computational efficiency.Thus both of these methods are robust and accurate so are preferred in low computation availability environments for Texture classification.

Though the techniques using manual texture descriptors are useful and have low computational dependency reach a bottleneck on accuracy even after increasing training data, so the need for better feature extraction and thus make use of Deep Neural networks to extract important features for classification through CNN’s combined and implemented with variations. Made use of Transfer learning to load pre-trained weights into our models for classification. InceptionNet provides solid evidence that moving to sparser architectures is feasible and useful idea in general. This suggests future work towards creating sparser and more refined structures in automated ways, as well as on applying the insights of the Inception architecture to other domains.We implemented one of the advanced versions based on the same architecture (InceptionV3).

We also implemented  B-CNN architecture that aggregates second-order statistics of CNN activations resulting in an order-less representation of an image. These networks can be trained in an end-to-end manner allowing both training from scratch on large datasets, and domain-specific fine-tuning for transfer learning. We also compared B-CNNs to both exact and approximate variants of deep texture representations and studied the accuracy and memory trade-offs they offer.We made use of all  possible combinations yielding best results(VGG d+d and VGG m+d).

To implement B-CNN on larger datasets,made use of sparse representation for all the features.We modeled bilinear pooling in a kernelized framework and utilised a compact representation,which allows back-propagation of gradients for end-to-end optimization of the classification pipeline. Our key experimental result is that an 8K dimensional TS(Tensor Sketch) feature has the same performance as a 262K bilinear feature, enabling a remarkable 96.5% compression. Further, TS reduces network and classification parameters memory significantly which can be critical e.g. for deployment on embedded systems.Our results yield solid evidence that approximating the expected optimal sparse structure by readily available dense building blocks is a viable method for improving neural networks for computer vision. The main advantage of this method is a significant quality gain at a modest increase of computational requirements compared to shallower and narrower architectures.

Hence all the above architectures improve further on accuracies and lay foundation for further research to option architectures with lower computational requirements and higher accuracies.


# MRI-Based Segmentation and Classification Approaches for Pediatric Brain Tumors: A Review

**Journal Target:** Journal of Taibah University Medical Sciences (Elsevier)  
**Article Type:** Review Article  

---

## TITLE PAGE

**Full Title:** MRI-Based Segmentation and Classification Approaches for Pediatric Brain Tumors: A Review  
**Short Title (Running Head):** Computational AI Pipelines for Pediatric Brain Tumor MRI  

**Authors:**  
Balaji Boddu$^1$, Sangeetha Saman$^{2,*}$, Syed Kashed$^1$, D. Jagadeesan$^1$, Kavitha S$^3$, Mohammed Mubasheer$^2$  

**Affiliations:**  
$^1$ School of Technology, The Apollo University, Murukampattu, Andhra Pradesh, India-517127  
$^2$ Vellore Institute of Technology, Vellore, Tamil Nadu, India-632014  
$^3$ Division of Allied Health Sciences, School of Health Sciences, The Apollo University, Murukampattu, Andhra Pradesh, India-517127  

**Corresponding Author:**  
Dr. Sangeetha Saman  
Vellore Institute of Technology, Vellore, Tamil Nadu, India-632014.  
Email: `s.sangeetha@vit.ac.in`  

**Manuscript Word Count & Metrics:**  
- **Abstract Word Count:** 232 words (Structured: Objectives, Methods, Results, Conclusions)  
- **Main Text Word Count:** ~5,250 words  
- **Number of Figures:** 7  
- **Number of Tables:** 2  
- **Number of References:** 125  

---

## ABSTRACT

**Objectives:** To improve the prognosis, rationality of treatment planning, and survival rate of children with pediatric brain tumors, an accurate diagnosis must be achieved as early as possible. However, current traditional radiological assessment is time-consuming and exhibits inter-observer variability in its results, leaving the field in urgent need of computer-aided diagnostic support systems.

**Methods:** This review systematically analyzes computational methods for pediatric brain tumor diagnosis based on magnetic resonance imaging (MRI), covering six core processing modules: image preprocessing, feature extraction, feature selection, dimensionality reduction, tumor segmentation, and tumor classification. We evaluate both conventional machine learning algorithms and advanced deep learning architectures (FCN, U-Net, 3D U-Net, ViTs, and GANs).

**Results:** Deep learning models demonstrate superior segmentation accuracy (Dice Similarity Coefficients up to 93–94%) and robust multi-class classification compared to traditional methods (Dice score 70–85%). However, severe class imbalance, data scarcity in rare pediatric cohorts, limited model interpretability, and lack of standardized multi-institutional evaluation remain critical hurdles.

**Conclusions:** Integrating multimodal imaging fusion, explainable artificial intelligence (XAI), and multi-task hybrid models represents the most promising direction to advance early detection, lower inter-observer variability, and deliver actionable clinical decision support in pediatric neuro-oncology.

**Keywords:** Pediatric brain tumors; Magnetic resonance imaging; Clinical decision support; Machine learning; Deep learning; Computer-aided diagnosis.

---

## 1. INTRODUCTION

Pediatric brain tumors are the most common solid childhood cancer, second only to leukemia in overall incidence. Most of these tumors are primary; they damage brain tissue based on their growth location, and can cause symptoms including headaches, vomiting, and gait disturbances [4,5,6]. Despite the advancements and innovations in medicine in brain cancer diagnosis and treatment, challenges still exist for early diagnostic assessment of brain tumors and managing their long-term adverse effects [7]. Various pediatric brain tumors have been classified, which exhibit differences in characteristics and incidence [8]. The most prevalent malignant brain tumor in children, medulloblastomas frequently start in the cerebellum [9]. Additionally, these tumors can spread through the CSF fluid [10]. In children, pilocytic astrocytoma is frequently identified as benign brain tumors that can be successfully treated with surgery [11]. High-grade brain tumors, such as diffuse intrinsic pontine glioma (DIPG), are uncommon, aggressive, and challenging to cure. They are extremely resistant to the majority of treatment strategies since they are almost impossible to eradicate [12,13]. Due to their positions and proximity to significant brain structures, other brain tumors such as ependymomas and craniopharyngiomas are thought to be challenging to treat in young populations and frequently call for a combination of radiation and surgery [14,15]. Multiple published studies [5,17] explicitly note that age is a core influencing factor that determines the biological characteristics and onset locations of brain tumors. Adult and pediatric brain tumors show fundamental differences at the levels of origin attributes, onset locations, and molecular features: primary brain tumors (linked to developmental processes or genetic abnormalities) have a high incidence in children, with medulloblastomas and ependymomas of the posterior fossa being common; secondary metastatic brain tumors are more frequently observed in adults, and glioblastomas in the supratentorial region are the most prevalent type.

Both the diagnosis and treatment of pediatric brain tumors depend on imaging. The Computed Tomography (CT) scan is the preferred scan in an emergency. CT scans are helpful in diagnosing diseases including bleeding and hydrocephalus. However, ionizing radiation is used during CT exams. Children are at serious risk from radiation exposure from CT scans. As a result, CT scans are mostly utilized in emergency situations to quickly diagnose serious problems [18]. The Magnetic Resonance Imaging (MRI) scan is the preferred test in non-emergency situations [21]. Ionizing radiation is not used during MRI exams, which can produce detailed images of brain tumors [22]. Children are also safer during MRI exams [23]. Diffusion-weighted imaging, functional MRI, and magnetic resonance spectroscopy are examples of advanced MRI techniques that offer comprehensive information on the tumor, supporting both diagnosis and treatment [5]. Sedation is frequently necessary for younger pediatric patients undergoing MRI exams in order to render the kid motionless [24,25]. An overview of pediatric brain tumor types emphasizing the primary categories, their unique characteristics, and their corresponding anatomical locations are illustrated in Fig 1. The figure also provides a concise summary of the role of diagnostic imaging modalities, specifically MRI and CT, in the identification of tumor features and the support of an accurate diagnosis.

> **Figure 1:** Overview of pediatric brain tumors types, key differences, and diagnostic Imaging.

### 1.1 Challenges in Diagnosing Pediatric Brain Tumors

The clinical appearance of pediatric brain tumors varies greatly among children, making diagnosis difficult [5]. Improving the prognosis of pediatric brain tumors requires early and precise diagnostic evaluation, despite a number of difficulties. The age of the patient, the type of tumor, and the tumor's location all affect how brain tumors manifest in pediatric patients. While older pediatric patients exhibit specific symptoms such headaches and seizures, newborns with these tumors have nonspecific symptoms like irritability and vomiting [26,27]. Because it can be challenging to communicate the symptoms in infants and younger pediatric patients, brain tumors in the posterior fossa, which are common in pediatric patients, also present with specific manifestations of cerebellar dysfunction and obstructive hydrocephalus, making early and accurate diagnosis more difficult [28]. Because brain tumors are uncommon, there is not enough data to train and evaluate machine learning algorithms, which makes identifying pediatric brain tumors computationally difficult [29]. Furthermore, standardizing computer-assisted brain tumor diagnosis tools is difficult due to variations in imaging methods and molecular features of pediatric brain tumors [30,3]. From a clinical perspective, sedation is frequently necessary for brain tumor imaging in young patients, which comes with dangers and limitations [31]. Similarly, using tissue biopsies to diagnose tumors has dangers and highlights the need for accurate, noninvasive diagnostic methods [32]. Despite advances in neuroimaging technologies, existing imaging modalities cannot accurately distinguish between the types and grades of brain tumors. The first section of this review discusses the importance of diagnosis for adolescent patients and the need for computer-aided tools, while the second section covers various machine learning and deep learning technologies. Various gaps in brain tumor analysis are covered in Section 3, and Section 4 concludes this investigation by exploring potential future enhancements to brain tumor analysis models.

---

## 2. COMPUTATIONAL TECHNIQUES FOR PEDIATRIC BRAIN TUMOR DIAGNOSIS

One of the main sources of cancer-related morbidity and mortality in the pediatric population is thought to be pediatric brain tumors [33]. The need for automated methods to identify pediatric brain tumors has grown due to the challenges of image interpretation [1]. By using sophisticated computational methods to identify these malignancies, these systems lower observer error and increase the precision of clinical teams' choices [34]. The several methods created for this goal are reviewed in this section.

> **Figure 2:** Comprehensive diagnostic pipeline for pediatric brain tumor analysis from data collection to clinical decision support.

Fig. 2 represents the systematic workflow of the complete diagnostic pipeline for pediatric brain tumor analysis, integrating the ten essential steps of the data analysis process discussed in this survey on computer-aided diagnosis systems. The workflow illustrates computer-aided diagnosis methods and demonstrates how each step advances the ultimate goal of offering precise and trustworthy clinical decision support in the field of pediatric neuro-oncology.

### 2.1 Methods of Preprocessing Data on Pediatric Brain Tumors

An essential step in the examination of pediatric brain tumor data is preprocessing. It describes the process of preparing raw data for further analysis. Clinical imaging and genetic data are examples of raw data used in pediatric brain tumor research. Preprocessing methods assist guarantee that the collected raw data is clean and prepared for further examination. Image preprocessing supports the realization of image normalization and noise removal. Subsequent sections will collate and review various preprocessing methods used in pediatric brain tumor research. Pattisapu et al. [35] mentioned that the core focus of the current field is improving the quality of clinical imaging and genomic data.

> **Figure 3:** Methods of Preprocessing for Pediatric Brain Tumors.

Fig. 3 presents an overview of preprocessing techniques for pediatric brain tumor data, encompassing medical imaging preparation, genomics data preprocessing, clinical data preprocessing, and multimodal data fusion preprocessing. The diagram delineates the essential procedures for improving data quality, standardizing diverse datasets, and integrating various data sources to enhance the accuracy and reliability of subsequent analysis and predictive modeling.

#### 2.1.1 Medical Image Preprocessing

In the clinical diagnosis and related research of pediatric brain tumors, core medical imaging data sources include MRI, CT, and PET scans. Medical image preprocessing is a core upstream step that guarantees the accuracy of subsequent tasks including tumor segmentation, classification, and prediction. This review sorts out five core preprocessing technologies in this field, each of which clarifies its specific role and implementation pathway for targeted application scenarios. The first is image denoising, which addresses the problem that modalities such as MRI are prone to generating noise that masks key tumor features. Denoising must be carried out on the premise of preserving the signals of the tumor region of interest. Common methods include wavelet denoising, median filtering, and Gaussian smoothing, and the rationality of this pathway has been confirmed by study [36]. The second is image normalization, which enables cross-cohort comparability of multi-source images by standardizing pixel intensities to a unified range, making it particularly suitable for research scenarios such as cross-modal fusion and multi-center collaboration. Mainstream methods include the z-score method, min-max method, and histogram equalization.

$$\text{Z-Score Normalization: } z = \frac{x - \mu}{\sigma}$$

$$\text{Min-Max Normalization: } x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

The third is skull stripping, which study [37] defines as the operation of removing non-brain tissues such as the skull from MRI and CT scans to retain only brain parenchyma. It is a necessary prerequisite for accurate tumor delineation, and common tools include FSL’s Brain Extraction Tool (BET) and ANTs (Advanced Normalization Tools), whose applicability has been verified by study [38]. The fourth is resampling, which unifies the scale consistency of the entire dataset by adjusting image resolution; the fifth is image registration, which aligns multiple images to the same coordinate system, serving as an indispensable core support for multi-modal image fusion research. For instance, it is necessary to merge MRI and PET scans. To register images obtained at different periods and with different modalities, image registration techniques like rigid, affine, and non-rigid registration are used [39]. A sophisticated algorithm called N4ITK (N4ITK Bias Field Correction) is used to adjust the bias field in MRI and other clinical images. The approach is an improvement over N3 (nonparametric nonuniform intensity normalization), another algorithm used to minimize low-frequency intensity inhomogeneities in clinical images [40]. The ITK library Insight Segmentation and Registration Toolkit is where N4ITK is implemented. Medical image analysis is done with this library. Iterative optimization techniques serve as the foundation for N4ITK's corrective mechanism [41].

#### 2.1.2 Genomic Data Preprocessing

A rare chance to learn more about the molecular features of pediatric brain tumors is provided by genomic data, which includes expression profiles and DNA sequencing.

1. **Quality Control (QC) of Sequencing Data:** Quality control procedures are carried out before the genetic data is analyzed to guarantee that the data is of high quality, i.e., to check for issues like low-quality, polluted, or aligned data mistakes. Because low-quality data can affect the outcomes of the subsequent analytic processes, FastQC and MultiQC are used for quality control of the sequencing data, removing low-quality data from the dataset [42].
2. **The process of normalizing gene expression data:** To remove biases from the gene expression profiles, which may occur during the experiment or during data processing, the expression profiles of the genes must be normalized. RPKM, TPM, and quantile normalization are the three most popular methods for normalizing the gene expression patterns derived from the RNA sequencing experiment [43].
3. **Data Imputation:** Genomic data sets, particularly those from high-throughput sequencing, frequently contain missing values. Missing values in genomic data sets are imputed using methods including mean imputation, multivariate imputation, and k-nearest neighbors imputation [44].
4. **Batch Effect Correction:** Due to differences in study design and data collection methods, genomic data sets are frequently impacted by batch effects. To make sure that biological variations are not overshadowed by technical variations, batch effects are eliminated using the ComBat method found in the R-SVA library [45].

#### 2.1.3 Clinical Data Preprocessing

Clinical data, including demographic data, clinical features, and treatment history, are crucial in predicting the outcome of the patients. Accordingly, preprocessing the data helps clean and standardize the data for analysis.

- **Data Cleaning and Standardization:** Clinical data may have several inconsistencies, including missing data, duplicates, and inaccuracies. Various preprocessing techniques are applied to clean the data. For example, categorical data, like tumor grade and type, are standardized using a coding system, like International Classification of Diseases for tumor type [46].
- **Handling Missing Data:** Missing data are common in clinical data, especially when dealing with a large population. Various techniques are applied to handle missing data, including mean imputation, multiple imputation, and using machine learning algorithms like random forest to predict missing data [37].
- **Feature Engineering:** Using domain expertise, a variety of features are developed during the preprocessing of clinical data. The machine learning algorithm may perform better as a result. To enhance the algorithm's performance for pediatric brain tumors, for instance, new features such as tumor progression and time-to-event features are developed based on the existing unprocessed data [47].

#### 2.1.4 Preprocessing for Multi-modal Data Fusion

The data used in pediatric brain tumor research may be multimodal. Multimodal data fusion encompasses three categories of data: clinical, genetic, and imaging data. Data preprocessing serves as a core, necessary precondition to ensure the standardized integration of data across all modalities. In multimodal data research on pediatric brain tumors, three core general preprocessing techniques must be implemented before integrated analysis can be conducted. The first is data harmonization: according to Jiang et al. [48], this technique unifies the scale and format of all data modalities, laying a foundation for multimodal integration. The second is dimensionality reduction: Nair et al. [49] indicated that the above approach addresses the difficulty of high dimensionality resulting from the integration of multisource data by compressing the dimensionality of combined features through methods such as PCA and t-SNE, hence mitigating complications in further research. The third is feature fusion, which is applied after all modalities complete basic preprocessing to generate a unified feature set that can support downstream analyses including classification and survival prediction. Preprocessing is the core step that converts multisource data into a clean, unified format. For this specific research field, three additional domain-specific operations must also be completed: image denoising, gene expression data normalization, and clinical data cleaning, which ensure the accuracy of data analysis and improve the quality of research results.

### 2.2 Pediatric Brain Tumor Segmentation

One of the most important tasks in medical imaging is the precise identification of pediatric brain tumors. There are several uses for this work, such as diagnosis and treatment. The goal of this work is to use CT or MRI scans to segment the tumor regions. Children with brain tumors face several difficulties. The shape, location, and surrounding normal tissues of brain tumors vary greatly. For this purpose, thresholding, region growth, and active contours were initially used. Nevertheless, these methods were constrained by their poor performance. As time has gone on, ML and DL-based methods have shown promising results for segmenting clinical images by directly extracting features from clinical images without the need for manual feature representation [50]. Recent developments in deep learning (DL), particularly the use of U-Net (Fig. 4), Fully Convolutional Networks (FCN), and 3D U-Net, have transformed the field of image segmentation by guaranteeing strong accuracy and robustness. The approach is highly useful for clinical applications since it uses large-scale datasets to train the models, which can subsequently execute the pixel-wise delineation of the tumor [51,52]. The outcomes have also improved with the use of the hybrid strategy, which combines the conventional methods with the DL, particularly when dealing with complex tumor locations [53].

> **Figure 4:** U-Net for Segmenting Pediatric Brain Tumors.

#### 2.2.1 Thresholding and Region Growing

Thresholding [50] is a primary approach used for tumor segmentation. Under this approach, the intensity value of the pixels is applied to differentiate the pixels into tumor and non-tumor. In region growing, pixels are added based on some similarity criteria. This strategy is efficient, quick, and easy. This strategy works well for high-contrast tumors. However, this approach fails when dealing with low-contrast and heterogeneous tumors. This strategy is also sensitive to noise. Under this approach, initialization is manual, which can be a problem.

#### 2.2.2 Active Contours (Snakes)

Active contour models [54], also called snakes, use image gradient information and energy minimization to evolve a curve that detects image boundaries. These architectures can better represent complex tumor shapes compared to simple thresholding methods. However, active contour models are sensitive to noise and initial contour placement. These methods are also resource-intensive. Furthermore, active contour models do not perform well in dealing with complex tumor shapes and anatomical structures.

#### 2.2.3 Level Set Methods

By reducing an energy functional, level set algorithms [52] evolve an image shape. Complex image segmentation issues like tumor splitting and merging can be successfully handled by these techniques. These techniques are therefore appropriate for challenging image segmentation issues. These techniques work well for early contour placement issues. Moreover, these techniques are capable of handling topological alterations in tumor areas. These techniques, however, need a lot of resources. For these techniques to produce the best outcomes, hyperparameter tweaking is also necessary. Furthermore, because of the computational complexity of these technologies, real-time applications are not possible.

#### 2.2.4 Graph-Based Techniques

In graph-based segmentation approaches, where each pixel is viewed as a node and the edges are used to indicate how similar the pixels are to one another. The energy function for tumor region segmentation is minimized using the graph cuts method. In addition to providing a worldwide solution for the image segmentation problem, graph-based approaches are highly effective at managing intricate tumor boundaries. However, the computational and memory demands of graph-based procedures, as well as the complexity of the graph cuts method's initialization, may limit their practical use.

> **Figure 5:** Detailed Overview of Segmentation Algorithms.

#### 2.2.5 Convolutional Neural Networks (CNNs)

CNNs employ deep learning algorithms for image segmentation, using the raw image as input and automatically learned features [55]. CNNs perform pixel-wise classification with good accuracy, and they are highly successful for the tumor segmentation problem [56]. The image segmentation challenge is made easier by CNNs' resilience to noise and lack of need for feature representation. However, the CNNs can limit practical use because they are computationally difficult, time-consuming, and require a significant amount of data for training [57].

#### 2.2.6 Fully Convolutional Networks (FCN)

A variation of CNNs known as completely convolutional networks substitutes convolutional layers for fully connected layers. This CNN variation works well for segmentation tasks at the pixel level. Features are developed manually in FCNs. Although FCNs have significant memory needs, require a big dataset for training, and can be computationally difficult for large images, they can also be employed for 3D clinical images [58].

#### 2.2.7 U-Net Architecture

A CNN design called U-Net is utilized for segmentation, especially in the biomedical industry. Because of its U-shaped form and usage of an encoder-decoder network with skip links, the U-Net can extract both local and global features [59,60]. Because of data augmentation and tumor boundary preservation, it has proven to be quite accurate in MRI segmentation, especially in the biomedical industry. It also performs well with limited datasets. Nevertheless, the U-Net requires a lot of resources and is prone to model overfitting, particularly when dealing with tiny datasets [59,61].

#### 2.2.8 3D CNNs and 3D U-Net

Particular deep learning architectures, such as 3D CNNs and 3D U-Net [62], are used to process volumetric clinical images, such as MRIs. The spatial links between neighboring image slices are recognized by these structures. As a result, these models are better suited for employing volumetric clinical MRI images to segment tumors. For precise tumor delineation utilizing volumetric MRI exams, these structures are especially helpful. These models, however, might be problematic in medical settings where computing infrastructure may be limited because they are computationally expensive and require strong technology, such as GPUs [63,64].

#### 2.2.9 Hybrid Models

The hybrid model [65] integrates deep learning-based image processing methods such as CNN and U-Net, alongside traditional image processing methods including active contours and graph cuts. After fusion, this model is adaptable to tumors of multiple types and sizes, and delivers superior output performance. However, it consumes large amounts of resources, and requires users to be proficient in both categories of technologies.

The experimental evaluation module of this study sequentially sorts out six core performance evaluation metrics in the field of medical image tumor segmentation:

$$\text{Dice Similarity Coefficient (DSC)} = \frac{2 \cdot |\hat{Y} \cap Y|}{|\hat{Y}| + |Y|} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

$$\text{Intersection over Union (IoU)} = \frac{|\hat{Y} \cap Y|}{|\hat{Y} \cup Y|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$

$$\text{Sensitivity (Recall)} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{Specificity (True Negative Rate)} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

$$\text{Precision (Positive Predictive Value)} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Sensitivity}}{\text{Precision} + \text{Sensitivity}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

The accuracy of the tumor regions, sensitivity to tumor regions, and overall classification performance are among the strengths of each strategy that can be objectively compared by evaluating the tumor segmentation methods using the aforementioned metrics (Table 1).

### Table 1: Comparative evaluation of segmentation algorithms on brain tumor datasets
| Study | Technique | Data Used | Dice Score (%) | IoU Score (%) | Sensitivity (%) | Specificity (%) | Precision (%) | F1 Score (%) | Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[50]** | Thresholding & Region Growing | MRI, CT examinations | 70 | 65 | 75 | 80 | 72 | 73 | Sensitive to noise and intensity variation |
| **[54]** | Active Contours (Snakes) | MRI, CT examinations | 80 | 75 | 77 | 82 | 78 | 79 | Sensitive to initialization and weak boundaries |
| **[52]** | Level Set Methods | MRI, CT examinations | 82 | 76 | 79 | 85 | 80 | 81 | Computationally expensive and requires tuning |
| **[28]** | Graph-Based Techniques | MRI, CT examinations | 85 | 80 | 81 | 88 | 83 | 84 | High memory and computational cost |
| **[29]** | CNNs | MRI, CT examinations | 88 | 85 | 84 | 90 | 86 | 87 | Requires large annotated datasets |
| **[30]** | FCN | MRI (3D) | 90 | 87 | 86 | 92 | 89 | 90 | High memory requirement for 3D images |
| **[31]** | U-Net | MRI, CT examinations | 92 | 89 | 88 | 93 | 90 | 91 | Prone to model overfitting on small datasets |
| **[62]** | 3D CNNs and 3D U-Net | MRI (3D) | 93 | 91 | 90 | 94 | 92 | 92 | Requires high GPU resources and long training time |
| **[65]** | Hybrid Models | MRI, CT examinations | 94 | 92 | 91 | 95 | 93 | 94 | Complex architecture and tuning challenges |
| **[66]** | CNN + Age Fusion | Pediatric MRI, CBTN dataset | - | - | - | - | - | - | Classification-focused study, not a pure segmentation study |
| **[67]** | DeepMedic / nnU-Net | Multi-institutional pediatric brain tumor MRI | 72-86 | - | - | - | - | - | Performance varies by tumor region and dataset |
| **[68]** | nnU-Net for Pediatric Medulloblastoma | Multiparametric pediatric MRI | 81–88 | - | - | - | - | - | Subregion segmentation remains challenging |
| **[69]** | U-Net + O-Net Ensemble | BraTS-PEDs 2023 pediatric MRI | ET: 55, TC: 70, WT: 79 | - | - | - | - | - | Lower performance for enhancing tumor region |
| **[70]** | Advanced nnU-Net | Pediatric and adult brain tumor MRI | Pediatric Dice: 71 | - | - | - | - | - | Pediatric tumor segmentation performance lower than adult glioma |
| **[71]** | Radiomics + ML | MRI / radiomics datasets | - | - | - | - | - | - | Mainly used for classification or prediction, not direct segmentation |

### Table 2: Summary of strengths and constraints of conventional and deep learning-based segmentation approaches for pediatric brain tumors
| Techniques | Advantages | Disadvantages |
| :--- | :--- | :--- |
| **Thresholding & Region Growing** | Simple and computationally inexpensive. Effective for relatively homogeneous regions in images. | Sensitive to noise, variations in intensity, and requires predefined thresholds which may not generalize well. |
| **Active Contours (Snakes)** | Provides accurate boundary detection and adapts to object shape, especially in images with complex boundaries. | Core technical flaws: initialization sensitivity, high computational power consumption, and poor boundary robustness. |
| **Level Set Methods** | This contour processing tool supports topological transformations and adapts to complex geometric scenarios. | This computing technology suffers from high costs and high complexity, is prone to falling into local minima, and requires fine-tuning. |
| **Graph-Based Techniques** | This image processing model excels at processing spatial relationships, resisting noise, and parsing complex image structures. | This machine learning segmentation model can automatically learn features, process noisy unstructured inputs, and achieve flexible segmentation limits. |
| **CNNs (Convolutional Neural Networks)** | Automatically learns features from data, allowing for flexible segmentation even with noisy or unstructured input. | Requires large annotated datasets for training and high computational power. |
| **FCN (Fully Convolutional Networks)** | End-to-end training, no need for manual feature extraction, handles varying input sizes. | High memory and computational requirements; may struggle with very small or complex structures. |
| **U-Net** | Highly effective for medical image segmentation, even with limited data. Provides precise localization and robust results. | Prone to overfitting in small datasets; may require significant computational resources for large volumes. |
| **3D CNNs and 3D U-Net** | Handles 3D spatial data and captures volumetric information, useful for full brain segmentation. | Extremely resource-intensive, both in terms of computation and memory, requires substantial training data. |
| **Hybrid Models** | Combine strengths of multiple techniques, improving segmentation performance and robustness. | More complex and may suffer from increased computational load and model tuning challenges. |

The advantages and limitations of several image segmentation techniques are thoroughly examined in Table 2, which is arranged by increasing complexity: pixel-level image processing techniques like thresholding and region growing; energy minimization techniques like active contours and level sets; optimization techniques like graph methods; and machine learning techniques like deep learning.

### 2.3 Feature Extraction Techniques

Feature representation is a key step when analyzing pediatric brain tumor information. This process enables researchers to transform raw information obtained from clinical images, clinical data, and genomic data into an easily analyzable format, thereby facilitating a better understanding of various tumor types [18]. This section summarizes feature representation techniques.

#### 2.3.1 Imaging Feature Extraction

Pediatric brain tumors are diagnosed by medical imaging methods as MRI (Magnetic Resonance Imaging), CT (Computed Tomography), and PET (Positron Emission Tomography) [5]. Tumor categorization and prognosis prediction can benefit from feature extraction of the tumor utilizing these methods [73]. Shape Features are the distinction between benign and malignant tumors is aided by the tumor's shape-based feature representation. In the field of medical imaging diagnosis for pediatric brain tumors, existing studies have confirmed that specific imaging attributes have unique advantages in identifying brain cancer and glioma in young patients. Currently, four types of well-established medical imaging features are applied to the identification, subtyping, and prognostic assessment of this disease, and all relevant conclusions are derived from publicly available academic literature: The first category is location features, extracted from MRI/CT data sources through morphological analysis and geometric modeling [74]; the second category is texture features, which adopts four indicators including the contrast and homogeneity of Haralick texture features, can distinguish between benign and malignant tumors as well as subtypes such as pediatric glioblastoma [75]; the third category is radiomic features, which integrates texture, morphology, and intensity attributes, relies on machine learning methods such as random forests and support vector machines, realizes tumor classification, prognosis prediction, and treatment efficacy monitoring based on MRI/PET data sources, and can predict the survival period of child patients [76]; the fourth category is functional imaging features, which combines fMRI, DTI and structural imaging, can analyze tumor-related changes in brain activity and evaluate the motor and cognitive outcomes of child patients [77].

#### 2.3.2 Genomic and Epigenetic Feature Extraction

The widespread availability of genomic data has greatly lowered the research threshold for studies on the molecular characteristics of pediatric brain tumors. Leveraging genomic feature extraction technologies, researchers can not only predict tumor traits, patient prognosis, and treatment outcomes, but also analyze the genetic properties of tumors. Gene expression profile analysis, which uses microarray and RNA sequencing technologies to extract differentially expressed genes across different tumor types to screen for biomarkers for tumor classification and survival prediction, has been applied to the mining of specific genes linked to pediatric glioma and medulloblastoma [78]; copy number variations (CNVs) refer to alterations in the copy number of specific genomic regions, which can influence tumor behavior and progression. CNV features identified via array comparative genomic hybridization and next-generation sequencing can predict the malignancy grade of tumors, making them critically important for assessing the development of pediatric brain tumors [79].

- **Single Nucleotide Polymorphisms (SNPs):** Single Nucleotide Polymorphisms refer to single nucleotide changes in the genome. This can have implications for individual susceptibility to disease. For pediatric brain tumors, features can be obtained that relate to specific SNPs. This can be useful in understanding genetic factors that lead to tumor development. Genome-wide association studies may be applied to relate specific SNPs to tumor and treatment response [79].
- **Epigenetic Features:** Tumorigenesis is significantly influenced by DNA methylation and histone modification, which are crucial regulators of gene expression. To comprehend the epigenetic changes that result in pediatric brain tumors, characteristics linked to DNA methylation patterns can be retrieved. Epigenetic profiling has shown promising results in identifying the many subtypes of brain tumors, particularly medulloblastomas and gliomas [80].

#### 2.3.3 Clinical Feature Extraction

Clinical features such patient demographics, symptom presentation, tumor location, and histological features are important in understanding the course of pediatric brain tumors as well as patient outcomes, in addition to imaging and genetic features.

- **Age and Tumor Location:** Depending on the age range, pediatric brain tumors have distinct features. In a similar vein, a crucial feature of pediatric brain tumors is their location. Both tumor kind and patient outcomes have a strong correlation with tumor site. For instance, aggressive tumor behavior and poor outcomes are linked to tumors in the posterior fossa in young individuals [81].
- **Histopathological Features:** Histopathological examination of tumor tissue samples is regarded the gold standard for tumor diagnosis and grading. Tumor features extracted from tissue samples, such as tumor cellularity, mitotic activity, and vascularization, may be used for grading and classification. Digital pathology and image analysis methods may be used to extract quantitative histological features for better tumor characterization and prognosis [82].
- **Symptom-based Features:** In youngsters, symptoms including headaches, vomiting, and seizures could be early signs of brain tumors. The prognostic outcome of a tumor and the accuracy of its early diagnosis rely on the duration band progression of tumor-related symptoms; when combined with tumor imaging, this information can be used to predict the tumor’s clinical stage and the outcomes of its diagnosis and treatment [83].

#### 2.3.4 Machine Learning-based Feature Extraction

This study finds that machine learning, especially deep learning models, can automatically generate feature representations from clinical imaging and genomics data, and their importance continues to grow.

- **Deep Learning for Feature Extraction:** Convolutional Neural Networks (CNNs) are frequently used to automate tasks involving the representation of features in clinical scan images, particularly MRI scans. Without the need for human intervention, this technique can automatically extract information from unprocessed clinical images. Studies on pediatric brain tumors have made extensive use of this technique. In medical image analysis, particularly radiological imaging images, deep learning has proven particularly helpful in identifying features that could otherwise go unnoticed [84].
- **Autoencoders in Dimensionality Reduction:** Autoencoders are neural networks that can be applied in unsupervised learning methods. These networks may be applied to extract features in an unsupervised manner. Autoencoders can also be used in dimension reduction. This is particularly useful in dealing with multi-modal datasets. This occurs because it is able to extract features that can be used in categorization of tumors as well as in making predictions of patient survival [85].
- **Feature Engineering:** This machine learning technique involves the creation of features by subject-matter experts. Experts in the field combine different elements to increase the accuracy of predictive models when treating pediatric brain tumors. Existing research [86] points out that applying extracted features to machine learning can improve the accuracy of pediatric brain tumor classification and patient prognosis prediction. Feature extraction is a core step in this field; information can be extracted from three categories of data sources: imaging, genomic, and clinical data to support the development of three types of machine learning models for brain tumor classification, prognosis, and treatment monitoring. Recent advances in genomic analysis, machine learning algorithms, and image acquisition technology [87,88] have optimized relevant feature representation strategies.

### 2.4 Feature Selection Techniques

Feature priority ranking is a core component of pediatric brain tumor analysis, and its importance has continued to grow in recent years as machine learning becomes widely applied in the field of medical diagnosis. This process can screen core features from four categories of multi-source data [89]: clinical, radiological, genetic, and histological data. It improves model efficiency and interpretability, and helps elevate the accuracy of treatment decision-making.

#### 2.4.1 Filter Methods

Filter methods are the simplest and most widely used feature prioritization strategies in the field of machine learning. This type of method does not rely on any specific learning algorithm, and can evaluate the correlation between dataset features and target variables solely through statistical methods. It has already been implemented in machine learning analyses of pediatric brain tumors, and can screen core research attributes from datasets covering clinical, genetic, and radiological dimensions. Current mainstream filter-based feature selection techniques fall into three main categories: The first is Correlation-based Feature Selection (CFS), which can eliminate feature redundancy and assess the quality of a feature set via the correlation between features and the target variable. One existing study [89] applied this method to pediatric brain tumor datasets such as glioma datasets, to screen core radiological features linked to tumor subtype, grade, and survival prognosis. The second is the chi-square test, which is mostly used for categorical data to judge the association between a feature and a target category (such as whether a tumor is benign or malignant). Multiple pediatric brain tumor studies [90] have adopted this technique to identify clinical features of different tumor types. The third is the mutual information method, which is used to judge the dependence between variables. This method has also been applied in pediatric brain cancer studies [91] to screen core features that can reflect tumor location and grade. Feature prioritization is a core component of pediatric brain tumor analysis. As the application of machine learning in medical diagnosis expands, its importance keeps rising. It can not only screen core variables that are critical to the identification and management of pediatric brain tumors from large, complex multi-dimensional datasets, but also improve the efficiency and interpretability of models, helping to increase the accuracy of treatment decision-making.

#### 2.4.2 Wrapper Methods

In the field of machine learning feature selection, wrapper approaches first train an algorithm using a feature subset, then use the generated outcomes to optimize that subset. While they require more computation time, they outperform filter approaches, and can help screen optimal features for childhood brain tumor research. A typical technique under this framework, Recursive Feature Elimination (RFE), works by recursively eliminating low-importance features before retraining the model. In pediatric glioma research, this approach has been utilized to identify key characteristics in MRI images that could be used to distinguish tumor kinds, grades, and patient survival outcomes [93]. Genetic Algorithm (GA) is another wrapper-based feature selection technique. Drawing on the logic of natural selection, it adopts a population-based approach to search for the optimal feature subset, with the core goal of maximizing model accuracy. This algorithm has been applied to genetic and multimodal imaging research on pediatric brain tumors, to support tasks including tumor classification prediction and survival prediction [94].

#### 2.4.3 Embedded Methods

Embedded feature selection methods differ from wrapper approaches, as they can complete feature prioritization at the same time a machine learning model is trained. This approach not only guarantees high model accuracy, but also outperforms wrapper methods in operational efficiency. Mainstream embedded algorithms fall into two categories: The first category is linear regression with an L1 penalty term, that is, Lasso Regression under L1 Regularization. Its penalty term can set the coefficients of low-importance features to zero, making it suitable for high-dimensional datasets such as gene expression profiles and multi-parametric MRI data. This method has been applied to adolescent brain cancer classification and survival prediction for brain tumor patients, as detailed in [95]; the second category is Random Forest, which ranks features according to their predictive importance for the target. It has been used for pediatric brain tumor classification to extract key features from clinical data including MRI images, age, and tumor location, as described in [96].

#### 2.4.4 Dimensionality Reduction Techniques

When processing typical high-dimensional data such as gene expression data and complex imaging features, dimensionality reduction techniques can compress the number of features while retaining the core information of the dataset. PCA, a method commonly used across the industry, completes the conversion from high-dimensional to low-dimensional data by retaining the principal components that preserve the largest amount of information, and it is currently an extremely frequently applied dimensionality reduction method. Numerous prior studies have applied dimensionality reduction methods to process multimodal imaging data of pediatric brain tumors, compressing data dimensions to extract effective patterns that support prognostic prediction and tumor classification; the t-SNE algorithm has also been used to integrate genetic and imaging data to achieve low-dimensional visual grouping of tumor subtypes.

#### 2.4.5 Hybrid Methods

Feature prioritization is a core process in the field of pediatric brain tumor data analysis. For the multi-dimensional data involved in these studies, a range of feature processing methods can be applied, including filter, wrapper, embedded, and hybrid methods. All individual single-type methods have their own limitations, while combined methods often achieve superior analytical performance. The academic community is currently exploring the potential of hybrid feature ranking methods to improve the accuracy and reliability of machine learning models. These methods are particularly well-suited for complex data types such as genetic information and multi-modal imaging data. Among them, the well-established filter-wrapper hybrid method first eliminates redundant features to reduce dimensionality via filter methods, then refines features using wrapper methods such as RFE. This approach has been successfully applied to pediatric brain tumor analysis, where it screens out imaging features capable of distinguishing tumor types and survival outcomes to improve classification accuracy. Ensemble methods also target improving model performance: they can integrate multiple machine learning techniques, as well as different feature ranking methods to identify the optimal features suitable for a range of models. Existing studies have integrated genetic, clinical, and radiological data to construct pediatric brain tumor prediction models. Both categories of combined methods can amplify the advantages of individual single methods, delivering stable and precise analytical outputs. Feature prioritization will continue to be crucial in enhancing brain tumor patients' diagnostic outcomes as machine learning methods continue to grow and evolve.

### 2.5 Pediatric Brain Tumor Classification

In the realm of medical imaging, the classification of pediatric brain tumors is crucial for precise diagnosis and therapy. To improve the classification accuracy of pediatric brain tumor diagnosis utilizing MRI and CT scan data, several machine learning and deep learning techniques have been put forth. Gliomas, medulloblastomas, and ependymomas are among the tumor types that these methods can automatically classify [53].

The complete diagnostic procedure for brain tumors is depicted in Fig. 6, which highlights the strategic application of several machine learning and deep learning algorithms for the identification, segmentation, and classification of tumor imaging data. Deep belief networks and generative adversarial networks are examples of detection algorithms; energy-based models and convolutional models are examples of segmentation techniques; and ensemble and neural networks are examples of classification methods for tumor type and grade classification [88,102].

> **Figure 6:** Overview of computational approaches for pediatric brain tumor diagnosis.

#### 2.5.1 Machine Learning Models

Due to their effectiveness in handling high-dimensional data, which is frequently present in medical image data, machine learning techniques including Support Vector Machine (SVM) [103], Random Forest (RF) [103], and K-Nearest Neighbor (KNN) [104] have also been employed. An effective machine learning approach for classifying pediatric brain tumor is SVM [103]. The idea behind SVM is to translate the data points to a higher-dimensional space in order to identify the best hyperplane for classifying the different types of tumors. High-dimensional data, which are frequently present in medical image data, have shown SVM's effectiveness. SVM works effectively with small data sets as well. When the right kernel is applied, SVM is also renowned for its resistance to model overfitting. But SVM also has the disadvantage of being computationally intensive, especially when dealing with big data sets. Even for hard-to-detect pediatric brain tumors, SVM improves classification performance by mapping the data points to higher-dimensional space [105]. In order to discriminate the data, Random Forests, an ensemble learning technique, combines the forecasts of several decision trees [106]. The capacity of this approach to function effectively with noisy and unbalanced data which is frequently the case in medical data, such as tumor imaging data makes it extremely beneficial. This approach is less vulnerable to model overfitting and may be applied to both continuous and categorical data. However, one disadvantage is that, due to the complexity of using many decision trees, the prediction may be slower than with decision trees and less interpretable. Using characteristics including shape, size, texture, and intensity of tumor imaging data from pediatric patients' MRI and CT exams, RF can categorize several types of brain tumors in pediatric patients. Based on characteristics taken from tumor imaging data, RF can be used to categorize and distinguish between benign and malignant pediatric brain tumors [107]. KNN [104] is a straightforward yet effective method that classifies new data points according to the majority class of their neighbors in the feature space. As a non-parametric technique, KNN works best with smaller data sets. Nevertheless, the KNN algorithm may perform less well with larger dimensions and be computationally costly during the prediction stage. For the classification of brain tumor areas in pediatric MRI and CT scans, KNN is a straightforward but effective method. The KNN algorithm determines the class based on the majority class after identifying the 'k' most similar data points in the data set. With the right feature representation approaches, KNN can be a straightforward but effective algorithm for the detection of pediatric brain tumors [108].

#### 2.5.2 Deep Learning Models

However, due to their capacity to directly learn hierarchical features from data without the need for human involvement, deep learning models particularly Convolutional Neural Networks (CNNs) [109] have exhibited positive performance in image classification tasks. Furthermore, certain hybrid models that combine the benefits of ML and DL approaches have demonstrated promising performance in increasing classification accuracy [110]. CNNs require a lot of data and processing resources to be used for image classification tasks, despite their purported high accuracy [111]. Conversely, conventional methods such as SVM and KNN can be complicated for high-dimensional clinical scan images, but they require less data and processing resources [112]. Deep learning networks that are particularly well-suited for image classification applications are CNNs [109]. CNNs are ideal for classification tasks involving pediatric brain tumors based on MRI or CT scan data because they can efficiently learn characteristics from images automatically. When it comes to classification jobs, CNNs typically offer excellent results. However, CNNs also have several disadvantages, such as the need for a large amount of training data and their resource-intensive nature, which may need the usage of GPUs. As the name implies, completely convolutional networks are a type of CNN in which convolutional layers are used in place of fully connected layers. Fully Convolutional Networks are therefore better suited for pixel-wise classification tasks, such as the detection of brain tumors in MRI. FCNs are more effective in identifying the boundaries of brain tumors because they preserve the spatial information of the images [113,114]. For 3D MRI images, the Fully Convolutional Networks are also more effective. However, the Fully Convolutional Networks' computational cost is increased, particularly for 3D scans, because they need large-scale datasets to train the models. When analyzing pediatric brain MRI or CT scans to detect brain tumors, CNNs may be more effective. CNNs are more effective at identifying brain tumors because of their feature representation capabilities, which allow them to automatically identify patterns of various brain tumors that affect children [115]. When utilized for semantic segmentation, U-Net has demonstrated remarkable efficacy in the segmentation of pediatric brain tumors through MRI scans. The U-Net model's architecture makes it ideal for applications requiring accurate pixel-by-pixel segmentation, such identifying tumor and healthy brain regions in young patients. The U-Net model is the most popular model for the task due to its accurate segmentation ability for places like tumor tissues [116]. Pre-trained models such as ResNet or VGGNet have shown promising results in the detection of pediatric brain tumors. These models can transfer knowledge learned from adult brain cancer recognition and general images to solve the core challenge of limited labeled data in this field, and researchers [117] also corroborates their extremely high application value in this domain. DBNs are generative models that can be used to use unlabeled data to learn feature representation. DBNs can be used to develop hierarchical representations from patient MRI scans, which can then be used to classify tumor locations by identifying patterns in pediatric brain tumors. In fields with little labeled data, DBNs' unsupervised learning capability can be rather beneficial [118]. RNNs, which are often applied to sequential data, can also be used to monitor the tumor's progression over time. RNNs can be used to detect brain tumors in pediatric patients by analyzing a series of MRI images to track the tumor's progression. Generative Adversarial Networks (GANs) have well-established mature applications in the field of brain tumor diagnosis and treatment, with relevant supporting research detailed in [119,59,120]. First, they support longitudinal monitoring of brain tumor patients to track the actual efficacy of treatment regimens; second, they generate synthetic MRI and CT images of pediatric brain tumor patients to expand diagnostic datasets, addressing the core clinical pain point of insufficient imaging data for rare pediatric brain tumors. Another type of hybrid approach [110] combines the advantages of machine learning and deep learning to improve classification accuracy: it uses CNNs to extract features, and produces decision outputs via random forests and SVMs. This model is compatible with all types of datasets and capable of identifying complex data patterns, but it has limitations of high training difficulty and high computational costs, with relevant research [121].

> **Figure 7:** Evolution of Pediatric Brain Tumor Segmentation Models (2000–2025).

Figure 7 demonstrates the evolution from traditional image processing and machine learning methodologies to advanced deep learning-based frameworks. The figure emphasizes significant advancements in segmentation approaches and their roles in enhancing tumor delineation accuracy, robustness, and clinical applicability.

### 2.6 Research Gaps

Although machine learning research in the field of pediatric medical artificial intelligence (AI) has made many major advances in recent years, the authors of this paper have sorted out six core unsolved problems that hinder the real-world implementation of relevant models:

1. First, the lack of large-scale annotated datasets of pediatric patients limits the development of effective models.
2. Second, the absence of standardized imaging technology increases the difficulty of model generalization.
3. Third, existing models rely on single-modal imaging; integrating multi-source data including MRI, CT, functional imaging, and genetic data can significantly improve their performance.
4. Fourth, it is necessary to develop automated real-time diagnostic systems that can integrate tasks across the entire workflow.
5. Fifth, the ability to classify tumors needs to be improved, and a customized classification system that integrates genomic data should be built.
6. Sixth, the problems of insufficient model interpretability (explainable AI) and excessively high computing power requirements must be solved, which accurately identifies the core research gaps in this field.

---

## 3. CONCLUSION AND FUTURE DIRECTIONS

Existing studies point out that early accurate diagnosis of pediatric brain tumors remains extremely challenging. Manual interpretation of MRI scan images requires a large amount of time and effort, which highlights the important value of cutting-edge computer-aided diagnostic methods. This review sorts out the key workflows for pediatric brain tumor identification and the application advantages of traditional machine learning methods; this review confirms that deep learning using CNN and U-Net architectures has achieved outstanding results in the diagnosis and assessment of malignant tumors. However, many obstacles in this field remain to be solved. In the future, more mature deep learning technologies need to be developed to support clinical diagnosis, ultimately improving the quality of care for pediatric patients and enhancing their quality of life.

In the field of clinical brain tumor diagnosis, core research priorities at the current stage must be advanced comprehensively, centered on the clinical adaptability of rolling out upgraded technologies. This study outlines seven core sub-directions and their implementation requirements:

1. First, develop multimodal imaging fusion technology that integrates MRI, CT, and molecular imaging to make up for the inherent flaws of single-modal imaging.
2. Second, build large-scale annotated datasets to support deep learning in completing tumor segmentation and classification tasks [123].
3. Third, develop hybrid models that combine the strengths of machine learning and deep learning to balance diagnostic accuracy and computational efficiency.
4. Fourth, create a real-time diagnostic system that integrates the functions of tumor detection, segmentation, and classification to reshape the brain tumor diagnosis and evaluation model.
5. Fifth, integrate genetic and imaging data to refine tumor typing and support personalized treatment.
6. Sixth, control model computational costs to guarantee the large-scale application of deep learning in medical scenarios [124].
7. Seventh, apply explainable AI technology to improve the clinical credibility and acceptance of models [125], ultimately promoting the early identification and early treatment of pediatric brain tumors and effectively improving patient prognosis.

---

## DECLARATIONS

### Funding and/or Conflicts of interests
The authors received no specific funding for this work. The authors declare that they have no known competing financial or non-financial interests that could have appeared to influence the work reported in this paper.

### Acknowledgments
The authors have no acknowledgments to declare.

### Research Involving Human Participants and/or Animals
This study did not involve human participants, human tissue, or animals. Therefore, ethical approval was not required.

### Informed Consent
Informed consent was not required as this study did not involve human participants.

### Data Availability
No datasets were generated or analyzed during the current study. Therefore, data sharing is not applicable.

### Author contributions
Sangeetha Saman conceptualized and designed the study and prepared the original manuscript. The authors Balaji Boddu and Syed Kashed conducted the literature survey, data collection, visualization and manuscript drafting. Mohammed Mubasheer participated in validation, proofreading, and critical editing of the manuscript. The authors D. Jagadeesan and Kavitha S reviewed, edited, and approved the final manuscript. All authors read and approved the final version of the manuscript.

---

## REFERENCES

[1] Mahajan A, Burrewar M, Agarwal U, Kss B, Mlv A, Guha A, et al. Deep learning based clinico-radiological model for paediatric brain tumor detection and subtype prediction. Exploration of Targeted Anti-Tumor Therapy 2023:669–84. doi:10.37349/etat.2023.00159.  
[2] Silva W, Gonçalves T, Härmä K, Schröder E, Obmann VC, Barroso MC, et al. Computer-aided diagnosis through medical image retrieval in radiology. Scientific Reports 2022;12. doi:10.1038/s41598-022-25027-2.  
[3] Lemaître G, Martí R, Freixenet J, Vilanova JC, Walker PM, Mériaudeau F. Computer-Aided Detection and diagnosis for prostate cancer based on mono and multi-parametric MRI: A review. Computers in Biology and Medicine 2015;60:8–31. doi:10.1016/j.compbiomed.2015.02.009.  
[4] Wang S, Jiang Y, Zheng R, Yu Y, Wei W, Li N. Mobilizing China and the Global Community to Confront the Treatment Desert for Pediatric Solid Tumors. Cancer Discovery 2024;14:26–9. doi:10.1158/2159-8290.cd-23-0451.  
[5] Nikam R, Yue X, Kaur G, Kandula V, Khair AM, Kecskemethy HH, et al. Advanced Neuroimaging Approaches to Pediatric Brain Tumors. Cancers 2022;14:3401–3401. doi:10.3390/cancers14143401.  
[6] Raucci U, Vecchia ND, Ossella C, Paolino MC, Villa MP, Reale A, et al. Management of Childhood Headache in the Emergency Department. Review of the Literature. Frontiers in Neurology 2019;10. doi:10.3389/fneur.2019.00886.  
[7] Walker D. Childhood brain tumors: It is the child’s brain that really matters. Frontiers in Oncology 2022;12. doi:10.3389/fonc.2022.982914.  
[8] Yang W, Cai Y, Chen J, Yang P, Ying Z, Liang Y, et al. Epidemiological characteristics, clinical presentations, and prognoses of pediatric brain tumors: Experiences of national center for children’s health. Frontiers in Oncology 2023;13. doi:10.3389/fonc.2023.1067858.  
[9] Huang S, Yang J. Targeting the Hedgehog Pathway in Pediatric Medulloblastoma. Cancers 2015;7:2110–23. doi:10.3390/cancers7040880.  
[10] Xu S, Cheng L, Huang C, Ye Y, Lai KN. The value of cerebrospinal fluid cytology in the diagnosis of atypical medulloblastoma: a case report and review of the literature. Frontiers in Medicine 2025;12. doi:10.3389/fmed.2025.1641160.  
[11] Collins VP, Jones D, Giannini C. Pilocytic astrocytoma: pathology, molecular mechanisms and markers. Acta Neuropathologica 2015;129:775–88. doi:10.1007/s00401-015-1410-7.  
[12] Khuong-Quang D-A, Buczkowicz P, Rakopoulos P, Liu X-Y, Fontebasso AM, Bouffet É, et al. K27M mutation in histone H3.3 defines clinically and biologically distinct subgroups of pediatric diffuse intrinsic pontine gliomas. Acta Neuropathologica 2012;124:439–47. doi:10.1007/s00401-012-0998-0.  
[13] Johung T, Monje M. Diffuse Intrinsic Pontine Glioma: New Pathophysiological Insights and Emerging Therapeutic Targets. Current Neuropharmacology 2016;15:88–97. doi:10.2174/1570159x14666160509123229.  
[14] Merchant TE, Li C, Xiong X, Kun LE, Boop FA, Sanford RA. Conformal radiotherapy after surgery for paediatric ependymoma: a prospective study. The Lancet Oncology 2009;10:258–66. doi:10.1016/s1470-2045(08)70342-5.  
[15] Schoenfeld AJ, Pekmezci M, Barnes MJ, Tihan T, Gupta N, Lamborn KR, et al. The superiority of conservative resection and adjuvant radiation for craniopharyngiomas. Journal of Neuro-Oncology 2012;108:133–9. doi:10.1007/s11060-012-0806-7.  
[16] Jessa S, Blanchet-Cohen A, Krug B, Vladoiu MC, Coutelier M, Faury D, et al. Stalled developmental programs at the root of pediatric brain tumors. Nature Genetics 2019;51:1702–13. doi:10.1038/s41588-019-0531-7.  
[17] Shida D. Editorial: Brain Metastases (Secondary Brain Tumor). Cancers 2025;17:3440–3440. doi:10.3390/cancers17213440.  
[18] Khalighi S, Reddy K, Midya A, Pandav K, Madabhushi A, Abedalthagafi M. Artificial intelligence in neuro-oncology: advances and challenges in brain tumor diagnosis, prognosis, and precision treatment. Npj Precision Oncology 2024;8. doi:10.1038/s41698-024-00575-0.  
[19] Scott N, Joy N, A. K Chimene, M. A Elizabeth, A. W David, Michael R, et al. Genomes for Kids: The Scope of Pathogenic Mutations in Pediatric Cancer Revealed by Comprehensive DNA and RNA Sequencing. Cancer Discovery 2021;11:3008–27. doi:10.1158/2159-8290.cd-20-1631.  
[20] Wu G, Parker M, Huether R, Zhang J, Gajjar A, Broniscer A, et al. Somatic histone H3 alterations in pediatric diffuse intrinsic pontine gliomas and non-brainstem glioblastomas. Nature Genetics 2012;44:251–3. doi:10.1038/ng.1102.  
[21] Bowles EJA, Miglioretti DL, Kwan ML, Bartels U, Furst A, Cheng SY, et al. Long-term medical imaging use in children with central nervous system tumors. PLoS ONE 2021;16. doi:10.1371/journal.pone.0248643.  
[22] Krishnan PT, Pradeep K, Khandelwal M, Gupta D, Nihaal A, Kumar TS. Enhancing brain tumor detection in MRI with a rotation invariant Vision Transformer. Frontiers in Neuroinformatics 2024;18. doi:10.3389/fninf.2024.1414925.  
[23] Kumar H, Green R, Cornfeld D, Condron P, Emsden T, Elsayed A, et al. Roadmap for an imaging and modelling paediatric study in rural NZ. Frontiers in Physiology 2023;14. doi:10.3389/fphys.2023.1104838.  
[24] Balani A, Sidpra J, Sudhakar S, Biswas A, Öztekin Ö, Capra V, et al. International Consensus Statement on the Radiologic Evaluation of Dysraphic Malformations of the Spine and Spinal Cord. American Journal of Neuroradiology 2024;45:673–80. doi:10.3174/ajnr.a8117.  
[25] Copeland A, Silver E, Korja R, Lehtola SJ, Merisaari H, Saukko E, et al. Infant and Child MRI: A Review of Scanning Procedures. Frontiers in Neuroscience 2021;15. doi:10.3389/fnins.2021.666020.  
[26] Aggarwal P, Luo W, Pehlivan KC, Hai H, Rajappa P, Cripe TP, et al. Pediatric versus adult high grade glioma: Immunotherapeutic and genomic considerations. Frontiers in Immunology 2022;13. doi:10.3389/fimmu.2022.1038096.  
[27] Toniutti M, Sasso AL, Carai A, Colafati GS, Piccirilli E, Baldo GD, et al. Central nervous system tumours in neonates: what should the neonatologist know? European Journal of Pediatrics 2024;183:1485–97. doi:10.1007/s00431-023-05404-3.  
[28] Jackson K, Packer RJ. Recent Advances in Pediatric Medulloblastoma. Current Neurology and Neuroscience Reports 2023;23:841–8. doi:10.1007/s11910-023-01316-9.  
[29] Martin D, Tong E, Kelly BS, Yeom KW, Yedavalli V. Current Perspectives of Artificial Intelligence in Pediatric Neuroradiology: An Overview. Frontiers in Radiology 2021;1. doi:10.3389/fradi.2021.713681.  
[30] Vossough A, Khalili N, Familiar A, Gandhi D, Viswanathan K, Tu W, et al. Training and Comparison of nnU-Net and DeepMedic Methods for Autosegmentation of Pediatric Brain Tumors. American Journal of Neuroradiology 2024;45:1081–9. doi:10.3174/ajnr.a8293.  
[31] Guimarães MD, Noschang J, Teixeira SR, Santos MK, Lederman HM, Tostes VS, et al. Whole-body MRI in pediatric patients with cancer. Cancer Imaging 2017;17. doi:10.1186/s40644-017-0107-7.  
[32] Puget S, Blauwblomme T, Grill J. Is Biopsy Safe in Children with Newly Diagnosed Diffuse Intrinsic Pontine Glioma? American Society of Clinical Oncology Educational Book 2012:629–33. doi:10.14694/edbook_am.2012.32.59.  
[33] Osorio DS, Qaddoumi I, Moreira DC. Editorial: Pediatric CNS tumors in low- and middle-income countries: expanding our understanding. Frontiers in Oncology 2025;15. doi:10.3389/fonc.2025.1548727.  
[34] Jahan, Shafi AA, Rusho MA, Hussain MdS, Dhrubo AFH. CerevianNet: parameter efficient multi-class brain tumor classification using custom lightweight CNN. Frontiers in Medicine 2026;12:1664673–1664673. doi:10.3389/fmed.2025.1664673.  
[35] Pattisapu MM, B. SA, P. VVR, A. VNCS. Image Preprocessing and Optimizer Sensitivity: Implications for Convolutional Neural Networks in Diagnosing Brain Tumors. Journal of Visualized Experiments 2026. doi:10.3791/69459.  
[36] Kalaiselvi T, Anitha T, Sriramakrishnan P. Data preprocessing techniques for MRI brain scans using deep learning models. Elsevier eBooks, Elsevier BV; 2022, p. 13–25. doi:10.1016/b978-0-323-91171-9.00006-5.  
[37] Warmuth‐Metz M. Imaging Guidelines for Pediatric Brain Tumor Patients, 2016, p. 55–67. doi:10.1007/978-3-319-42503-0_4.  
[38] Javaid I, Zhang S, Isselmou AEK, Kamhi S, Kulsum U, Ahmad IS. Hybrid Automated Brain Tumor Detection by Using FKM, KFCM Algorithm with Skull Stripping 2020:112–9. doi:10.1145/3431943.3431962.  
[39] AMoghe A, Singhai J. Image Registration: A Review of Elastic Registration Methods Applied to Medical Imaging. International Journal of Computer Applications 2013;70:6–11. doi:10.5120/11972-7827.  
[40] Xu Y, Hu S, Sun J. A Segmentation-Based Bias Field Correction Network 2026:260–4. doi:10.1109/emc68537.2026.11441642.  
[41] Tustison NJ, Avants B, Cook PA, Zheng Y, Egan A, Yushkevich PA, et al. N4ITK: Improved N3 Bias Correction. IEEE Transactions on Medical Imaging 2010;29:1310–20. doi:10.1109/tmi.2010.2046908.  
[42] Liberati F. Trimmed quality scores (FastQC and MULTIQC). Figshare 2026. doi:10.6084/m9.figshare.26764621.  
[43] Said HM. Determination human brain tumor marker gene carbonic anhydrase 9 (CA9) gene expression in different type of brain tumor cells 2013;58:403–7. doi:10.1109/iccme.2013.6548278.  
[44] Mohit MK, Devpriya DS. Handling Missing values through Imputation techniques: A Quantitative Survey. Research Square 2023. doi:10.21203/rs.3.rs-3401489/v1.  
[45] Manoranjan B. Brain Tumor Genomics. Cancer Genomics, 2014, p. 321–38.  
[46] Meisler SL, Kahana MJ, Ezzyat Y. Does data cleaning improve brain state classification? Journal of Neuroscience Methods 2019;328:108421–108421. doi:10.1016/j.jneumeth.2019.108421.  
[47] Feature Level Fusion for Brain Tumor Detection using Deep Neural Network 2019. doi:10.35625/am960127i.  
[48] Jiang Y, Zhang S, Chi J. Multi-Modal Brain Tumor Data Completion Based on Reconstruction Consistency Loss. PubMed Central 2023;36:1794–807. doi:10.1007/s10278-022-00697-6.  
[49] Nair AM, Kumar LS, Vimina ER. Multiclass Brain Tumor Classification of MR-Images Using ResNet-50 and Dimensionality Reduction Techniques 2023:1–6. doi:10.1109/conit59222.2023.10205710.  
[50] Bhunia B. Region Growing Segmentation for Brain Tumor Segmentation-Based MRI Images. International Journal for Research in Applied Science and Engineering Technology 2023;11:2725–8. doi:10.22214/ijraset.2023.54121.  
[51] Grady L. Targeted Image Segmentation Using Graph Methods, 2017, p. 111–40. doi:10.1201/b12281-5.  
[52] Yu H, Wang D, Tan Z. Level set methods and image segmentation 2002:204–8. doi:10.1109/miar.2001.930287.  
[53] Ramu B, Swamy MS, Teegala SP, Veerla S, Prasad PS, David S. Brain Tumor Detection In MRI Scans Using Hybrid AI Techniques. International Journal of Advances in Signal and Image Sciences 2026;12:654–64. doi:10.29284/00wyv075.  
[54] EXTRACTION OF REGION BOUNDARY PATTERNS WITH ACTIVE CONTOURS 2012:240–8. doi:10.5220/0003826102400248.  
[55] Ganesh S, Gomathi RM, Kannadhasan S. Brain tumor segmentation and detection in MRI using convolutional neural networks and VGG16. Cancer Biomarkers 2025;42. doi:10.1177/18758592241311184.  
[56] Elsharkawy M, Abdelhalim I, Azam MT, Abdelrahim M, Aboudessouki A, Sharaby I, et al. Deep learning and classical segmentation techniques for lung cancer imaging, 2026, p. 7–1. doi:10.1088/978-0-7503-3359-7ch7.  
[57] Dey N, Rajinikanth V. CNN-based segmentation of brain tumor from T2-weighted MRI slices. Magnetic Resonance Imaging 2022:127–45. doi:10.1016/b978-0-12-823401-3.00005-5.  
[58] Dweik M, Ferretti R. Integrating Anisotropic Filtering, Level Set Methods and Convolutional Neural Networks for Fully Automatic Segmentation of Brain Tumors in Magnetic Resonance Imaging. SSRN Electronic Journal 2022. doi:10.2139/ssrn.4037968.  
[59] Koc ABB, Akgün D. A Review of U-Net Based Deep Learning Frameworks for MRI-Based Brain Tumor Segmentation. Diagnostics 2026;16:506–506. doi:10.3390/diagnostics16040506.  
[60] Chaki J. Brain MRI segmentation using deep learning: background study and challenges. Elsevier eBooks 2022:1–12. doi:10.1016/b978-0-323-91171-9.00012-0.  
[61] Borkar S. Performance Evaluation of U-Net Architectures for Biomedical Image Segmentation 2022;20:2065–71. doi:10.48047/nq.2022.20.9.nq44239.  
[62] Kong L, Zhang Y. Multi-modal Brain Tumor Segmentation Using Cascaded 3D U-Net. 2021 6th International Conference on Image, Vision and Computing (ICIVC), 2021, p. 129–33. doi:10.1109/icivc52351.2021.9526932.  
[63] Ren X, Wang L, Nie D, Shen D, Lin W, Lao C. 3-D Fully Convolutional Networks for Multimodal Isointense Infant Brain Image Segmentation. UNC Libraries 2026. doi:10.17615/snqv-gd13.  
[64] Khan G, Khan N. Brain MRI Segmentation using efficient 3D Fully Convolutional Neural Networks 2018;17:2351–6. doi:10.1109/bibm.2018.8621509.  
[65] Wang G, Li J. Brain tumor image segmentation algorithm based on hybrid model feature extraction 2024;70:27–27. doi:10.1117/12.3020759.  
[66] Tampu IE, Bianchessi T, Blystad I, Lundberg P, Nyman PO, Eklund A, et al. Pediatric brain tumor classification using deep learning on MR images with age fusion. PubMed Central 2024;7. doi:10.1093/noajnl/vdae205.  
[67] Kazerooni AF, Khalili N, Liu X, Haldar D, Jiang Z, Zapaishchykova A, et al. BraTS-PEDs: Results of the Multi-Consortium International Pediatric Brain Tumor Segmentation Challenge 2023. The Journal of Machine Learning for Biomedical Imaging 2025;3:72–87. doi:10.59275/j.melba.2025-f6fg.  
[68] Bareja R, Ismail M, Martin D, Nayate A, Yadav I, Labbad M, et al. nnU-Net–based Segmentation of Tumor Subcompartments in Pediatric Medulloblastoma Using Multiparametric MRI: A Multi-institutional Study. PubMed Central 2024;6. doi:10.1148/ryai.230115.  
[69] Javaji SR, Gosai A, Mohapatra S, Schlaug G. Automated Ensemble Method for Pediatric Brain Tumor Segmentation. Lecture notes in computer science, Springer Science+Business Media; 2024, p. 211–20. doi:10.1007/978-3-031-76163-8_19.  
[70] Zhou M, Wagner M, Tabori U, Hawkins C, Ertl‐Wagner B, Khalvati F. Generating 3D brain tumor regions in MRI using vector-quantization Generative Adversarial Networks. OPUS (Augsburg University) 2024;185:109502–109502. doi:10.1016/j.compbiomed.2024.109502.  
[71] Pacchiano F, Tortora M, Doneda C, Izzo G, Arrigoni F, Ugga L, et al. Radiomics and artificial intelligence applications in pediatric brain tumors. World Journal of Pediatrics 2024;20:747–63. doi:10.1007/s12519-024-00823-0.  
[72] Liu L, Ouyang W, Wang X, Fieguth P, Chen J, Liu X, et al. Deep Learning for Generic Object Detection: A Survey. International Journal of Computer Vision 2019;128:261–318. doi:10.1007/s11263-019-01247-4.  
[73] Zhou M, Scott JG, Chaudhury B, Hall L, Goldgof DB, Yeom KW, et al. Radiomics in Brain Tumor: Image Assessment, Quantitative Feature Descriptors, and Machine-Learning Approaches. American Journal of Neuroradiology 2017;39:208–16. doi:10.3174/ajnr.a5391.  
[74] Nageshwaran V. Optimizing Feature Selection for Brain Tumor Classification using a Hybrid ACCSA and Utilizing Machine Learning Algorithms for classification 2024:1299–304. doi:10.1109/icosec61587.2024.10722749.  
[75] Uçar MK. Eta Correlation Coefficient Based Feature Selection Algorithm for Machine Learning: E-Score Feature Selection Algorithm. Journal of Intelligent Systems Theory and Applications 2019;2:7–12. doi:10.38016/jista.498799.  
[76] Gholami B, Norton I, Tannenbaum A, Agar NYR. Recursive feature elimination for brain tumor classification using desorption electrospray ionization mass spectrometry imaging 2012;2012:5258–61. doi:10.1109/embc.2012.6347180.  
[77] Eberhardt M, Kossebau FWH, König A. Automatic Feature Selection by Genetic Algorithms. Artificial Neural Nets and Genetic Algorithms, 2001, p. 256–9. doi:10.1007/978-3-7091-6230-9_63.  
[78] Takefuji Y. Mitigating biases in feature selection and importance assessments in predictive models using LASSO regression. Oral Oncology 2024;159:107090–107090. doi:10.1016/j.oraloncology.2024.107090.  
[79] Wang Y. The Functional Design and Application of Computerized Accounting System in the University Financial Management 2017. doi:10.2991/emcs-17.2017.259.  
[80] Phillips KR, Filippidis A, Mackel CE, Enríquez-Marulanda A, Vega RA. Octogenarian Brain Tumor Registry: Single-Institution Surgical Outcomes and Mortality Study. Brain Tumor Research and Treatment 2023;11:114–114. doi:10.14791/btrt.2023.0007.  
[81] Furata T, Tabuchi A, Adachi Y, Mizumatsu S, Tamesa N, Ichikawa T, et al. Primary brain tumors in children under age 3 years. Brain Tumor Pathology 1998;15:7–12. doi:10.1007/bf02482094.  
[82] Akhtar M. Histopathological features. Elsevier eBooks, Elsevier BV; 1989, p. 59–70. doi:10.1016/b978-0-7236-0941-4.50012-2.  
[83] Johnson AH, Rice M, Turner‐Henson A, Haase JE, Azuero A. Perceived Stress and the Fatigue Symptom Cluster in Childhood Brain Tumor Survivors. Oncology Nursing Forum 2018;45:775–85. doi:10.1188/18.onf.775-785.  
[84] Hanumanthappa S, Guruprakash CD. Feature Extraction from Brain MR Images for Detecting Brain Tumor using Deep Learning Techniques. International Research Journal on Advanced Science Hub 2023;5:242–7. doi:10.47392/irjash.2023.049.  
[85] Biggs M, Wang Y, Soni N, Priya S, Bathla G, Canahuate G. Evaluating Autoencoders for Dimensionality Reduction of MRI-derived Radiomics and Classification of Malignant Brain Tumors 2023;2023:1–11. doi:10.1145/3603719.3603737.  
[86] Chen Y. Multimodal Brain Tumor Segmentation Based on Multi-Scale Feature Extraction Network. Applied and Computational Engineering 2024;106:87–92. doi:10.54254/2755-2721/106/20241311.  
[87] FAROOQ U, Jing C. Recent Advances in Artificial Intelligence and Hybrid Model for Enhanced Brain Tumor Classification Using MRI. World Journal of Biology Pharmacy and Health Sciences 2026;25:192–203. doi:10.30574/wjbphs.2026.25.1.0045.  
[88] Ünlüleblebici S, Kaçmaz RN, Öztürk C. A Systematic Review of Machine Learning and Deep Learning Techniques for Brain Tumor Classification, and Grading. IEEE Access 2026;14:18075–98. doi:10.1109/access.2026.3659641.  
[89] Yu J, Nguyen J, Fang J, Naeem A, Cao Z, Sanjay K, et al. PathMoE: Interpretable Multimodal Interaction Experts for Pediatric Brain Tumor Classification. arXiv (Cornell University) 2026.  
[90] Vaart A van der. Rank, Sign, and Permutation Statistics 1998:173–91. doi:10.1017/cbo9780511802256.014.  
[91] Sahu TP, Ankita. Correlation Feature Selection (CFS) and Feature Weighting (CFW) Based Improved BPSO for Gene Selection and Cancer Classification 2023:208–14. doi:10.1145/3625156.3625199.  
[92] Mitra PP, Bokil H. Entropy and Mutual Information, 2007, p. 333–42. doi:10.1093/acprof:oso/9780195178081.003.0014.  
[93] Zulfikar AF, Supriyadi D, Heryadi Y, Lukas L. Comparison Performance of Decision Tree Classification Model for Spam Filtering with or without the Recursive Feature Elimination (RFE) Approach 2019:311–6. doi:10.1109/icitisee48480.2019.9004001.  
[94] Judson R. Genetic Algorithms. Encyclopedia of Optimization, 2008, p. 1254–7. doi:10.1007/978-0-387-74759-0_218.  
[95] Huang H, Liu X, Liang Y. Feature Selection and Cancer Classification via Sparse Logistic Regression with the Hybrid L1/2 +2 Regularization. PLoS ONE 2016;11. doi:10.1371/journal.pone.0149675.  
[96] Amiri S, Mahjoub MA, Rekik I. Bayesian Network and Structured Random Forest Cooperative Deep Learning for Automatic Multi-label Brain Tumor Segmentation 2018:183–90. doi:10.5220/0006629901830190.  
[97] Xia J, Bu H. Tumor Classification by Using PCA with Hybrid Feature Selection Mechanism. Energy Procedia 2011;11:4966–72. doi:10.1016/j.egypro.2011.10.936.  
[98] Balamurali M. t-Distributed Stochastic Neighbor Embedding. Encyclopedia of earth sciences series/Encyclopedia of earth sciences, Springer Nature (Netherlands); 2021, p. 1–9. doi:10.1007/978-3-030-26050-7_446-1.  
[99] Anand MD, Kumanan S, Girish R, Selvaraj T, Asokan P, Valle CD, et al. An Enhanced Genetic Algorithm for Assembly Planning. International Journal of Recent Technology and Engineering (IJRTE) 2019;8:36–42. doi:10.35940/ijrte.b1007.0782s319.  
[100] Wang H, Liu S. An Effective Feature Selection Approach Using the Hybrid Filter Wrapper. International Journal of Hybrid Information Technology 2016;9:119–28. doi:10.14257/ijhit.2016.9.1.11.  
[101] Alfaro E, Gámez M, García N. Ensemble Classifiers Methods 2018:31–50. doi:10.1002/9781119421566.ch3.  
[102] Sultan H, Ullah Z, Park KI, Kim J. Deep learning-based classification and segmentation of brain tumor progression with clinical pipeline by generative artificial intelligence. Expert Systems with Applications 2026;318:131988–131988. doi:10.1016/j.eswa.2026.131988.  
[103] Bhagat N, Kaur G. MRI brain tumor image classification with support vector machine. Materials Today: Proceedings, vol. 51, 2022, p. 2233–44.  
[104] Fayaz M, Shah AS, Wahid F, Shah A. A Robust Technique of Brain MRI Classification using Color Features and K-Nearest Neighbors Algorithm. International Journal of Signal Processing Image Processing and Pattern Recognition 2016;9:11–20. doi:10.14257/ijsip.2016.9.10.02.  
[105] Kharrat A, Halima MB, Ayed MB. MRI brain tumor classification using Support Vector Machines and meta-heuristic method 2015:446–51. doi:10.1109/isda.2015.7489271.  
[106] Abdullah O, Alnaggar MF, Jagadale BN, Narayan SH, Bondy M, Scheurer M, et al. MRI Brain Tumor Detection Using Boosted Crossbred Random Forests and Chimp Optimization Algorithm Based Convolutional Neural Networks. International Journal of Intelligent Engineering and Systems 2022;15:36–46. doi:10.22266/ijies2022.0430.04.  
[107] Alam A, Sobuj MdAK, Akbar MA, Saha A, Ghosh SK, Rafid STS. AI-Optimized Ensemble Model with Hyperparameter Tuning for Brain Tumor Detection 2026:1–6. doi:10.1109/icecte69292.2026.11429396.  
[108] Ramdlon RH, Kusumaningtyas EM, Karlita T. Brain Tumor Classification Using MRI Images with K-Nearest Neighbor Method 2019:660–7. doi:10.1109/elecsym.2019.8901560.  
[109] Mohanty N, Sarmadi M. Brain tumor MRI classification and identification using an image classification model via Convolutional Neural Networks. medRxiv 2024. doi:10.1101/2024.09.13.23299832.  
[110] Rao TC, Prasanna DL, Yeshoda B, Bhavani V, Kashyapu P, Kiran Y, et al. BRAIN TUMOR DETECTION AND CLASSIFICATION USING HYBRID DEEP LEARNING MODELS. International Research Journal of Modernization in Engineering Technology and Science 2024. doi:10.56726/irjmets51077.  
[111] Pereira S, Oliveira A, Alves V, Silva CA. On hierarchical brain tumor segmentation in MRI using fully convolutional neural networks: A preliminary study 2017:1–4. doi:10.1109/enbeng.2017.7889452.  
[112] Zhou W, Tianyun S, Ping L, Ma X. A General Framework and Key Technology for Intelligent Railway Video Recognition Platform 2019;39:256–60. doi:10.1109/icaibd.2019.8836987.  
[113] N S. Brain Tumor Detection from MRI Using Deep Learning. International Journal for Research in Applied Science and Engineering Technology 2026;14:1723–8. doi:10.22214/ijraset.2026.77630.  
[114] Sivakumar M, Sivaguruathan S. Transformative Fusion of VGG-16 and U-Net for Efficient Multimodal MRI Brain Tumor Substructure Segmentation. International Research Journal of Multidisciplinary Scope 2026;7:1262–72. doi:10.47857/irjms.2026.v07i01.08173.  
[115] Özer E. Brain Tumor Detection using Deep CNNs and Ensemble Algorithms over MRI Images. Computer Science 2024. doi:10.53070/bbd.1455902.  
[116] Ronneberger O. Invited Talk: U-Net Convolutional Networks for Biomedical Image Segmentation. Informatik aktuell, Springer Nature; 2017, p. 3–3. doi:10.1007/978-3-662-54345-0_3.  
[117] He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition 2016:770–8. doi:10.1109/cvpr.2016.90.  
[118] Hinton GE, Salakhutdinov R. Reducing the Dimensionality of Data with Neural Networks. Science 2006;313:504–7. doi:10.1126/science.1127647.  
[119] Graves A. Supervised Sequence Labelling with Recurrent Neural Networks. Springer Nature; 2012. doi:10.1007/978-3-642-24797-2.  
[120] Agravat RR, Raval MS. A Survey and Analysis on Automated Glioma Brain Tumor Segmentation and Overall Patient Survival Prediction. arXiv (Cornell University) 2021;28:4117–52. doi:10.1007/s11831-021-09559-w.  
[121] Sharma DrSKS, Tanushka Gupta, Priyanka. A Hybrid Machine Learning Model for Enhanced Classification Accuracy. Zenodo (CERN European Organization for Nuclear Research) 2026. doi:10.5281/zenodo.18168538.  
[122] Liu F, Guo H, Wang J. Efficient hybrid fuzzy weighted 3D FCNN with TSO PSO optimization for accurate multi modal MRI brain tumor classification. Scientific Reports 2025;15. doi:10.1038/s41598-025-19889-5.  
[123] Akbar MU, Larsson M, Blystad I, Eklund A. Brain tumor segmentation using synthetic MR images - A comparison of GANs and diffusion models. Scientific Data 2024;11. doi:10.1038/s41597-024-03073-x.  
[124] Sapoval N, Aghazadeh A, Nute M, Antunes DA, Balaji A, Baraniuk RG, et al. Current progress and open challenges for applying deep learning across the biosciences. Nature Communications 2022;13:1728–1728. doi:10.1038/s41467-022-29268-7.  
[125] Hossain MdI, Zamzmi G, Mouton PR, Salekin MS, Sun Y, Goldgof DB. Explainable AI for Medical Data: Current Methods, Limitations, and Future Directions. ACM Computing Surveys 2023;57:1–46. doi:10.1145/3637487.  

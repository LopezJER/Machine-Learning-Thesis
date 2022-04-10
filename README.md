# Improving Tomato Leaf Disease Classification with Synthetic Data Based on Generative Adversarial Networks

## Table of contents
1. [Concept Note](#concept_note)
2. [Dataset](#dataset)
3. [Guidelines](#guidelines)


## Concept Note <a name="concept_note">
Plant disease accounts for up to 40% of crop damages worldwide, contributing to a loss of about $220 billion dollars each year (Savary et al., 2019). To curb the mounting costs of crop loss, it is necessary to promote the precise and early diagnosis of plant diseases (Buja et al., 2019; Mahlein, 2016). A promising avenue to achieve this goal is the use of deep learning techniques. Deep learning is a subset of artificial intelligence that models the human brain in making predictions about unstructured, real-world data (Kai et al, 2013). An important application in this domain is image classification, whereby deep learning models are trained to extract relevant features of images and classify them given various classes (Dhaware, Chaitali and Wanjale, 2016). This application is  particularly relevant in crop disease detection as symptoms often have visual manifestations that show up in various parts of the plant.

The potentials of deep learning in crop disease classification, however, are greatly hampered by limited    image datasets. To build robust classification models, it is important to obtain datasets that are complete and representative (Picard et al., 2020). Yet, Barbedo (2018) points out that many of the crop disease datasets used in deep learning   only have a small number of labeled disease classes. Further, he reports that of the datasets that do have   significant number of disease classes, their images were taken in carefully controlled conditions that are not  truly representative of infected samples found in practice (Barbedo, 2016). For example, the widely used PlantVillage dataset compiled over 50, 000 images of healthy and unhealthy leaves across 38 crop-disease classes but nonetheless resulted in models with poor generalizability (Mohanty, Hughes, and Salathe, 2016). The limitation of the PlantVillage dataset, among others, is that all of its images had controlled lighting and backgrounds so that they could never represent most real-life infected samples (Barbedo, 2016). Consequently, models trained primarily under the PlantVillage dataset scored less than 50% accuracy when exposed to leaf images outside the dataset (Singh et al, 2020). This finding highlights the need for datasets that not only have a high quantity of data but are also diverse enough to cope   with the wide range of conditions found in the field.

In this regard, a relevant approach to diversifying datasets is data augmentation. Data augmentation is the process of warping or repeating existing data in order to expand a given dataset (Shorten and Khoshgoftaar, 2019). For example, in an image dataset, one may add new images simply through basic manipulations such as flipping, rotating, scaling, adjusting contrast and brightness, and color transformations (Perez and Wang, 2017). Such basic augmentation techniques are indispensable to crop disease classification where visual diagnoses may be done from different angles, with varying backgrounds, and at different parts of the day. In fact, a study by Enkvetchakul and Surinta (2021) has shown that adding zoom and rotations to a leaf disease dataset may increase accuracy of classification models. However, since much of the transformations are linear, Bi and Hu (2020) note that the diversity   afforded by these techniques may be quite limited . Mi and colleagues (2020) also advise that these basic techniques are better used for resisting noise rather than improving classification, for which sufficiently novel data must be obtained.

In view of the limitations of basic data augmentation, one may harness advances in deep learning to synthesize new image data that may be used for expanding limited datasets. The chief technology used for generative modeling in deep learning is the generative adversarial network (GAN). GANs are a family of machine learning frameworks that are used to generate synthetic data by learning the distribution of a given dataset (Goodfellow, 2020). GANs have been used to generate fake images of handwritten digits (Jha and Cecotti, 2020) , celebrity faces (Curto et al., 2020), and flora and fauna (Colton and Ferrer, 2021). An important use case for GANs is augmenting datasets for which data is expected to be scarce as in medical images (Uzunova et al., 2019) or images of rare plant diseases (Wang and Wang, 2021). Hence, GANs represent an opportunity for artificially diversifying limited datasets in crop pathology to improve classification models. Indeed, Bi and Hu (2020) have shown that adding GAN-generated leaf disease images to a limited crop dataset improves classification models by as much as 2%. Though, an important caveat in this finding   is that the authors have used the PlantVillage dataset in training, which we have previously shown is unfit for practical use. It remains to be seen if GANs may be used for augmenting crop disease datasets whose images have greater noise and variety, as is generally expected in most field condition images. 
Hence, my  thesis proposes the use of GANs to augment a more representative crop dataset for improved crop disease classification models. In contrast with Bi and Hu’s study (2020) which used the PlantVillage dataset, my thesis uses Huang and Chang’s (2020) tomato leaf disease dataset which includes images with multiple leaves and noisy backgrounds. Using this dataset is also strategic as tomato happens to be an economically important crop in the Philippines (Altoveros and Borromeo, 2007) but whose yield is threatened by pest and disease problems (Gorme et al., 2017). Further, as a perennial and tropical crop, the costs associated with its damage due to pests and diseases are higher compared to other crops (Agrios, 2005). Therefore, building an effective tomato disease classifier strengthened by GANs would aid local farmers in improving yield and economic productivity. 

Proving the efficacy of GANs in crop disease datasets is an important step in improving automated crop disease classification. Should GANs hold up to their effectiveness in data augmentation, the constraints of manual data collection could be relaxed as variety and quantity could in fact be achieved artificially. Instead of spending a great amount of time collecting sample images, agricultural practitioners may simply expand existing datasets through generative modeling, adding real images only for exceptional or noteworthy symptoms. This would contribute to an efficient training process of building robust classification models, helping farmers systematically diagnose infected crops encountered in the field. The systematic diagnosis of crop diseases is an important component of integrated crop disease management and has consequences for long-term food security and agricultural productivity (Mutanga et al., 2017). On the other hand, even if GANs could not at present, be proven to significantly impact data augmentation, the insights from this thesis can be used to further improve existing GAN architectures and eventually help improve data and model quality in crop   disease classification.
 
<details>
<summary>References</summary>
<br>
Agrios, G. (2005). Plant pathology. Elsevier.
  
Altoveros, N., & Borromeo, T. (2007). Country report on the state of plant genetic resources for food and agriculture philippines. The state of the plant genetic resources for food and agriculture of the philippines (1997- 2006).
  
Barbedo, J. (2016). A review on the main challenges in automatic plant disease identification based on visible range images. Biosystems Engineering, 144, 52–60. https://doi.org/10.1016/j.biosystemseng.2016.01.017
  
Barbedo, J. G. A. (2018). Impact of dataset size and variety on the effectiveness of deep learning and transfer learning for plant disease classification. Computers and Electronics in Agriculture, 153, 46–53. https://doi.org/10.1016/j.compag.2018.08.013
  
Bi, L., & Hu, G. (2020). Improving image-based plant disease classification with generative adversarial network under limited training set. Frontiers in Plant Science, 11. https://doi.org/10.3389/fpls.2020.583438
  
Buja, I., Sabella, E., Monteduro, A. G., Chiriacò, M. S., Bellis, D., Luvisi, A., & Maruccio, G. (2021). Advances in plant disease detection and monitoring: From traditional assays to in-field diagnostics. Sensors, 21, 2129. https://doi.org/10.3390/s21062129
  
Colton, S., & Ferrer, B. (2021). GANlapse generative photography. https://computationalcreativity.net/iccc21/wp-content/uploads/2021/09/ICCC_2021_paper_120.pdf
Curtó, D., Zarza, C., de la Torre, F., King, I., & Lyu, M. R. (2020, April). High-resolution deep convolutional generative adversarial networks. ArXiv:1711.06491 [Cs]. https://arxiv.org/abs/1711.06491
  
Dhaware, C., & Wanjale, M. (2016). Survey on image classification methods in image processing. International Journal of Computer Science Trends and Technology (IJCS T), 4. http://www.ijcstjournal.org/volume-4/issue-3/IJCST-V4I3P40.pdf
  
Enkvetchakul, P., & Surinta, O. (2021). Effective data augmentation and training techniques for improving deep learning in plant leaf disease recognition. Applied Science and Engineering Progress. https://doi.org/10.14416/j.asep.2021.01.003
  
Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2020). Generative adversarial networks. Communications of the ACM, 63, 139–144. https://doi.org/10.1145/3422622
  
Gorme, A. L., Gonzaga, Z., Capuno, O., Rom, J., McDougall, S., Goldwater, A., & Rogers, G. (2017). Growth and yield of tomato (Solanum lycopersicum) as influenced by different soil organic amendments and types of cultivation. Annals of Tropical Research, 116–128. https://doi.org/10.32945/atr39sb9.2017
  
Huang, Mei-Ling; Chang, Ya-Han (2020), “Dataset of Tomato Leaves”, Mendeley Data, V1, doi: 10.17632/ngdgg79rzb.1
  
Jha, G., & Cecotti, H. (2020). Data augmentation for handwritten digit recognition using generative adversarial networks. Multimedia Tools and Applications. https://doi.org/10.1007/s11042-020-08883-w
  
Kai, Yu, Lei, J., Yuqiang, C., Wei, & Xu. (2013). Deep learning: Yesterday, today, and tomorrow. Journal of Computer Research and Development, 50, 1799. https://crad.ict.ac.cn/EN/abstract/abstract1340.shtml
  
Mahlein, A.-K. (2016). Plant disease detection by imaging sensors – parallels and specific demands for precision agriculture and plant phenotyping. Plant Disease, 100, 241–251. https://doi.org/10.1094/pdis-03-15-0340-fe
  
Mi, J., Hao, X., Yang, S., Gao, W., Li, M., & Minjuan, W. (2020). Res-wgan: Image classification for plant small-scale datasets. https://doi.org/10.21203/rs.2.23790/v1
  
Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. Frontiers in Plant Science, 7. https://doi.org/10.3389/fpls.2016.01419
  
Mutanga, O., Dube, T., & Galal, O. (2017). Remote sensing of crop health for food security in Africa: Potentials and constraints. Remote Sensing Applications: Society and Environment, 8, 231–239. https://doi.org/10.1016/j.rsase.2017.10.004
  
Perez, L., & Wang, J. (2017, December). The effectiveness of data augmentation in image classification using deep learning. ArXiv:1712.04621 [Cs]. https://arxiv.org/abs/1712.04621
  
Picard, S., Chapdelaine, C., Cappi, C., Gardes, L., Jenn, E., Lefevre, B., & Soumarmon, T. (2020). Ensuring dataset quality for machine learning certification. IEEE Xplore, 275–282. https://doi.org/10.1109/ISSREW51248.2020.00085
  
Savary, S., Willocquet, L., Pethybridge, S. J., Esker, P., McRoberts, N., & Nelson, A. (2019). The global burden of pathogens and pests on major food crops. Nature Ecology & Evolution, 3, 430–439. https://doi.org/10.1038/s41559-018-0793-y
  
Shorten, C., & Khoshgoftaar, Taghi M. (2019). A survey on image data augmentation for deep learning. Journal of Big Data, 6. https://doi.org/10.1186/s40537-019-0197-0
  
Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). Plantdoc: a dataset for visual plant disease detection (pp. 249–253).
Uzunova, H., Ehrhardt, J., Jacob, F., Frydrychowicz, A., & Handels, H. (2019). Multi-scale GANs for memory-efficient generation of high resolution medical images. Lecture Notes in Computer Science, 112–120. https://doi.org/10.1007/978-3-030-32226-7_13
  
Wang, Y., & Wang, S. (2021). IMAL: An improved meta-learning approach for few-shot classification of plant diseases. IEEE Xplore, 1–7. https://doi.org/10.1109/BIBE52308.2021.9635575

</details>
</a>

## Dataset <a name="dataset">
Primary Dataset: [Huang and Chang's Taiwan Tomato Dataset](https://data.mendeley.com/datasets/ngdgg79rzb/1)
  
This dataset is used for generating fake images of tomato leaves. It also contains the official train and test datasets for image classification of the thesis.
  
Experimental Dataset: [PlantVillage Dataset](https://www.kaggle.com/emmarex/plantdisease)
  
This dataset was used for initial experiments. Image classifiers trained under this dataset scored nearly 100% accuracy, yet offered poor generalizability to real-life samples. Hence, we use the primary dataset with more diversified images for more accurate metrics.

## Guidelines <a name="guidelines">
1. Download the above datasets and place them in the appropriate directories. I recommend using Kaggle as the paths match the built-in folders in Kaggle.
2. The generative model takes up to 2 straight weeks to train. Save and load model checkpoints in the generative model to restore progress of generator and discriminator.
3. Use remote Jupyter notebook services to take advantage of free GPUs and background execution.

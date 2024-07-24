# Detecting without Training: An Open-vocabulary Object Detection Method for Identifying Hazardous Objects on the Construction Site
# Our contribution
•	The creation of the first dataset focused on egocentric views of construction site hazards, comprising over 5,000 images from various construction sites in both indoor and outdoor environments. This dataset serves as a significant resource for future research in construction safety.
•	An innovative OVOD-based construction hazard detection and classification method that avoids time-consuming training processes while showing exceptional robustness and a solid ability to generalize.
•	Extensive experiments have been conducted to evaluate the proposed system and compared with existing popular classification networks in various settings. These experiments validate the high quality and diversity of the dataset and the proposed system’s superior performance.

# Workflow
Our comprehensive system is structured into three main sections to address the challenge of identifying construction hazards on sites: 'EgoCon' system setup, data preparation and preprocessing, and experimental evaluations. The process begins with the hardware setup, where wide-angle cameras are mounted on safety helmets to collect data from various construction sites. The collected data is then filtered and manually annotated depending on whether it contains construction hazards. The final stage involves testing the system's effectiveness and assessing its usability by comparing its performance with existing deep-learning networks. Each of these stages will be explained in detail in the subsequent sections.

<p align="center"> <img src="imgs/workflow.png" width="100%"> </p>

# Dataset
Initially, the researchers collected more than 5500 images and manually removed those invalid images, following the above standards, resulting in a final collection of 4799 images. The dataset was categorized into two groups: ‘Hazard’ and ‘Non-hazard,’ in which 1652 images were labeled as ‘Hazards’ and 3147 images were labeled as ‘Non-hazards’. The dataset was also categorized into indoor and outdoor based on the environment, which contained a collection of 1610 indoor images and 3189 outdoor images. The source dataset comprised images captured directly with a wide-angle camera, while the correction dataset included these source images after performing camera calibration to correct distortions. From the comparative experiments on the source dataset and correction dataset, the authors found that camera calibration can significantly improve the performance of the proposed system. To compare the models’ performance in different settings, we extended the two datasets to four datasets: The source dataset categorized by scene (Source Scene), the correction dataset categorized by scene (Correction Scene), the source dataset categorized by indoor/outdoor, and correction dataset categorized by indoor/outdoor. In Source Scene and Correction Scene, the images have been systematically categorized according to the various sites from which they were sourced.

<p align="center"> <img src="imgs/datadistribution.png" width="100%"> </p>
<p align="center"> <img src="imgs/sampleimages.png" width="100%"> </p>

# Fine-tuning Deep Learning Networks
## Scene datasets with/without data augmentation
<p align="center"> <img src="imgs/table1.png" width="100%"> </p>

## Indoor/outdoor datasets with/without data augmentation
<p align="center"> <img src="imgs/table2.png" width="100%"> </p>

# The contrast between the best fine-tuned Deep Learning Networks and the proposed Ground-dino-based system on our four datasets
<p align="center"> <img src="imgs/table3.png" width="100%"> </p>

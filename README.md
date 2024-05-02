# USIS10K & USIS-SAM
![issues](https://img.shields.io/github/issues/LiamLian0727/USIS10K)
![forks](https://img.shields.io/github/forks/LiamLian0727/USIS10K)
![stars](https://img.shields.io/github/stars/LiamLian0727/USIS10K)
![license](https://img.shields.io/github/license/LiamLian0727/USIS10K)

This repository is the official implementation of "[Diving into Underwater: Segment Anything Model Guided Underwater Salient Instance Segmentation and A Large-scale Dataset]()", authored by Shijie Lian, Ziyi Zhang, Hua Li, Wenjie Li, Laurence Tianruo Yang, Sam Kwong, and Runmin Cong, which has been accepted by ICML2024! 🎉🎉🎉

If you found this project useful, please give us a star ⭐️ or cite us in your paper, this is the greatest support and encouragement for us.

### :rocket: Highlights:
- **USIS10K dataset**: We construct the first large-scale USIS10K dataset for the underwater salient instance segmentation task, which contains 10,632 images and pixel-level annotations of 7 categories. As far as we know, this is the **largest salient instance segmentation dataset** available that simultaneously includes Class-Agnostic and Multi-Class labels.
  
  ![dataset img](figs/usis_dataset.png)
- **SOTA performance**: We first attempt to apply SAM to underwater salient instance segmentation and propose USIS-SAM, aiming to improve the segmentation accuracy in complex underwater scenes. Extensive public evaluation criteria and large numbers of experiments verify the effectiveness of our USIS10K dataset and USIS-SAM model.
- 
  ![framework_img](figs/framework.png)

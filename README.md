# SuperEEM: A Super-Resolution Reconstruction Framework for Rapid and Reliable Excitation-Emission Matrix Fluorescence Analysis
## About
Excitation-emission matrix (EEM) fluorescence spectroscopy is widely employed for characterizing complex samples in food analysis and environmental monitoring. However, its practical utility is hindered by an inherent trade-off: high-resolution EEM collection is time-consuming, while rapid low-resolution measurements sacrifice critical spectral details. Herein, we introduce SuperEEM, an EEM-specific super-resolution framework that combines a modified U-Net–GAN architecture with residual learning, channel attention, and a quantitative-fidelity-oriented hybrid loss to recover high-resolution spectral features while preserving fluorescence-intensity consistency and excitation-emission relationships.

-------------
![image](https://github.com/JunZeng1999/SuperEEM/edit/blob/Images/Figure1.jpg)

#### The flowchart of this work.

-------------
<img width="1920" height="2880" alt="Figure2" src="https://github.com/user-attachments/assets/cc259fb8-3583-4576-81e7-4da78a419d67" />

#### Comparison of EEMs (PAHs) reconstructed by different super-resolution methods. (a) Original high-resolution EEM. (b-f) EEMs reconstructed from low-resolution inputs using (b) nearest neighbor interpolation, (c) bilinear interpolation, (d) U-Net, (e) SRGAN, and (f) SuperEEM. (g) Representative emission profiles and (h) excitation profiles extracted from the dotted lines in (a).

-------------
<img width="2952" height="3096" alt="Figure3" src="https://github.com/user-attachments/assets/27b422bb-f3a5-4cd7-888f-84cc05bf95f1" />

#### The resolved normalized excitation spectral profiles and normalized emission spectral profiles obtained by ATLD algorithm based on EEMs (PAHs) reconstructed by different methods. (a) Original high-resolution EEMs. (b) Original low-resolution EEMs. (c-g) EEMs reconstructed from low-resolution inputs using (c) nearest neighbor interpolation, (d) bilinear interpolation, (e) U-Net, (f) SRGAN, and (g) SuperEEM.

-------------
<img width="3312" height="3720" alt="Figure4" src="https://github.com/user-attachments/assets/0054e10b-9f77-4557-a7dd-daa093a0fb7a" />

#### Classification performance and spectral fidelity of reconstructed EEMs. Radar plots illustrating classification performance for (a) the edible oil test set and (b) the beer prediction set, evaluated using accuracy, macro-recall, macro-precision, macro-specificity, macro-AUC, and macro-F1-score. (c-f) Representative HR, LR, bilinear interpolated, and SuperEEM reconstructed EEMs, respectively. Red dashed circles and arrows highlight the recovered fine spectral structures and characteristic shoulder peaks.

-------------
<img width="1352" height="862" alt="Figure5" src="https://github.com/user-attachments/assets/ff2ceaaf-806a-4e33-9f8d-701db34ca5f4" />

#### Graphical user interface of SuperEEM

-------------

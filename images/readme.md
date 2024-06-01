# Visualization

## Top 100 images flagged by different SOTA methods. Note that they fail to capture the failure mode of duplicate images with conflicting labels.

Use ctrl + scroll (or use pinch to zoom on mobile) to zoom into images.

### Top CIFAR100 images detected by SSFT as easy to forget
![SSFT](./ssft_cifar100_200.png)

Figure 1: Top CIFAR100 images detected by SSFT as easy to forget

### Top CIFAR100 images detected by SimiFeat as likely noisy
![SimiFeat](./simifeat_cifar100_200.png)

Figure 2: Top CIFAR100 images detected by SimiFeat as likely noisy

### High curvature samples from training set according to Slo-Curves (Garg & Roy, 2023)
![Slo-Curves](./slo-curves-rank.png)

Figure 3: High curvature samples from training set according to Slo-Curves (Garg & Roy, 2023). Obtained from ResNet18 trained without weight decay on CIFAR100

## Calibration curves: Early stopping using curvature of training set as a criterion yields better-calibrated networks.

### Model calibration at lowest validation loss stopping (Epoch 17)
![Epoch 17 Calibration](./epoch17_cal.png)

Figure 4: Model calibration at lowest validation loss stopping (Epoch 17).

### Model calibration at highest curvature stopping (Epoch 19)
![Epoch 19 Calibration](./epoch19_cal.png)

Figure 5: Model calibration at highest curvature stopping (Epoch 19).

### Model calibration at end of training (overconfident model)
![End of Training Calibration](./epoch200_cal.png)

Figure 6: Model calibration at end of training (overconfident model).

## Landscape Visualization
Visualizing the loss landscape and decision boundary during training on a toy dataset.

![Visualization GIF](./2D_Viz_input_space.gif)


## Higher Quality Images
### ImageNet, High Curvature Images
![ImageNet, High Curvature Images](./imagenet_highcurv.png)

### ImageNet, Low Curvature Images
![ImageNet, Low Curvature Images](./imagenet_lowcurv.png)

## Higher Quality Version of Most Memorized according to FZ scores on CIFAR100

![Most Memorized according to FZ scores on CIFAR100](./cf100_feld_worst.png)


## Higher Quality Version of Highest Curvature with Weight Decay on CIFAR100
![Highest Curvature with Weight Decay on CIFAR100](./cf100_feld_worst.png)
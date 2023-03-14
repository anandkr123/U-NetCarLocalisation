# U-NetCarLocalisation

- Dataset of 123 grayscale car images (200*300) with their corresponding **localization target as (x,y) co-ordinates**.

- Converted the (x, y) co-ordinates to a defined radius circle of 16 units and preparation of masked images (RoI) from it.

- Used the U-Net model architectire using Convolutional Neural Networks (CNNs), pooling, and features concatenation. Model trained till 400 iterations with   a batch size of 10; validation after every epoch.

- Precise localization of area of Region of Interest and minimized the loss to 0.02 and 0.04 for training and validation sets respectively.


EXAMPLE IMAGE 

![image_0010](https://user-images.githubusercontent.com/23450113/50496084-bbad1780-0a2d-11e9-8f06-57a6072028be.jpg)

MASKED FILE 
  
![image_0010_masked](https://user-images.githubusercontent.com/23450113/50496087-bea80800-0a2d-11e9-8b12-ed94c2fd88c3.png)

TRAINING AND VALIDATION LOSS (minimised upto 2.2% for TRAINING & 4.3% for VALIDATION SET

![val-loss_train-loss](https://user-images.githubusercontent.com/23450113/80147637-1ae45300-85b4-11ea-913e-3bb1eb5f8a19.png)

TEST RESULT (Localisation area is under the car, between front and rear tyres)

Input image             Predicted RoI               Masked Image
![final](https://user-images.githubusercontent.com/23450113/80147519-e8d2f100-85b3-11ea-856d-81b81905a3e1.png)

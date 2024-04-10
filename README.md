# Age and Gender Deep Learning Classification - CMPE258

## Group Members:

- Markus Valvur
- Alan Park
- Pallavi Suma

# Project Milestone Report

## List of Achievements So Far (Please see .ipynb file for code)

Thus far we have been able to successfully create a deep learning model that can predict both the age and gender of a person based on an image. While we do have a working model that does in essence what we want to achieve, there is still much room for improvement moving forward. After training our model using the 24000 images over 10 epochs, we found our results to be a good starting point that show promise. With regards to age prediction, the mean absolute error (MAE) on the test dataset was found to be 6.33. With regards to gender prediction, the MAE was 0.41. These values demonstrate that our model is properly predicting the age and gender of a person based on the image, which is a great achievement. However, we believe that these values can be drastically improved upon. For the majority of our modules, we would classify them still in the development stage.

The first module to discuss is our CustomAgeGenderDataset class, which potentially is our only module that can be considered fully functional. This is due to the fact that it properly takes the images (our data) and prepares it to be used for our model. We do not believe that any further work will be needed on this module. The next module, our AgeGenderPredictionModel, is definitely still in the development phase. Given our results, we believe that we could potentially make our model more robust, which should help increase the effectiveness. Additionally, our training module can be considered as still in the development phase. This is because there is a lot of room for us to experiment and change our training methodology with regards to optimization, regularization, hyperparameter tuning, and much more. Lastly, our evaluation module is essentially functional, however, we may consider adding additional metrics in order to get more information regarding how our model is performing.

Overall, there is still plenty of work to be done. While our modules are essentially still in the development phase, we have seen a lot of promise based on our initial results. Having a working age and gender prediction model is a great achievement, and improving upon what we have worked on should yield even better results. We are pleased with our progress to this point and look forward to continuing the work moving forward.

## Detailed Description of the Identified ‘Baseline’ Modules

CustomAgeGenderDataset(Dataset): This module performs a basic form of "pre-processing" of our selected image dataset. It extracts age and gender from filenames, and normalizes gender to 0 or 1 for male/female.

AgeGenderPredictionModel(nn.Module): This module creates a CNN, utilizing ResNet18 as a pre-trained base model. We use 'fc_age' as an age prediction layer, and 'fc_gender' as a gender prediction layer. The 'forward' method dictates how the input data moves through the model.

Training Module: We run a selected number of epochs to train our model and evaluate its performance. It calculates the loss and updates model weights using back propagation.

Evaluation Module: We evaluate the model's performance/accuracy through Mean Loss and Mean Absolute Errors on our test set. The train-test split is 80/20%. We utilize torch.save() to save our final model.

## Challenges

Currently, our MAE scores are not ideal and much can be improved in our model's accuracy/performance. We are satisfied with having a running model, but would like to work on lowering these values. The challenges lie within experimenting and finding ways to improve our model in order to achieve the results that we are looking for.

## Plans to Overcome Challenges

Essentially, our plan to overcome our current challenges is to continue to work on improving our model itself and how we are training it. The model could be drastically improved by increasing the robustness/complexity, which should ensure that it can effectively learn from the training and perform much better when it is being evaluated. Additionally, we really want to focus on how we are training our model. By tuning hyperparameters and experimenting with various optimization techniques, we hope that our model will be able to better learn about age and gender prediction from our dataset, which should hopefully improve our models performance and yield better results.

## References

Dataset (24k cropped images): https://www.kaggle.com/datasets/abhikjha/utk-face-cropped

PyTorch: https://pytorch.org/

Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2017). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10), 1499-1503.

Castrillon Santana, M., Lorenzo Navarro, J., Lorenzo Navarro, J., & Hernández Tejera, M. (2017). Age and gender classification from facial images: A comparative study. Expert Systems with Applications, 78, 173-181.

OpenFace: https://github.com/cmusatyalab/openface


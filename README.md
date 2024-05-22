# Age and Gender Deep Learning Classification - CMPE258

## Group Members:

- Markus Valvur
- Alan Park
- Pallavi Suma

### To access dataset, please visit the following link and download: https://www.kaggle.com/datasets/abhikjha/utk-face-cropped

# AGE GENDER PREDICTION   REPORT
	
## Github Repo:  https://github.com/mhvalvur/cmpe258project

### Abstract (Focus Area)

This work presents a deep learning approach for simultaneously predicting age and gender from facial images. Given the widespread applications of facial analysis in areas like security, biometrics, and human-computer interaction, accurate age and gender classification from visual data is an important problem. However, building robust models that can generalize well across diverse age groups, ethnicities, and imaging conditions remains challenging. 

Our proposed model is based on a convolutional neural network (CNN) architecture that takes cropped facial images as input and learns discriminative feature representations for the two tasks of age regression and gender classification. We explore several network backbones pre-trained on large face recognition datasets to efficiently transfer learned facial features. Multi-task learning is employed to share representations across the age and gender prediction branches.

The model is trained and evaluated on a large dataset of over 20,000 facial images labeled with ground truth age and gender. Extensive data augmentation techniques including random crops, flips, and color jittering are used to improve generalization. We systematically evaluate the performance of the model on held-out test sets in terms of mean absolute error (MAE) for both age and gender prediction tasks. 

### Dataset

We utilize the UTKFace dataset for training and evaluating our age and gender prediction model. The UTKFace dataset (https://susanqq.github.io/UTKFace/) is a large-scale and diverse collection of over 23,000 facial images labeled with age, gender, and ethnicity information.

The dataset was constructed over a long period spanning age ranges from 0 to 116 years old. It contains a high degree of variation in terms of pose, facial expression, illumination, occlusion, resolution, background, and other imaging conditions. The age and gender ground truth labels were carefully annotated through a manual process.
Some key attributes of the UTKFace dataset that make it well-suited for this task are:
- Large size of over 23,000 images to train deep neural networks effectively
- Wide age range from infants to elderly to learn discriminative age features  
- Balanced gender distribution (51% males, 49% females)
- Multi-ethnic with images from White, Black, Asian, Indian, and others
- Diverse imaging conditions like varying illumination, resolutions, occlusions
- Range of poses from frontal to profile views
- Variation in expressions, makeup, accessories like glasses/hats
The diversity of the UTKFace dataset promotes learning representations that generalize well across different age groups, ethnicities, and imaging environments encountered in real-world scenarios. Its scale and careful annotation make it an ideal benchmark for age and gender facial analysis algorithms.


### Architectures 

### Requirements

### Hardware Requirements :

GPU with >=8GB VRAM (Recommended)
High-end CPU (Intel i7/AMD Ryzen 7 or better)
16GB+ RAM
Sufficient Storage (SSD preferred)

Software Requirements: 

  Ubuntu 18.04+ (Linux) or macOS 10.14+ or Windows 10  
  Python 3.11
  PyTorch 1.13+ (with CUDA if using GPU)
  CUDA Toolkit 11.x+ (if using NVIDIA GPU)
  Libraries: NumPy, Pandas, Matplotlib, OpenCV, Pillow, tqdm, torchvision.
  Adam Optimizer
  ReLU Activation

Additional:

  Updated GPU drivers (if using GPU)
  Python IDE (PyCharm/VS Code/Jupyter)
  Git for version control

### Methodology

### Model Architecture:
Convolutional Neural Network (CNN) with ResNet-18 as the backbone architecture.
Fully connected layers for age and gender prediction tasks.
Each fully connected layer consists of multiple linear layers.
Non-linear ReLU activation functions after linear layers.
Single neuron output layer for age prediction.
Single neuron output layer for gender prediction with sigmoid activation to constrain output between 0 and 1.

Training Process:
Model trained on the UTKFace dataset containing 23,000 images.
Training performed over 10 epochs.
Mini-batch gradient descent optimization approach used.

Loss Functions:

Mean Squared Error (MSE) loss for age prediction (regression task)
Binary Cross-Entropy loss for gender prediction (classification task)

Evaluation Metrics:
Mean Absolute Error (MAE) for age prediction
Mean Absolute Error (MAE) for gender prediction (between 0 and 1)


The ResNet-18 architecture with fully connected heads allows learning discriminative facial features useful for both age and gender prediction tasks in a multi-task learning setup. Systematic evaluation on held-out test sets provides insights into the model's performance across diverse age groups and genders.

### Results 

Results:

Result Interpretations

We are pleased with the performance of our age and gender prediction model, having achieved the initial goals set for this project. The key results are as follows:

Age Prediction:
- Mean Absolute Error (MAE): 5.4 years
- Goal: MAE less than 6 years ✅ Achieved

Gender Prediction: 
- Mean Absolute Error (MAE): 0.24
- Goal: MAE less than 0.3 ✅ Achieved

Age Prediction Analysis:
Predicting a person's age from facial images is a challenging task, even for humans. Factors such as makeup, facial hair, ethnicity, and individual variations can make age estimation difficult. In this context, achieving an MAE of 5.4 years on the diverse UTKFace dataset is a commendable result. It demonstrates the model's ability to learn discriminative age-related patterns from facial features effectively.

Gender Prediction Analysis:
Gender prediction, while still non-trivial, is generally considered an easier task compared to age estimation. Our model achieved an MAE of 0.24 for gender prediction, which is satisfactory but leaves room for improvement. Potential factors affecting the performance could be the presence of gender-ambiguous faces, occlusions, or biases in the dataset. Further fine-tuning and analysis of failure cases could lead to better gender prediction capabilities.

### Future Work

While we have met our initial performance targets, there is always scope for improvement in such complex tasks. For age prediction, exploring more advanced architectures, larger datasets, or incorporating additional modalities (e.g., voice data) could further enhance the model's accuracy. As for gender prediction, a deeper analysis of errors, dataset biases, and techniques like adversarial training could help mitigate potential issues and improve the model's robustness.

Overall, we are delighted with the progress made in this project and the promising results achieved. The insights gained from this work will pave the way for more advanced and robust age and gender prediction systems in the future.


### Roles and Responsibilities 

Team Members

Markus
- Model Architecture/Design
- Model training

Alan
- Hyperparameter tuning
- Model testing/evaluation

Pallavi
- Research and Data Acquisition
- Dataset initialization 
- Tried training different Models like VGGnet, Basic Resent.

### References

[1] Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2017). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10), 1499-1503.

[2] Castrillon Santana, M., Lorenzo Navarro, J., Lorenzo Navarro, J., & Hernández Tejera, M. (2017). Age and gender classification from facial images: A comparative study. Expert Systems with Applications, 78, 173-181.

[3] OpenFace: https://github.com/cmusatyalab/openface

[4] PyTorch: https://pytorch.org/

[5] UTKFace: https://susanqq.github.io/UTKFace/


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


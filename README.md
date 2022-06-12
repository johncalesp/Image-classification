# Computer vision with tensorflow
This a classification model using transfer learning with tensorflow.

## Model Development
1. I build the model using the resources from kaggle on Intel Image Classification
2. Preprocess the images and load them using the data API from tensorflow
3. Import Xception pretrained model, than trained the last layers keeping the lower layers unchanged
4. Finally, I set layer.trainable = True for all the layers and trained the full model
5. Export the model to serve it as a web application

Link to model development https://www.kaggle.com/code/johnsnow27/intel-img-clf
## Model deployment
1. I use streamlit to make a prototype of a web application where a user can upload and image and the model can predict a label
2. Serve the model predictions in the backend
3. Deploy the application in Heroku.

Live Demo: https://tf-ml-classification.herokuapp.com/

![image](https://user-images.githubusercontent.com/81483067/173214322-ad9ff2dd-22a9-428b-84f3-d70485e35709.png)


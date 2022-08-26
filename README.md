# Computer vision with Tensorflow
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

![image](https://user-images.githubusercontent.com/81483067/186813045-adafd916-ec73-4696-ac25-4b66bc56a32c.png)

**After uploading an image, we obtain a prediction from the model**

![image](https://user-images.githubusercontent.com/81483067/186813336-1265fea7-7210-4918-bdec-062c9175db70.png)



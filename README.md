# COVID19-real time mask detection

The Real-Time Mask Detection System is designed to enhance public safety during the COVID-19 pandemic by providing a reliable solution for monitoring mask compliance in crowded environments. This system can be seamlessly integrated with surveillance cameras to automatically detect whether individuals are wearing masks. By analyzing video feeds in real time, it enables authorities to monitor compliance levels and take appropriate action to limit the spread of the COVID-19 virus.

Wearing masks has proven to be one of the most effective measures in reducing virus transmission, as they significantly decrease the likelihood of respiratory droplets spreading in the air. By implementing this system, organizations can ensure better adherence to health guidelines, thereby protecting the health of employees, customers, and the wider community.

The importance of this technology lies not only in its ability to facilitate compliance but also in fostering a safer environment as societies strive to navigate the ongoing challenges of the pandemic. By actively monitoring mask usage, we can collectively work towards reducing the risk of outbreaks and promoting public health.

## Model Development

The machine learning model for the Real-Time Mask Detection System was developed using a dataset consisting of 10,000 images of individuals wearing and not wearing masks. This data was sourced from reputable platforms such as Kaggle, DatasetNinja, and Flickr, ensuring a diverse representation of mask usage.

To identify faces in real-time video feeds, we utilized OpenCV along with the haarcascade_frontalface_default classifier. The model architecture was built using TensorFlow, employing a MobileNet pretrained model as the backbone of our deep neural network (DNN).

## Performance Metrics

Training Accuracy: 95.5%

Testing Accuracy: 93.5%

Despite the impressive accuracy metrics, the model is lightweight, with a file size of just 13.2 MB. This small size allows for fast processing and makes it easily integrable with low-level microcontrollers, ensuring seamless deployment in various surveillance applications.

## Demonstration 

<img width="215" height="175" alt="Screenshot 2024-11-04 at 3 13 06 AM" src="https://github.com/user-attachments/assets/18c30ac1-6e89-45a6-9b82-36199d5ad689">   <img width="215" height="175" alt="Screenshot 2024-11-04 at 3 13 12 AM" src="https://github.com/user-attachments/assets/0de4e79f-f32e-4bbb-b743-fa2173eec1aa">   <img width="215" height="175" alt="Screenshot 2024-11-04 at 3 13 21 AM" src="https://github.com/user-attachments/assets/d846df37-6c9f-448c-8ffd-e581876b141c">


## How to run the code

To view the code and performance details, please refer to the mask.ipynb file. To execute the face mask detection application, ensure that a webcam is connected. Run the video.py file to check whether an individual is wearing a face mask or not.

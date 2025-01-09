# iChef_Smart_IOT_Kitchen

## Project Description

iChef – A Smart Kitchen is an AI-powered cooking assistant designed to revolutionize the way people cook, customize, and experience food. With a focus on automation, personalization, and social engagement, iChef tackles common challenges faced by individuals balancing busy lifestyles and the desire for homemade-quality meals. Designed a machine that user controls through a mobile application that can cook any recipe according to the user's taste and also integrated an Image Processing algorithm using Convolutional Neural Networks (CNN), classifying the food as cooked or burnt with 96% accuracy.

## Hardware Implementation

1. **Hardware model outlook** <br>
The iChef prototype integrates precision mechanical components with IoT-enabled sensors to create an efficient automated cooking system. It features a funnel-based ingredient release mechanism, solenoid valve-controlled fluid flow, and servo motor-driven stirrer for real-time control, all coordinated by an Arduino Mega for seamless cooking automation.<br>
  ![image](https://github.com/user-attachments/assets/f3c7ad32-1bea-42f6-9883-4568706bd92d)
2. **Pin Connections**<br>
* **Induction stove - Relay modules**<br>
  The induction stove is controlled through relay modules connected to Arduino Mega, enabling functions like power control and temperature adjustments via specific pin connections.<br>
  ![image](https://github.com/user-attachments/assets/a9e7bcba-6582-4d1d-ac2c-c0dc232f7fa7)
* **Relay modules - Arduino MEGA**<br>
  The relay modules are interfaced with the Arduino Mega, where each relay's input, power, and ground pins manage cooking operations by controlling connected components.<br>
  ![image](https://github.com/user-attachments/assets/33ca9a9a-0046-4ccb-8fac-dfc082c3cf8d)
* **Arduino MEGA - HX77 Load Cell** <br>
  The Arduino Mega connects to the HX711 load cell module to accurately measure ingredient weight, ensuring precise control over quantities dispensed during cooking.<br>
  ![image](https://github.com/user-attachments/assets/5d3d08bf-cab8-4ac4-bec7-07ba9b4a94d7)
* **Arduino MEGA – Servo motor SG90 – High Torque Servo motor MG996R**<br>
The Arduino Mega controls the SG90 and MG996R servo motors, enabling precise angular movements for ingredient stirring and rotation with varying torque requirements.<br>
  ![image](https://github.com/user-attachments/assets/1d648570-f6de-48c6-892f-14f1163174b2)

## Software Implementation

1. **Application development to control cooking**<br>
* The iChef mobile application, enhanced using Figma, allows users to control the cooking process by selecting or customizing recipes, adjusting ingredient quantities, and setting induction stove temperatures. Users can monitor live video feeds of the cooking process, virtually stir ingredients, and end the cooking session with a simple interface for seamless automation.<br>
   ![image](https://github.com/user-attachments/assets/8a556414-547a-45a8-b287-be3d85b8ced9)
   ![image](https://github.com/user-attachments/assets/c3f145c3-a1af-4130-911b-6f8bf878022b)
   ![image](https://github.com/user-attachments/assets/db5c5a9d-0649-48c7-beee-eced273c77ab)
3. **Image Classification**
* **Block Diagram**<br>
The image classification block diagram outlines the process flow, including data collection, pre-processing, CNN model training, and prediction to classify food as either cooked or burnt.<br>
  ![image](https://github.com/user-attachments/assets/d5b7884c-2f87-4098-95b6-6a3a5de8bf1b)
* **Results**<br>
The image classification model achieved a high accuracy of 96.02%, effectively distinguishing between cooked and burnt food using a CNN-based approach.<br>
  ![image](https://github.com/user-attachments/assets/a11963bf-8f18-4901-ab74-4adea8c7041d)

## Technologies Used
* **Hardware**: Arduino Mega 2560, Induction Stove, Relay Modules, Load Cell (HX711), Servo Motor SG90, High Torque Servo Motor MG996R, Solenoid Valve
* **Software**: MIT App Inventor, Figma, TensorFlow, Keras, Google Colab, Arduino IDE











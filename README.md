# Reliability Test and Improvement of a Sensor System for Object Detection

## Overview
This study introduces an innovative approach to develop an algorithm that will act as an energy saving intelligent smart lighting system utilizing ADC data from ultrasonic sensor (FIUS). 
Unlike conventional motion-sensor based lighting, which often fails to accommodate scenarios where individuals remain stationary for extended periods, our system offers a nuanced solution that enhances user convenience while minimizing energy consumption.  
This system uses ultrasonic sensing systems through the integration of machine learning algorithms. This report delves into an experimental study that leverages the capabilities of ultrasonic sensors, specifically the Red Pitaya STEM Lab board and the Ultrasonic Sensor SRF02, 
to detect human presence and motion dynamics, ensuring that lights remain active even when slightest human motion is detected. 
The research employs a multifaceted signal analysis methodology, integrating Ultrasonic Sensors (FIUS) for distance calculation and surface characterization, alongside advanced signal processing techniques such as FFT and the Hilbert Transform. 
The core of the analysis involves using ML algorithms for distance measurement using Multilayer Perceptron (MLP) and classification of motion and steady environment using Rain Forest algorithm. These models were trained on ultrasonic signal data to classify object types and measure distances accurately. 
Additionally, enhancements to the ultrasonic measurement system are outlined, focusing on decision speed, accuracy, and usability improvements through optimized algorithms and advanced signal processing. 
## Installation

 **Clone the repository:**
```bash
   git clone <repository-url>
```
Ensure Python 3.9 or above is installed on your system. You can download it from the official Python website.
Install the required packages:
```bash
   pip install -r requirements.txt
```
Ensure the necessary data files are in the correct directories as expected by each script.

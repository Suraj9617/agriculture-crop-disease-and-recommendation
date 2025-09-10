# Agriculture Crop Disease Detection and Recommendation System

## Project Overview
This project is an AI-based system for detecting crop diseases in real-time and providing treatment recommendations. It leverages deep learning models, image processing, and recommendation algorithms to assist farmers in identifying crop health issues and suggesting appropriate actions.

## Features
- Real-time crop disease detection from images
- Accurate classification using pre-trained EfficientNet models
- Provides treatment and care recommendations for detected diseases
- User-friendly interface for easy interaction
- Supports multiple crop types with scalable architecture

## Technologies Used
- **Python**: Core programming language
- **TensorFlow & Keras**: Deep learning implementation
- **EfficientNet**: Pre-trained CNN model for image classification
- **OpenCV**: Image preprocessing and handling
- **LLMs (Large Language Models)**: Generating recommendations

## Methodology
1. **Image Preprocessing**: Resize, normalize, and augment images for better accuracy
2. **Disease Detection**: Classify images into healthy or diseased categories
3. **Recommendation System**: Suggest treatment based on detected disease
4. **Logging & Reports**: Keep track of detections and recommendations

### Workflow
1. Load pre-trained EfficientNet model
2. Pass crop images through the model
3. Detect and classify disease
4. Generate actionable treatment recommendations
5. Display results and logs

## Sample Results
**Sample Output 1**  
- Crop: Tomato  
- Disease: Early Blight  
- Recommendation: Apply fungicide, maintain proper irrigation  

## Sample Results

<div align="center">
  <strong>Sample Output 1</strong><br>
  <img src="https://github.com/user-attachments/assets/a5c46ba9-4aad-43ba-aabe-28faa66f0e11" width="600"><br>
  <em>HOME PAGE</em><br>
  <p style="max-width:600px;">This is the homepage of the Crop Disease Detection system showing the main interface.</p>
</div>

<div align="center">
  <strong>Sample Output 2</strong><br>
  <img src="https://github.com/user-attachments/assets/51fb5838-54d8-4dcc-93d0-301f91a1341e" width="600"><br>
  <em>UNHEALTHY LEAF RESULT</em><br>
  <p style="max-width:600px;">The system identifies an unhealthy leaf and displays the detected disease along with recommendations.</p>
</div>

<div align="center">
  <strong>Sample Output 3</strong><br>
  <img src="https://github.com/user-attachments/assets/f23f86c6-e3b7-4eee-9282-1811ea493780" width="600"><br>
  <em>HEALTHY LEAF RESULT</em><br>
  <p style="max-width:600px;">The system confirms a healthy leaf with no detected diseases, ensuring accurate monitoring.</p>
</div>

---

---

## Conclusion

The Crop Disease Detection system delivers **real-time crop health monitoring** using deep learning and **LLM-based recommendations**. It accurately detects both healthy and diseased leaves, provides treatment suggestions, and presents **clear visual results**. With its user-friendly interface, the system empowers farmers to make **timely, data-driven decisions**, improving crop yield and overall productivity.

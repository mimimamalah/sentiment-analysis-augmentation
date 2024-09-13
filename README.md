![Project Status](https://img.shields.io/badge/Project-Completed-brightgreen)

## **Sentiment Analysis and Data Augmentation with RoBERTa**

This project demonstrates fine-tuning a pre-trained RoBERTa model for sentiment analysis on product reviews and explores advanced model evaluation techniques. Additionally, the project incorporates data augmentation through paraphrasing using a T5 model, enhancing the dataset to improve sentiment analysis results.

### **Project Overview**
- **Sentiment Analysis:** Fine-tuned the RoBERTa model on a sentiment analysis task to classify product reviews as positive or negative.
- **Shortcut Identification:** Analyzed and identified potential shortcuts and salient features that the RoBERTa model might have learned during training.
- **Data Annotation:** Annotated new data points and measured annotation agreement between multiple annotators for consistency.
- **Data Augmentation:** Expanded the training dataset using a T5 paraphrasing model to generate additional training data and improved model performance with this augmented data.

### **Key Components**
- **Part 1: Sentiment Analysis** 
  - Dataset Processing
  - Model Training and Evaluation
  - Fine-Grained Validation
- **Part 2: Shortcut Identification** 
  - N-gram Pattern Extraction
  - Case Study and Distillation of Useful Patterns
- **Part 3: Data Annotation** 
  - Cross-annotation of New Data Points
  - Agreement Measure between Annotators
- **Part 4: Data Augmentation** 
  - Paraphrasing using T5
  - Model Retraining with Augmented Data

### **Technical Details**
- **Pre-trained Model:** [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta) fine-tuned on the sentiment analysis task.
- **Data Augmentation:** Used the [T5 paraphrase model](https://huggingface.co/docs/transformers/en/model_doc/t5) to augment the training dataset by paraphrasing the original data.
- **Annotation and Evaluation:** Cross-annotated datasets to calculate agreement statistics, ensuring high-quality data labeling.
- **Model Framework:** Utilized Hugging Face’s Transformers library, and Python for model training and evaluation.
- **Training Optimization:** Managed large files with Git LFS and employed GPU acceleration via Colab for faster model training.

### **Codebase File Structure**

```txt
.
├── data
│   ├── ...
├── models
│   ├── ...
├── predictions
│   ├── ...
├── tensorboard
│   └── ...
├── README.md
├── requirements.txt
├── sa.py
├── requirements.txt
├── shortcut.py
├── testA2.py
└── train.ipynb (main notebook)
```

### **Results**
- **Model Performance:** Improved the performance of the RoBERTa model with data augmentation, reducing overfitting to shortcuts.
- **Fine-Grained Evaluation:** Conducted thorough validation and analysis of sentiment prediction accuracy across multiple domains.
- **Agreement Measure:** Ensured high-quality data annotations with strong agreement measures between annotators.

This project demonstrates a complete workflow from data preprocessing and annotation to advanced model training and data augmentation, showcasing skills in NLP, machine learning, and data analysis.
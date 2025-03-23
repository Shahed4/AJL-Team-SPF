
# Equitable AI For Dermatology Team SPF

### **👥 Team Members**

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| Brianna Anaya  | @Briannanaya | Worked with EfficientNet CNN |
| Veronica Zhao | @verozhao | Worked with Ensembled MobileNetV2, EfficientNetB0, and DenseNet121 |
| Shahed Ahmed | @Shahed4 | Worked with Sequential, EfficientNetB0, InceptionV3, and MobileNetV2 |
| Maame Abena |  |  |
| Samin Chowdhury |  |  |
| Khadija Dial | @Kdial17 | Worked with the Sequential & Inceptionv3 model |

---
<!--
## **🎯 Project Highlights**

**Example:**

* Built a \[insert model type\] using \[techniques used\] to solve \[Kaggle competition task\]
* Achieved an F1 score of \[insert score\] and a ranking of \[insert ranking out of participating teams\] on the final Kaggle Leaderboard
* Used \[explainability tool\] to interpret model decisions
* Implemented \[data preprocessing method\] to optimize results within compute constraints

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)
🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

---
-->
## **🏗️ Project Overview**

* This project is part of a Kaggle competition sponsored by Break Through Tech and the Algorithmic Justice League, aiming to build more inclusive AI models in dermatology. The competition involves Break Through Tech AI Fellows from various programs, including Virtual, MIT, and UCLA, with mentorship from data science and AI TAs.
* The primary goal of this competition is to develop a machine learning model capable of classifying 21 different skin conditions across diverse skin tones. Given the historical underrepresentation of darker skin tones in dermatology AI datasets, this project seeks to address bias and improve model performance across all demographics. The competition evaluates submissions based on a weighted average F1 score.
* AI-driven diagnostic tools are increasingly used in healthcare, yet biases in training data can lead to disparities in diagnosis and treatment. By developing an inclusive dermatology AI model, this project contributes to reducing healthcare inequities, ensuring better diagnostic accuracy for historically marginalized communities. Our work aligns with ongoing research at institutions such as Stanford and MIT Media Lab, promoting fairness and explainability in AI models.

---

## **📊 Data Exploration**

**Dataset description**

* The competition dataset consists of dermatological images labeled with 21 different skin conditions. It includes metadata such as skin tone distribution and lesion types. To enhance model performance and fairness, we may incorporate external datasets and employ augmentation techniques.

**Data Exploration & Preprocessing Approaches**
- **Handling Class Imbalance:** Given the known imbalance in skin tone representation, we utilize data augmentation techniques such as flipping, rotation, and brightness adjustments to ensure fair representation across all skin tones.
- **Transfer Learning:** We fine-tune pre-trained convolutional neural networks (CNNs) to adapt them to our classification task, optimizing performance while reducing computational costs.
- **Fairness and Explainability:** Using AI fairness tools like Fairlearn, we analyze model biases and adjust our preprocessing pipeline accordingly.

**Challenges**

##### Class Imbalance  
- The dataset has an uneven distribution of skin tones, with lighter skin tones being overrepresented.  
- Certain skin conditions appear less frequently, making it difficult for the model to learn their patterns effectively.  

##### Variability in Image Quality  
- Images vary in resolution, lighting, and focus, which can affect model performance.  
- Some images may have noise that obscure skin conditions.  

##### Bias and Fairness Concerns  
- Existing dermatology AI models have historically struggled with darker skin tones due to biased training data.  
- Need to ensure that the model performs equitably across all demographic groups.  


### EDA Visualizations
**Class Distribution:** A bar chart displaying the frequency of each skin condition in the dataset, highlighting any class imbalances.
<img width="1019" alt="image" src="https://github.com/user-attachments/assets/122c5c4f-1394-4beb-a522-9988e0e914c3" />

**Skin Tone Representation:** A pie chart or histogram illustrating the distribution of skin tones across the dataset, revealing potential biases (1-lightest skin, 6-darkest skin).
   <img width="718" alt="image" src="https://github.com/user-attachments/assets/6a7fcd90-4c0d-4776-a69d-1e9e4f3be146" />
**Sample Image Augmentation:** A grid showcasing examples of augmented images, demonstrating transformations used to balance the dataset.
<img width="793" alt="image" src="https://github.com/user-attachments/assets/fa835906-78f5-493b-bed5-a4905525d84b" />

---
<!--
## **🧠 Model Development**

**Describe (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)

---

## **📈 Results & Key Findings**

**Describe (as applicable):**

* Performance metrics (e.g., Kaggle Leaderboard score, F1-score)
* How your model performed overall
* How your model performed across different skin tones (AJL)
* Insights from evaluating model fairness (AJL)

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## **🖼️ Impact Narrative**

**Answer the relevant questions below based on your competition:**

**WiDS challenge:**

1. What brain activity patterns are associated with ADHD; are they different between males and females, and, if so, how?
2. How could your work help contribute to ADHD research and/or clinical care?

**AJL challenge:**

As Dr. Randi mentioned in her challenge overview, “Through poetry, art, and storytelling, you can reach others who might not know enough to understand what’s happening with the machine learning model or data visualizations, but might still be heavily impacted by this kind of work.”
As you answer the questions below, consider using not only text, but also illustrations, annotated visualizations, poetry, or other creative techniques to make your work accessible to a wider audience.
Check out [this guide](https://drive.google.com/file/d/1kYKaVNR\_l7Abx2kebs3AdDi6TlPviC3q/view) from the Algorithmic Justice League for inspiration!

1. What steps did you take to address [model fairness](https://haas.berkeley.edu/wp-content/uploads/What-is-fairness_-EGAL2.pdf)? (e.g., leveraging data augmentation techniques to account for training dataset imbalances; using a validation set to assess model performance across different skin tones)
2. What broader impact could your work have?

---

## **🚀 Next Steps & Future Improvements**

**Address the following:**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## **📄 References & Additional Resources**

* Cite any relevant papers, articles, or tools used in your project

---
-->

## **📄 References & Additional Resources**

* Cite any relevant papers, articles, or tools used in your project

---

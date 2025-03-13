# Automating-the-Classification-of-Disengagements-using-Convolutional-Neural-Networks
This is the official repository for the Batchlor's Thesis of Elisabet Hein, titled "Automating the Classification of Disengagements using Convolutional Neural Networks".

The repository contains 8 binary classification models and data preprocessing script files.

## The Model Files

The models and their respective disengagement classes are as follows:

| Disengagement Class        | Numerical Mapping          | Model File  |
| ------------- |-------------| -----|
| 'OBS'      | 0 | “model_class_0_final.h5” |
| 'Safety'      | 1      |   “model_class_1_final.h5” |
| 'Turnback' | 2      |    “model_class_2_final.h5” |
| 'Pedestrian crossing'      | 3 | “model_class_3_final.h5” |
| 'Localization'      | 4      |   “model_class_4_final.h5” |
| 'Give way' | 6      |    “model_class_6_final.h5” |
| 'SPEED'      | 7 | “model_class_7_final.h5” |
| 'STOP'      | 8      |   “model_class_8_final.h5” |

There is no class 5 because the original "Bad Engage" label was removed and replaced with the most recent disengagement reason prior to the transition. This adjustment was made to ensure cleaner data for model training, as the "Bad Engage" label did not represent a distinct disengagement reason but rather an ongoing issue linked to a previously occurring class. By reassigning these instances to their preceding disengagement category, the dataset maintains a more precise and meaningful classification structure.

## The Script Files

Additionally, several preprocessing scripts were developed:

1. The script **"combining_topics.ipynb"** was developed to preprocess and unify key control and localization topics from multiple test drive datasets. It merges the selected topics into a single CSV file per test drive, aligning data based on timestamps while handling missing values through forward and backward filling to ensure continuity. Additionally, it filters out irrelevant columns containing metadata such as headers, component types, and frame identifiers, simplifying the dataset without compromising essential control and localization information. This streamlined dataset serves as a foundation for further analysis and model development.

2. The script **"adding_disengagement_types.ipynb"** was developed to enhance the previously unified CSV files by incorporating manually labeled disengagement data from the ADL. It detects disengagement transitions based on changes in the ‘drivemode’ column and adds three new columns: ‘disengagement’ (indicating whether a transition occurred), ‘disengagement_reason’ (mapping textual reasons to numerical values), and ‘disengagement_type’ (differentiating planned and unplanned disengagements). Additionally, the script replaces the "Bad Engage" label with the last seen disengagement reason before transition to prevent unintended learning patterns,  as that label indicates a disengagement happening immediately after engagement, suggesting an unresolved issue rather than a distinct cause. This refined dataset improves the analysis of disengagement events for further model development.

3. The script **"processing_for_waypoints_and_objects.ipynb"** was designed to extract and preprocess complex environmental perception and planning topics, such as detected objects, obstacles, and planned trajectories, into separate CSV files per ride. Since these topics contained nested JSON structures, the script parsed and transformed them into a structured format suitable for further analysis.

4. Building on this, the script **"expanding_with_objects_waypoints.ipynb"** merged these processed datasets into a unified format, ensuring they were properly structured for integration into the training pipeline. Instead of generating additional CSV files, this step utilized Pandas DataFrames to streamline data handling and improve efficiency. Additionally, this script extracted five-second time windows preceding each disengagement event, creating focused datasets that capture critical moments leading up to disengagements, enhancing their usefulness for model training and analysis.

5. The "**binary_models_training.ipynb"** script handles the full pipeline for training binary classification models for disengagement prediction. It standardizes disengagement event data, converts categorical features to numerical values, normalizes numerical data, and applies data augmentation if needed. The dataset is then split into training and validation sets.
For model training, separate 1D CNN-based binary classifiers are built for each disengagement class, using Adam optimization, binary cross-entropy loss, and EarlyStopping to enhance performance and prevent overfitting. After training, models are validated, and SHAP analysis is used to interpret feature importance, ensuring transparency in predictions.

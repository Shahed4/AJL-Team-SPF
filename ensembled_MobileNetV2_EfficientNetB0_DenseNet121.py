import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import glob
import matplotlib.pyplot as plt
import random
import math
from tqdm import tqdm
from tensorflow.keras import layers, models, optimizers, regularizers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, DenseNet121
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils import class_weight

def set_seeds(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma) * y_true
        focal = alpha * weight * cross_entropy
        return tf.reduce_sum(focal, axis=-1)
    return focal_loss_fixed

def mixup(x, y, alpha=0.2):
    batch_size = tf.shape(x)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    
    weight = tf.random.uniform([batch_size], minval=0, maxval=alpha)
    x_weight = tf.reshape(weight, [batch_size, 1, 1, 1])
    y_weight = tf.reshape(weight, [batch_size, 1])
    
    x_mixup = x * (1 - x_weight) + tf.gather(x, indices) * x_weight
    y_mixup = y * (1 - y_weight) + tf.gather(y, indices) * y_weight
    
    return x_mixup, y_mixup

def enhance_skin_image(image_path, size=224):
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        img = cv2.resize(img, (size, size))
        return img.astype(np.float32) / 255.0
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return np.zeros((size, size, 3), dtype=np.float32)

def prepare_data(folder_path, img_size=224):
    train_csv = os.path.join(folder_path, "train.csv")
    test_csv = os.path.join(folder_path, "test.csv")
    image_dir = os.path.join(folder_path, "train", "train")
    test_dir = os.path.join(folder_path, "test", "test")
    
    print("Loading datasets...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    train_df['file_path'] = train_df.apply(
        lambda row: os.path.join(image_dir, row['label'], f"{row['md5hash']}.jpg"), 
        axis=1
    )
    
    test_df['file_path'] = test_df.apply(
        lambda row: os.path.join(test_dir, f"{row['md5hash']}.jpg"), 
        axis=1
    )
    
    train_df = train_df[train_df['file_path'].apply(os.path.exists)].reset_index(drop=True)
    test_df = test_df[test_df['file_path'].apply(os.path.exists)].reset_index(drop=True)
    
    fitzpatrick_dummies = pd.get_dummies(train_df['fitzpatrick_scale'], prefix='fitz')
    train_df = pd.concat([train_df, fitzpatrick_dummies], axis=1)
    
    unique_classes = sorted(train_df['label'].unique())
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    
    train_df['label_idx'] = train_df['label'].map(class_to_idx)
    
    class_counts = train_df['label'].value_counts()
    print("\nClass distribution:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count} samples")
    
    weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_df['label_idx']),
        y=train_df['label_idx']
    )
    class_weight_dict = {i: min(w * 0.75, 5.0) for i, w in enumerate(weights)}
    
    return train_df, test_df, class_to_idx, idx_to_class, class_weight_dict

def create_mobilenetv2_model(img_size, num_classes):
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_efficientnet_model(img_size, num_classes):
    base_model = EfficientNetB0(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_densenet_model(img_size, num_classes):
    base_model = DenseNet121(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = True
    for layer in base_model.layers[:-15]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(192, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_train_generator(train_df, img_size=224, batch_size=16):
    datagen = ImageDataGenerator(
        preprocessing_function=lambda x: x * 2.0 - 1.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode='constant',
        cval=0
    )
    
    return datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='file_path',
        y_col='label',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

def create_val_generator(val_df, img_size=224, batch_size=16):
    datagen = ImageDataGenerator(
        preprocessing_function=lambda x: x * 2.0 - 1.0
    )
    
    return datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='file_path',
        y_col='label',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

def fix_test_generator(test_df, img_size=256, batch_size=16):
    test_df['temp_label'] = 'dummy_class'
    
    datagen = ImageDataGenerator(
        preprocessing_function=lambda x: x * 2.0 - 1.0
    )
    
    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='file_path',
        y_col='temp_label',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_df.drop('temp_label', axis=1, inplace=True)
    
    return test_generator

def f1_metric(y_true, y_pred):
    y_pred_classes = tf.cast(tf.argmax(y_pred, axis=-1), tf.int64)
    y_true_classes = tf.cast(tf.argmax(y_true, axis=-1), tf.int64)
    
    num_classes = tf.shape(y_true)[-1]
    
    def compute_f1(class_index):
        class_index = tf.cast(class_index, tf.int64)
        
        y_pred_binary = tf.cast(tf.equal(y_pred_classes, class_index), tf.float32)
        y_true_binary = tf.cast(tf.equal(y_true_classes, class_index), tf.float32)
        
        true_positives = tf.reduce_sum(y_true_binary * y_pred_binary)
        false_positives = tf.reduce_sum((1 - y_true_binary) * y_pred_binary)
        false_negatives = tf.reduce_sum(y_true_binary * (1 - y_pred_binary))
        
        precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
        recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())
        
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return f1
    
    f1_scores = tf.map_fn(
        compute_f1, 
        tf.range(tf.cast(num_classes, tf.int64), dtype=tf.int64),
        fn_output_signature=tf.float32
    )
    
    macro_f1 = tf.reduce_mean(f1_scores)
    
    return macro_f1

def weighted_f1_metric(y_true, y_pred):
    y_pred_classes = tf.cast(tf.argmax(y_pred, axis=-1), tf.int64)
    y_true_classes = tf.cast(tf.argmax(y_true, axis=-1), tf.int64)
    
    num_classes = tf.shape(y_true)[-1]
    
    weights = tf.reduce_sum(y_true, axis=0)
    weights = weights / (tf.reduce_sum(weights) + tf.keras.backend.epsilon())
    
    def compute_class_f1(class_index):
        class_index = tf.cast(class_index, tf.int64)
        
        y_pred_binary = tf.cast(tf.equal(y_pred_classes, class_index), tf.float32)
        y_true_binary = tf.cast(tf.equal(y_true_classes, class_index), tf.float32)
        
        true_positives = tf.reduce_sum(y_true_binary * y_pred_binary)
        false_positives = tf.reduce_sum((1 - y_true_binary) * y_pred_binary)
        false_negatives = tf.reduce_sum(y_true_binary * (1 - y_pred_binary))
        
        precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
        recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())
        
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return f1
    
    f1_scores = tf.map_fn(
        compute_class_f1, 
        tf.range(tf.cast(num_classes, tf.int64), dtype=tf.int64),
        fn_output_signature=tf.float32
    )
    
    weighted_f1 = tf.reduce_sum(f1_scores * weights)
    
    return weighted_f1

def create_callbacks(model_name):
    return [
        ModelCheckpoint(
            f'{model_name}_best.keras',
            monitor='val_f1_metric',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_f1_metric',
            patience=7,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_f1_metric',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode='max',
            verbose=1
        )
    ]

def train_model(model, model_name, train_generator, val_generator, class_weights, epochs=30):
    print(f"\nTraining {model_name}...")
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss=focal_loss(gamma=2.0),
        metrics=['accuracy', f1_metric, weighted_f1_metric]
    )
    
    callbacks = create_callbacks(model_name)
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    model.load_weights(f'{model_name}_best.keras')
    
    return model, history

def evaluate_model(model, val_generator, class_names):
    y_pred = model.predict(val_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_generator.classes
    
    weighted_f1 = f1_score(y_true, y_pred_classes, average='weighted')
    macro_f1 = f1_score(y_true, y_pred_classes, average='macro')
    
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    report = classification_report(
        y_true, y_pred_classes, 
        target_names=class_names, 
        output_dict=True
    )
    
    return weighted_f1, macro_f1, report

def train_ensemble(train_df, val_df, class_to_idx, class_weights, img_size=224, batch_size=16):
    num_classes = len(class_to_idx)
    
    train_generator = create_train_generator(train_df, img_size, batch_size)
    val_generator = create_val_generator(val_df, img_size, batch_size)
    
    models = {
        'mobilenetv2': create_mobilenetv2_model(img_size, num_classes),
        'efficientnet': create_efficientnet_model(img_size, num_classes),
        'densenet': create_densenet_model(img_size, num_classes)
    }
    
    trained_models = {}
    f1_scores = {}
    
    for name, model in models.items():
        trained_model, _ = train_model(
            model, name, train_generator, val_generator, 
            class_weights, epochs=20
        )
        
        print(f"\nEvaluating {name}...")
        weighted_f1, _, _ = evaluate_model(
            trained_model, val_generator, list(class_to_idx.keys())
        )
        
        trained_models[name] = trained_model
        f1_scores[name] = weighted_f1
    
    return trained_models, f1_scores, train_generator, val_generator

def ensemble_predict(models, generator):
    all_predictions = []
    
    for name, model in models.items():
        print(f"Getting predictions from {name}...")
        predictions = model.predict(generator)
        all_predictions.append(predictions)
    
    ensemble_preds = np.mean(all_predictions, axis=0)
    return ensemble_preds

def create_submission(predictions, test_df, idx_to_class, output_file='ensemble_submission.csv'):
    pred_classes = np.argmax(predictions, axis=1)
    pred_labels = [idx_to_class[idx] for idx in pred_classes]
    
    submission_df = pd.DataFrame({
        'md5hash': test_df['md5hash'],
        'label': pred_labels
    })
    
    submission_df.to_csv(output_file, index=False)
    print(f"Submission file created: {output_file}")
    return submission_df

def run_ensemble_pipeline(folder_path, img_size=224, batch_size=16):
    set_seeds(42)
    
    train_df, test_df, class_to_idx, idx_to_class, class_weights = prepare_data(folder_path, img_size)
    
    train_split, val_split = train_test_split(
        train_df, 
        test_size=0.2, 
        stratify=train_df['label'],
        random_state=42
    )
    
    print(f"Training set: {len(train_split)} samples")
    print(f"Validation set: {len(val_split)} samples")
    
    models, f1_scores, train_generator, val_generator = train_ensemble(
        train_split, val_split, class_to_idx, class_weights, img_size, batch_size
    )
    
    fixed_test_generator = fix_test_generator(test_df, img_size, batch_size)
    
    print("\nMaking ensemble predictions on validation data...")
    val_predictions = ensemble_predict(models, val_generator)
    val_pred_classes = np.argmax(val_predictions, axis=1)
    
    y_true = val_generator.classes
    ensemble_weighted_f1 = f1_score(y_true, val_pred_classes, average='weighted')
    ensemble_macro_f1 = f1_score(y_true, val_pred_classes, average='macro')
    
    print(f"\nEnsemble Weighted F1 Score: {ensemble_weighted_f1:.4f}")
    print(f"Ensemble Macro F1 Score: {ensemble_macro_f1:.4f}")
    
    print("\nMaking ensemble predictions on test data...")
    test_predictions = ensemble_predict(models, fixed_test_generator)
    
    submission_df = create_submission(test_predictions, test_df, idx_to_class)
    
    return models, ensemble_weighted_f1, submission_df

def create_ensemble_submission(folder_path='/content/drive/MyDrive/bttai-ajl-2025', img_size=256, batch_size=16):
    train_csv = os.path.join(folder_path, "train.csv")
    train_df = pd.read_csv(train_csv)
    
    unique_classes = sorted(train_df['label'].unique())
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    
    test_csv = os.path.join(folder_path, "test.csv")
    test_df = pd.read_csv(test_csv)
    
    test_dir = os.path.join(folder_path, "test", "test")
    test_df['file_path'] = test_df.apply(
        lambda row: os.path.join(test_dir, f"{row['md5hash']}.jpg"), 
        axis=1
    )
    
    test_df = test_df[test_df['file_path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"Found {len(test_df)} valid test files.")
    
    print("Loading the pre-trained models...")
    models = {
        'mobilenetv2': tf.keras.models.load_model('mobilenetv2_best.keras', compile=False),
        'efficientnet': tf.keras.models.load_model('efficientnet_best.keras', compile=False),
        'densenet': tf.keras.models.load_model('densenet_best.keras', compile=False)
    }
    
    print("Creating a fixed test generator...")
    fixed_test_generator = fix_test_generator(test_df, img_size, batch_size)
    
    print("Making ensemble predictions on test data...")
    test_predictions = ensemble_predict(models, fixed_test_generator)
    
    print("Creating submission file...")
    submission_df = create_submission(test_predictions, test_df, idx_to_class)
    
    print("Done! Submission file created successfully.")
    return submission_df

if __name__ == "__main__":
    folder_path = '/content/drive/MyDrive/bttai-ajl-2025'
    
    if os.path.exists(folder_path):
        print(f"Folder '{folder_path}' found.")
        
        models, f1_score, submission = run_ensemble_pipeline(folder_path, img_size=256, batch_size=16)
        print(f"Final ensemble F1 score: {f1_score:.4f}")
        
        submission = create_ensemble_submission(folder_path, img_size=256, batch_size=16)
    else:
        print(f"Folder '{folder_path}' not found in your Google Drive.")
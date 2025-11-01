"""
Pneumonia Detection from Chest X-Ray Images
Main Training Script
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PneumoniaDetector:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_dir = os.path.join(data_dir, 'train')
        self.val_dir = os.path.join(data_dir, 'val')
        self.test_dir = os.path.join(data_dir, 'test')
        self.history = {}
        self.results = {}
        
    def prepare_data(self):
        """Prepare data generators with augmentation"""
        print("Preparing data generators...")
        
        # Training data with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation and test data (only rescaling)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        self.val_generator = val_test_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        self.test_generator = val_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        
    def build_custom_cnn(self):
        """Build a custom CNN architecture"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
        )
        
        return model
    
    def build_vgg16_transfer(self):
        """Build VGG16 transfer learning model"""
        base_model = VGG16(weights='imagenet', include_top=False, 
                          input_shape=(*self.img_size, 3))
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
        )
        
        return model
    
    def build_resnet50_transfer(self):
        """Build ResNet50 transfer learning model"""
        base_model = ResNet50(weights='imagenet', include_top=False, 
                             input_shape=(*self.img_size, 3))
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
        )
        
        return model
    
    def train_model(self, model, model_name, epochs=20):
        """Train a model with callbacks"""
        print(f"\nTraining {model_name}...")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
            ModelCheckpoint(f'models/{model_name}_best.h5', 
                          monitor='val_accuracy', save_best_only=True)
        ]
        
        history = model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history[model_name] = history.history
        return model
    
    def evaluate_model(self, model, model_name):
        """Evaluate model on test set"""
        print(f"\nEvaluating {model_name}...")
        
        # Get predictions
        self.test_generator.reset()
        y_pred_prob = model.predict(self.test_generator, verbose=1)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_true = self.test_generator.classes
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_pred_prob)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob.flatten()
        }
        
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        
        return self.results[model_name]
    
    def plot_training_history(self, save_path='plots/training_history.png'):
        """Plot training history for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for model_name, history in self.history.items():
            # Accuracy
            axes[0, 0].plot(history['accuracy'], label=f'{model_name} - Train')
            axes[0, 0].plot(history['val_accuracy'], label=f'{model_name} - Val', linestyle='--')
            
            # Loss
            axes[0, 1].plot(history['loss'], label=f'{model_name} - Train')
            axes[0, 1].plot(history['val_loss'], label=f'{model_name} - Val', linestyle='--')
            
            # AUC
            if 'auc' in history:
                axes[1, 0].plot(history['auc'], label=f'{model_name} - Train')
                axes[1, 0].plot(history['val_auc'], label=f'{model_name} - Val', linestyle='--')
        
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Comparison bar chart
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        model_names = list(self.results.keys())
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, model_name in enumerate(model_names):
            values = [self.results[model_name][m] for m in metrics]
            axes[1, 1].bar(x + i*width, values, width, label=model_name)
        
        axes[1, 1].set_title('Model Comparison')
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x + width)
        axes[1, 1].set_xticklabels(metrics, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history saved to {save_path}")
    
    def plot_confusion_matrices(self, save_path='plots/confusion_matrices.png'):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(result['y_true'], result['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xticklabels(['Normal', 'Pneumonia'])
            axes[idx].set_yticklabels(['Normal', 'Pneumonia'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrices saved to {save_path}")
    
    def plot_roc_curves(self, save_path='plots/roc_curves.png'):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, result in self.results.items():
            fpr, tpr, _ = roc_curve(result['y_true'], result['y_pred_prob'])
            auc = result['auc_roc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curves saved to {save_path}")
    
    def save_results(self, save_path='results/model_comparison.csv'):
        """Save comparison results to CSV"""
        df_results = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [r['accuracy'] for r in self.results.values()],
            'Precision': [r['precision'] for r in self.results.values()],
            'Recall': [r['recall'] for r in self.results.values()],
            'F1-Score': [r['f1_score'] for r in self.results.values()],
            'AUC-ROC': [r['auc_roc'] for r in self.results.values()]
        })
        
        os.makedirs('results', exist_ok=True)
        df_results.to_csv(save_path, index=False)
        print(f"\nResults saved to {save_path}")
        print(df_results.to_string(index=False))

def main():
    # Configuration
    DATA_DIR = 'chest_xray'  # Update this path to your dataset location
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Initialize detector
    detector = PneumoniaDetector(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    detector.prepare_data()
    
    # Train multiple models
    models_to_train = {
        'Custom_CNN': detector.build_custom_cnn(),
        'VGG16_Transfer': detector.build_vgg16_transfer(),
        'ResNet50_Transfer': detector.build_resnet50_transfer()
    }
    
    # Train and evaluate each model
    for model_name, model in models_to_train.items():
        trained_model = detector.train_model(model, model_name, epochs=EPOCHS)
        detector.evaluate_model(trained_model, model_name)
        trained_model.save(f'models/{model_name}_final.h5')
    
    # Generate visualizations
    detector.plot_training_history()
    detector.plot_confusion_matrices()
    detector.plot_roc_curves()
    detector.save_results()
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()

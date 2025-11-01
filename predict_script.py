"""
Inference Script for Pneumonia Detection
Load trained models and make predictions on new X-ray images
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import argparse

class PneumoniaPredictor:
    def __init__(self, model_path):
        """Initialize predictor with trained model"""
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        self.img_size = (224, 224)
        print("Model loaded successfully!")
        
    def preprocess_image(self, img_path):
        """Preprocess image for prediction"""
        img = image.load_img(img_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    
    def predict(self, img_path, threshold=0.5):
        """Make prediction on single image"""
        img_array, original_img = self.preprocess_image(img_path)
        
        # Get prediction
        prediction_prob = self.model.predict(img_array, verbose=0)[0][0]
        prediction_class = 'PNEUMONIA' if prediction_prob > threshold else 'NORMAL'
        confidence = prediction_prob if prediction_prob > threshold else 1 - prediction_prob
        
        result = {
            'image_path': img_path,
            'prediction': prediction_class,
            'confidence': confidence,
            'probability_pneumonia': prediction_prob,
            'probability_normal': 1 - prediction_prob
        }
        
        return result, original_img
    
    def predict_batch(self, image_folder, threshold=0.5):
        """Make predictions on all images in a folder"""
        results = []
        image_files = [f for f in os.listdir(image_folder) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\nProcessing {len(image_files)} images...")
        
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            try:
                result, _ = self.predict(img_path, threshold)
                results.append(result)
                print(f"✓ {img_file}: {result['prediction']} "
                      f"(confidence: {result['confidence']:.2%})")
            except Exception as e:
                print(f"✗ Error processing {img_file}: {e}")
        
        return results
    
    def visualize_prediction(self, img_path, threshold=0.5, save_path=None):
        """Visualize prediction with probability"""
        result, original_img = self.predict(img_path, threshold)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display image
        axes[0].imshow(original_img)
        axes[0].axis('off')
        axes[0].set_title(f'Input X-Ray Image\n{os.path.basename(img_path)}')
        
        # Display prediction
        classes = ['NORMAL', 'PNEUMONIA']
        probabilities = [result['probability_normal'], result['probability_pneumonia']]
        colors = ['green' if result['prediction'] == 'NORMAL' else 'lightgreen',
                 'red' if result['prediction'] == 'PNEUMONIA' else 'lightcoral']
        
        bars = axes[1].bar(classes, probabilities, color=colors, alpha=0.7)
        axes[1].set_ylabel('Probability')
        axes[1].set_title('Prediction Probabilities')
        axes[1].set_ylim(0, 1)
        axes[1].axhline(y=threshold, color='gray', linestyle='--', 
                       label=f'Threshold ({threshold})')
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{prob:.2%}', ha='center', va='bottom', fontsize=12)
        
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add prediction text
        prediction_text = f"Prediction: {result['prediction']}\n" \
                         f"Confidence: {result['confidence']:.2%}"
        fig.text(0.5, 0.02, prediction_text, ha='center', fontsize=14, 
                fontweight='bold',
                color='red' if result['prediction'] == 'PNEUMONIA' else 'green')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return result

def main():
    parser = argparse.ArgumentParser(
        description='Pneumonia Detection from Chest X-Ray Images'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.h5 file)')
    parser.add_argument('--image', type=str,
                       help='Path to single image for prediction')
    parser.add_argument('--folder', type=str,
                       help='Path to folder containing multiple images')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of prediction')
    parser.add_argument('--output', type=str,
                       help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = PneumoniaPredictor(args.model)
    
    # Single image prediction
    if args.image:
        if args.visualize:
            result = predictor.visualize_prediction(
                args.image, 
                args.threshold,
                args.output
            )
        else:
            result, _ = predictor.predict(args.image, args.threshold)
            
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Image: {result['image_path']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probability (Normal): {result['probability_normal']:.2%}")
        print(f"Probability (Pneumonia): {result['probability_pneumonia']:.2%}")
        print("="*60)
    
    # Batch prediction
    elif args.folder:
        results = predictor.predict_batch(args.folder, args.threshold)
        
        # Summary statistics
        total = len(results)
        pneumonia_count = sum(1 for r in results if r['prediction'] == 'PNEUMONIA')
        normal_count = total - pneumonia_count
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print("\n" + "="*60)
        print("BATCH PREDICTION SUMMARY")
        print("="*60)
        print(f"Total Images: {total}")
        print(f"Predicted PNEUMONIA: {pneumonia_count} ({pneumonia_count/total:.1%})")
        print(f"Predicted NORMAL: {normal_count} ({normal_count/total:.1%})")
        print(f"Average Confidence: {avg_confidence:.2%}")
        print("="*60)
    
    else:
        print("Error: Please specify either --image or --folder argument")
        parser.print_help()

if __name__ == "__main__":
    # Example usage when run without arguments
    if len(os.sys.argv) == 1:
        print("Pneumonia Detection - Prediction Script")
        print("="*60)
        print("\nUsage Examples:")
        print("\n1. Single image prediction with visualization:")
        print("   python predict.py --model models/resnet50_final.h5 \\")
        print("                     --image test_image.jpg \\")
        print("                     --visualize \\")
        print("                     --output result.png")
        print("\n2. Single image prediction (text only):")
        print("   python predict.py --model models/vgg16_final.h5 \\")
        print("                     --image test_image.jpg")
        print("\n3. Batch prediction on folder:")
        print("   python predict.py --model models/custom_cnn_final.h5 \\")
        print("                     --folder chest_xray/test/NORMAL")
        print("\n4. Custom threshold:")
        print("   python predict.py --model models/resnet50_final.h5 \\")
        print("                     --image test_image.jpg \\")
        print("                     --threshold 0.6")
        print("\n" + "="*60)
    else:
        main()

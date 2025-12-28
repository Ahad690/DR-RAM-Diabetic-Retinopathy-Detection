"""
=============================================================================
DR-RAM: Load Pre-trained Model and Test on Sample Image
=============================================================================

This script demonstrates how to:
1. Load a pre-trained DR-RAM model from a checkpoint file
2. Preprocess a retinal fundus image
3. Run inference to get DR grade prediction
4. Visualize the attention trajectory (where the model looked)

Author: Ahad Imran et al.
Course: CS480 - Reinforcement Learning
=============================================================================
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models

# =============================================================================
# STEP 1: DEFINE THE MODEL ARCHITECTURE
# =============================================================================
# We need to define the exact same architecture that was used during training.
# The model won't load if the architecture doesn't match!

class SpatialAttention(nn.Module):
    """Spatial attention module - same as training"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attn = self.conv(x)
        return x * attn, attn


class RecurrentAttentionModel(nn.Module):
    """
    The DR-RAM model architecture.
    This MUST match exactly what was used during training!
    """
    def __init__(self, hidden_size=512, num_classes=5, num_glimpses=6):
        super().__init__()
        self.num_glimpses = num_glimpses
        self.hidden_size = hidden_size
        
        # Pretrained backbone (EfficientNet-B0)
        backbone = models.efficientnet_b0(weights=None)  # Don't load pretrained weights
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.feat_dim = 1280
        
        # Spatial attention
        self.spatial_attn = SpatialAttention(self.feat_dim)
        
        # Location predictor
        self.loc_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Tanh()
        )
        
        # Recurrent module
        self.gru = nn.GRUCell(self.feat_dim, hidden_size)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Storage for visualization
        self.attention_maps = []
        self.locations = []
        
    def extract_glimpse(self, features, loc, size=7):
        """Extract glimpse from feature map at location"""
        B, C, H, W = features.shape
        
        theta = torch.zeros(B, 2, 3, device=features.device)
        theta[:, 0, 0] = size / W * 2
        theta[:, 1, 1] = size / H * 2
        theta[:, 0, 2] = loc[:, 0]
        theta[:, 1, 2] = loc[:, 1]
        
        grid = F.affine_grid(theta, [B, C, size, size], align_corners=False)
        glimpse = F.grid_sample(features, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        return glimpse
    
    def forward(self, x):
        B = x.size(0)
        device = x.device
        
        # Extract features
        features = self.features(x)
        
        # Initialize
        h = torch.zeros(B, self.hidden_size, device=device)
        loc = torch.zeros(B, 2, device=device)
        
        self.attention_maps = []
        self.locations = [loc.detach().cpu()]
        
        # Recurrent glimpses
        for t in range(self.num_glimpses):
            glimpse = self.extract_glimpse(features, loc)
            glimpse_attn, attn_map = self.spatial_attn(glimpse)
            self.attention_maps.append(attn_map.detach().cpu())
            
            glimpse_vec = F.adaptive_avg_pool2d(glimpse_attn, 1).flatten(1)
            h = self.gru(glimpse_vec, h)
            
            if t < self.num_glimpses - 1:
                loc = self.loc_net(glimpse_attn)
                loc = torch.clamp(loc, -0.8, 0.8)
                self.locations.append(loc.detach().cpu())
        
        logits = self.classifier(h)
        return logits
    
    def get_attention_info(self):
        """Return attention maps and locations for visualization"""
        return self.attention_maps, self.locations


# =============================================================================
# STEP 2: IMAGE PREPROCESSING FUNCTIONS
# =============================================================================

def preprocess_image(image_path, size=380):
    """
    Preprocess a retinal fundus image exactly as done during training.
    
    Steps:
    1. Read the image
    2. Crop black borders
    3. Apply Ben Graham's preprocessing
    4. Resize to target size
    5. Normalize using ImageNet statistics
    
    Args:
        image_path: Path to the image file
        size: Target size (default 380x380)
    
    Returns:
        tensor: Preprocessed image tensor ready for model input [1, 3, H, W]
        original: Original image for visualization
    """
    # Step 1: Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Convert BGR (OpenCV) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = img.copy()  # Keep original for visualization
    
    # Step 2: Crop black borders
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > 10  # Pixels brighter than 10 are considered part of the retina
    coords = np.column_stack(np.where(mask))
    if len(coords) > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img = img[y0:y1, x0:x1]
    
    # Step 3: Resize
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    
    # Step 4: Ben Graham's preprocessing
    # This enhances blood vessels and normalizes illumination
    img = cv2.addWeighted(
        img, 4,                                    # Original image weight
        cv2.GaussianBlur(img, (0, 0), 10), -4,    # Subtract blurred version
        128                                        # Add constant to avoid negative values
    )
    
    # Step 5: Convert to tensor and normalize
    # ImageNet normalization statistics
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Normalize to [0, 1] first, then apply ImageNet normalization
    img_normalized = img.astype(np.float32) / 255.0
    img_normalized = (img_normalized - mean) / std
    
    # Convert to tensor: [H, W, C] -> [C, H, W] -> [1, C, H, W]
    tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
    
    return tensor, original


# =============================================================================
# STEP 3: LOAD THE TRAINED MODEL
# =============================================================================

def load_model(checkpoint_path, device='cuda'):
    """
    Load a trained DR-RAM model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the .pth file containing model weights
        device: 'cuda' for GPU or 'cpu' for CPU
    
    Returns:
        model: Loaded model ready for inference
    """
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create model with same architecture as training
    print("Creating model architecture...")
    model = RecurrentAttentionModel(
        hidden_size=512,
        num_classes=5,
        num_glimpses=6
    )
    
    # Load the saved weights
    print(f"Loading weights from: {checkpoint_path}")
    
    # torch.load() loads the saved state dictionary
    # map_location ensures it works even if trained on GPU but testing on CPU
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # If the checkpoint is a full training state (with optimizer, epoch, etc.)
    # we need to extract just the model weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # If it's just the state dict directly
        state_dict = checkpoint
    
    # Load weights into model
    model.load_state_dict(state_dict)
    
    # Move model to device (GPU or CPU)
    model = model.to(device)
    
    # Set to evaluation mode (disables dropout, etc.)
    # IMPORTANT: Always call .eval() before inference!
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


# =============================================================================
# STEP 4: RUN INFERENCE
# =============================================================================

def predict(model, image_tensor, device='cuda'):
    """
    Run inference on a preprocessed image.
    
    Args:
        model: Loaded DR-RAM model
        image_tensor: Preprocessed image tensor [1, 3, H, W]
        device: 'cuda' or 'cpu'
    
    Returns:
        predicted_class: Integer 0-4 representing DR grade
        probabilities: Probability for each class
        locations: List of attention locations for visualization
    """
    # Move image to device
    image_tensor = image_tensor.to(device)
    
    # Disable gradient computation for inference (faster and uses less memory)
    with torch.no_grad():
        # Forward pass
        logits = model(image_tensor)
        
        # Convert logits to probabilities using softmax
        probabilities = F.softmax(logits, dim=1)
        
        # Get predicted class (highest probability)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Get attention locations for visualization
        _, locations = model.get_attention_info()
    
    return predicted_class, probabilities.cpu().numpy()[0], locations


# =============================================================================
# STEP 5: VISUALIZATION
# =============================================================================

def visualize_prediction(image, predicted_class, probabilities, locations, save_path=None):
    """
    Visualize the prediction with attention trajectory.
    
    Args:
        image: Original image (numpy array)
        predicted_class: Predicted DR grade (0-4)
        probabilities: Class probabilities
        locations: Attention locations
        save_path: Optional path to save the figure
    """
    # DR grade names
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ----- LEFT: Attention Trajectory -----
    ax1 = axes[0]
    ax1.imshow(image)
    
    H, W = image.shape[:2]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(locations)))
    
    for i, (loc, color) in enumerate(zip(locations, colors)):
        # Convert normalized coordinates [-1, 1] to pixel coordinates
        loc_np = loc[0].numpy() if hasattr(loc[0], 'numpy') else loc[0]
        y = int((loc_np[1] + 1) / 2 * H)
        x = int((loc_np[0] + 1) / 2 * W)
        
        # Draw attention point
        ax1.scatter(x, y, c=[color], s=300, marker='o', edgecolors='white', linewidths=2, zorder=5)
        ax1.annotate(str(i+1), (x, y), color='white', fontsize=12, 
                    ha='center', va='center', fontweight='bold', zorder=6)
        
        # Draw line connecting to previous point
        if i > 0:
            prev_loc = locations[i-1][0].numpy() if hasattr(locations[i-1][0], 'numpy') else locations[i-1][0]
            prev_y = int((prev_loc[1] + 1) / 2 * H)
            prev_x = int((prev_loc[0] + 1) / 2 * W)
            ax1.plot([prev_x, x], [prev_y, y], c=color, linewidth=2, alpha=0.7, zorder=4)
    
    ax1.set_title(f'Attention Trajectory\nPrediction: {class_names[predicted_class]}', fontsize=14)
    ax1.axis('off')
    
    # ----- RIGHT: Probability Bar Chart -----
    ax2 = axes[1]
    colors = ['green' if i == predicted_class else 'steelblue' for i in range(5)]
    bars = ax2.barh(class_names, probabilities, color=colors)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Class Probabilities', fontsize=14)
    
    # Add probability values on bars
    for bar, prob in zip(bars, probabilities):
        ax2.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{prob:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


# =============================================================================
# STEP 6: MAIN FUNCTION - PUT IT ALL TOGETHER
# =============================================================================

def main():
    """
    Main function demonstrating the complete inference pipeline.
    """
    print("=" * 60)
    print("DR-RAM: Diabetic Retinopathy Detection")
    print("Loading Pre-trained Model and Running Inference")
    print("=" * 60)
    
    # ----- Configuration -----
    CHECKPOINT_PATH = 'results/checkpoints/drram_best.pth'  # Path to your saved model
    
    # Use one of the sample images if they exist, otherwise fallback
    SAMPLE_DIR = 'sample_images'
    if os.path.exists(SAMPLE_DIR) and os.listdir(SAMPLE_DIR):
        IMAGE_PATH = os.path.join(SAMPLE_DIR, os.listdir(SAMPLE_DIR)[0])
    else:
        IMAGE_PATH = 'sample_image.png'
        
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {DEVICE}")
    
    # ----- Step 1: Load Model -----
    print("\n[1/4] Loading model...")
    try:
        model = load_model(CHECKPOINT_PATH, device=DEVICE)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure you have the checkpoint file at the specified path.")
        print("You can download it from the GitHub repository or train the model yourself.")
        return
    
    # ----- Step 2: Load and Preprocess Image -----
    print("\n[2/4] Preprocessing image...")
    try:
        image_tensor, original_image = preprocess_image(IMAGE_PATH)
        print(f"   Image shape: {image_tensor.shape}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease provide a valid image path.")
        return
    
    # ----- Step 3: Run Inference -----
    print("\n[3/4] Running inference...")
    predicted_class, probabilities, locations = predict(model, image_tensor, device=DEVICE)
    
    # Print results
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    print(f"\n" + "=" * 40)
    print("PREDICTION RESULTS")
    print("=" * 40)
    print(f"Predicted Class: {predicted_class} ({class_names[predicted_class]})")
    print(f"Confidence: {probabilities[predicted_class]:.2%}")
    print("\nAll Probabilities:")
    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
        marker = "→" if i == predicted_class else " "
        print(f"  {marker} {name}: {prob:.4f}")
    
    # ----- Step 4: Visualize -----
    print("\n[4/4] Generating visualization...")
    visualize_prediction(
        original_image, 
        predicted_class, 
        probabilities, 
        locations,
        save_path='prediction_result.png'
    )
    
    print("\n✅ Inference complete!")


# =============================================================================
# RUN THE SCRIPT
# =============================================================================

if __name__ == "__main__":
    main()

"""
## Usage Examples

### Example 1: Command Line
```bash
python inference.py
```

### Example 2: In Jupyter Notebook
```python
# Load model once
model = load_model('checkpoints/drram_best.pth', device='cuda')

# Test multiple images
test_images = [
    'data/test_images/image1.png',
    'data/test_images/image2.png',
    'data/test_images/image3.png'
]

for img_path in test_images:
    tensor, original = preprocess_image(img_path)
    pred_class, probs, locs = predict(model, tensor, device='cuda')
    
    print(f"{img_path}: Class {pred_class} (confidence: {probs[pred_class]:.2%})")
    visualize_prediction(original, pred_class, probs, locs)
```

### Example 3: Quick Single Image Test
```python
# One-liner inference (after imports)
model = load_model('checkpoints/drram_best.pth')
tensor, img = preprocess_image('my_image.png')
pred, probs, locs = predict(model, tensor)
print(f"Prediction: {['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'][pred]} ({probs[pred]:.1%})")
```
"""
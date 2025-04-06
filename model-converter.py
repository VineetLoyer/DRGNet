import tensorflow as tf
import os
import time
import sys

# Check for vit-keras
try:
    import vit
except ImportError:
    print("Installing vit-keras...")
    os.system("pip install vit-keras tensorflow-addons")
    try:
        import vit
    except ImportError:
        print("❌ Failed to install vit-keras. Please install it manually:")
        print("pip install vit-keras tensorflow-addons")
        sys.exit(1)

# Print header
print("\n" + "="*60)
print("🔄 Vision Transformer Model Converter")
print("="*60)
print("This utility helps you convert ViT models to a deployable format")
print("by properly registering the custom layers used in Vision Transformers.")
print("="*60 + "\n")

# Check if model exists
model_path = 'model.h5'
if not os.path.exists(model_path):
    print(f"❌ Model file '{model_path}' not found.")
    print("Please place your model file in the current directory as 'model.h5'")
    sys.exit(1)

print("✅ Model file found.")
print("✅ vit-keras package is available.")

print("\nStep 1: Creating a dummy ViT model to register custom layers...")
try:
    # Create a dummy ViT model to register the custom layers
    dummy_vit = vit.vit_b16(
        image_size=224,
        activation='softmax',
        include_top=False,
        pretrained=False
    )
    print("✅ Dummy ViT model created successfully.")
except Exception as e:
    print(f"❌ Error creating dummy ViT model: {e}")
    sys.exit(1)

print("\nStep 2: Loading your model...")
try:
    # Now try to load the actual model
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nThis could be due to:")
    print("1. Custom layers not properly registered")
    print("2. Incompatible TensorFlow versions")
    print("3. Other dependencies missing")
    sys.exit(1)

print("\nStep 3: Model information:")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")
print(f"Number of layers: {len(model.layers)}")

print("\nStep 4: Converting model to SavedModel format...")
saved_model_dir = 'saved_model'
try:
    model.save(saved_model_dir, save_format='tf')
    print(f"✅ Model successfully converted and saved to '{saved_model_dir}'")
except Exception as e:
    print(f"❌ Error saving model: {e}")
    sys.exit(1)

print("\nStep 5: Testing that the saved model can be loaded...")
try:
    # Load the saved model without custom objects
    loaded_model = tf.keras.models.load_model(saved_model_dir)
    print("✅ Saved model loaded successfully without custom objects!")
    print("This means the model can now be used in your Streamlit app.")
except Exception as e:
    print(f"⚠️ Warning: {e}")
    print("The SavedModel still requires custom objects when loading.")

print("\nStep 6: Would you like to try converting to TFLite format? (y/n)")
response = input("> ")

if response.lower() == 'y':
    try:
        print("Converting to TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)
        print("✅ TensorFlow Lite model saved as 'model.tflite'")
    except Exception as e:
        print(f"❌ Error converting to TFLite: {e}")
        print("Vision Transformer models can be complex to convert to TFLite.")

print("\n" + "="*60)
print("🎉 Conversion Complete!")
print("="*60)
print("\nNext Steps:")
print("1. Copy the 'saved_model' directory to your Streamlit app folder")
print("2. Update your Streamlit app to load the model with:")
print("   ```python")
print("   model = tf.keras.models.load_model('saved_model')")
print("   ```")
print("3. If you still encounter issues, make sure your app has the necessary")
print("   dependencies installed:")
print("   ```bash")
print("   pip install streamlit tensorflow opencv-python vit-keras")
print("   ```")
print("\nGood luck with your retinal disease detection application!")
print("="*60)
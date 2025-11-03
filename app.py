import streamlit as st
import numpy as np
import pandas as pd
import json
import pickle
import os
from tensorflow import keras
from itertools import product

# Page configuration
st.set_page_config(
    page_title="Boolean Logic Truth Table Converter",
    page_icon="🔢",
    layout="wide"
)

# Load model and preprocessing files
@st.cache_resource
def load_model_files():
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load tokenizer configuration
        tokenizer_path = os.path.join(script_dir, 'tokenizer.json')
        with open(tokenizer_path, 'r') as f:
            tokenizer_config = json.load(f)
        
        # Try to load label encoder
        try:
            encoder_path = os.path.join(script_dir, 'label_encoder.pkl')
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load label encoder ({str(e)}). Using default gate types.")
            # Create a simple replacement label encoder
            class SimpleLabelEncoder:
                def __init__(self):
                    self.classes_ = np.array(['AND', 'BUFFER', 'NAND', 'NOR', 'NOT', 'OR', 'XNOR', 'XOR'])
                    
                def inverse_transform(self, y):
                    return [self.classes_[i] for i in y]
            
            label_encoder = SimpleLabelEncoder()
        
        # Load LSTM model
        model_path = os.path.join(script_dir, 'lstm_gate_model.h5')
        model = keras.models.load_model(model_path)
        
        return tokenizer_config, label_encoder, model
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.error(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        return None, None, None

# Text preprocessing function
def preprocess_text(text, tokenizer_config):
    """Preprocess text using tokenizer configuration"""
    # Parse word_index if it's a string (from JSON)
    word_index = tokenizer_config['config']['word_index']
    if isinstance(word_index, str):
        word_index = json.loads(word_index)
    
    filters = tokenizer_config['config']['filters']
    lower = tokenizer_config['config']['lower']
    
    # Apply filters and lowercase
    if lower:
        text = text.lower()
    
    for char in filters:
        text = text.replace(char, ' ')
    
    # Split and convert to sequences
    words = text.split()
    sequence = [word_index.get(word, 1) for word in words]  # 1 is <OOV>
    
    return sequence

# Pad sequences
def pad_sequence(sequence, maxlen=50):
    """Pad sequence to fixed length"""
    if len(sequence) > maxlen:
        return sequence[:maxlen]
    else:
        return [0] * (maxlen - len(sequence)) + sequence

# Generate truth table
def generate_truth_table(gate_type, num_inputs=2):
    """Generate truth table for given gate type"""
    if num_inputs == 1:
        inputs = [[0], [1]]
    else:
        inputs = list(product([0, 1], repeat=num_inputs))
    
    outputs = []
    
    for input_combo in inputs:
        if gate_type.upper() == 'AND':
            output = all(input_combo)
        elif gate_type.upper() == 'OR':
            output = any(input_combo)
        elif gate_type.upper() == 'NOT':
            output = not input_combo[0]
        elif gate_type.upper() == 'NAND':
            output = not all(input_combo)
        elif gate_type.upper() == 'NOR':
            output = not any(input_combo)
        elif gate_type.upper() == 'XOR':
            output = sum(input_combo) % 2 == 1
        elif gate_type.upper() == 'XNOR':
            output = sum(input_combo) % 2 == 0
        elif gate_type.upper() == 'BUFFER':
            output = input_combo[0]
        else:
            output = 0
        
        outputs.append(int(output))
    
    return inputs, outputs

# Main app
def main():
    st.title("🔢 Boolean Logic Expression to Truth Table Converter")
    st.markdown("### Convert Boolean logic expressions into truth tables using AI")
    
    # Load model files
    tokenizer_config, label_encoder, model = load_model_files()
    
    if tokenizer_config is None or label_encoder is None or model is None:
        st.error("Failed to load model files. Please ensure tokenizer.json, label_encoder.pkl, and lstm_gate_model.h5 are in the same directory.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Text Description", "Example Expressions"]
        )
        
        if input_method == "Text Description":
            user_input = st.text_area(
                "Enter Boolean logic description:",
                placeholder="e.g., 'This gate produces logic 1 only when both inputs are true'",
                height=150
            )
        else:
            example = st.selectbox(
                "Select an example:",
                [
                    "This gate produces logic 1 only when both inputs are true",
                    "Output is high when at least one input is high",
                    "This gate inverts the input signal",
                    "Output is 1 when inputs differ",
                    "Output is low when all inputs are high",
                    "Passes input directly to output"
                ]
            )
            user_input = example
        
        # Number of inputs
        num_inputs = st.selectbox("Number of inputs:", [1, 2], index=1)
        
        # Predict button
        predict_button = st.button("🔍 Generate Truth Table", type="primary")
    
    with col2:
        st.subheader("Output")
        
        if predict_button and user_input:
            with st.spinner("Processing..."):
                try:
                    # Preprocess input
                    sequence = preprocess_text(user_input, tokenizer_config)
                    padded_sequence = pad_sequence(sequence)
                    input_array = np.array([padded_sequence])
                    
                    # Make prediction
                    prediction = model.predict(input_array, verbose=0)
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    confidence = np.max(prediction) * 100
                    
                    # Decode prediction
                    gate_type = label_encoder.inverse_transform([predicted_class])[0]
                    
                    # Display prediction with confidence warning
                    st.markdown(f"**Detected Gate:** `{gate_type.upper()}`")
                    if confidence < 60:
                        st.warning(f"**Confidence:** {confidence:.2f}% (Low confidence - prediction may be uncertain)")
                        
                        # Show top 3 predictions for low confidence
                        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
                        st.markdown("**Top 3 Predictions:**")
                        for i, idx in enumerate(top_3_indices):
                            gate_name = label_encoder.inverse_transform([idx])[0]
                            conf = prediction[0][idx] * 100
                            st.write(f"{i+1}. {gate_name.upper()}: {conf:.2f}%")
                    else:
                        st.markdown(f"**Confidence:** {confidence:.2f}%")
                    
                    # Generate truth table
                    inputs, outputs = generate_truth_table(gate_type, num_inputs)
                    
                    # Debug information
                    st.markdown(f"**Debug Info:** Using {num_inputs} input(s), Generated {len(inputs)} rows")
                    
                    # Create DataFrame
                    if num_inputs == 1:
                        df = pd.DataFrame({
                            'Input A': [i[0] for i in inputs],
                            'Output': outputs
                        })
                    else:
                        df = pd.DataFrame({
                            'Input A': [i[0] for i in inputs],
                            'Input B': [i[1] for i in inputs],
                            'Output': outputs
                        })
                    
                    # Display truth table
                    st.markdown("**Truth Table:**")
                    st.dataframe(df, use_container_width=True)
                    
                    # Show expected XOR truth table for comparison if detected as XOR
                    if gate_type.upper() == 'XOR' and num_inputs == 2:
                        st.markdown("**Expected XOR Truth Table for Reference:**")
                        expected_df = pd.DataFrame({
                            'Input A': [0, 0, 1, 1],
                            'Input B': [0, 1, 0, 1],
                            'Output': [0, 1, 1, 0]
                        })
                        st.dataframe(expected_df, use_container_width=True)
                    
                    # Additional information
                    with st.expander("ℹ️ Gate Information"):
                        gate_info = {
                            'AND': "Output is 1 only when all inputs are 1",
                            'OR': "Output is 1 when at least one input is 1",
                            'NOT': "Output is the inverse of the input",
                            'NAND': "Output is 0 only when all inputs are 1 (NOT AND)",
                            'NOR': "Output is 1 only when all inputs are 0 (NOT OR)",
                            'XOR': "Output is 1 when inputs differ (odd number of 1s)",
                            'XNOR': "Output is 1 when inputs match (even number of 1s)",
                            'BUFFER': "Output equals input (no change)"
                        }
                        st.write(gate_info.get(gate_type.upper(), "No information available"))
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
        
        elif predict_button:
            st.warning("Please enter a Boolean logic description.")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a trained LSTM model to:
        - Analyze Boolean logic descriptions
        - Identify the gate type
        - Generate corresponding truth tables
        
        **Supported Gates:**
        - AND, OR, NOT
        - NAND, NOR
        - XOR, XNOR
        - BUFFER
        """)
        
        st.header("📊 Model Info")
        if model is not None:
            st.write(f"**Model Type:** LSTM")
            st.write(f"**Vocabulary Size:** {tokenizer_config['config']['num_words']}")
            st.write(f"**Classes:** {len(label_encoder.classes_)}")
        
        st.header("💡 Tips")
        st.markdown("""
        - Use clear, descriptive language
        - Mention input conditions
        - Describe output behavior
        - Try different phrasings
        """)

if __name__ == "__main__":
    main()
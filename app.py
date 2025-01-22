import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



# Load the pre-trained models
models = {
    "Random Forest": joblib.load(r"C:\Users\Shrey\Downloads\random_forest_model.pkl"),
    "Naive Bayes": joblib.load(r"C:\Users\Shrey\Downloads\naive_bayes_model.pkl"),
    "Decision Tree": joblib.load(r"C:\Users\Shrey\Downloads\decision_tree_model.pkl"),
    "Gradient Boosting": joblib.load(r"C:\Users\Shrey\Downloads\gradient_boosting_model.pkl"),
    "Logistic Regression": joblib.load(r"C:\Users\Shrey\Downloads\logistic_regression_model.pkl")


    
}

# Encoding mapping (from the original script)
encoding_mapping = {
    'odor': {'a': 0, 'c': 1, 'f': 2, 'l': 3, 'm': 4, 'n': 5, 'p': 6, 's': 7, 'y': 8},
    'spore-print-color': {'b': 0, 'h': 1, 'k': 2, 'n': 3, 'o': 4, 'r': 5, 'u': 6, 'w': 7, 'y': 8},
    'gill-color': {'b': 0, 'e': 1, 'g': 2, 'h': 3, 'k': 4, 'n': 5, 'o': 6, 'p': 7, 'r': 8, 'u': 9, 'w': 10, 'y': 11},
    'stalk-surface-above-ring': {'f': 0, 'k': 1, 's': 2, 'y': 3},
    'stalk-surface-below-ring': {'f': 0, 'k': 1, 's': 2, 'y': 3},
    'ring-type': {'e': 0, 'f': 1, 'l': 2, 'n': 3, 'p': 4}
}

def preprocess_input(input_data):
    """Preprocess the input data using the same encoding as in training"""
    processed_data = []
    for column in ['odor', 'spore-print-color', 'gill-color', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'ring-type']:
        processed_data.append(encoding_mapping[column][input_data[column]])
    return np.array(processed_data).reshape(1, -1)

def main():
    st.title('🍄 Mushroom Classifier')
    st.write('Predict whether a mushroom is poisonous or edible based on its characteristics')

    # Create input widgets for each feature
    col1, col2 = st.columns(2)

    with col1:
        odor = st.selectbox('Odor', list(encoding_mapping['odor'].keys()), 
                             format_func=lambda x: {
                                 'a': 'Almond', 'c': 'Creosote', 'f': 'Foul', 
                                 'l': 'Anise', 'm': 'Musty', 'n': 'None', 
                                 'p': 'Pungent', 's': 'Fishy', 'y': 'Spicy'
                             }[x])
        
        spore_print_color = st.selectbox('Spore Print Color', list(encoding_mapping['spore-print-color'].keys()),
                                          format_func=lambda x: {
                                              'b': 'Black', 'h': 'Chocolate', 'k': 'Black', 
                                              'n': 'Brown', 'o': 'Orange', 'r': 'Green', 
                                              'u': 'Purple', 'w': 'White', 'y': 'Yellow'
                                          }[x])
        
        gill_color = st.selectbox('Gill Color', list(encoding_mapping['gill-color'].keys()),
                                   format_func=lambda x: {
                                       'b': 'Black', 'e': 'Red', 'g': 'Gray', 
                                       'h': 'Chocolate', 'k': 'Black', 'n': 'Brown', 
                                       'o': 'Orange', 'p': 'Pink', 'r': 'Green', 
                                       'u': 'Purple', 'w': 'White', 'y': 'Yellow'
                                   }[x])

    with col2:
        stalk_surface_above_ring = st.selectbox('Stalk Surface Above Ring', list(encoding_mapping['stalk-surface-above-ring'].keys()),
                                                 format_func=lambda x: {
                                                     'f': 'Fibrous', 'k': 'Knobby', 
                                                     's': 'Smooth', 'y': 'Scaly'
                                                 }[x])
        
        stalk_surface_below_ring = st.selectbox('Stalk Surface Below Ring', list(encoding_mapping['stalk-surface-below-ring'].keys()),
                                                 format_func=lambda x: {
                                                     'f': 'Fibrous', 'k': 'Knobby', 
                                                     's': 'Smooth', 'y': 'Scaly'
                                                 }[x])
        
        ring_type = st.selectbox('Ring Type', list(encoding_mapping['ring-type'].keys()),
                                  format_func=lambda x: {
                                      'e': 'Evanescent', 'f': 'Flaring', 
                                      'l': 'Large', 'n': 'None', 'p': 'Pendant'
                                  }[x])

    # Prediction button
    if st.button('Predict Mushroom Type'):
        # Prepare input data
        input_data = {
            'odor': odor,
            'spore-print-color': spore_print_color,
            'gill-color': gill_color,
            'stalk-surface-above-ring': stalk_surface_above_ring,
            'stalk-surface-below-ring': stalk_surface_below_ring,
            'ring-type': ring_type
        }

        # Preprocess input
        processed_input = preprocess_input(input_data)

        # Make predictions using different models
        predictions = {}
        for model_name, model in models.items():
            pred = model.predict(processed_input)[0]
            predictions[model_name] = 'Edible' if pred == 0 else 'Poisonous'

        # Display predictions
        st.subheader('Prediction Results')
        for model_name, prediction in predictions.items():
            st.write(f"{model_name}: {prediction}")

        # Majority voting
        from collections import Counter
        final_prediction = Counter(predictions.values()).most_common(1)[0][0]
        st.success(f'🍄 Final Prediction: {final_prediction}')

if __name__ == "__main__":
    main()
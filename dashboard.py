import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import os
from PIL import Image, ImageFilter


from tensorflow.keras.models import load_model

# Charger le modèle sauvegardé
model_loaded = load_model('dog_breed_classifier_6.keras')

# Fonction pour faire des prédictions
def predict(image):
    image = image.resize((224, 224))  # Adapter la taille de l'image
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normaliser l'image
    predictions = model_loaded.predict(image)
    return predictions

# Interface utilisateur avec Streamlit
st.title('Prédiction de race de chien')
# Récupérer les noms des sous-dossiers
# classes = os.listdir(dogImages_folder)
# classes : list(validation_generator.class_indices.keys())
classes = ['Chihuahua', 'French_bulldog', 'German_shepherd', 'Saint_Bernard', 'Saluki', 'Siberian_husky']

# Ajout d'une boîte de sélection pour les images d'exemple
example_images = {
    'German shepherd': './data_/examples/image1.jpg',
    'Saluki': './data_/examples/image2.jpg',
    'Chihuahua': './data_/examples/image3.jpg',
    'French bulldog': './data_/examples/image4.jpg',
    'Saint Bernard': './data_/examples/image5.jpg',
    'Siberian husky': './data_/examples/image6.jpg'
}

 # Ajouter une ligne vide au début de la liste des options
example_image_options = [''] + list(example_images.keys())

selected_example = st.selectbox("Choisissez une image de race de chiens...", example_image_options)

if selected_example:
    example_image_path = example_images[selected_example]
    image = Image.open(example_image_path)

     # Transformation: equalization
    equalized_image = Image.fromarray(np.uint8(np.interp(img_to_array(image), (0, 255), (0, 255))))
    
    # Transformation: floutage
    blurred_image = image.filter(ImageFilter.GaussianBlur(5))
    
    # Afficher les images côte à côte
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(equalized_image, caption='Image égalisée', use_column_width=True, output_format="PNG")
    
    with col2:
        st.image(blurred_image, caption='Image floutée', use_column_width=True, output_format="PNG")
    

    
    st.write("Classification après transformation...")
    predictions = predict(image)
    predicted_class = classes[np.argmax(predictions)]
    st.write(f'La race prédite est : {predicted_class} avec une probabilité de {np.max(predictions)*100:.2f}%')


st.write('Téléchargez une image de chien et le modèle prédit sa race.')

# Téléchargement de l'image par l'utilisateur
uploaded_file = st.file_uploader("Choisissez une image de chien...", type="jpg")

# Chemin vers le dossier contenant les images de chiens
dogImages_folder = 'data/selectedDogs2'


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)
    st.write("Classification...")
    st.write(f'Les races : {classes}')
    predictions = predict(image)
    # Afficher les résultats
    predicted_class = classes[np.argmax(predictions)]
    st.write(f'La race prédite est : {predicted_class} avec une probabilité de {np.max(predictions)*100:.2f}%')



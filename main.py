import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import io
import os
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
import exifread
import requests
from geopy.geocoders import Nominatim
from googlesearch import search
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import spacy
from textblob import TextBlob
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon.zip')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Load ResNet50 model for image classification
model = ResNet50(weights='imagenet')

# Set page configuration
st.set_page_config(
    page_title="Multipurpose Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a feature", ["Home", "Image Analysis", "Text Analysis", "CSV Analysis", "Web Search"])

# Home page
if page == "Home":
    st.title("🔍 Multipurpose Analyzer")
    st.markdown("""
    Welcome to the Multipurpose Analyzer! This application provides real-time analysis capabilities for:
    
    * 📸 **Image Analysis**: Upload an image to analyze its properties, detect objects, and identify brands
    * 📝 **Text Analysis**: Enter text for sentiment analysis, summarization, and key metrics
    * 📊 **CSV Analysis**: Upload a CSV file for automatic data analysis and insights
    * 🌐 **Web Search**: Search for relevant websites based on your query
    
    Select a feature from the sidebar to get started!
    """)
    
    st.image("https://via.placeholder.com/800x400.png?text=Multipurpose+Analyzer", use_column_width=True)

# Image Analysis page
elif page == "Image Analysis":
    st.title("📸 Image Analysis")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        try:
            # Read the image
            file_bytes = uploaded_file.getvalue()
            
            # Get image from bytes for PIL
            pil_image = Image.open(io.BytesIO(file_bytes))
            
            # Convert PIL image to numpy array for OpenCV
            np_image = np.array(pil_image)
            
            # Display original image
            st.subheader("Original Image")
            st.image(pil_image, caption="Uploaded Image", use_column_width=True)
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Basic Info", "Color Analysis", "Object Detection", "Brand Recognition", "EXIF Data"])
            
            with tab1:
                st.subheader("Basic Information")
                
                # Get image dimensions
                width, height = pil_image.size
                mode = pil_image.mode
                format_type = pil_image.format
                
                # Display basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Width", f"{width}px")
                with col2:
                    st.metric("Height", f"{height}px")
                with col3:
                    st.metric("Resolution", f"{width}x{height}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mode", mode)
                with col2:
                    st.metric("Format", format_type)
                with col3:
                    st.metric("File Size", f"{len(file_bytes)/1024:.2f} KB")
                
                # Convert to grayscale
                if len(np_image.shape) == 3:
                    gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray_image = np_image
                
                # Calculate histogram
                hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
                
                # Plot histogram
                st.subheader("Grayscale Histogram")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(hist)
                ax.set_xlim([0, 256])
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
                
            with tab2:
                st.subheader("Color Analysis")
                
                # Show color channels
                if len(np_image.shape) == 3:
                    # Convert BGR to RGB if needed
                    if np_image.shape[2] == 3:
                        # Split channels
                        r, g, b = cv2.split(np_image)
                        
                        # Create color histograms
                        st.subheader("Color Histograms")
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
                        
                        ax1.hist(r.ravel(), bins=256, color='red', alpha=0.7)
                        ax1.set_title('Red Channel')
                        ax1.set_xlim([0, 256])
                        
                        ax2.hist(g.ravel(), bins=256, color='green', alpha=0.7)
                        ax2.set_title('Green Channel')
                        ax2.set_xlim([0, 256])
                        
                        ax3.hist(b.ravel(), bins=256, color='blue', alpha=0.7)
                        ax3.set_title('Blue Channel')
                        ax3.set_xlim([0, 256])
                        
                        st.pyplot(fig)
                        
                        # Display color channels
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.image(r, caption="Red Channel", use_column_width=True)
                        with col2:
                            st.image(g, caption="Green Channel", use_column_width=True)
                        with col3:
                            st.image(b, caption="Blue Channel", use_column_width=True)
                        
                        # Color distribution
                        st.subheader("Color Distribution")
                        
                        # Get the dominant colors
                        resized = cv2.resize(np_image, (100, 100))
                        pixels = resized.reshape(-1, 3)
                        
                        # Number of colors to display
                        n_colors = min(10, len(np.unique(pixels, axis=0)))
                        
                        # Use k-means to find dominant colors
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
                        flags = cv2.KMEANS_RANDOM_CENTERS
                        _, labels, palette = cv2.kmeans(np.float32(pixels), n_colors, None, criteria, 10, flags)
                        
                        # Count occurrences of each color
                        counts = Counter(labels.flatten())
                        
                        # Sort by count
                        sorted_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                        
                        # Display dominant colors
                        st.subheader(f"Top {n_colors} Dominant Colors")
                        
                        # Create a color bar
                        bar_height = 50
                        bar_width = 500
                        color_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
                        
                        start_x = 0
                        
                        for i, (idx, count) in enumerate(sorted_colors):
                            # Calculate percentage
                            percentage = count / len(labels.flatten()) * 100
                            
                            # Calculate width of this color's section
                            section_width = int(percentage / 100 * bar_width)
                            
                            # Fill the section with this color
                            if section_width > 0:
                                color = palette[idx].astype(np.uint8)
                                color_bar[:, start_x:start_x+section_width] = color
                                start_x += section_width
                                
                                # Display color info
                                st.write(f"Color {i+1}: RGB {color} - {percentage:.1f}%")
                        
                        # Display the color bar
                        st.image(color_bar, caption="Color Distribution", use_column_width=True)
                        
                    else:
                        st.write("Image doesn't have RGB channels for color analysis.")
                else:
                    st.write("Image is grayscale, no color analysis available.")
                    
            with tab3:
                st.subheader("Object Detection")
                
                # Convert to grayscale for edge detection
                if len(np_image.shape) == 3:
                    gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray_image = np_image
                    
                # Apply Canny edge detection
                edges = cv2.Canny(gray_image, 100, 200)
                
                # Display edge detection
                st.subheader("Edge Detection")
                st.image(edges, caption="Edge Detection", use_column_width=True)
                
                # Face detection
                try:
                    st.subheader("Face Detection")
                    
                    # Load face cascade
                    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    face_cascade = cv2.CascadeClassifier(face_cascade_path)
                    
                    # Convert to grayscale
                    if len(np_image.shape) == 3:
                        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = np_image
                    
                    # Detect faces
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    # Draw rectangles around faces
                    img_with_faces = np_image.copy()
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Display results
                    st.image(img_with_faces, caption=f"Detected {len(faces)} faces", use_column_width=True)
                    
                except Exception as e:
                    st.write("Face detection could not be performed.")
                    
            with tab4:
                st.subheader("Brand Recognition")
                
                try:
                    # Preprocess image for ResNet50
                    img = pil_image.resize((224, 224))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)
                    
                    # Make prediction
                    predictions = model.predict(img_array)
                    decoded_predictions = decode_predictions(predictions, top=5)[0]
                    
                    # Display predictions
                    st.write("Top 5 Detected Objects:")
                    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                        st.write(f"{i+1}. {label}: {score*100:.2f}%")
                        
                        # If it's a brand-related object, try to identify the brand
                        if any(brand in label.lower() for brand in ['watch', 'jewelry', 'ring', 'necklace', 'bracelet', 'earring']):
                            st.write(f"Possible brands for this {label}:")
                            # Add brand recognition logic here
                            # This would typically involve using a specialized brand recognition model
                            st.write("Note: Brand recognition requires a specialized model")
                    
                except Exception as e:
                    st.write("Brand recognition could not be performed.")
                    
            with tab5:
                st.subheader("EXIF Data")
                
                try:
                    exif_data = {}
                    
                    # Use exifread to extract EXIF data
                    tags = exifread.process_file(io.BytesIO(file_bytes))
                    
                    if tags:
                        # Extract relevant EXIF data
                        exif_data = {
                            "Camera Model": tags.get("Image Model", "Unknown"),
                            "Date Taken": tags.get("EXIF DateTimeOriginal", "Unknown"),
                            "Exposure Time": tags.get("EXIF ExposureTime", "Unknown"),
                            "Aperture": tags.get("EXIF FNumber", "Unknown"),
                            "ISO": tags.get("EXIF ISOSpeedRatings", "Unknown"),
                            "Focal Length": tags.get("EXIF FocalLength", "Unknown"),
                            "Flash": tags.get("EXIF Flash", "Unknown"),
                            "Image Width": tags.get("EXIF ExifImageWidth", "Unknown"),
                            "Image Height": tags.get("EXIF ExifImageHeight", "Unknown"),
                            "GPS Latitude": tags.get("GPS GPSLatitude", "Unknown"),
                            "GPS Longitude": tags.get("GPS GPSLongitude", "Unknown"),
                        }
                        
                        # Display EXIF data
                        st.json(exif_data)
                        
                        # Try to extract GPS coordinates
                        if "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
                            try:
                                # Function to convert GPS coordinates
                                def convert_to_decimal(value):
                                    if isinstance(value, exifread.utils.Ratio):
                                        return float(value.num) / float(value.den)
                                    return value
                                
                                def get_decimal_from_dms(dms, ref):
                                    degrees = convert_to_decimal(dms[0])
                                    minutes = convert_to_decimal(dms[1]) / 60.0
                                    seconds = convert_to_decimal(dms[2]) / 3600.0
                                    
                                    if ref in ['S', 'W']:
                                        return -1 * (degrees + minutes + seconds)
                                    return degrees + minutes + seconds
                                
                                lat = get_decimal_from_dms(tags["GPS GPSLatitude"].values, 
                                                          tags.get("GPS GPSLatitudeRef", "N").values)
                                lon = get_decimal_from_dms(tags["GPS GPSLongitude"].values, 
                                                          tags.get("GPS GPSLongitudeRef", "E").values)
                                
                                st.subheader("Location Information")
                                st.write(f"Latitude: {lat}")
                                st.write(f"Longitude: {lon}")
                                
                                # Try to get location name
                                try:
                                    geolocator = Nominatim(user_agent="multipurpose_analyzer")
                                    location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True)
                                    
                                    if location:
                                        st.write("Location:")
                                        st.write(location.address)
                                        
                                        # Display map
                                        st.subheader("Location Map")
                                        df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
                                        st.map(df)
                                except Exception as e:
                                    st.write("Could not determine location name.")
                            except Exception as e:
                                st.write("Could not parse GPS coordinates.")
                    else:
                        st.write("No EXIF data found in the image.")
                except Exception as e:
                    st.write("Error extracting EXIF data:", str(e))
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Text Analysis page
elif page == "Text Analysis":
    st.title("📝 Text Analysis")
    
    text_input = st.text_area("Enter text for analysis", height=200)
    
    if text_input:
        try:
            # Initialize sentiment analyzer
            sia = SentimentIntensityAnalyzer()
            
            # Analyze sentiment
            sentiment = sia.polarity_scores(text_input)
            
            # Tokenize text
            words = word_tokenize(text_input.lower())
            sentences = sent_tokenize(text_input)
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
            
            # Lemmatize words
            lemmatizer = WordNetLemmatizer()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
            
            # Count word frequencies
            word_freq = Counter(lemmatized_words)
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Basic Metrics", "Sentiment Analysis", "Word Cloud", "Text Summary", "Named Entities"])
            
            with tab1:
                st.subheader("Basic Text Metrics")
                
                # Calculate basic metrics
                word_count = len(words)
                sentence_count = len(sentences)
                avg_word_length = sum(len(word) for word in words) / max(1, len(words))
                avg_sentence_length = len(words) / max(1, len(sentences))
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Word Count", word_count)
                    st.metric("Average Word Length", f"{avg_word_length:.2f} characters")
                with col2:
                    st.metric("Sentence Count", sentence_count)
                    st.metric("Average Sentence Length", f"{avg_sentence_length:.2f} words")
                
                # Display word frequency
                st.subheader("Top 10 Words")
                
                # Create a word frequency dataframe
                word_freq_df = pd.DataFrame(word_freq.most_common(10), columns=['Word', 'Frequency'])
                
                # Plot word frequency
                fig = px.bar(word_freq_df, x='Word', y='Frequency', color='Frequency', 
                             color_continuous_scale='Viridis')
                st.plotly_chart(fig)
            
            with tab2:
                st.subheader("Sentiment Analysis")
                
                # Determine sentiment
                compound_score = sentiment['compound']
                
                # Display sentiment score
                st.metric("Compound Sentiment Score", f"{compound_score:.2f}")
                
                # Create sentiment gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = compound_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Sentiment"},
                    gauge = {
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': "black"},
                        'steps': [
                            {'range': [-1, -0.5], 'color': "red"},
                            {'range': [-0.5, 0.5], 'color': "yellow"},
                            {'range': [0.5, 1], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': compound_score
                        }
                    }
                ))
                
                st.plotly_chart(fig)
                
                # Display detailed sentiment scores
                st.subheader("Detailed Sentiment Scores")
                
                # Create a bar chart for sentiment scores
                sentiment_df = pd.DataFrame({
                    'Score': [sentiment['pos'], sentiment['neu'], sentiment['neg']],
                    'Category': ['Positive', 'Neutral', 'Negative']
                })
                
                fig = px.bar(sentiment_df, x='Category', y='Score', color='Category',
                             color_discrete_map={'Positive': 'green', 'Neutral': 'yellow', 'Negative': 'red'})
                st.plotly_chart(fig)
                
                # TextBlob sentiment analysis
                blob = TextBlob(text_input)
                st.subheader("TextBlob Sentiment Analysis")
                st.write(f"Polarity: {blob.sentiment.polarity:.2f}")
                st.write(f"Subjectivity: {blob.sentiment.subjectivity:.2f}")
            
            with tab3:
                st.subheader("Word Cloud")
                
                # Create word cloud
                if lemmatized_words:
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(lemmatized_words))
                    
                    # Display word cloud
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.write("Not enough words to generate a word cloud.")
            
            with tab4:
                st.subheader("Text Summary")
                
                # Create a basic extractive summary
                # Score sentences based on word frequency
                sentence_scores = {}
                for sentence in sentences:
                    for word in word_tokenize(sentence.lower()):
                        if word in word_freq:
                            if sentence not in sentence_scores:
                                sentence_scores[sentence] = word_freq[word]
                            else:
                                sentence_scores[sentence] += word_freq[word]
                
                # Get top 3 sentences or fewer if there are less than 3
                summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:min(3, len(sentence_scores))]
                
                # Recreate the summary in original order
                summary = ' '.join([sentence for sentence, score in sorted(summary_sentences, key=lambda x: sentences.index(x[0]))])
                
                st.write("Summary:")
                st.write(summary)
                
                # Display key phrases
                st.subheader("Key Phrases")
                
                # Extract key phrases using spaCy
                doc = nlp(text_input)
                key_phrases = []
                
                # Extract noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) >= 2:  # Only include phrases with 2 or more words
                        key_phrases.append(chunk.text)
                
                # Extract named entities
                for ent in doc.ents:
                    if len(ent.text.split()) >= 2:  # Only include entities with 2 or more words
                        key_phrases.append(ent.text)
                
                # Remove duplicates and display
                key_phrases = list(set(key_phrases))
                st.write("Key phrases found in the text:")
                for phrase in key_phrases[:5]:  # Display top 5 phrases
                    st.write(f"- {phrase}")
            
            with tab5:
                st.subheader("Named Entity Recognition")
                
                # Process text with spaCy
                doc = nlp(text_input)
                
                # Extract entities by type
                entity_dict = {}
                for ent in doc.ents:
                    if ent.label_ not in entity_dict:
                        entity_dict[ent.label_] = []
                    entity_dict[ent.label_].append(ent.text)
                
                # Display entities
                for entity_type, entities in entity_dict.items():
                    if entities:
                        st.write(f"**{entity_type}:**")
                        for entity in entities:
                            st.write(f"- {entity}")
                
                # Display entity visualization
                st.subheader("Entity Visualization")
                
                # Create a network graph of entities
                G = nx.Graph()
                
                # Add nodes and edges
                for ent in doc.ents:
                    G.add_node(ent.text, type=ent.label_)
                
                # Add edges between entities that appear in the same sentence
                for sent in doc.sents:
                    sent_ents = [ent for ent in sent.ents]
                    for i, ent1 in enumerate(sent_ents):
                        for ent2 in sent_ents[i+1:]:
                            G.add_edge(ent1.text, ent2.text)
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                       node_size=2000, font_size=8, font_weight='bold', ax=ax)
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error analyzing text: {str(e)}")

# CSV Analysis page
elif page == "CSV Analysis":
    st.title("📊 CSV Data Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Load the data
            df = pd.read_csv(uploaded_file)
            
            # Display basic info
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Display data shape
            st.write(f"**Data Shape:** {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["Data Summary", "Visualizations", "Correlation Analysis", "Insights"])
            
            with tab1:
                # Data types
                st.subheader("Data Types")
                st.write(df.dtypes)
                
                # Missing values
                st.subheader("Missing Values")
                missing_data = df.isnull().sum().reset_index()
                missing_data.columns = ['Column', 'Missing Values']
                missing_data['Percentage'] = (missing_data['Missing Values'] / len(df)) * 100
                
                st.dataframe(missing_data)
                
                # Numerical summary
                st.subheader("Numerical Summary")
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if not numeric_cols.empty:
                    st.dataframe(df[numeric_cols].describe())
                else:
                    st.write("No numerical columns found.")
                
                # Categorical summary
                st.subheader("Categorical Summary")
                categorical_cols = df.select_dtypes(include=['object']).columns
                
                if not categorical_cols.empty:
                    for col in categorical_cols:
                        st.write(f"**{col}**")
                        st.write(df[col].value_counts().head(10))
                else:
                    st.write("No categorical columns found.")
            
            with tab2:
                # Visualizations
                st.subheader("Visualizations")
                
                # Check if there are any numeric columns
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                if numeric_cols:
                    # Histogram
                    st.subheader("Histograms")
                    selected_numeric_col = st.selectbox("Select a numerical column for histogram", numeric_cols)
                    
                    fig = px.histogram(df, x=selected_numeric_col, marginal="box")
                    st.plotly_chart(fig)
                    
                    # Box plot
                    st.subheader("Box Plots")
                    selected_numeric_cols = st.multiselect("Select numerical columns for box plot", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
                    
                    if selected_numeric_cols:
                        fig = px.box(df, y=selected_numeric_cols)
                        st.plotly_chart(fig)
                
                if categorical_cols:
                    # Bar chart
                    st.subheader("Bar Charts")
                    selected_cat_col = st.selectbox("Select a categorical column for bar chart", categorical_cols)
                    
                    value_counts = df[selected_cat_col].value_counts().reset_index()
                    value_counts.columns = [selected_cat_col, 'Count']
                    
                    fig = px.bar(value_counts.head(10), x=selected_cat_col, y='Count', color=selected_cat_col)
                    st.plotly_chart(fig)
                
                # Scatter plot
                if len(numeric_cols) >= 2:
                    st.subheader("Scatter Plot")
                    x_col = st.selectbox("Select X-axis column", numeric_cols, index=0)
                    y_col = st.selectbox("Select Y-axis column", numeric_cols, index=min(1, len(numeric_cols)-1))
                    
                    color_col = None
                    if categorical_cols:
                        color_col = st.selectbox("Select color column (optional)", ['None'] + categorical_cols)
                    
                    if color_col == 'None':
                        fig = px.scatter(df, x=x_col, y=y_col)
                    else:
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
                    
                    st.plotly_chart(fig)
            
            with tab3:
                # Correlation analysis
                st.subheader("Correlation Analysis")
                
                # Check if there are at least 2 numeric columns
                if len(numeric_cols) >= 2:
                    # Correlation matrix
                    corr_matrix = df[numeric_cols].corr()
                    
                    # Plot correlation heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                    st.pyplot(fig)
                    
                    # Get top correlations
                    st.subheader("Top Correlations")
                    
                    # Get upper triangle of correlation matrix
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    
                    # Find top correlations
                    top_corr = upper.unstack().sort_values(ascending=False).dropna().head(10)
                    
                    # Display top correlations
                    for (col1, col2), corr_value in top_corr.items():
                        st.write(f"**{col1} - {col2}:** {corr_value:.3f}")
                        
                        # Scatter plot for top correlations
                        fig = px.scatter(df, x=col1, y=col2, trendline="ols")
                        st.plotly_chart(fig)
                else:
                    st.write("Need at least 2 numeric columns for correlation analysis.")
            
            with tab4:
                # Automated insights
                st.subheader("Automated Insights")
                
                # Check for adequate data
                if len(df) > 0:
                    insights = []
                    
                    # Basic dataset insights
                    insights.append(f"📊 Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
                    
                    # Missing values
                    missing_counts = df.isnull().sum()
                    missing_cols = missing_counts[missing_counts > 0]
                    if not missing_cols.empty:
                        insights.append(f"⚠️ Found {len(missing_cols)} columns with missing values.")
                        for col, count in missing_cols.items():
                            insights.append(f"   - {col}: {count} missing values ({count/len(df)*100:.1f}%)")
                    else:
                        insights.append("✅ No missing values found in the dataset.")
                    
                    # Numeric columns insights
                    if numeric_cols:
                        insights.append(f"🔢 Found {len(numeric_cols)} numeric columns.")
                        
                        # Find columns with outliers
                        for col in numeric_cols:
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                            
                            if len(outliers) > 0:
                                insights.append(f"⚠️ Found {len(outliers)} outliers in {col} column.")
                        
                        # Find correlations
                        corr_matrix = df[numeric_cols].corr()
                        high_corr = np.where(np.abs(corr_matrix) > 0.7)
                        high_corr = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                                    for x, y in zip(*high_corr) if x != y and x < y]
                        
                        if high_corr:
                            insights.append("🔍 Strong correlations found:")
                            for var1, var2, corr in high_corr:
                                insights.append(f"   - {var1} and {var2}: {corr:.2f}")
                    
                    # Categorical columns insights
                    if categorical_cols:
                        insights.append(f"📝 Found {len(categorical_cols)} categorical columns.")
                        
                        for col in categorical_cols:
                            unique_values = df[col].nunique()
                            if unique_values < 10:
                                insights.append(f"   - {col} has {unique_values} unique values.")
                    
                    # Display insights
                    for insight in insights:
                        st.write(insight)
                    
                    # Additional analysis based on data type
                    if numeric_cols:
                        # Time series analysis if date column exists
                        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                        if date_cols:
                            st.subheader("Time Series Analysis")
                            date_col = st.selectbox("Select date column", date_cols)
                            value_col = st.selectbox("Select value column", numeric_cols)
                            
                            try:
                                df[date_col] = pd.to_datetime(df[date_col])
                                fig = px.line(df, x=date_col, y=value_col)
                                st.plotly_chart(fig)
                            except Exception as e:
                                st.write("Could not create time series plot.")
                        
                        # Distribution analysis
                        st.subheader("Distribution Analysis")
                        selected_col = st.selectbox("Select column for distribution analysis", numeric_cols)
                        
                        fig = px.histogram(df, x=selected_col, marginal="box")
                        st.plotly_chart(fig)
                        
                        # Calculate skewness and kurtosis
                        skewness = df[selected_col].skew()
                        kurtosis = df[selected_col].kurtosis()
                        
                        st.write(f"Skewness: {skewness:.2f}")
                        st.write(f"Kurtosis: {kurtosis:.2f}")
                        
                        if abs(skewness) > 1:
                            st.write("⚠️ The distribution is significantly skewed.")
                        if abs(kurtosis) > 2:
                            st.write("⚠️ The distribution has significant kurtosis.")
                    
                    # Clustering analysis for numeric data
                    if len(numeric_cols) >= 2:
                        st.subheader("Clustering Analysis")
                        
                        # Select columns for clustering
                        cluster_cols = st.multiselect("Select columns for clustering", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
                        
                        if len(cluster_cols) >= 2:
                            # Prepare data
                            X = df[cluster_cols].copy()
                            X = X.fillna(X.mean())
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                            # Perform clustering
                            n_clusters = st.slider("Number of clusters", 2, 5, 3)
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            df['Cluster'] = kmeans.fit_predict(X_scaled)
                            
                            # Visualize clusters
                            if len(cluster_cols) == 2:
                                fig = px.scatter(df, x=cluster_cols[0], y=cluster_cols[1], color='Cluster')
                            else:
                                fig = px.scatter_3d(df, x=cluster_cols[0], y=cluster_cols[1], z=cluster_cols[2], color='Cluster')
                            
                            st.plotly_chart(fig)
                            
                            # Display cluster statistics
                            st.write("Cluster Statistics:")
                            cluster_stats = df.groupby('Cluster')[cluster_cols].mean()
                            st.dataframe(cluster_stats)
                else:
                    st.write("Not enough data for analysis.")
                    
        except Exception as e:
            st.error(f"Error analyzing CSV file: {str(e)}")

# Web Search page
elif page == "Web Search":
    st.title("🌐 Web Search")
    
    # Search input
    search_query = st.text_input("Enter your search query")
    
    if search_query:
        try:
            # Perform web search
            st.subheader("Top 5 Relevant Websites")
            
            # Get search results
            search_results = list(search(search_query, num_results=5))
            
            # Display results
            for i, url in enumerate(search_results, 1):
                st.write(f"{i}. [{url}]({url})")
                
                # Try to get page title
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(response.text, 'html.parser')
                        title = soup.title.string if soup.title else "No title available"
                        st.write(f"   Title: {title}")
                except:
                    st.write("   Could not fetch page title")
                
                st.write("---")
                
        except Exception as e:
            st.error(f"Error performing web search: {str(e)}")
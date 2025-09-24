import os
import ollama
import google.generativeai as genai
import requests
from PIL import Image, ImageFilter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration, CLIPProcessor, CLIPModel
import scipy.io.wavfile
import torchvision.transforms as transforms
import torchvision.models as models
from sentence_transformers import SentenceTransformer
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import soundfile as sf
import textwrap
import time
import io
import datetime
import csv
import json
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import euclidean
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# --- 1. ENHANCED CONFIGURATION ---
GOOGLE_API_KEY = "AIzaSyDFs6XbvLFFdwU_RoYnZwT1rriQdgDzFx8"

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error configuring Google AI API: {e}\nPlease ensure your API key is correct.")
    exit()

# Global models
music_processor = None
music_model = None
clip_processor = None
clip_model = None

class AdvancedMultimodalAnalyzer:
    def __init__(self):
        self.text_model = None
        self.image_model = None
        self.audio_features = {}
        self.text_features = {}
        self.image_features = {}
        self.correlation_data = {}
        
    def load_models(self):
        """Load all required models"""
        print("🔄 Loading advanced analysis models...")
        try:
            # Text analysis model
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Image analysis model (ResNet + CLIP)
            self.image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.image_model.eval()
            
            # CLIP for better text-image alignment
            global clip_processor, clip_model
            if clip_processor is None:
                clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            print("✅ All models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")

    def extract_advanced_text_features(self, text):
        """Extract comprehensive text features"""
        print("📝 Extracting advanced text features...")
        
        # Basic embedding
        embedding = self.text_model.encode([text])[0]
        
        # Sentiment and emotion analysis (basic)
        emotion_words = {
            'joy': ['happy', 'joyful', 'bright', 'cheerful', 'delighted', 'elated'],
            'sadness': ['sad', 'melancholy', 'dark', 'gloomy', 'sorrowful', 'mournful'],
            'fear': ['fear', 'afraid', 'terror', 'horror', 'dread', 'anxiety'],
            'anger': ['angry', 'rage', 'fury', 'wrath', 'irritated', 'mad'],
            'surprise': ['surprise', 'astonished', 'amazed', 'shocked', 'startled'],
            'calm': ['peaceful', 'serene', 'tranquil', 'quiet', 'still', 'calm']
        }
        
        text_lower = text.lower()
        emotion_scores = {}
        for emotion, words in emotion_words.items():
            score = sum(1 for word in words if word in text_lower)
            emotion_scores[emotion] = score / len(words)
        
        # Literary features
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Color and mood indicators
        color_words = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'gold', 'silver', 'crimson', 'azure']
        color_score = sum(1 for color in color_words if color in text_lower) / len(color_words)
        
        self.text_features = {
            'embedding': embedding,
            'emotions': emotion_scores,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'color_richness': color_score,
            'text_length': len(text)
        }
        
        return self.text_features

    def extract_advanced_image_features(self, image_path):
        """Extract comprehensive image features"""
        print("🖼️ Extracting advanced image features...")
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Basic ResNet features
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                resnet_features = self.image_model(image_tensor).squeeze().numpy()
            
            # Color analysis
            img_array = np.array(image)
            avg_color = np.mean(img_array, axis=(0,1))
            color_std = np.std(img_array, axis=(0,1))
            brightness = np.mean(avg_color)
            contrast = np.mean(color_std)
            
            # Color temperature (simplified)
            red_ratio = avg_color[0] / (avg_color[0] + avg_color[1] + avg_color[2])
            blue_ratio = avg_color[2] / (avg_color[0] + avg_color[1] + avg_color[2])
            warmth = red_ratio - blue_ratio
            
            # Texture analysis using edge detection
            gray_image = image.convert('L')
            edges = gray_image.filter(ImageFilter.FIND_EDGES)
            edge_intensity = np.mean(np.array(edges))
            
            # CLIP features for better semantic understanding
            clip_inputs = clip_processor(images=image, return_tensors="pt", padding=True)
            clip_features = clip_model.get_image_features(**clip_inputs).detach().numpy().flatten()
            
            self.image_features = {
                'resnet_embedding': resnet_features,
                'clip_embedding': clip_features,
                'avg_color': avg_color,
                'brightness': brightness,
                'contrast': contrast,
                'warmth': warmth,
                'edge_intensity': edge_intensity,
                'color_std': color_std
            }
            
            return self.image_features
            
        except Exception as e:
            print(f"❌ Error extracting image features: {e}")
            return None

    def extract_advanced_audio_features(self, audio_path):
        """Extract comprehensive audio features"""
        print("🎵 Extracting advanced audio features...")
        
        try:
            y, sr = librosa.load(audio_path, sr=None, duration=30)  # Load first 30 seconds
            
            # Spectral features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Rhythm features
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            
            # --- FIX: Ensure tempo is a single scalar value by averaging if it's an array ---
            if isinstance(tempo, np.ndarray):
                tempo = np.mean(tempo)
            # --- END FIX ---
            
            # Harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Energy and dynamics
            rms_energy = librosa.feature.rms(y=y)
            
            # Aggregate features
            features = {
                'mfcc_mean': np.mean(mfccs, axis=1),
                'mfcc_std': np.std(mfccs, axis=1),
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
                'chroma_mean': np.mean(chroma, axis=1),
                'tempo': tempo, # Now guaranteed to be a single number
                'rms_energy_mean': np.mean(rms_energy),
                'harmonic_ratio': np.mean(y_harmonic**2) / (np.mean(y_harmonic**2) + np.mean(y_percussive**2)),
                'duration': len(y) / sr
            }
            
            # Create a single feature vector for correlation analysis
            feature_vector = np.concatenate([
                features['mfcc_mean'],
                [features['spectral_centroid_mean'], features['spectral_rolloff_mean'],
                features['zero_crossing_rate_mean'], features['tempo'],
                features['rms_energy_mean'], features['harmonic_ratio']]
            ])
            
            self.audio_features = {
                'feature_vector': feature_vector,
                'detailed_features': features
            }
            
            return self.audio_features
            
        except Exception as e:
            print(f"❌ Error extracting audio features: {e}")
            return None

    def calculate_advanced_correlations(self, text, image_path, audio_path):
        """Calculate comprehensive multimodal correlations"""
        print("🔍 Calculating advanced correlations...")
        
        # Extract features
        text_feats = self.extract_advanced_text_features(text)
        image_feats = self.extract_advanced_image_features(image_path)
        audio_feats = self.extract_advanced_audio_features(audio_path)
        
        if not all([text_feats, image_feats, audio_feats]):
            return None
        
        # Prepare feature vectors for correlation
        text_vec = text_feats['embedding']
        image_vec = image_feats['resnet_embedding']
        clip_vec = image_feats['clip_embedding']
        audio_vec = audio_feats['feature_vector']
        
        # Normalize vector lengths
        max_len = max(len(text_vec), len(image_vec), len(clip_vec), len(audio_vec))
        
        def pad_vector(vec, target_len):
            if len(vec) < target_len:
                return np.pad(vec, (0, target_len - len(vec)), 'constant')
            else:
                return vec[:target_len]
        
        text_vec_pad = pad_vector(text_vec, max_len)
        image_vec_pad = pad_vector(image_vec, max_len)
        clip_vec_pad = pad_vector(clip_vec, max_len)
        audio_vec_pad = pad_vector(audio_vec, max_len)
        
        # Calculate various correlation metrics
        vectors = {
            'Text (Semantic)': text_vec_pad,
            'Image (ResNet)': image_vec_pad,
            'Image (CLIP)': clip_vec_pad,
            'Audio (Spectral)': audio_vec_pad
        }
        
        # Correlation matrices
        cosine_matrix = np.zeros((len(vectors), len(vectors)))
        pearson_matrix = np.zeros((len(vectors), len(vectors)))
        euclidean_matrix = np.zeros((len(vectors), len(vectors)))
        
        labels = list(vectors.keys())
        
        for i, (label1, vec1) in enumerate(vectors.items()):
            for j, (label2, vec2) in enumerate(vectors.items()):
                # Cosine similarity
                cosine_matrix[i, j] = cosine_similarity([vec1], [vec2])[0][0]
                
                # Pearson correlation
                pearson_corr, _ = pearsonr(vec1, vec2)
                pearson_matrix[i, j] = pearson_corr if not np.isnan(pearson_corr) else 0
                
                # Euclidean distance (inverted and normalized)
                euclidean_dist = euclidean(vec1, vec2)
                max_dist = np.sqrt(len(vec1) * 4)  # Theoretical max distance
                euclidean_matrix[i, j] = 1 - (euclidean_dist / max_dist)
        
        # Cross-modal semantic analysis using CLIP
        text_clip_similarity = 0
        if clip_processor and clip_model:
            try:
                inputs = clip_processor(text=[text], images=Image.open(image_path), 
                                     return_tensors="pt", padding=True)
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                text_clip_similarity = torch.softmax(logits_per_image, dim=1).item()
            except:
                text_clip_similarity = 0
        
        self.correlation_data = {
            'cosine_similarity': cosine_matrix,
            'pearson_correlation': pearson_matrix,
            'euclidean_similarity': euclidean_matrix,
            'labels': labels,
            'text_image_clip_similarity': text_clip_similarity,
            'detailed_features': {
                'text': text_feats,
                'image': image_feats,
                'audio': audio_feats
            }
        }
        
        return self.correlation_data

# Enhanced generation functions (keeping your original ones but with improvements)
def generate_novel_excerpt(prompt: str, model: str = "gemini-1.5-flash") -> str:
    """Generates a short novel excerpt using the Gemini API."""
    print(f"📚 Generating novel excerpt with Gemini ({model})...")
    try:
        gemini_model = genai.GenerativeModel(model)
        
        # We can reuse the same high-quality prompt structure
        enhanced_prompt = f"""Write a vivid, atmospheric opening excerpt (80-120 words) for: '{prompt}'. 
        Focus on:
        - Rich sensory details (visual, auditory, tactile)
        - Strong emotional tone
        - Evocative imagery and metaphors
        - Clear setting and mood establishment
        Make it cinematic and immersive."""
        
        response = gemini_model.generate_content(enhanced_prompt)
        story = response.text
        
        print("✅ Enhanced excerpt generated successfully.")
        return story.strip()
    except Exception as e:
        print(f"❌ Error with Gemini API for excerpt generation: {e}")
        return "Failed to generate story due to an API error."

def create_enhanced_image_prompt(excerpt: str, model: str = "gemini-1.5-flash") -> str:
    print(f"🎨 Creating enhanced artistic prompt with Gemini...")
    try:
        gemini_model = genai.GenerativeModel(model)
        prompt_enhancer = f"""Create a detailed, artistic prompt for AI image generation based on this excerpt:

"{excerpt}"

Include specific details about:
- Lighting (golden hour, dramatic shadows, soft diffused light, etc.)
- Color palette (warm/cool tones, specific colors, saturation levels)
- Composition (rule of thirds, leading lines, focal points)
- Art style (cinematic, painterly, photorealistic, etc.)
- Camera angle and perspective
- Atmospheric elements (fog, particles, weather)
- Texture and material details

Make it highly detailed and evocative. Focus on visual storytelling."""
        
        response = gemini_model.generate_content(prompt_enhancer)
        detailed_prompt = response.text
        print("✅ Enhanced artistic prompt created successfully.")
        return detailed_prompt.strip()
    except Exception as e:
        print(f"❌ Error with Gemini API: {e}")
        return "A cinematic, dramatically lit scene with rich colors and atmospheric depth."

def create_enhanced_music_prompt(excerpt: str, model: str = "gemini-1.5-flash") -> str:
    """Uses Gemini to translate a literary mood into a CONCISE music prompt."""
    print("🎼 Creating enhanced music prompt with Gemini...")
    try:
        gemini_model = genai.GenerativeModel(model)
        prompt_enhancer = f"""Based on this excerpt, create a detailed music prompt.

"{excerpt}"

Specify key elements like:
- Primary instruments (e.g., strings, piano, synth)
- Tempo and mood (e.g., slow, melancholic, epic, tense)
- Key stylistic elements (e.g., orchestral, ambient, electronic)

Crucially, **keep the final prompt concise and under 50 words.** This is a strict limit.
"""
        
        response = gemini_model.generate_content(prompt_enhancer)
        music_prompt = response.text
        print("✅ Enhanced music prompt created successfully.")
        return music_prompt.strip()
    except Exception as e:
        print(f"❌ Error with Gemini API: {e}")
        return "A cinematic orchestral piece with emotional depth."

def generate_image(prompt: str, save_path: str):
    """Enhanced image generation with better parameters"""
    print("🖼️ Generating high-quality image...")
    try:
        # Add style enhancers
        enhanced_prompt = f"{prompt}, high quality, detailed, cinematic lighting, professional photography"
        url_encoded_prompt = requests.utils.quote(enhanced_prompt)
        
        # Use higher quality settings
        image_url = f"https://image.pollinations.ai/prompt/{url_encoded_prompt}?width=1024&height=768&seed=42&model=flux"
        
        response = requests.get(image_url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ High-quality image saved to '{save_path}'.")
        return save_path
    except Exception as e:
        print(f"❌ Failed to generate image: {e}")
        return None

def generate_music(prompt: str, save_path: str):
    """Enhanced music generation"""
    global music_processor, music_model
    print("🎵 Generating music with enhanced MusicGen...")
    try:
        if music_processor is None or music_model is None:
            print("   -> Loading MusicGen model...")
            model_name = "facebook/musicgen-small"
            music_processor = AutoProcessor.from_pretrained(model_name)
            music_model = MusicgenForConditionalGeneration.from_pretrained(model_name)
            print("   -> Model loaded successfully.")
        
        inputs = music_processor(text=[prompt], padding=True, return_tensors="pt")
        print("   -> Generating enhanced audio...")
        
        # Enhanced generation parameters
        audio_values = music_model.generate(
            **inputs, 
            do_sample=True, 
            guidance_scale=4.0,  # Higher guidance for better prompt adherence
            max_new_tokens=756,  # Longer generation
            temperature=0.8
        )
        
        sampling_rate = music_model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write(save_path, rate=sampling_rate, data=audio_values[0, 0].numpy())
        
        print(f"✅ Enhanced music saved to '{save_path}'.")
        return save_path
    except Exception as e:
        print(f"❌ Error during music generation: {e}")
        return None

def create_advanced_visualizations(analyzer, novel_prompt, excerpt, image_path, image_prompt, 
                                 music_path, music_prompt, analysis_text, output_dir):
    """Create comprehensive visualization suite"""
    print("📊 Creating advanced visualization suite...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Main Dashboard
    create_main_dashboard(analyzer, novel_prompt, excerpt, image_path, image_prompt, 
                         music_path, music_prompt, analysis_text, output_dir)
    
    # 2. Correlation Analysis Dashboard
    create_correlation_dashboard(analyzer, output_dir)
    
    # 3. Feature Analysis Dashboard
    create_feature_dashboard(analyzer, output_dir)
    
    # 4. Interactive Plotly Visualizations
    create_interactive_visualizations(analyzer, output_dir)
    
    # 5. Word Cloud and Text Analysis
    create_text_analysis_dashboard(analyzer, excerpt, output_dir)

def create_main_dashboard(analyzer, novel_prompt, excerpt, image_path, image_prompt, 
                         music_path, music_prompt, analysis_text, output_dir):
    """Enhanced main dashboard"""
    fig = plt.figure(figsize=(24, 16), constrained_layout=True)
    gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[1, 1.5, 1.5, 1.2])
    
    # Title
    fig.suptitle(f"Advanced Multimodal Analysis Dashboard\n'{novel_prompt}'", 
                fontsize=28, weight='bold', ha='center', y=0.98)
    
    # Row 1 - Text Analysis
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("Generated Novel Excerpt & Analysis", fontsize=18, weight='bold')
    
    # Create text with background
    text_content = f"EXCERPT:\n{textwrap.fill(excerpt, 100)}"
    if hasattr(analyzer, 'text_features') and analyzer.text_features:
        emotions = analyzer.text_features.get('emotions', {})
        dominant_emotion = max(emotions.items(), key=lambda x: x[1]) if emotions else ('neutral', 0)
        text_content += f"\n\nDOMINANT EMOTION: {dominant_emotion[0].title()} ({dominant_emotion[1]:.2f})"
        text_content += f" | WORD COUNT: {analyzer.text_features.get('word_count', 'N/A')}"
        text_content += f" | AVG SENTENCE LENGTH: {analyzer.text_features.get('avg_sentence_length', 'N/A'):.1f}"
    
    ax1.text(0.05, 0.5, text_content, ha='left', va='center', fontsize=12, 
             wrap=True, fontfamily='serif', transform=ax1.transAxes,
             bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.7))
    ax1.axis('off')
    
    # Row 2 - Image and Prompts
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Generated Image", fontsize=16, weight='bold')
    try:
        img = Image.open(image_path)
        ax2.imshow(img)
    except Exception:
        ax2.text(0.5, 0.5, "Image Not Available", ha='center', va='center', fontsize=14)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[1, 1:])
    ax3.set_title("AI-Generated Prompts", fontsize=16, weight='bold')
    prompt_text = f"IMAGE PROMPT:\n{textwrap.fill(image_prompt[:400] + '...', 60)}\n\n"
    prompt_text += f"MUSIC PROMPT:\n{textwrap.fill(music_prompt[:400] + '...', 60)}"
    ax3.text(0.05, 0.95, prompt_text, ha='left', va='top', fontsize=11, 
             transform=ax3.transAxes, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax3.axis('off')
    
    # Row 3 - Correlation Matrices
    correlation_data = analyzer.correlation_data
    if correlation_data:
        matrices = [
            ('Cosine Similarity', correlation_data['cosine_similarity']),
            ('Pearson Correlation', correlation_data['pearson_correlation']),
            ('Euclidean Similarity', correlation_data['euclidean_similarity'])
        ]
        
        for i, (title, matrix) in enumerate(matrices):
            ax = fig.add_subplot(gs[2, i])
            ax.set_title(title, fontsize=14, weight='bold')
            
            im = ax.imshow(matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
            
            # Add correlation values
            for row in range(len(correlation_data['labels'])):
                for col in range(len(correlation_data['labels'])):
                    ax.text(col, row, f'{matrix[row, col]:.2f}', 
                           ha='center', va='center', fontweight='bold',
                           color='white' if abs(matrix[row, col]) > 0.5 else 'black')
            
            ax.set_xticks(range(len(correlation_data['labels'])))
            ax.set_yticks(range(len(correlation_data['labels'])))
            ax.set_xticklabels([l.split('(')[0].strip() for l in correlation_data['labels']], 
                              rotation=45, ha='right')
            ax.set_yticklabels([l.split('(')[0].strip() for l in correlation_data['labels']])
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 4 - Analysis and Audio Info
    ax4 = fig.add_subplot(gs[3, :2])
    ax4.set_title("Qualitative Alignment Analysis", fontsize=16, weight='bold')
    ax4.text(0.05, 0.95, textwrap.fill(analysis_text, 80), ha='left', va='top', 
             fontsize=11, transform=ax4.transAxes, wrap=True,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[3, 2])
    ax5.set_title("Generated Music Info", fontsize=16, weight='bold')
    music_info = f"🎵 AI-Generated Music\n\nFile: {os.path.basename(music_path)}\n"
    if hasattr(analyzer, 'audio_features') and analyzer.audio_features:
        details = analyzer.audio_features.get('detailed_features', {})
        music_info += f"Tempo: {details.get('tempo', 'N/A'):.1f} BPM\n"
        music_info += f"Duration: {details.get('duration', 'N/A'):.1f}s\n"
        music_info += f"Energy: {details.get('rms_energy_mean', 'N/A'):.3f}\n"
    music_info += "\n(Audio file saved in output folder)"
    
    ax5.text(0.5, 0.5, music_info, ha='center', va='center', fontsize=12,
             transform=ax5.transAxes,
             bbox=dict(boxstyle="round,pad=1", facecolor="lightcoral", alpha=0.8))
    ax5.axis('off')
    
    # Save
    dashboard_path = os.path.join(output_dir, "advanced_main_dashboard.png")
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Main dashboard saved to '{dashboard_path}'")

def create_correlation_dashboard(analyzer, output_dir):
    """Create detailed correlation analysis dashboard"""
    if not hasattr(analyzer, 'correlation_data') or not analyzer.correlation_data:
        return
    
    correlation_data = analyzer.correlation_data
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Correlation Analysis', fontsize=20, weight='bold')
    
    matrices = [
        ('Cosine Similarity', correlation_data['cosine_similarity'], 'RdYlBu_r'),
        ('Pearson Correlation', correlation_data['pearson_correlation'], 'coolwarm'),
        ('Euclidean Similarity', correlation_data['euclidean_similarity'], 'viridis')
    ]
    
    # Top row - correlation matrices
    for i, (title, matrix, cmap) in enumerate(matrices):
        ax = axes[0, i]
        im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(title, fontsize=14, weight='bold')
        
        # Add values
        for row in range(len(correlation_data['labels'])):
            for col in range(len(correlation_data['labels'])):
                ax.text(col, row, f'{matrix[row, col]:.3f}', 
                       ha='center', va='center', fontweight='bold')
        
        ax.set_xticks(range(len(correlation_data['labels'])))
        ax.set_yticks(range(len(correlation_data['labels'])))
        ax.set_xticklabels([l.split('(')[0].strip() for l in correlation_data['labels']], 
                          rotation=45, ha='right')
        ax.set_yticklabels([l.split('(')[0].strip() for l in correlation_data['labels']])
        plt.colorbar(im, ax=ax)
    
    # Bottom row - additional analyses
    # Cross-modal similarity heatmap
    ax = axes[1, 0]
    cross_modal = correlation_data['cosine_similarity'][:3, :3]  # Focus on main modalities
    sns.heatmap(cross_modal, annot=True, cmap='RdBu_r', center=0, ax=ax,
                xticklabels=['Text', 'Image', 'Audio'], 
                yticklabels=['Text', 'Image', 'Audio'])
    ax.set_title('Cross-Modal Similarity Focus', fontsize=14, weight='bold')
    
    # Correlation distribution
    ax = axes[1, 1]
    all_correlations = []
    for matrix in [correlation_data['cosine_similarity'], correlation_data['pearson_correlation']]:
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        all_correlations.extend(matrix[mask])
    
    ax.hist(all_correlations, bins=20, alpha=0.7, edgecolor='black')
    ax.set_title('Correlation Distribution', fontsize=14, weight='bold')
    ax.set_xlabel('Correlation Value')
    ax.set_ylabel('Frequency')
    ax.axvline(np.mean(all_correlations), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_correlations):.3f}')
    ax.legend()
    
    # Feature importance radar chart
    ax = axes[1, 2]
    if 'detailed_features' in correlation_data:
        features = correlation_data['detailed_features']
        
        # Create radar chart data
        categories = ['Semantic\nRichness', 'Visual\nComplexity', 'Audio\nDynamics', 
                     'Emotional\nIntensity', 'Structural\nCoherence']
        
        # Calculate scores (normalized)
        text_score = len(features['text']['embedding']) / 1000 if 'text' in features else 0
        image_score = features['image']['edge_intensity'] / 100 if 'image' in features else 0
        audio_score = features['audio']['detailed_features']['rms_energy_mean'] if 'audio' in features else 0
        emotion_score = max(features['text']['emotions'].values()) if 'text' in features and 'emotions' in features['text'] else 0
        coherence_score = np.mean([correlation_data['cosine_similarity'][i,j] for i in range(3) for j in range(i+1, 3)])
        
        values = [text_score, image_score, audio_score, emotion_score, coherence_score]
        values = [min(1, max(0, v)) for v in values]  # Normalize to 0-1
        
        # Create polar plot
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label='Analysis Scores')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Multimodal Feature Analysis', fontsize=14, weight='bold')
        ax.grid(True)
    
    plt.tight_layout()
    correlation_path = os.path.join(output_dir, "correlation_analysis_dashboard.png")
    plt.savefig(correlation_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Correlation dashboard saved to '{correlation_path}'")

def create_feature_dashboard(analyzer, output_dir):
    """Create detailed feature analysis dashboard"""
    if not hasattr(analyzer, 'correlation_data') or not analyzer.correlation_data:
        return
    
    features = analyzer.correlation_data.get('detailed_features', {})
    if not features:
        return
    
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1])
    fig.suptitle('Detailed Feature Analysis Dashboard', fontsize=20, weight='bold')
    
    # Text Features Analysis
    if 'text' in features:
        text_feats = features['text']
        
        # Emotion analysis
        ax1 = fig.add_subplot(gs[0, 0])
        if 'emotions' in text_feats:
            emotions = text_feats['emotions']
            emotion_names = list(emotions.keys())
            emotion_scores = list(emotions.values())
            
            bars = ax1.bar(emotion_names, emotion_scores, color=sns.color_palette("husl", len(emotions)))
            ax1.set_title('Text Emotional Profile', fontsize=14, weight='bold')
            ax1.set_ylabel('Emotion Score')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, score in zip(bars, emotion_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.2f}', ha='center', va='bottom')
        
        # Text statistics
        ax2 = fig.add_subplot(gs[0, 1])
        stats = ['Word Count', 'Sentences', 'Avg Length', 'Color Richness']
        values = [
            text_feats.get('word_count', 0),
            text_feats.get('sentence_count', 0), 
            text_feats.get('avg_sentence_length', 0),
            text_feats.get('color_richness', 0) * 100
        ]
        
        bars = ax2.bar(stats, values, color='skyblue', edgecolor='navy')
        ax2.set_title('Text Statistics', fontsize=14, weight='bold')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom')
    
    # Image Features Analysis
    if 'image' in features:
        image_feats = features['image']
        
        # Color analysis
        ax3 = fig.add_subplot(gs[0, 2])
        if 'avg_color' in image_feats:
            colors = ['Red', 'Green', 'Blue']
            rgb_values = image_feats['avg_color']
            
            bars = ax3.bar(colors, rgb_values, color=['red', 'green', 'blue'], alpha=0.7)
            ax3.set_title('Average Color Profile', fontsize=14, weight='bold')
            ax3.set_ylabel('Color Intensity (0-255)')
            
            for bar, value in zip(bars, rgb_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{value:.0f}', ha='center', va='bottom')
        
        # Image properties
        ax4 = fig.add_subplot(gs[1, 0])
        properties = ['Brightness', 'Contrast', 'Warmth', 'Edge Intensity']
        prop_values = [
            image_feats.get('brightness', 0),
            image_feats.get('contrast', 0),
            image_feats.get('warmth', 0) + 0.5,  # Shift warmth to positive
            image_feats.get('edge_intensity', 0)
        ]
        
        bars = ax4.bar(properties, prop_values, color='lightcoral', edgecolor='darkred')
        ax4.set_title('Image Properties', fontsize=14, weight='bold')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        for bar, value in zip(bars, prop_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(prop_values)*0.01,
                    f'{value:.2f}', ha='center', va='bottom')
    
    # Audio Features Analysis
    if 'audio' in features:
        audio_feats = features['audio']['detailed_features']
        
        # Spectral features
        ax5 = fig.add_subplot(gs[1, 1])
        spectral_features = ['Spectral Centroid', 'Spectral Rolloff', 'ZCR', 'RMS Energy']
        spectral_values = [
            audio_feats.get('spectral_centroid_mean', 0) / 1000,  # Scale down
            audio_feats.get('spectral_rolloff_mean', 0) / 1000,   # Scale down
            audio_feats.get('zero_crossing_rate_mean', 0) * 100,  # Scale up
            audio_feats.get('rms_energy_mean', 0) * 100           # Scale up
        ]
        
        bars = ax5.bar(spectral_features, spectral_values, color='lightgreen', edgecolor='darkgreen')
        ax5.set_title('Audio Spectral Features', fontsize=14, weight='bold')
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
        
        for bar, value in zip(bars, spectral_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + max(spectral_values)*0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # MFCC visualization
        ax6 = fig.add_subplot(gs[1, 2])
        mfcc_mean = audio_feats.get('mfcc_mean', np.zeros(13))
        mfcc_indices = range(len(mfcc_mean))
        
        ax6.plot(mfcc_indices, mfcc_mean, 'o-', color='purple', linewidth=2, markersize=6)
        ax6.set_title('MFCC Profile', fontsize=14, weight='bold')
        ax6.set_xlabel('MFCC Coefficient')
        ax6.set_ylabel('Mean Value')
        ax6.grid(True, alpha=0.3)
        
        # Tempo and rhythm
        ax7 = fig.add_subplot(gs[2, 0])
        rhythm_features = ['Tempo (BPM)', 'Harmonic Ratio', 'Duration (s)']
        rhythm_values = [
            audio_feats.get('tempo', 0),
            audio_feats.get('harmonic_ratio', 0) * 100,  # Scale up
            audio_feats.get('duration', 0)
        ]
        
        bars = ax7.bar(rhythm_features, rhythm_values, color='gold', edgecolor='orange')
        ax7.set_title('Rhythm & Timing Features', fontsize=14, weight='bold')
        plt.setp(ax7.get_xticklabels(), rotation=45, ha='right')
        
        for bar, value in zip(bars, rhythm_values):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + max(rhythm_values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Chroma features (if available)
        ax8 = fig.add_subplot(gs[2, 1])
        chroma_mean = audio_feats.get('chroma_mean', np.zeros(12))
        chroma_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        bars = ax8.bar(chroma_names, chroma_mean, color='mediumpurple', edgecolor='indigo')
        ax8.set_title('Chroma (Pitch Class) Profile', fontsize=14, weight='bold')
        ax8.set_ylabel('Intensity')
        
        for bar, value in zip(bars, chroma_mean):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + max(chroma_mean)*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Feature vector dimensionality comparison
    ax9 = fig.add_subplot(gs[2, 2])
    modalities = []
    dimensions = []
    
    if 'text' in features:
        modalities.append('Text Embedding')
        dimensions.append(len(features['text']['embedding']))
    if 'image' in features:
        modalities.append('Image ResNet')
        dimensions.append(len(features['image']['resnet_embedding']))
        modalities.append('Image CLIP')
        dimensions.append(len(features['image']['clip_embedding']))
    if 'audio' in features:
        modalities.append('Audio Features')
        dimensions.append(len(features['audio']['feature_vector']))
    
    if modalities and dimensions:
        bars = ax9.bar(modalities, dimensions, color='teal', edgecolor='darkslategray')
        ax9.set_title('Feature Vector Dimensions', fontsize=14, weight='bold')
        ax9.set_ylabel('Dimension Count')
        plt.setp(ax9.get_xticklabels(), rotation=45, ha='right')
        
        for bar, dim in zip(bars, dimensions):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height + max(dimensions)*0.01,
                    f'{dim}', ha='center', va='bottom')
    
    plt.tight_layout()
    feature_path = os.path.join(output_dir, "feature_analysis_dashboard.png")
    plt.savefig(feature_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Feature dashboard saved to '{feature_path}'")

def create_interactive_visualizations(analyzer, output_dir):
    """Create interactive Plotly visualizations"""
    if not hasattr(analyzer, 'correlation_data') or not analyzer.correlation_data:
        return
    
    correlation_data = analyzer.correlation_data
    
    # 3D correlation visualization
    fig_3d = go.Figure()
    
    # Create 3D scatter plot of correlations
    labels = correlation_data['labels']
    cosine_sim = correlation_data['cosine_similarity']
    
    x_coords, y_coords, z_coords = [], [], []
    hover_texts = []
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:  # Exclude self-correlations
                x_coords.append(i)
                y_coords.append(j)
                z_coords.append(cosine_sim[i, j])
                hover_texts.append(f"{labels[i]} ↔ {labels[j]}<br>Similarity: {cosine_sim[i, j]:.3f}")
    
    fig_3d.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(
            size=8,
            color=z_coords,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Similarity"),
            opacity=0.8
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        name='Cross-Modal Correlations'
    ))
    
    fig_3d.update_layout(
        title='3D Cross-Modal Correlation Space',
        scene=dict(
            xaxis_title='Modality 1',
            yaxis_title='Modality 2', 
            zaxis_title='Correlation Strength',
            xaxis=dict(tickmode='array', tickvals=list(range(len(labels))), 
                      ticktext=[l.split('(')[0].strip() for l in labels]),
            yaxis=dict(tickmode='array', tickvals=list(range(len(labels))), 
                      ticktext=[l.split('(')[0].strip() for l in labels])
        ),
        width=800, height=600
    )
    
    # Save interactive plot
    interactive_path = os.path.join(output_dir, "interactive_3d_correlations.html")
    fig_3d.write_html(interactive_path)
    
    # Correlation network diagram
    fig_network = go.Figure()
    
    # Create network layout
    import math
    n = len(labels)
    node_positions = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = math.cos(angle)
        y = math.sin(angle)
        node_positions.append((x, y))
    
    # Add edges (connections between modalities)
    edge_trace = []
    for i in range(n):
        for j in range(i+1, n):
            correlation = cosine_sim[i, j]
            if abs(correlation) > 0.1:  # Only show significant correlations
                x0, y0 = node_positions[i]
                x1, y1 = node_positions[j]
                
                # Line width based on correlation strength
                line_width = abs(correlation) * 10
                line_color = 'red' if correlation > 0 else 'blue'
                
                fig_network.add_trace(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=line_width, color=line_color),
                    opacity=0.6,
                    hoverinfo='none',
                    showlegend=False
                ))
    
    # Add nodes
    node_x = [pos[0] for pos in node_positions]
    node_y = [pos[1] for pos in node_positions]
    
    fig_network.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=30, color='lightblue', line=dict(width=2, color='navy')),
        text=[l.split('(')[0].strip() for l in labels],
        textposition="middle center",
        hoverinfo='text',
        hovertext=labels,
        name='Modalities'
    ))
    
    fig_network.update_layout(
        title='Multimodal Correlation Network',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=800, height=600,
        plot_bgcolor='white'
    )
    
    network_path = os.path.join(output_dir, "correlation_network.html")
    fig_network.write_html(network_path)
    
    print(f"✅ Interactive visualizations saved:")
    print(f"   - 3D Correlations: '{interactive_path}'")
    print(f"   - Network Diagram: '{network_path}'")

def create_text_analysis_dashboard(analyzer, excerpt, output_dir):
    """Create text-specific analysis dashboard with word cloud"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Text Analysis Dashboard', fontsize=18, weight='bold')
    
    # Word cloud
    ax1 = axes[0, 0]
    try:
        wordcloud = WordCloud(width=400, height=300, background_color='white',
                             colormap='viridis', max_words=50).generate(excerpt)
        ax1.imshow(wordcloud, interpolation='bilinear')
        ax1.set_title('Word Cloud', fontsize=14, weight='bold')
        ax1.axis('off')
    except Exception as e:
        ax1.text(0.5, 0.5, f'Word Cloud\nGeneration Error:\n{str(e)}', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.axis('off')
    
    # Word frequency
    ax2 = axes[0, 1]
    words = excerpt.lower().replace('.', '').replace(',', '').split()
    word_freq = {}
    for word in words:
        if len(word) > 3:  # Only words longer than 3 characters
            word_freq[word] = word_freq.get(word, 0) + 1
    
    if word_freq:
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        words_list, freqs = zip(*top_words)
        
        bars = ax2.bar(words_list, freqs, color='steelblue', edgecolor='navy')
        ax2.set_title('Top Word Frequencies', fontsize=14, weight='bold')
        ax2.set_ylabel('Frequency')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        for bar, freq in zip(bars, freqs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{freq}', ha='center', va='bottom')
    
    # Sentence length distribution
    ax3 = axes[1, 0]
    sentences = [s.strip() for s in excerpt.split('.') if s.strip()]
    sentence_lengths = [len(s.split()) for s in sentences]
    
    if sentence_lengths:
        ax3.hist(sentence_lengths, bins=max(5, len(sentence_lengths)//2), 
                alpha=0.7, color='lightcoral', edgecolor='darkred')
        ax3.set_title('Sentence Length Distribution', fontsize=14, weight='bold')
        ax3.set_xlabel('Words per Sentence')
        ax3.set_ylabel('Frequency')
        ax3.axvline(np.mean(sentence_lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(sentence_lengths):.1f}')
        ax3.legend()
    
    # Reading complexity metrics
    ax4 = axes[1, 1]
    if hasattr(analyzer, 'text_features') and analyzer.text_features:
        metrics = ['Avg Sentence\nLength', 'Word Count', 'Color\nRichness', 'Text\nLength']
        values = [
            analyzer.text_features.get('avg_sentence_length', 0),
            analyzer.text_features.get('word_count', 0) / 10,  # Scale down
            analyzer.text_features.get('color_richness', 0) * 100,  # Scale up
            analyzer.text_features.get('text_length', 0) / 100  # Scale down
        ]
        
        bars = ax4.bar(metrics, values, color='gold', edgecolor='orange')
        ax4.set_title('Text Complexity Metrics', fontsize=14, weight='bold')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    text_analysis_path = os.path.join(output_dir, "text_analysis_dashboard.png")
    plt.savefig(text_analysis_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Text analysis dashboard saved to '{text_analysis_path}'")

# --- NEW: Custom JSON encoder to handle NumPy data types ---
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_comprehensive_data(analyzer, novel_prompt, excerpt, image_prompt, music_prompt, 
                          analysis_text, output_dir):
    """Save all analysis data in structured formats"""
    
    # JSON data export
    data_export = {
        'metadata': {
            'novel_prompt': novel_prompt,
            'generation_timestamp': datetime.datetime.now().isoformat(),
            'analysis_version': '2.0'
        },
        'content': {
            'novel_excerpt': excerpt,
            'image_prompt': image_prompt,
            'music_prompt': music_prompt,
            'qualitative_analysis': analysis_text
        },
        'quantitative_analysis': {}
    }
    
    # Add correlation data if available
    if hasattr(analyzer, 'correlation_data') and analyzer.correlation_data:
        corr_data = analyzer.correlation_data
        data_export['quantitative_analysis'] = {
            'cosine_similarity_matrix': corr_data['cosine_similarity'],
            'pearson_correlation_matrix': corr_data['pearson_correlation'],
            'euclidean_similarity_matrix': corr_data['euclidean_similarity'],
            'modality_labels': corr_data['labels'],
            'text_image_clip_similarity': corr_data.get('text_image_clip_similarity', 0)
        }
        
        # Add feature summaries
        if 'detailed_features' in corr_data:
            features = corr_data['detailed_features']
            feature_summary = {}
            
            if 'text' in features:
                feature_summary['text'] = {
                    'dominant_emotion': max(features['text'].get('emotions', {}).items(), 
                                          key=lambda x: x[1], default=('neutral', 0)),
                    'word_count': features['text'].get('word_,
        "Echoes of the Void Warden: A space opera about the last guardian of interdimensional rifts in a dying universe", 
        "The Clockwork Heart: A steampunk romance between a mechanical inventor and a time-displaced Victorian poet",
        "Shadows of New Eden: A biopunk thriller set in a world where genetic memories can be harvested and traded"count', 0),
                    'avg_sentence_length': features['text'].get('avg_sentence_length', 0),
                    'color_richness': features['text'].get('color_richness', 0)
                }
            
            if 'image' in features:
                feature_summary['image'] = {
                    'brightness': features['image'].get('brightness', 0),
                    'contrast': features['image'].get('contrast', 0),
                    'warmth': features['image'].get('warmth', 0),
                    'edge_intensity': features['image'].get('edge_intensity', 0)
                }
            
            if 'audio' in features:
                audio_details = features['audio'].get('detailed_features', {})
                feature_summary['audio'] = {
                    'tempo': audio_details.get('tempo', 0),
                    'duration': audio_details.get('duration', 0),
                    'rms_energy': audio_details.get('rms_energy_mean', 0),
                    'harmonic_ratio': audio_details.get('harmonic_ratio', 0)
                }
            
            data_export['feature_summary'] = feature_summary
    
    # Save JSON
    json_path = os.path.join(output_dir, "comprehensive_analysis_data.json")
    with open(json_path, 'w') as f:
        # --- THIS IS THE CORRECTED LINE ---
        json.dump(data_export, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    # Save detailed CSV reports
    if hasattr(analyzer, 'correlation_data') and analyzer.correlation_data:
        corr_data = analyzer.correlation_data
        
        # Correlation matrices CSV
        for matrix_name, matrix in [
            ('cosine_similarity', corr_data['cosine_similarity']),
            ('pearson_correlation', corr_data['pearson_correlation']),
            ('euclidean_similarity', corr_data['euclidean_similarity'])
        ]:
            df = pd.DataFrame(matrix, 
                            index=[l.split('(')[0].strip() for l in corr_data['labels']],
                            columns=[l.split('(')[0].strip() for l in corr_data['labels']])
            csv_path = os.path.join(output_dir, f"{matrix_name}_matrix.csv")
            df.to_csv(csv_path)
    
    print(f"✅ Comprehensive data saved:")
    print(f"   - JSON Export: '{json_path}'")
    print(f"   - CSV Matrices: '{output_dir}/*_matrix.csv'")

# Enhanced analysis function
def analyze_alignment_gemini_advanced(excerpt: str, image_prompt: str, music_prompt: str, 
                                    correlation_data: dict = None) -> str:
    print("🤖 Performing advanced qualitative alignment analysis...")
    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Enhanced analysis prompt with quantitative data integration
        quantitative_info = ""
        if correlation_data:
            avg_correlation = np.mean([
                correlation_data['cosine_similarity'][i,j] 
                for i in range(min(3, len(correlation_data['labels']))) 
                for j in range(i+1, min(3, len(correlation_data['labels'])))
            ])
            quantitative_info = f"\n\n**Quantitative Context:** The cross-modal correlation analysis shows an average similarity score of {avg_correlation:.3f}, indicating {'strong' if avg_correlation > 0.7 else 'moderate' if avg_correlation > 0.4 else 'weak'} alignment between modalities."
        
        analysis_prompt = f'''You are an expert in multimodal art analysis and cross-media translation. Provide a comprehensive analysis of the semantic and thematic alignment between a novel excerpt and its AI-generated image and music prompts.

**Original Novel Excerpt:** "{excerpt}"

**Generated Image Prompt:** "{image_prompt}"

**Generated Music Prompt:** "{music_prompt}"{quantitative_info}

**Your Comprehensive Analysis:**

Provide a detailed analysis with these sections:

1. **Thematic Coherence Analysis:** 
   - How well do all three modalities capture the core themes and emotional essence?
   - Identify specific thematic elements that translate across modalities
   - Note any thematic disconnects or opportunities for improvement

2. **Sensory Translation Assessment:**
   - Evaluate how textual sensory descriptions translate to visual and auditory elements
   - Assess the effectiveness of cross-modal sensory mapping
   - Identify the strongest and weakest sensory translations

3. **Emotional Resonance Evaluation:**
   - Compare the emotional trajectories across text, image, and music
   - Assess mood consistency and emotional intensity alignment
   - Evaluate the multimodal emotional impact

4. **Artistic Stylistic Harmony:**
   - Analyze the coherence of artistic styles and aesthetic choices
   - Evaluate genre, period, and cultural consistency
   - Assess the overall artistic vision unity

5. **Narrative and Structural Analysis:**
   - Evaluate how narrative elements translate to visual composition and musical structure
   - Assess pacing, rhythm, and structural coherence across modalities
   - Identify narrative strengths and gaps

6. **Technical Quality Assessment:**
   - Evaluate the specificity and implementability of the generated prompts
   - Assess the level of detail and artistic direction provided
   - Comment on the potential for high-quality output generation

7. **Overall Multimodal Synergy:**
   - Provide a comprehensive evaluation of the three-way interaction
   - Rate the overall coherence on a scale of 1-10 with justification
   - Suggest specific improvements for enhanced multimodal alignment

Please be specific, analytical, and constructive in your assessment.'''

        response = gemini_model.generate_content(analysis_prompt)
        print("✅ Advanced alignment analysis complete.")
        return response.text.strip()
    except Exception as e:
        print(f"❌ Failed to perform advanced alignment analysis: {e}")
        return "Advanced analysis could not be performed due to API limitations."

# Main execution function
def run_advanced_pipeline():
    """Main execution function for the advanced pipeline"""
    
    # Configuration
    novel_concepts = [
        "Karnataka, the land of glory"
    ]
    
    # Allow user selection or use default
    selected_concept = novel_concepts[0]  # Default selection
    
    print("=" * 80)
    print("🚀 ADVANCED MULTIMODAL NOVEL ANALYSIS PIPELINE 🚀")
    print("=" * 80)
    print(f"Selected Novel Concept: {selected_concept}")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"advanced_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output Directory: {output_dir}")
    
    # Initialize analyzer
    analyzer = AdvancedMultimodalAnalyzer()
    analyzer.load_models()
    
    print("\n" + "-" * 50)
    print("PHASE 1: CONTENT GENERATION")
    print("-" * 50)
    
    # Generate content
    novel_excerpt = generate_novel_excerpt(selected_concept)
    if "Failed" in novel_excerpt:
        print("❌ Pipeline failed at excerpt generation")
        return
    
    image_prompt = create_enhanced_image_prompt(novel_excerpt)
    music_prompt = create_enhanced_music_prompt(novel_excerpt)
    
    # Generate media files
    image_file_path = os.path.join(output_dir, "generated_image.png")
    music_file_path = os.path.join(output_dir, "generated_music.wav")
    
    image_file = generate_image(image_prompt, image_file_path)
    music_file = generate_music(music_prompt, music_file_path)
    
    if not image_file or not music_file:
        print("❌ Pipeline failed at media generation")
        return
    
    print("\n" + "-" * 50)
    print("PHASE 2: ADVANCED FEATURE EXTRACTION & CORRELATION ANALYSIS")
    print("-" * 50)
    
    # Perform advanced correlation analysis
    correlation_data = analyzer.calculate_advanced_correlations(novel_excerpt, image_file, music_file)
    
    if not correlation_data:
        print("❌ Correlation analysis failed")
        return
    
    print("\n" + "-" * 50)
    print("PHASE 3: QUALITATIVE ANALYSIS")
    print("-" * 50)
    
    # Enhanced qualitative analysis
    alignment_analysis = analyze_alignment_gemini_advanced(
        novel_excerpt, image_prompt, music_prompt, correlation_data
    )
    
    print("\n" + "-" * 50)
    print("PHASE 4: COMPREHENSIVE VISUALIZATION GENERATION")
    print("-" * 50)
    
    # Create all visualizations
    create_advanced_visualizations(
        analyzer, selected_concept, novel_excerpt, image_file, image_prompt,
        music_file, music_prompt, alignment_analysis, output_dir
    )
    
    print("\n" + "-" * 50)
    print("PHASE 5: DATA EXPORT & DOCUMENTATION")
    print("-" * 50)
    
    # Save comprehensive data
    save_comprehensive_data(
        analyzer, selected_concept, novel_excerpt, image_prompt, 
        music_prompt, alignment_analysis, output_dir
    )
    
    # Generate summary report
    generate_summary_report(analyzer, selected_concept, novel_excerpt, output_dir)
    
    print("\n" + "=" * 80)
    print("🎉 ADVANCED PIPELINE EXECUTION COMPLETE! 🎉")
    print("=" * 80)
    print(f"\n📁 All outputs saved in: '{output_dir}'")
    print("\n📊 Generated Files:")
    print("   • advanced_main_dashboard.png - Main analysis dashboard")
    print("   • correlation_analysis_dashboard.png - Detailed correlation analysis")
    print("   • feature_analysis_dashboard.png - Feature extraction results")
    print("   • text_analysis_dashboard.png - Text-specific analysis")
    print("   • interactive_3d_correlations.html - 3D interactive correlations")
    print("   • correlation_network.html - Interactive network diagram")
    print("   • comprehensive_analysis_data.json - Complete data export")
    print("   • *_matrix.csv - Individual correlation matrices")
    print("   • analysis_summary_report.txt - Human-readable summary")
    print("   • generated_image.png - AI-generated image")
    print("   • generated_music.wav - AI-generated music")
    print("\n🔍 Open the HTML files in a web browser for interactive exploration!")

def generate_summary_report(analyzer, novel_prompt, excerpt, output_dir):
    """Generate a comprehensive summary report"""
    report_path = os.path.join(output_dir, "analysis_summary_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ADVANCED MULTIMODAL NOVEL ANALYSIS - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Novel Concept: {novel_prompt}\n\n")
        f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-" * 50 + "\n")
        f.write("GENERATED CONTENT SUMMARY\n")
        f.write("-" * 50 + "\n\n")
        
        f.write(f"Novel Excerpt ({len(excerpt.split())} words):\n")
        f.write(f'"{excerpt}"\n\n')
        
        # Feature summary
        if hasattr(analyzer, 'correlation_data') and analyzer.correlation_data:
            features = analyzer.correlation_data.get('detailed_features', {})
            
            f.write("-" * 50 + "\n")
            f.write("QUANTITATIVE ANALYSIS SUMMARY\n")
            f.write("-" * 50 + "\n\n")
            
            if 'text' in features:
                text_feats = features['text']
                f.write("TEXT FEATURES:\n")
                f.write(f"  • Word Count: {text_feats.get('word_count', 'N/A')}\n")
                f.write(f"  • Average Sentence Length: {text_feats.get('avg_sentence_length', 'N/A'):.1f}\n")
                f.write(f"  • Color Richness Score: {text_feats.get('color_richness', 'N/A'):.3f}\n")
                
                if 'emotions' in text_feats:
                    emotions = text_feats['emotions']
                    dominant = max(emotions.items(), key=lambda x: x[1])
                    f.write(f"  • Dominant Emotion: {dominant[0].title()} ({dominant[1]:.3f})\n")
                f.write("\n")
            
            if 'image' in features:
                image_feats = features['image']
                f.write("IMAGE FEATURES:\n")
                f.write(f"  • Brightness: {image_feats.get('brightness', 'N/A'):.2f}\n")
                f.write(f"  • Contrast: {image_feats.get('contrast', 'N/A'):.2f}\n")
                f.write(f"  • Color Warmth: {image_feats.get('warmth', 'N/A'):.3f}\n")
                f.write(f"  • Edge Intensity: {image_feats.get('edge_intensity', 'N/A'):.2f}\n")
                f.write("\n")
            
            if 'audio' in features:
                audio_details = features['audio'].get('detailed_features', {})
                f.write("AUDIO FEATURES:\n")
                f.write(f"  • Tempo: {audio_details.get('tempo', 'N/A'):.1f} BPM\n")
                f.write(f"  • Duration: {audio_details.get('duration', 'N/A'):.1f} seconds\n")
                f.write(f"  • RMS Energy: {audio_details.get('rms_energy_mean', 'N/A'):.4f}\n")
                f.write(f"  • Harmonic Ratio: {audio_details.get('harmonic_ratio', 'N/A'):.3f}\n")
                f.write("\n")
            
            # Correlation summary
            corr_data = analyzer.correlation_data
            f.write("-" * 50 + "\n")
            f.write("CORRELATION ANALYSIS SUMMARY\n")
            f.write("-" * 50 + "\n\n")
            
            # Average correlations
            cosine_avg = np.mean([corr_data['cosine_similarity'][i,j] 
                                for i in range(min(3, len(corr_data['labels']))) 
                                for j in range(i+1, min(3, len(corr_data['labels'])))])
            
            f.write(f"Average Cross-Modal Correlation: {cosine_avg:.3f}\n")
            f.write(f"Correlation Strength: {'Strong' if cosine_avg > 0.7 else 'Moderate' if cosine_avg > 0.4 else 'Weak'}\n\n")
            
            # Strongest correlations
            correlations = []
            labels = corr_data['labels']
            for i in range(len(labels)):
                for j in range(i+1, len(labels)):
                    correlations.append((labels[i], labels[j], corr_data['cosine_similarity'][i,j]))
            
            correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            f.write("Top 3 Strongest Correlations:\n")
            for i, (mod1, mod2, corr) in enumerate(correlations[:3]):
                f.write(f"  {i+1}. {mod1.split('(')[0].strip()} ↔ {mod2.split('(')[0].strip()}: {corr:.3f}\n")
            
            f.write("\n")
        
        f.write("-" * 50 + "\n")
        f.write("FILES GENERATED\n")
        f.write("-" * 50 + "\n\n")
        
        files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        for file in sorted(files):
            f.write(f"  • {file}\n")
        
        f.write("\n" + "-" * 50 + "\n")
        f.write("ANALYSIS COMPLETE\n")
        f.write("-" * 50 + "\n")
        
    print(f"✅ Summary report saved to '{report_path}'")

# Entry point
if __name__ == "__main__":
    try:
        run_advanced_pipeline()
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
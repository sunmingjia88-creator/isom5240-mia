"""
Storytelling Application for Kids (Age 3-10)
Using Hugging Face Transformers Pipelines
- Image Captioning: Salesforce/blip-image-captioning-base
- Story Generation: gpt2 (better for creative stories)
- Text-to-Speech: gTTS (Google Text-to-Speech)
"""

import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
import tempfile
import os
import torch

# ============================================
# Helper Functions
# ============================================

@st.cache_resource
def load_captioning_model():
    """Load the image captioning model (cached for performance)"""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_text_generation_model():
    """
    Load GPT-2 model for story generation
    GPT-2 is much better at generating creative, non-repetitive stories
    """
    return pipeline(
        "text-generation", 
        model="gpt2",
        device=-1  # Use CPU (Streamlit Cloud doesn't have GPU)
    )


def img2text(image, captioning_pipeline):
    """
    Convert image to text caption using BLIP model
    Args:
        image: PIL Image object
        captioning_pipeline: Loaded image-to-text pipeline
    Returns:
        caption text (string)
    """
    try:
        result = captioning_pipeline(image)
        caption = result[0]["generated_text"]
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return "A happy scene with animals and nature"


def text2story(caption, text_pipeline):
    """
    Expand caption into a short story (50-100 words)
    Using GPT-2 with careful parameters to avoid repetition
    Args:
        caption: Short caption from image
        text_pipeline: Loaded text generation pipeline
    Returns:
        Story text (string)
    """
    # Create a better prompt for story generation
    prompt = f"Once upon a time, there was a scene with {caption}. "
    prompt += "This is a short story for kids about what happened: "
    
    try:
        # Generate story with parameters that reduce repetition
        result = text_pipeline(
            prompt,
            max_length=150,  # Generate enough tokens for a full story
            min_length=50,   # Ensure minimum length
            do_sample=True,
            temperature=0.85,  # Slightly higher for more creativity
            top_k=50,          # Limit to top 50 tokens for coherence
            top_p=0.95,        # Nucleus sampling
            repetition_penalty=1.2,  # Penalty for repeating tokens (crucial!)
            no_repeat_ngram_size=3,   # Prevent repeating 3-word phrases
            pad_token_id=50256,       # GPT-2's eos token id
            truncation=True
        )
        
        # Extract generated text
        generated_text = result[0]["generated_text"]
        
        # Remove the prompt from the beginning if present
        story = generated_text.replace(prompt, "").strip()
        
        # If story is empty or too short, try alternative
        if len(story) < 30:
            story = generate_fallback_story(caption)
        
        # Clean up any obvious repetition patterns
        story = clean_repetitive_text(story)
        
        # Ensure length is appropriate (50-100 words)
        word_count = len(story.split())
        if word_count < 45:
            story = story + " They were very happy and had a wonderful time together. The end!"
        elif word_count > 110:
            words = story.split()[:100]
            story = " ".join(words) + "... The end!"
        
        return story
        
    except Exception as e:
        st.error(f"Error generating story: {e}")
        return generate_fallback_story(caption)


def generate_fallback_story(caption):
    """Generate a simple fallback story when the model fails"""
    templates = [
        f"Once upon a time, there was {caption}. Every day was full of adventure and joy. "
        f"They played and laughed with their friends. They learned that being kind and sharing "
        f"makes everyone happy. The end!",
        
        f"One sunny day, {caption}. All the friends came together to have fun. "
        f"They explored, played games, and helped each other. "
        f"It was the best day ever, and they promised to be friends forever. The end!",
        
        f"In a magical land, there was {caption}. Every moment brought new discoveries. "
        f"They showed courage, kindness, and creativity. And they lived happily ever after. The end!"
    ]
    import random
    return random.choice(templates)


def clean_repetitive_text(text):
    """
    Clean up repetitive patterns in generated text
    """
    sentences = text.split('.')
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 5:
            continue
        # Skip if very similar to recently seen sentences
        if sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
    
    cleaned = '. '.join(unique_sentences)
    if not cleaned.endswith('.'):
        cleaned += '.'
    
    return cleaned


def text2audio(story_text):
    """
    Convert story text to audio using gTTS
    Args:
        story_text: Story string to convert
    Returns:
        Path to temporary audio file
    """
    try:
        # Create a temporary file for audio
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # Generate speech with slower speed for kids
        tts = gTTS(text=story_text, lang="en", slow=True)  # slow=True is better for kids
        tts.save(temp_audio_path)
        
        return temp_audio_path
    except Exception as e:
        st.error(f"Error converting text to audio: {e}")
        return None


# ============================================
# Streamlit UI
# ============================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Kids Storyteller",
        page_icon="📖",
        layout="centered"
    )
    
    # App title
    st.title("📖 Kids Storyteller")
    st.markdown("### Turn any picture into a magical story!")
    st.markdown("For kids aged 3-10 🎈")
    
    # Sidebar information
    with st.sidebar:
        st.header("🌟 How to Use")
        st.markdown("""
        1. Upload a picture (JPG, JPEG, or PNG)
        2. Click **'Generate Story'** button
        3. Read the story or listen to it!
        4. Download the audio to save it
        """)
        st.divider()
        st.markdown("**Example pictures to try:**")
        st.markdown("- A dog playing in the park 🐕")
        st.markdown("- Children on a beach 🏖️")
        st.markdown("- A beautiful rainbow 🌈")
        st.markdown("- Animals in a forest 🦊")
        st.divider()
        st.markdown("**Tips:**")
        st.markdown("- Bright, clear pictures work best")
        st.markdown("- Wait a moment for the story to generate")
    
    # File uploader
    uploaded_image = st.file_uploader(
        "🎨 Upload an image", 
        type=["jpg", "jpeg", "png"],
        help="Upload any picture, and I'll create a story about it!"
    )
    
    # Display uploaded image
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Your Picture", use_container_width=True)
    
    # Generate button
    if uploaded_image is not None:
        if st.button("✨ Generate Story ✨", type="primary"):
            # Step 1: Load models (with loading indicators)
            with st.spinner("🖼️ Loading AI models... (first time may take a moment)"):
                captioning_pipeline = load_captioning_model()
                text_pipeline = load_text_generation_model()
            
            # Step 2: Image to Caption
            with st.spinner("🔍 Looking at your picture..."):
                caption = img2text(image, captioning_pipeline)
                st.info(f"📷 I see: *{caption}*")
            
            # Step 3: Caption to Story
            with st.spinner("✍️ Writing a magical story for you..."):
                story = text2story(caption, text_pipeline)
            
            # Display the story
            st.success("📖 Your Story is Ready!")
            st.markdown("---")
            st.subheader("✨ The Story ✨")
            
            # Display story in a nice box
            st.markdown(
                f"""
                <div style="background-color: #f0f8ff; padding: 20px; border-radius: 15px; border-left: 5px solid #ff6b6b;">
                    <p style="font-size: 18px; line-height: 1.6; font-family: 'Comic Sans MS', cursive;">{story}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Word count check
            word_count = len(story.split())
            st.caption(f"📏 Word count: {word_count} words (target: 50-100)")
            st.markdown("---")
            
            # Step 4: Text to Audio
            with st.spinner("🔊 Converting story to audio..."):
                audio_path = text2audio(story)
            
            if audio_path:
                # Play audio
                st.subheader("🎧 Listen to the Story")
                col1, col2 = st.columns([3, 1])
                with col1:
                    with open(audio_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3")
                
                with col2:
                    st.download_button(
                        label="📥 Download",
                        data=audio_bytes,
                        file_name="story_audio.mp3",
                        mime="audio/mpeg",
                        help="Save the audio to your device"
                    )
                
                # Clean up temp file
                try:
                    os.unlink(audio_path)
                except:
                    pass
            else:
                st.warning("⚠️ Audio generation failed. You can still read the story above!")
    
    else:
        # Show placeholder when no image uploaded
        st.info("👆 Please upload an image to begin your storytelling adventure!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 12px;'>"
        "Made with ❤️ for young storytellers | Powered by Hugging Face & gTTS"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

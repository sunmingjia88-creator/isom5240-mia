"""
Storytelling Application for Kids (Age 3-10)
Using Hugging Face Transformers Pipelines
- Image Captioning: Salesforce/blip-image-captioning-base
- Story Generation: DistilGPT-2 with kid-friendly prompting
- Text-to-Speech: gTTS (Google Text-to-Speech)
"""

import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
import tempfile
import os
import re

# ============================================
# Helper Functions
# ============================================

@st.cache_resource
def load_captioning_model():
    """Load the image captioning model (cached for performance)"""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


@st.cache_resource
def load_story_model():
    """Load DistilGPT-2 for simple story generation (smaller and faster)"""
    return pipeline("text-generation", model="distilgpt2")


def img2text(image, captioning_pipeline):
    """
    Convert image to text caption using BLIP model
    """
    try:
        result = captioning_pipeline(image)
        caption = result[0]["generated_text"]
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return "a happy dog"


def generate_kid_story(caption, story_pipeline):
    """
    Generate a simple, kid-friendly story based on the image caption.
    Uses simple words and stays close to the image content.
    """
    caption = caption.strip().lower()
    
    # Simple prompt with easy words
    prompt = f"Once upon a time, there was {caption}."
    
    try:
        # Generate story with kid-friendly settings
        result = story_pipeline(
            prompt,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=50256
        )
        
        # Extract generated text
        full_text = result[0]["generated_text"]
        
        # Remove the prompt
        story = full_text.replace(prompt, "").strip()
        
        # Clean up any weird characters
        story = re.sub(r'[^\w\s\.\,\!\?]', '', story)
        
        # Make sure story is not too long
        words = story.split()
        if len(words) > 70:
            story = ' '.join(words[:67]) + '.'
        
        # If story is too short, add a simple ending
        if len(words) < 25:
            story = story + f" The {caption} was happy. Everyone had fun. The end."
        
        # Make sure first letter is capital
        if story and len(story) > 0:
            story = story[0].upper() + story[1:]
        
        # Add period at end if needed
        if story and story[-1] not in ['.', '!', '?']:
            story += '.'
        
        return story
        
    except Exception as e:
        st.error(f"Error generating story: {e}")
        return create_simple_fallback(caption)


def create_simple_fallback(caption):
    """
    Simple fallback stories with easy words for kids
    """
    caption_lower = caption.lower()
    
    # Detect what's in the picture
    if 'dog' in caption_lower or 'puppy' in caption_lower:
        return f"This is a picture of {caption}. The dog is cute and happy. It likes to play and run. What a nice dog to see in this photo!"
    
    elif 'cat' in caption_lower or 'kitten' in caption_lower:
        return f"I see {caption}. The cat is soft and pretty. It likes to nap in the sun. This cat looks very friendly and nice!"
    
    elif 'boy' in caption_lower or 'girl' in caption_lower or 'child' in caption_lower:
        return f"Look at this photo of {caption}. The child is having fun and smiling. Playing outside is the best thing to do on a sunny day!"
    
    elif 'flower' in caption_lower or 'tree' in caption_lower:
        return f"Here is {caption}. The flowers and trees look so pretty. Nature is full of beautiful colors. This picture makes me feel happy!"
    
    elif 'car' in caption_lower or 'truck' in caption_lower or 'bike' in caption_lower:
        return f"This photo shows {caption}. The vehicle is red and shiny. It looks fast and cool. I like looking at this picture!"
    
    elif 'food' in caption_lower or 'cake' in caption_lower or 'pizza' in caption_lower:
        return f"Yum! This picture is about {caption}. The food looks so tasty. Eating good food with friends is always fun!"
    
    else:
        return f"This is a nice picture of {caption}. The colors are bright and pretty. I like this photo. It makes me smile!"


def text2audio(story_text):
    """
    Convert story text to audio using gTTS
    """
    try:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # Use slower speed for kids to understand better
        tts = gTTS(text=story_text, lang="en", slow=True)
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
    
    # Custom CSS
    st.markdown("""
        <style>
        .story-box {
            background-color: #FFF8DC !important;
            padding: 25px !important;
            border-radius: 20px !important;
            border-left: 8px solid #FF6B6B !important;
            border-right: 8px solid #4ECDC4 !important;
            font-size: 20px !important;
            line-height: 1.8 !important;
            font-family: 'Comic Neue', 'Comic Sans MS', 'Chalkboard SE', cursive !important;
            color: #1A1A2E !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        }
        .story-title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            color: #FF6B6B !important;
            margin-bottom: 20px;
        }
        .caption-box {
            background-color: #E8F8F5 !important;
            padding: 12px !important;
            border-radius: 15px !important;
            font-size: 16px !important;
            margin: 10px 0 !important;
            color: #2C3E50 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # App title
    st.markdown('<div class="story-title">📖 Kids Storyteller 📖</div>', unsafe_allow_html=True)
    st.markdown("### Turn any picture into a story for kids!")
    st.markdown("🎈 For ages 3-10 | 🎨 Easy words | 🔊 Listen and read")
    
    # Sidebar
    with st.sidebar:
        st.header("🌟 How to Use")
        st.markdown("""
        1. **Upload a picture** (JPG or PNG)
        2. **Click 'Make a Story'**
        3. **Read the story** (easy words!)
        4. **Listen to the story**
        5. **Download the audio**
        """)
        st.divider()
        st.markdown("**📸 Good pictures to try:**")
        st.markdown("- 🐕 A dog or cat")
        st.markdown("- 👧 A child smiling")
        st.markdown("- 🌸 Flowers or trees")
        st.markdown("- 🚗 A car or bike")
        st.markdown("- 🍰 Cake or food")
    
    # File uploader
    uploaded_image = st.file_uploader(
        "🎨 **Upload a picture**", 
        type=["jpg", "jpeg", "png"],
        help="Pick any picture!"
    )
    
    # Display uploaded image
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Your picture", use_container_width=True)
        
        # Generate Story Button
        if st.button("📖 Make a Story 📖", type="primary", use_container_width=True):
            
            # Step 1: Look at the picture
            with st.spinner("👀 Looking at your picture..."):
                captioning_pipeline = load_captioning_model()
                caption = img2text(image, captioning_pipeline)
                
                st.markdown(f"""
                <div class="caption-box">
                🖼️ <strong>I see:</strong> {caption}
                </div>
                """, unsafe_allow_html=True)
            
            # Step 2: Write a story
            with st.spinner("✏️ Writing a story for you..."):
                story_pipeline = load_story_model()
                story = generate_kid_story(caption, story_pipeline)
            
            # Show the story
            st.markdown("---")
            st.markdown('<div class="story-title">✨ The Story ✨</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="story-box">
            {story}
            </div>
            """, unsafe_allow_html=True)
            
            # Word count
            word_count = len(story.split())
            st.caption(f"📖 {word_count} words")
            st.markdown("---")
            
            # Step 3: Make audio
            with st.spinner("🔊 Making audio..."):
                audio_path = text2audio(story)
            
            if audio_path:
                st.subheader("🎧 Listen")
                
                with open(audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
                
                st.download_button(
                    label="💾 Download Audio",
                    data=audio_bytes,
                    file_name="my_story.mp3",
                    mime="audio/mpeg"
                )
                
                try:
                    os.unlink(audio_path)
                except:
                    pass
            else:
                st.warning("⚠️ Audio not ready. You can still read the story!")
    
    else:
        st.info("👆 **Upload a picture to start!**")
        
        with st.expander("📷 How it works", expanded=False):
            st.markdown("""
            1. Take or pick a picture
            2. The computer looks at your picture
            3. The computer writes a simple story
            4. You can read and listen!
            
            **Example:** A picture of a dog 🐕
            
            *"This is a picture of a dog. The dog is cute and happy. It likes to play and run. What a nice dog to see in this photo!"*
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #95A5A6; font-size: 13px;'>"
        "Made with ❤️ for kids | Easy words | Fun stories"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

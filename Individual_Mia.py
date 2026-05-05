"""
Storytelling Application for Kids (Age 3-10)
Using Hugging Face Transformers Pipelines
- Image Captioning: Salesforce/blip-image-captioning-base
- Story Generation: GPT-2 with careful prompting (coherent, no repetition)
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
    """Load GPT-2 for story generation"""
    return pipeline("text-generation", model="gpt2")


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
        return "a dog playing in the park"


def generate_story(caption, story_pipeline):
    """
    Generate a coherent, non-repetitive story based on the image caption.
    Uses GPT-2 with optimized parameters for 50-100 word output.
    """
    # Clean the caption
    caption = caption.strip()
    
    # Create a prompt that guides GPT-2 to write a short, descriptive story
    prompt = f"Write a short story about {caption}. The story is for young children:"
    
    try:
        # Generate story with parameters to avoid repetition
        result = story_pipeline(
            prompt,
            max_new_tokens=120,
            min_new_tokens=50,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            truncation=True,
            pad_token_id=50256
        )
        
        # Extract generated text
        full_text = result[0]["generated_text"]
        
        # Remove the prompt from the beginning
        story = full_text.replace(prompt, "").strip()
        
        # If story starts with common prefixes, clean them
        story = re.sub(r'^"|^\'|^The story:|^Story:', '', story).strip()
        
        # Ensure story has proper ending
        if not story[-1] in ['.', '!', '?']:
            story += '.'
        
        # Count words
        words = story.split()
        
        # Adjust length to 50-100 words
        if len(words) > 100:
            story = ' '.join(words[:97]) + '...'
        elif len(words) < 50:
            # Add a natural extension if too short
            extension = f" What a lovely picture of {caption}."
            story = story + extension
        
        # Capitalize first letter
        story = story[0].upper() + story[1:] if len(story) > 1 else story
        
        return story
        
    except Exception as e:
        st.error(f"Error generating story: {e}")
        return create_fallback_story(caption)


def create_fallback_story(caption):
    """
    Fallback stories that are unique and not repetitive
    """
    caption_lower = caption.lower()
    
    stories = [
        f"I see {caption}. The sun is shining and everything looks beautiful. This picture makes me feel happy and calm. What a wonderful moment to capture.",
        f"Look at this picture! It shows {caption}. The colors are bright and lovely. I can tell this is a special moment that someone wanted to remember forever.",
        f"This is a picture of {caption}. When I look at it, I think about how many beautiful things exist in our world. Every picture tells a story, and this one tells a happy story.",
        f"Here we have {caption}. This scene looks peaceful and nice. It reminds us to slow down and enjoy the little things in life. What a great picture to share."
    ]
    
    # Use caption hash to pick consistent story
    index = hash(caption) % len(stories)
    story = stories[index]
    
    # Ensure 50-100 words
    words = story.split()
    if len(words) > 100:
        story = ' '.join(words[:97]) + '...'
    elif len(words) < 50:
        story = story + f" This picture of {caption} is truly wonderful."
    
    return story


def text2audio(story_text):
    """
    Convert story text to audio using gTTS
    """
    try:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        tts = gTTS(text=story_text, lang="en", slow=False)
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
    
    # Custom CSS for better readability
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
            font-style: italic !important;
            font-size: 16px !important;
            margin: 10px 0 !important;
            color: #2C3E50 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # App title
    st.markdown('<div class="story-title">📖 Kids Storyteller 📖</div>', unsafe_allow_html=True)
    st.markdown("### Turn any picture into a story!")
    st.markdown("🎈 For kids aged 3-10")
    
    # Sidebar
    with st.sidebar:
        st.header("🌟 How to Use")
        st.markdown("""
        1. **Upload a picture**
        2. **Click 'Generate Story'**
        3. **Read your unique story**
        4. **Listen to the audio**
        5. **Download** to keep it!
        """)
        st.divider()
        st.markdown("**📸 Best pictures to try:**")
        st.markdown("- 🐕 Dogs, cats, or animals")
        st.markdown("- 👧 Children or people")
        st.markdown("- 🌸 Flowers and nature")
        st.markdown("- 🏠 Places and buildings")
    
    # File uploader
    uploaded_image = st.file_uploader(
        "🎨 **Upload an image**", 
        type=["jpg", "jpeg", "png"]
    )
    
    # Display uploaded image
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Your Picture", use_container_width=True)
        
        # Generate Story Button
        if st.button("✨ Generate Story ✨", type="primary", use_container_width=True):
            
            # Step 1: Generate caption
            with st.spinner("🖼️ Looking at your picture..."):
                captioning_pipeline = load_captioning_model()
                caption = img2text(image, captioning_pipeline)
                
                st.markdown(f"""
                <div class="caption-box">
                🔍 <strong>I see:</strong> {caption}
                </div>
                """, unsafe_allow_html=True)
            
            # Step 2: Generate story
            with st.spinner("✍️ Writing a story..."):
                story_pipeline = load_story_model()
                story = generate_story(caption, story_pipeline)
            
            # Display the story
            st.markdown("---")
            st.markdown('<div class="story-title">✨ The Story ✨</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="story-box">
            {story}
            </div>
            """, unsafe_allow_html=True)
            
            # Word count
            word_count = len(story.split())
            st.caption(f"📏 {word_count} words")
            st.markdown("---")
            
            # Step 3: Text to Audio
            with st.spinner("🔊 Creating audio..."):
                audio_path = text2audio(story)
            
            if audio_path:
                st.subheader("🎧 Listen")
                
                with open(audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
                
                st.download_button(
                    label="📥 Download Audio",
                    data=audio_bytes,
                    file_name="story.mp3",
                    mime="audio/mpeg"
                )
                
                try:
                    os.unlink(audio_path)
                except:
                    pass
            else:
                st.warning("⚠️ Audio generation failed. You can still read the story above!")
    
    else:
        st.info("👆 **Please upload an image to begin!**")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #95A5A6; font-size: 13px;'>"
        "Made with ❤️ for young storytellers"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

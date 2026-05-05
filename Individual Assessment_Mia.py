"""
Storytelling Application for Kids (Age 3-10)
Using Hugging Face Transformers Pipelines
- Image Captioning: Salesforce/blip-image-captioning-base
- Story Generation: Uses the caption directly + simple extensions
- Text-to-Speech: gTTS (simple and reliable)
"""

import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
import tempfile
import os

# ============================================
# Helper Functions
# ============================================

@st.cache_resource
def load_captioning_model():
    """Load the image captioning model (cached for performance)"""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


def img2text(image, captioning_pipeline):
    """
    Convert image to text caption using BLIP model
    Returns a detailed caption describing the image
    """
    try:
        result = captioning_pipeline(image)
        caption = result[0]["generated_text"]
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return "a beautiful scene"


def create_story_from_caption(caption):
    """
    Create a simple story that directly describes what's in the picture.
    No extra fantasy or unrelated content - just describe the image.
    """
    # Clean up the caption
    caption = caption.strip()
    
    # Simple story templates that just describe the picture
    # Each template puts the caption at the center
    
    template1 = f"""Look at this picture. In this picture, I see {caption}. This is what is happening in the image. That is all I can see in this photo. The end."""
    
    template2 = f"""This is a nice picture. The picture shows {caption}. This is what the camera captured. Thank you for sharing this photo with me. The end."""
    
    template3 = f"""I am looking at this photo. In this photo, there is {caption}. This is the scene in front of me. I hope you like this description of your picture. The end."""
    
    template4 = f"""Let me tell you about this picture. The picture contains {caption}. That is exactly what I see when I look at this image. The end."""
    
    template5 = f"""Here is what I see: {caption}. This picture shows this scene clearly. That is my description of your photo. The end."""
    
    # Choose a template based on caption length for variety
    templates = [template1, template2, template3, template4, template5]
    
    # Use the same caption to always get the same template (for consistency)
    template_index = hash(caption) % len(templates)
    story = templates[template_index]
    
    # Ensure word count is between 50-100
    words = story.split()
    if len(words) > 100:
        # Trim to exactly 100 words
        story = " ".join(words[:97]) + " The end."
    elif len(words) < 50:
        # Add a simple sentence if too short
        story = story.replace("The end.", "This picture is very nice to look at. The end.")
    
    return story


def text2audio(story_text):
    """
    Convert story text to audio using gTTS
    """
    try:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # Use slow=False for natural speed
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
        page_title="Kids Storyteller - Picture to Story",
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
            text-shadow: none !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        }
        .story-box * {
            color: #1A1A2E !important;
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
        .caption-box * {
            color: #2C3E50 !important;
        }
        .stMarkdown, .stMarkdown p {
            color: inherit !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # App title
    st.markdown('<div class="story-title">📖 Kids Storyteller 📖</div>', unsafe_allow_html=True)
    st.markdown("### Turn any picture into a simple story!")
    st.markdown("🎈 For kids aged 3-10 | 🌟 Easy to read and understand")
    
    # Sidebar
    with st.sidebar:
        st.header("🌟 How to Use")
        st.markdown("""
        1. **Upload a picture** (JPG, JPEG, or PNG)
        2. **Click 'Generate Story'** 
        3. **Read what the AI sees** in your picture
        4. **Listen to the audio** version
        5. **Download** to keep the story!
        """)
        st.divider()
        st.markdown("**📸 Best pictures to try:**")
        st.markdown("- 🐕 A dog or cat")
        st.markdown("- 👧 A child or people")
        st.markdown("- 🌸 Flowers or nature")
        st.markdown("- 🏠 A house or building")
        st.markdown("- 🍕 Food or objects")
        st.divider()
        st.markdown("**💡 Tip:** The story will simply describe what the AI sees in your picture!")
    
    # File uploader
    uploaded_image = st.file_uploader(
        "🎨 **Upload an image**", 
        type=["jpg", "jpeg", "png"],
        help="Upload any picture, and I'll describe what I see!"
    )
    
    # Display uploaded image
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="🖼️ Your Picture", use_container_width=True)
        
        # Generate Story Button
        if st.button("✨ Generate Story ✨", type="primary", use_container_width=True):
            
            # Step 1: Load model and generate caption
            with st.spinner("🖼️ Looking at your picture..."):
                captioning_pipeline = load_captioning_model()
                caption = img2text(image, captioning_pipeline)
                
                # Display the caption (what the AI sees)
                st.markdown(f"""
                <div class="caption-box">
                🔍 <strong>The AI sees:</strong> {caption}
                </div>
                """, unsafe_allow_html=True)
            
            # Step 2: Create simple story from caption
            with st.spinner("✍️ Writing a simple description..."):
                story = create_story_from_caption(caption)
            
            # Display the story
            st.markdown("---")
            st.markdown('<div class="story-title">✨ The Story ✨</div>', unsafe_allow_html=True)
            
            # Story in a nice box
            st.markdown(f"""
            <div class="story-box">
            {story}
            </div>
            """, unsafe_allow_html=True)
            
            # Word count
            word_count = len(story.split())
            if 50 <= word_count <= 100:
                st.success(f"📏 Word count: {word_count} words (Perfect! ✓)")
            else:
                st.info(f"📏 Word count: {word_count} words (Target: 50-100)")
            
            st.markdown("---")
            
            # Step 3: Text to Audio
            with st.spinner("🔊 Converting story to audio..."):
                audio_path = text2audio(story)
            
            if audio_path:
                st.subheader("🎧 Listen to the Story")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    with open(audio_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3")
                
                with col2:
                    st.download_button(
                        label="📥 Download Audio",
                        data=audio_bytes,
                        file_name="story_audio.mp3",
                        mime="audio/mpeg",
                        use_container_width=True
                    )
                
                # Clean up temp file
                try:
                    os.unlink(audio_path)
                except:
                    pass
            else:
                st.warning("⚠️ Audio generation failed. You can still read the story above!")
    
    else:
        # Placeholder when no image
        st.info("👆 **Please upload an image to begin!**")
        
        # Show example
        with st.expander("📷 How it works", expanded=False):
            st.markdown("""
            1. Upload any picture
            2. The AI looks at your picture
            3. The AI describes exactly what it sees
            4. You get a simple story that matches your picture!
            
            **Example:** If you upload a picture of a dog, the story will describe the dog.
            No extra fantasy or made-up content - just a clear description of your picture.
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #95A5A6; font-size: 13px;'>"
        "Made with ❤️ for young storytellers | 📖 Pictures become simple stories | 🔊 Stories become audio"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

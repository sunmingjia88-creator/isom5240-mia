"""
Storytelling Application for Kids (Age 3-10)
Using Hugging Face Transformers Pipelines
- Image Captioning: Salesforce/blip-image-captioning-base
- Story Generation: google/flan-t5-small
- Text-to-Speech: gTTS (Google Text-to-Speech)
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

@st.cache_resource
def load_text_generation_model():
    """Load the text generation model (cached for performance)"""
    return pipeline("text2text-generation", model="google/flan-t5-small")


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
        return "A happy scene with animals and nature."


def text2story(caption, text_pipeline):
    """
    Expand caption into a short story (50-100 words)
    Args:
        caption: Short caption from image
        text_pipeline: Loaded text generation pipeline
    Returns:
        Story text (string)
    """
    # Create a prompt suitable for kids
    prompt = f"""
Write a short and fun story for young kids (3-10 years old) based on this scene: {caption}

The story should:
- Be 50 to 100 words long
- Have happy and positive content
- Use simple words
- End with a moral or a positive message

Story:
"""
    
    try:
        result = text_pipeline(
            prompt, 
            max_length=200, 
            do_sample=True, 
            temperature=0.7
        )
        story = result[0]["generated_text"]
        
        # Clean up the story (remove prompt artifacts if any)
        story = story.replace("Story:", "").strip()
        
        # Ensure story length is reasonable (50-100 words)
        word_count = len(story.split())
        if word_count < 40:
            story = story + " Always remember to be kind and enjoy the little moments!"
        elif word_count > 120:
            story = " ".join(story.split()[:100]) + "..."
        
        return story
    except Exception as e:
        st.error(f"Error generating story: {e}")
        return f"Once upon a time, there was a {caption}. They had a wonderful day full of joy, laughter, and friendship. The end!"


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
        
        # Generate speech
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
            st.markdown(f"> {story}")
            
            # Word count check
            word_count = len(story.split())
            st.caption(f"📏 Word count: {word_count} words")
            st.markdown("---")
            
            # Step 4: Text to Audio
            with st.spinner("🔊 Converting story to audio..."):
                audio_path = text2audio(story)
            
            if audio_path:
                # Play audio
                st.subheader("🎧 Listen to the Story")
                with open(audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
                
                # Download button
                st.download_button(
                    label="📥 Download Audio",
                    data=audio_bytes,
                    file_name="story_audio.mp3",
                    mime="audio/mpeg"
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

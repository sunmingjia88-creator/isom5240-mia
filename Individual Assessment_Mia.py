"""
Storytelling Application for Kids (Age 3-10)
Using Hugging Face Transformers Pipelines
- Image Captioning: Salesforce/blip-image-captioning-base
- Story Generation: pranavpsv/genre-story-generator-v2 (from class demo)
- Text-to-Speech: Matthijs/mms-tts-eng (from class demo)
"""

import streamlit as st
from PIL import Image
from transformers import pipeline
import numpy as np

# ============================================
# Helper Functions
# ============================================

@st.cache_resource
def load_captioning_model():
    """Load the image captioning model (cached for performance)"""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


@st.cache_resource
def load_story_generator():
    """Load the story generation model from class demo"""
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")


@st.cache_resource
def load_audio_generator():
    """Load the text-to-audio model from class demo"""
    return pipeline("text-to-audio", model="Matthijs/mms-tts-eng")


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
        return "a happy scene with children playing in a beautiful place"


def caption_to_story(caption, story_generator):
    """
    Convert image caption to a full story using story generation model
    The prompt is carefully crafted to tell a story BASED ON the image content
    """
    # Create a prompt that instructs the model to tell a story about the image
    prompt = f"""Tell me a short, happy story for young children about what you see in this picture: {caption}

The story should be 50-100 words, simple, and fun for kids age 3-10.

Story:"""
    
    try:
        # Generate story using the model
        result = story_generator(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            truncation=True
        )
        
        # Extract the generated story
        full_text = result[0]["generated_text"]
        
        # Extract only the story part (after "Story:")
        if "Story:" in full_text:
            story = full_text.split("Story:")[-1].strip()
        else:
            story = full_text.replace(prompt, "").strip()
        
        # Clean up any common issues
        story = story.replace("  ", " ")
        
        # Ensure story has a proper ending
        if not story.endswith((".", "!", "?")):
            story += " The end!"
        
        # Trim to 50-100 words
        words = story.split()
        if len(words) > 100:
            story = " ".join(words[:97]) + "... The end!"
        elif len(words) < 45:
            # Add a nice sentence if too short
            story = story + " This is a beautiful picture full of joy and happiness! The end!"
        
        return story
        
    except Exception as e:
        st.error(f"Error generating story: {e}")
        # Fallback story based on caption
        return create_fallback_story(caption)


def create_fallback_story(caption):
    """Create a simple fallback story based on the caption"""
    caption = caption.lower()
    
    if "dog" in caption or "puppy" in caption:
        return f"Once upon a time, there was a cute {caption}. The puppy played and ran all day long. It made many friends and everyone was happy. The little puppy learned that being kind and friendly makes every day special. The end!"
    
    elif "cat" in caption or "kitten" in caption:
        return f"Look at this lovely picture of {caption}. The cat was soft and cuddly. It loved to play with yarn and nap in the sunshine. Every day was a happy adventure for this sweet cat. The end!"
    
    elif "child" in caption or "boy" in caption or "girl" in caption or "kid" in caption:
        return f"What a wonderful picture of {caption}! The child was playing and having so much fun. They laughed, smiled, and enjoyed every moment. This picture reminds us that happiness comes from simple joys. The end!"
    
    elif "park" in caption or "beach" in caption or "garden" in caption:
        return f"Wow! Look at this beautiful {caption}. The sun was shining, flowers were blooming, and everything felt magical. It was the perfect day to be outside, explore nature, and make happy memories with friends. The end!"
    
    else:
        return f"Look at this amazing picture! It shows {caption}. Everyone in this picture is having a wonderful time. They are smiling, playing, and enjoying life. This happy scene teaches us to always find joy in the little things. The end!"


def text2audio_generator(story_text, audio_pipeline):
    """
    Convert story text to audio using the MMS TTS model from class demo
    """
    try:
        # Generate audio using the pipeline
        speech_output = audio_pipeline(story_text)
        
        # Extract audio array and sample rate
        audio_array = speech_output["audio"]
        sample_rate = speech_output["sampling_rate"]
        
        return audio_array, sample_rate
        
    except Exception as e:
        st.error(f"Error converting text to audio: {e}")
        return None, None


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
    st.markdown("### Turn any picture into a magical story!")
    st.markdown("🎈 For kids aged 3-10 | 🌟 Fun and family-friendly")
    
    # Sidebar
    with st.sidebar:
        st.header("🌟 How to Use")
        st.markdown("""
        1. **Upload a picture** (JPG, JPEG, or PNG)
        2. **Click 'Generate Story'** 
        3. **Read the story** about what's in your picture
        4. **Listen to the audio** version
        """)
        st.divider()
        st.markdown("**📸 Best pictures to try:**")
        st.markdown("- 🐕 A dog or cat playing")
        st.markdown("- 👧 Children smiling or playing")
        st.markdown("- 🌈 Rainbow, flowers, or nature")
        st.markdown("- 🏖️ Beach, park, or playground")
        st.markdown("- 🎂 Birthday party or celebration")
        st.divider()
        st.markdown("**💡 Tip:** Clear, bright pictures work best!")
    
    # File uploader
    uploaded_image = st.file_uploader(
        "🎨 **Upload an image**", 
        type=["jpg", "jpeg", "png"],
        help="Upload any picture, and I'll create a story that matches what's in the picture!"
    )
    
    # Display uploaded image
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="🖼️ Your Picture", use_container_width=True)
        
        # Generate Story Button
        if st.button("✨ Generate Story ✨", type="primary", use_container_width=True):
            
            # Step 1: Load models and generate caption
            with st.spinner("🖼️ Looking at your picture..."):
                captioning_pipeline = load_captioning_model()
                caption = img2text(image, captioning_pipeline)
                
                # Display the caption (what the AI sees)
                st.markdown(f"""
                <div class="caption-box">
                🔍 <strong>The AI sees:</strong> {caption}
                </div>
                """, unsafe_allow_html=True)
            
            # Step 2: Generate story from caption using the story generator model
            with st.spinner("✍️ Writing a magical story for you..."):
                story_generator = load_story_generator()
                story = caption_to_story(caption, story_generator)
            
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
            
            # Step 3: Convert story to audio using MMS TTS model
            with st.spinner("🔊 Converting story to audio..."):
                audio_generator = load_audio_generator()
                audio_array, sample_rate = text2audio_generator(story, audio_generator)
            
            if audio_array is not None:
                st.subheader("🎧 Listen to the Story")
                st.audio(audio_array, sample_rate=sample_rate)
                st.info("💡 Click the play button above to listen to the story!")
            else:
                st.warning("⚠️ Audio generation failed. You can still read the story above!")
    
    else:
        # Placeholder when no image
        st.info("👆 **Please upload an image to begin your storytelling adventure!**")
        
        # Show example
        with st.expander("📷 See example", expanded=False):
            st.markdown("""
            Try uploading a picture of:
            - A child playing in a park
            - A cute animal
            - A beautiful sunset or rainbow
            - Friends or family having fun
            
            The AI will look at your picture and then tell a story about what it sees!
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #95A5A6; font-size: 13px;'>"
        "Made with ❤️ for young storytellers | 📖 Pictures become stories | 🔊 Stories become audio"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

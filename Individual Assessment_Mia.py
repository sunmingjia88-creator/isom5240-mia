"""
Storytelling Application for Kids (Age 3-10)
Using Hugging Face Transformers Pipelines
- Image Captioning: Salesforce/blip-image-captioning-base (generates detailed caption)
- Story Generation: Using a template-based approach with caption insertion (avoids hallucination)
- Text-to-Speech: gTTS (Google Text-to-Speech)
"""

import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
import tempfile
import os
import random

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
        return "a happy scene with children playing"


def create_story_from_caption(caption):
    """
    Create a coherent, engaging story based on the image caption.
    Uses structured story templates that incorporate the actual image content.
    This avoids hallucination and ensures the story matches the picture.
    """
    
    # Clean up the caption
    caption = caption.lower().strip()
    
    # Define story templates for different types of scenes
    # Each template uses the actual caption to ensure relevance
    
    # Template Set 1: Adventure style
    templates_adventure = [
        f"""Once upon a time, there was {caption}. 

Every day was a new adventure! The sun was shining brightly as our friend explored the wonderful world around them. They met new friends, discovered beautiful flowers, and laughed with joy. 

What made this day so special was all the happiness that filled the air. Our friend learned that every moment is precious when you share it with others. 

And so, this beautiful {caption} reminds us to always find joy in simple things. The end!""",

        f"""In a magical land not so far away, there was {caption}. 

The day began with a warm, golden sun rising in the sky. Our little hero felt excited and curious about everything around. They played, they smiled, and they made wonderful memories. 

This {caption} taught everyone that being brave and kind makes the world a better place. Every day brings new chances to be happy and spread love.

And they all lived happily, enjoying {caption}. The end!"""
    ]
    
    # Template Set 2: Friendship style
    templates_friendship = [
        f"""Look at this wonderful picture! It shows {caption}. 

This reminds us of how special friendship and happiness can be. Everyone in this {caption} is having such a good time together. They are learning, growing, and sharing beautiful moments. 

The most important thing is to always be kind to one another. When we share our happiness, everyone feels warm and loved. 

Let's remember this happy {caption} whenever we need a smile. The end!""",

        f"""What a beautiful scene! Here we can see {caption}. 

This picture tells a story of joy and wonder. Every little detail shows how amazing our world can be. The colors, the smiles, and the magic of {caption} make our hearts feel full of happiness. 

Remember: every day is special, just like this {caption}. Always look for the good things around you, because happiness is everywhere if you know where to look.

The end!"""
    ]
    
    # Template Set 3: Simple and sweet (for younger kids)
    templates_simple = [
        f"""I see {caption}. 

This is such a nice picture! Everything about {caption} makes me feel happy and warm inside. 

When I look at this picture, I think about all the wonderful things in the world. The sun, the smiles, and the love that surrounds us every single day. 

Let's always remember this happy {caption} and carry its joy in our hearts. 

The end!""",

        f"""Wow! Look at this amazing picture of {caption}. 

Isn't it beautiful? This picture shows us how wonderful life can be when we take time to enjoy the little things. 

Whether it's playing, laughing, or just being together, moments like {caption} are what make life special. 

So let's celebrate this beautiful {caption} and all the happiness it brings. 

The end!"""
    ]
    
    # Combine all templates
    all_templates = templates_adventure + templates_friendship + templates_simple
    
    # Randomly select a template (but same seed for same caption to keep consistency)
    # Using hash of caption to select template consistently
    template_index = hash(caption) % len(all_templates)
    story = all_templates[template_index]
    
    # Post-process to ensure the story flows well
    story = story.replace("  ", " ")
    
    # Ensure word count is between 50-100
    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:97]) + "... The end!"
    elif len(words) < 50:
        # Add a sentence if too short
        story = story + " This happy picture reminds us to always smile and be grateful for every moment we share with others. The end!"
    
    return story


def create_detailed_story(caption, image):
    """
    Create an even more detailed story by extracting more information from the image
    This is a fallback/enhanced version
    """
    
    # Basic story using the caption
    base_story = create_story_from_caption(caption)
    
    # Add some variation based on image dimensions/colors (simple enhancement)
    try:
        width, height = image.size
        if width > height:
            orientation = "wide and beautiful"
        else:
            orientation = "tall and wonderful"
        
        # Enhance the story with orientation detail
        enhancement = f" This {orientation} scene captures a perfect moment."
        
        # Insert enhancement at a natural point
        sentences = base_story.split('. ')
        if len(sentences) > 2:
            sentences.insert(2, enhancement)
            enhanced_story = '. '.join(sentences)
            return enhanced_story
    except:
        pass
    
    return base_story


def text2audio(story_text):
    """
    Convert story text to audio using gTTS
    """
    try:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # Use slow=True for better clarity for kids
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
        page_title="Kids Storyteller - Picture to Story",
        page_icon="📖",
        layout="centered"
    )
    
    # Custom CSS for better readability
    st.markdown("""
        <style>
        .story-box {
            background-color: #FFF8DC;
            padding: 25px;
            border-radius: 20px;
            border-left: 8px solid #FF6B6B;
            border-right: 8px solid #4ECDC4;
            font-size: 20px;
            line-height: 1.8;
            font-family: 'Comic Neue', 'Comic Sans MS', 'Chalkboard SE', cursive;
            color: #2C3E50;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .story-title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            color: #FF6B6B;
            margin-bottom: 20px;
        }
        .caption-box {
            background-color: #E8F8F5;
            padding: 12px;
            border-radius: 15px;
            font-style: italic;
            font-size: 16px;
            margin: 10px 0;
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
        3. **Read the story** that matches your picture
        4. **Listen to the audio** version
        5. **Download** to keep the story!
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
            
            # Step 2: Create story from caption
            with st.spinner("✍️ Writing a story based on your picture..."):
                story = create_detailed_story(caption, image)
            
            # Display the story
            st.markdown("---")
            st.markdown('<div class="story-title">✨ The Story ✨</div>', unsafe_allow_html=True)
            
            # Story in a nice box with larger, readable font
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
                        file_name="my_story.mp3",
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
        st.info("👆 **Please upload an image to begin your storytelling adventure!**")
        
        # Show example
        with st.expander("📷 See example", expanded=False):
            st.markdown("""
            Try uploading a picture of:
            - A child playing in a park
            - A cute animal
            - A beautiful sunset or rainbow
            - Friends or family having fun
            
            The AI will look at your picture and create a story that matches what it sees!
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

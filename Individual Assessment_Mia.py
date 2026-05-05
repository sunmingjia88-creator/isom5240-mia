"""
Storytelling Application for Kids (Age 3-10)
Using Hugging Face Transformers Pipelines
- Image Captioning: Salesforce/blip-image-captioning-base
- Story Generation: Describes what's actually in the picture using simple, kid-friendly language
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
        return "a happy scene"


def extract_key_elements(caption):
    """
    Extract key nouns and elements from caption to use in the story
    """
    # Common words to remove
    stop_words = {'a', 'an', 'the', 'and', 'of', 'with', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'has', 'have'}
    
    words = caption.lower().split()
    key_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Try to identify what's in the picture
    elements = []
    
    # People/animals
    people_indicators = ['child', 'children', 'kid', 'kids', 'boy', 'girl', 'person', 'people', 'man', 'woman', 'friend', 'family']
    animal_indicators = ['dog', 'cat', 'bird', 'rabbit', 'horse', 'cow', 'pig', 'duck', 'chicken', 'elephant', 'lion', 'tiger', 'bear', 'monkey', 'fish', 'butterfly', 'bee']
    
    for word in words:
        if word in people_indicators:
            elements.append(('people', word))
        elif word in animal_indicators:
            elements.append(('animal', word))
    
    # Actions
    action_indicators = ['playing', 'running', 'jumping', 'sitting', 'standing', 'eating', 'drinking', 'sleeping', 'walking', 'smiling', 'laughing', 'crying', 'reading', 'drawing', 'singing', 'dancing', 'swimming']
    
    for word in words:
        if word in action_indicators:
            elements.append(('action', word))
    
    # Objects/Things
    object_indicators = ['ball', 'toy', 'book', 'table', 'chair', 'car', 'bike', 'house', 'tree', 'flower', 'grass', 'water', 'sun', 'cloud', 'rainbow', 'cake', 'birthday', 'present', 'gift']
    
    for word in words:
        if word in object_indicators:
            elements.append(('object', word))
    
    # Places
    place_indicators = ['park', 'beach', 'forest', 'garden', 'zoo', 'school', 'home', 'house', 'store', 'restaurant', 'kitchen', 'bedroom', 'playground']
    
    for word in words:
        if word in place_indicators:
            elements.append(('place', word))
    
    return elements, key_words


def create_story_from_caption(caption):
    """
    Create a kid-friendly story that ACTUALLY describes what's in the picture
    """
    caption = caption.lower().strip()
    elements, key_words = extract_key_elements(caption)
    
    # Determine what's in the picture
    has_people = any(e[0] == 'people' for e in elements)
    has_animal = any(e[0] == 'animal' for e in elements)
    has_action = any(e[0] == 'action' for e in elements)
    has_place = any(e[0] == 'place' for e in elements)
    has_object = any(e[0] == 'object' for e in elements)
    
    # Extract specific items for better description
    people_word = next((e[1] for e in elements if e[0] == 'people'), 'someone')
    animal_word = next((e[1] for e in elements if e[0] == 'animal'), None)
    action_word = next((e[1] for e in elements if e[0] == 'action'), 'being happy')
    place_word = next((e[1] for e in elements if e[0] == 'place'), None)
    object_word = next((e[1] for e in elements if e[0] == 'object'), None)
    
    # Start building the story
    story = ""
    
    # Opening sentence - describe what we see
    if has_people:
        if people_word == 'child' or people_word == 'kid':
            story += f"Look at this happy picture! I see a little {people_word} {action_word}. "
        elif people_word == 'children' or people_word == 'kids':
            story += f"Look at this happy picture! I see {people_word} {action_word}. "
        else:
            story += f"Look at this wonderful picture! I see {people_word} {action_word}. "
    elif has_animal:
        story += f"Look at this cute picture! I see a {animal_word} {action_word}. "
    else:
        story += f"Look at this nice picture! I see {caption}. "
    
    # Second sentence - describe the place or objects
    if has_place:
        story += f"This is happening in a {place_word}. "
    elif has_object:
        story += f"I can also see a {object_word} nearby. "
    
    # Third sentence - add some fun details
    if has_people and has_action:
        if action_word in ['playing', 'running', 'jumping']:
            story += f"The {people_word} looks so happy and full of energy! "
        elif action_word in ['eating', 'drinking']:
            story += f"The {people_word} is enjoying a tasty treat! "
        elif action_word in ['reading', 'drawing']:
            story += f"The {people_word} is learning something new and having fun! "
        else:
            story += f"This makes everyone feel warm and happy inside! "
    elif has_animal:
        story += f"The {animal_word} is so cute and friendly! "
    
    # Fourth sentence - add a simple lesson or feeling
    if has_people:
        story += f"We can learn that playing and smiling with others is the best thing to do. "
    elif has_animal:
        story += f"We can learn to be kind to animals and love nature. "
    else:
        story += f"We can learn to enjoy the beautiful things around us every day. "
    
    # Fifth sentence - positive ending
    if has_people:
        story += f"I hope the {people_word} has a wonderful day full of joy and laughter! The end!"
    elif has_animal:
        story += f"I hope this sweet {animal_word} stays happy and healthy forever! The end!"
    else:
        story += f"I hope you enjoyed this beautiful picture and story! The end!"
    
    # Clean up any "a a" or duplicate issues
    story = re.sub(r'\ba\s+a\b', 'a', story)
    story = re.sub(r'\ban\s+an\b', 'an', story)
    
    # Ensure word count is between 50-100
    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:97]) + "... The end!"
    elif len(words) < 50:
        # Add a simple sentence if too short
        if has_people:
            story = story.replace("The end!", "Every day is special when we share it with friends! The end!")
        else:
            story = story.replace("The end!", "Let's always be happy and kind to everyone we meet! The end!")
    
    return story


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

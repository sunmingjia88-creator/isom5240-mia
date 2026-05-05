"""
Storytelling Application for Kids (Age 3-10)
Using Hugging Face Transformers Pipelines
- Image Captioning: Salesforce/blip-image-captioning-base
- Story Generation: Dynamically creates story based on actual image content
- Text-to-Speech: gTTS (Google Text-to-Speech)
"""

import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
import tempfile
import os
import re
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
        return "a happy scene"


def create_kid_friendly_story(caption):
    """
    Create a story that truly describes what's in the picture
    Each story is built dynamically based on the actual caption content
    """
    original_caption = caption
    caption = caption.lower().strip()
    
    # Remove common prefixes that BLIP adds
    caption = re.sub(r'^a photo of |^an image of |^a picture of |^a man |^a woman ', '', caption)
    
    # Split into words for analysis
    words = caption.split()
    
    # === Identify what's in the picture ===
    
    # Detect main subject (what is the picture about?)
    subjects = []
    animals = {'dog': '🐕', 'cat': '🐱', 'bird': '🐦', 'rabbit': '🐰', 'horse': '🐴', 
               'cow': '🐮', 'pig': '🐷', 'duck': '🦆', 'elephant': '🐘', 'lion': '🦁',
               'tiger': '🐯', 'bear': '🐻', 'monkey': '🐵', 'fish': '🐟', 'butterfly': '🦋',
               'bee': '🐝', 'puppy': '🐕', 'kitten': '🐱', 'bunny': '🐰'}
    
    people = {'boy': '👦', 'girl': '👧', 'child': '🧒', 'children': '👧👦', 
              'kid': '🧒', 'kids': '🧒🧒', 'man': '👨', 'woman': '👩', 
              'person': '🧑', 'people': '👥', 'friend': '🤝', 'family': '👨‍👩‍👧‍👦'}
    
    actions = {'playing': 'playing', 'running': 'running', 'jumping': 'jumping', 
               'sitting': 'sitting', 'standing': 'standing', 'eating': 'eating',
               'drinking': 'drinking', 'sleeping': 'sleeping', 'walking': 'walking',
               'smiling': 'smiling', 'laughing': 'laughing', 'reading': 'reading',
               'drawing': 'drawing', 'singing': 'singing', 'dancing': 'dancing',
               'swimming': 'swimming', 'flying': 'flying', 'barking': 'barking'}
    
    places = {'park': 'park 🌳', 'beach': 'beach 🏖️', 'forest': 'forest 🌲', 
              'garden': 'garden 🌸', 'zoo': 'zoo 🦒', 'school': 'school 🏫',
              'home': 'home 🏠', 'house': 'house 🏡', 'store': 'store 🛒',
              'restaurant': 'restaurant 🍽️', 'kitchen': 'kitchen 🍳', 
              'bedroom': 'bedroom 🛏️', 'playground': 'playground 🎠', 'grass': 'grass 🌿',
              'water': 'water 💧', 'pool': 'pool 🏊'}
    
    objects = {'ball': 'ball ⚽', 'toy': 'toy 🧸', 'book': 'book 📚', 
               'table': 'table 🪑', 'chair': 'chair 💺', 'car': 'car 🚗',
               'bike': 'bike 🚲', 'tree': 'tree 🌳', 'flower': 'flower 🌸',
               'sun': 'sun ☀️', 'cloud': 'cloud ☁️', 'rainbow': 'rainbow 🌈',
               'cake': 'cake 🎂', 'present': 'present 🎁', 'gift': 'gift 🎁',
               'balloon': 'balloon 🎈', 'ice cream': 'ice cream 🍦', 'pizza': 'pizza 🍕'}
    
    # Find main subject
    main_subject = None
    subject_emoji = ""
    subject_type = None
    
    for word in words:
        if word in animals:
            main_subject = word
            subject_emoji = animals[word]
            subject_type = 'animal'
            break
        elif word in people:
            main_subject = word
            subject_emoji = people[word]
            subject_type = 'person'
            break
    
    # Find action
    main_action = None
    for word in words:
        if word in actions:
            main_action = actions[word]
            break
    
    # Find place
    main_place = None
    place_display = None
    for word in words:
        if word in places:
            main_place = word
            place_display = places[word]
            break
    
    # Find object
    main_object = None
    object_display = None
    for word in words:
        if word in objects:
            main_object = word
            object_display = objects[word]
            break
    
    # Count how many subjects (plural detection)
    is_plural = 'children' in caption or 'kids' in caption or 'people' in caption or 'dogs' in caption or 'cats' in caption
    
    # If no specific subject found, use the whole caption
    if not main_subject:
        main_subject = caption[:30] if len(caption) > 30 else caption
        subject_type = 'scene'
    
    # === Build the story dynamically ===
    
    story_parts = []
    
    # Opening sentence - describe what we see
    if subject_type == 'person':
        if is_plural:
            story_parts.append(f"Wow! Look at this picture. I see {main_subject}s {main_action if main_action else 'having fun'} {subject_emoji}")
        else:
            story_parts.append(f"Wow! Look at this picture. I see a {main_subject} {main_action if main_action else 'playing'} {subject_emoji}")
    elif subject_type == 'animal':
        story_parts.append(f"Aww! Look at this cute picture. I see a {main_subject} {main_action if main_action else 'being happy'} {subject_emoji}")
    else:
        story_parts.append(f"Wow! Look at this beautiful picture. I see {original_caption}")
    
    # Second sentence - describe the place or objects
    if main_place and place_display:
        story_parts.append(f"Look where they are - in a {place_display} 🏞️")
    elif main_object and object_display:
        story_parts.append(f"I also see a {object_display} in this picture")
    else:
        # Describe the scene more
        if 'outside' in caption or 'outdoor' in caption:
            story_parts.append(f"Everything looks so bright and sunny outside ☀️")
        elif 'inside' in caption or 'indoor' in caption:
            story_parts.append(f"This is happening inside a cozy place 🏠")
        elif 'color' in caption or 'colorful' in caption:
            story_parts.append(f"The colors in this picture are so pretty and bright! 🎨")
    
    # Third sentence - describe what's happening
    if main_action:
        if main_action == 'playing':
            story_parts.append(f"{main_subject.capitalize()} is playing and having so much fun! It looks like a great time 🎉")
        elif main_action == 'smiling':
            story_parts.append(f"Everyone is smiling - that makes me happy too! 😊")
        elif main_action == 'eating':
            story_parts.append(f"Yum! {main_subject.capitalize()} is enjoying some delicious food 🍽️")
        elif main_action == 'running':
            story_parts.append(f"{main_subject.capitalize()} is running so fast! What a great energy ⚡")
        elif main_action == 'sleeping':
            story_parts.append(f"Aww, {main_subject.capitalize()} looks so peaceful and cozy 😴")
        elif main_action == 'reading':
            story_parts.append(f"Reading is fun! {main_subject.capitalize()} is learning new things 📖")
        else:
            story_parts.append(f"{main_subject.capitalize()} is {main_action} and seems very happy about it 🌟")
    else:
        story_parts.append(f"This is such a wonderful moment to look at 🌟")
    
    # Fourth sentence - add a fun observation
    if main_subject:
        if subject_type == 'animal':
            fun_phrases = [
                f"I wonder what the {main_subject} is thinking right now 🤔",
                f"The {main_subject} looks so soft and cuddly 🫶",
                f"This {main_subject} reminds me to be happy every day 💛"
            ]
        else:
            fun_phrases = [
                f"I wonder what fun things will happen next 🎈",
                f"This makes me want to go out and play too! 🏃",
                f"Moments like this are the best memories ✨"
            ]
        story_parts.append(random.choice(fun_phrases))
    else:
        story_parts.append(f"This picture makes my heart feel warm and happy 💛")
    
    # Fifth sentence - unique ending based on picture
    if subject_type == 'person':
        endings = [
            f"I hope the {main_subject}s in this picture have the best day ever! Goodbye for now 👋",
            f"What a lovely picture of {main_subject}s. May every day be as happy as this one! 🌈",
            f"Thank you for sharing this beautiful moment with me. Have a wonderful day! 🦋"
        ]
    elif subject_type == 'animal':
        endings = [
            f"I hope this sweet {main_subject} gets lots of hugs and treats today! Bye bye 🐾",
            f"What a good {main_subject}! Animals make our world so much brighter. Stay happy! 🫶",
            f"Thank you for showing me this cute {main_subject}. Sending love to your furry friend! 💕"
        ]
    elif main_place:
        endings = [
            f"What a beautiful day at the {main_place}. I hope you get to visit places like this too! 🗺️",
            f"I love seeing pretty places like this. Thanks for sharing this adventure with me! 🌍"
        ]
    else:
        endings = [
            f"This picture is so nice. Thank you for showing me! Keep smiling every day 😊",
            f"Every picture tells a story, and this one tells a happy story. Bye for now! 🌟"
        ]
    
    story_parts.append(random.choice(endings))
    
    # Join all parts
    story = " ".join(story_parts)
    
    # Clean up any double spaces or issues
    story = re.sub(r'\s+', ' ', story).strip()
    
    # Ensure first letter is capital
    story = story[0].upper() + story[1:] if len(story) > 1 else story
    
    # Ensure word count is between 50-100
    words_count = len(story.split())
    if words_count > 100:
        # Trim to about 95 words and add ending
        story_words = story.split()[:95]
        story = " ".join(story_words) + " The end!"
    elif words_count < 50:
        # Add a sweet sentence if too short
        story = story + " Every picture has a story, and this one is full of happiness and love. The end! 💝"
    
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
            with st.spinner("✍️ Writing a magical story for you..."):
                story = create_kid_friendly_story(caption)
            
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

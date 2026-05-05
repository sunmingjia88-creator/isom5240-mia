"""
Storytelling Application for Kids (Age 3-10)
Using Hugging Face Transformers Pipelines
- Image Captioning: Salesforce/blip-image-captioning-base
- Story Generation: Rich description based on caption (no hallucination)
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
        return "a happy scene with beautiful things"


def create_rich_story_from_caption(caption):
    """
    Create a rich, descriptive story based on the image caption.
    Stays true to the image content - no made-up stories or fantasy.
    Just describes what is in the picture in a more engaging way.
    """
    original_caption = caption
    caption_lower = caption.lower().strip()
    
    # Remove common prefixes
    caption_clean = re.sub(r'^a photo of |^an image of |^a picture of |^a |an ', '', caption_lower)
    
    # Split into words for analysis
    words = caption_lower.split()
    
    # === Identify key elements in the picture ===
    
    # Find the main subject (what is this picture about?)
    animals = ['dog', 'puppy', 'cat', 'kitten', 'bird', 'rabbit', 'horse', 'cow', 'pig', 
               'duck', 'elephant', 'lion', 'tiger', 'bear', 'monkey', 'fish', 'butterfly']
    
    people = ['boy', 'girl', 'child', 'children', 'kid', 'kids', 'man', 'woman', 
              'person', 'people', 'baby', 'friend', 'family', 'mother', 'father', 'parent']
    
    actions = ['playing', 'running', 'jumping', 'sitting', 'standing', 'eating', 'drinking',
               'sleeping', 'walking', 'smiling', 'laughing', 'reading', 'drawing', 'singing',
               'dancing', 'swimming', 'flying', 'barking', 'looking']
    
    places = ['park', 'beach', 'forest', 'garden', 'zoo', 'school', 'home', 'house',
              'store', 'restaurant', 'kitchen', 'bedroom', 'playground', 'grass', 'water',
              'pool', 'street', 'road', 'field', 'farm']
    
    objects = ['ball', 'toy', 'book', 'table', 'chair', 'car', 'bike', 'tree', 'flower',
               'sun', 'cloud', 'sky', 'rainbow', 'cake', 'present', 'gift', 'balloon',
               'ice cream', 'pizza', 'food', 'plate', 'cup', 'phone', 'computer']
    
    # Find subject
    subject = None
    subject_type = None
    for word in words:
        if word in animals:
            subject = word
            subject_type = 'animal'
            break
        elif word in people:
            subject = word
            subject_type = 'person'
            break
    
    # If no specific subject found, use the main noun from caption
    if not subject:
        # Try to get the first noun
        for word in words:
            if len(word) > 2 and word not in ['the', 'and', 'with', 'this', 'that']:
                subject = word
                subject_type = 'thing'
                break
    
    # Find action
    action = None
    for word in words:
        if word in actions:
            action = word
            break
    
    # Find place
    place = None
    for word in words:
        if word in places:
            place = word
            break
    
    # Find objects
    found_objects = []
    for word in words:
        if word in objects and word != subject:
            found_objects.append(word)
    
    # === Build a rich, descriptive story ===
    
    story_parts = []
    
    # Part 1: Opening - What we see
    if subject:
        if subject_type == 'animal':
            story_parts.append(f"Look at this beautiful picture. In this picture, I can see a {subject}.")
        elif subject_type == 'person':
            if subject in ['child', 'kid', 'boy', 'girl']:
                story_parts.append(f"Look at this lovely picture. In this picture, I can see a little {subject}.")
            else:
                story_parts.append(f"Look at this nice picture. In this picture, I can see a {subject}.")
        else:
            story_parts.append(f"Look at this wonderful picture. In this picture, I can see {subject}.")
    else:
        story_parts.append(f"Look at this picture. In this picture, I can see {original_caption}.")
    
    # Part 2: Describe what is happening (action)
    if action:
        story_parts.append(f"The {subject if subject else 'scene'} is {action} right now in this photo.")
    else:
        if subject:
            story_parts.append(f"The {subject} is staying still and looking nice in this picture.")
    
    # Part 3: Describe the location (place)
    if place:
        story_parts.append(f"This is happening in or near a {place}. The {place} looks like a nice place to be.")
    else:
        if 'outside' in caption_lower or 'outdoor' in caption_lower:
            story_parts.append(f"This picture was taken outside. The outdoors looks bright and natural.")
        elif 'inside' in caption_lower or 'indoor' in caption_lower:
            story_parts.append(f"This picture was taken inside a building or room. It looks cozy and comfortable.")
        else:
            story_parts.append(f"The background of this picture is also very interesting to look at.")
    
    # Part 4: Describe any objects in the picture
    if found_objects:
        if len(found_objects) == 1:
            story_parts.append(f"I can also see a {found_objects[0]} in this picture.")
        elif len(found_objects) >= 2:
            objects_list = ", ".join(found_objects[:-1]) + f" and {found_objects[-1]}"
            story_parts.append(f"I can also see {objects_list} in this picture.")
    else:
        if subject_type == 'animal':
            story_parts.append(f"The {subject} is the main thing I notice when I look at this picture.")
        elif subject_type == 'person':
            story_parts.append(f"The {subject} is doing something interesting in this photo.")
    
    # Part 5: Describe colors if mentioned in caption
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'pink', 'brown', 'black', 'white', 'orange']
    found_colors = [c for c in colors if c in caption_lower]
    if found_colors:
        story_parts.append(f"I see {', '.join(found_colors)} colors in this picture. The colors make this photo look very nice.")
    
    # Part 6: Positive observation
    if subject_type == 'animal':
        story_parts.append(f"This {subject} looks very cute and friendly. I like looking at this animal picture.")
    elif subject_type == 'person':
        if action in ['smiling', 'laughing', 'playing']:
            story_parts.append(f"The {subject} looks very happy and full of joy in this picture. That makes me feel happy too.")
        else:
            story_parts.append(f"This is a nice picture of a person. It is always good to see people having a good time.")
    else:
        story_parts.append(f"This is a very nice picture to look at. Every picture tells us something about the world around us.")
    
    # Part 7: Closing
    story_parts.append(f"Thank you for sharing this picture with me. That is what I see when I look at this photo of {original_caption}.")
    story_parts.append("The end.")
    
    # Join all parts
    story = " ".join(story_parts)
    
    # Clean up
    story = story.replace("  ", " ")
    story = story.replace("the the", "the")
    story = story.replace("a a", "a")
    
    # Ensure word count is between 50-100
    words_count = len(story.split())
    if words_count > 100:
        # Trim to about 100 words
        story_words = story.split()[:97]
        story = " ".join(story_words) + " The end."
    elif words_count < 50:
        # Add a few more descriptive sentences
        story = story.replace("The end.", f" This picture shows {original_caption} clearly. I hope you enjoyed this description of your photo. The end.")
    
    # Capitalize first letter
    story = story[0].upper() + story[1:]
    
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
    st.markdown("### Turn any picture into a rich story!")
    st.markdown("🎈 For kids aged 3-10 | 🌟 Clear descriptions about your picture")
    
    # Sidebar
    with st.sidebar:
        st.header("🌟 How to Use")
        st.markdown("""
        1. **Upload a picture** (JPG, JPEG, or PNG)
        2. **Click 'Generate Story'** 
        3. **Read the rich description** of what's in your picture
        4. **Listen to the audio** version
        5. **Download** to keep the story!
        """)
        st.divider()
        st.markdown("**📸 Best pictures to try:**")
        st.markdown("- 🐕 A dog or cat")
        st.markdown("- 👧 Children or people")
        st.markdown("- 🌸 Flowers, trees, nature")
        st.markdown("- 🏠 Houses, buildings, places")
        st.markdown("- 🍕 Food, toys, objects")
        st.divider()
        st.markdown("**💡 How it works:**")
        st.markdown("- The AI looks at your picture")
        st.markdown("- It describes what it sees")
        st.markdown("- The story stays true to your picture")
    
    # File uploader
    uploaded_image = st.file_uploader(
        "🎨 **Upload an image**", 
        type=["jpg", "jpeg", "png"],
        help="Upload any picture, and I'll create a rich description of what I see!"
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
            
            # Step 2: Create rich story from caption
            with st.spinner("✍️ Writing a rich description of your picture..."):
                story = create_rich_story_from_caption(caption)
            
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
        with st.expander("📷 See an example", expanded=False):
            st.markdown("""
            **Example:** If you upload a picture of a dog sitting on grass
            
            **The story will describe:**
            - What animal is in the picture (a dog)
            - What the dog is doing (sitting)
            - Where the dog is (on grass)
            - What colors are in the picture
            - A positive, happy message about the picture
            
            No made-up stories - just a clear, rich description of YOUR picture!
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #95A5A6; font-size: 13px;'>"
        "Made with ❤️ for young storytellers | 📖 Pictures become rich descriptions | 🔊 Stories become audio"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

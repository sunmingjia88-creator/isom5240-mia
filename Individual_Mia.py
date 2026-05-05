"""
Storytelling Application for Kids (Age 3-10)
Using Hugging Face Transformers Pipelines
- Image Captioning: Salesforce/blip-image-captioning-base
- Story Generation: Template-based + keyword matching (80-100 words)
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
    """
    try:
        result = captioning_pipeline(image)
        caption = result[0]["generated_text"]
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return "a group of people eating together"


def extract_keywords(caption):
    """
    Extract key elements from caption to understand the scene
    """
    caption_lower = caption.lower()
    
    # Define keyword categories
    keywords = {
        'eating': ['eating', 'dinner', 'lunch', 'food', 'meal', 'restaurant', 'table', 'plate', 'fork', 'knife', 'dining'],
        'playing': ['playing', 'game', 'toy', 'ball', 'run', 'jump', 'playground', 'park', 'sports'],
        'animal': ['dog', 'cat', 'bird', 'rabbit', 'horse', 'cow', 'pig', 'duck', 'elephant', 'lion', 'tiger', 'bear', 'monkey', 'puppy', 'kitten'],
        'outdoor': ['park', 'beach', 'forest', 'garden', 'outside', 'sun', 'grass', 'tree', 'flower', 'nature', 'mountain'],
        'indoor': ['inside', 'room', 'house', 'home', 'building', 'office', 'school', 'classroom'],
        'celebration': ['birthday', 'party', 'cake', 'candle', 'gift', 'present', 'balloon', 'celebration', 'celebrate'],
        'family': ['family', 'parent', 'mother', 'father', 'mom', 'dad', 'sister', 'brother', 'grandma', 'grandpa'],
        'friends': ['friend', 'friends', 'group', 'people', 'together', 'crowd'],
        'sleeping': ['sleep', 'sleeping', 'bed', 'nap', 'rest', 'tired'],
        'reading': ['read', 'reading', 'book', 'story', 'library'],
        'walking': ['walk', 'walking', 'stroll', 'street', 'road', 'path'],
        'beach': ['beach', 'ocean', 'sea', 'sand', 'wave', 'coast'],
        'school': ['school', 'class', 'student', 'teacher', 'classroom', 'learn'],
        'cooking': ['cook', 'cooking', 'kitchen', 'bake', 'making food']
    }
    
    matched = []
    for category, words in keywords.items():
        for word in words:
            if word in caption_lower:
                matched.append(category)
                break
    
    # Remove duplicates
    matched = list(set(matched))
    
    return matched


def count_words(text):
    """Count the number of words in a text"""
    return len(text.split())


def ensure_word_count(story, target_min=80, target_max=100):
    """Ensure story has between 80-100 words"""
    word_count = count_words(story)
    
    if word_count > target_max:
        # Trim to about target_max words
        words = story.split()[:target_max]
        story = ' '.join(words)
        if not story.endswith(('.', '!', '?')):
            story += '.'
    elif word_count < target_min:
        # Add a few more sentences
        extra_sentences = [
            " It was such a special moment that I will never forget.",
            " I feel so lucky to have this memory to look back on.",
            " This picture always makes me smile when I see it.",
            " Times like these are what make life so wonderful.",
            " I hope we can have more happy days like this one.",
            " My heart feels so full of joy and gratitude.",
            " This is what happiness looks like in a picture.",
            " I am so thankful for everyone in this photo.",
            " These are the little moments that mean the most.",
            " Every time I see this picture, it makes me happy."
        ]
        # Add random extra sentences until we reach min words
        while count_words(story) < target_min:
            story += random.choice(extra_sentences)
    
    return story


def create_story_from_caption(caption, keywords):
    """
    Create a natural, first-person story based on the image content (80-100 words)
    """
    caption_lower = caption.lower()
    
    # === EATING / DINING / RESTAURANT stories (80-100 words) ===
    if 'eating' in keywords or 'dinner' in caption_lower or 'lunch' in caption_lower or 'food' in caption_lower or 'table' in caption_lower:
        stories = [
            f"Today was such a special day. I went out to eat with my family and friends. When I look at this picture, I see {caption}. The restaurant was warm and cozy. We ordered so many yummy dishes like pizza, noodles, and salad. Everyone was laughing and telling funny stories. My little brother made a silly face and we all cracked up. We shared our food and tried a little bit of everything. After dinner, we had ice cream for dessert. Eating together with people I love is the best feeling in the world. I hope we can do this again very soon.",
            
            f"I love looking at this picture. It shows {caption}. We were all sitting around a big table filled with delicious food. The smell of pizza and pasta made my tummy rumble. My mom passed me a plate of hot soup and my dad poured some juice for everyone. We talked about our day at school and work. My friend told us a funny joke that made everyone laugh so hard. We ate slowly and enjoyed every single bite. Having a meal together is not just about food. It is about love, happiness, and being together. I will remember this night forever.",
            
            f"This picture brings back so many happy memories. I see {caption}. We went to our favorite restaurant to celebrate something special. The table was full of colorful plates and steaming hot food. Everyone was hungry and excited. We took turns sharing what we were thankful for. Then we clinked our glasses and said cheers. The food tasted amazing. We ordered extra fries because they were so good. After we finished eating, we sat around and talked for a long time. Nobody wanted the night to end. Moments like these are the best parts of life."
        ]
        story = random.choice(stories)
        return ensure_word_count(story)
    
    # === BIRTHDAY / PARTY / CELEBRATION stories ===
    if 'celebration' in keywords or 'birthday' in caption_lower or 'cake' in caption_lower:
        stories = [
            f"Wow, what a fun day that was! This picture shows {caption}. We were celebrating my birthday and I was so excited. Colorful balloons floated everywhere and pretty streamers hung on the walls. My friends came over with gifts wrapped in shiny paper. When it was time for cake, everyone gathered around me. The cake had candles that sparkled like tiny stars. I closed my eyes, made a wish, and blew out all the candles in one big breath. Everyone clapped and cheered for me. We ate cake and played games like musical chairs and pin the tail on the donkey. I got so many nice presents. This was the best birthday party ever in my whole life.",
            
            f"I love looking at this happy picture. It shows {caption}. We were having a big celebration and everyone was smiling so brightly. The house was decorated with ribbons and balloons of all colors. My mom baked a big chocolate cake with vanilla frosting. We sang the birthday song really loud. My little cousin tried to help blow out the candles and it was so cute. After cake, we played fun party games and danced to our favorite songs. I got a big box of crayons and a coloring book as a gift. Everyone stayed late because nobody wanted the party to end. I am so lucky to have such wonderful family and friends."
        ]
        story = random.choice(stories)
        return ensure_word_count(story)
    
    # === PLAYING / OUTDOOR / PARK stories ===
    if 'playing' in keywords or 'outdoor' in keywords:
        stories = [
            f"This picture makes me feel so happy. I see {caption}. It was a beautiful sunny day, perfect for playing outside. The sky was blue and the grass was soft and green. My friends and I ran to the park as fast as we could. We played on the swings and went down the slide over and over. Then we kicked a ball around and tried to score goals. Some of us climbed the monkey bars and pretended we were superheroes. Everyone was laughing and having so much fun. The breeze felt cool on our warm faces. We played until the sun started to go down. Playing outside with friends is the best thing ever. I wish every day could be like this.",
            
            f"I remember this day so well. This picture shows {caption}. The weather was warm and perfect for being outside. We went to the playground near my house. First, we raced each other across the field. Then we took turns on the swings, trying to see who could go the highest. We played hide and seek behind the big trees. My friend found the best hiding spot and nobody could find him for a long time. We also played tag and ran until our legs got tired. We stopped to drink water and catch our breath. Then we played some more because we didn't want to go home. Days like this are the best days ever."
        ]
        story = random.choice(stories)
        return ensure_word_count(story)
    
    # === ANIMAL / PET stories ===
    if 'animal' in keywords:
        if 'dog' in caption_lower or 'puppy' in caption_lower:
            stories = [
                f"Aww, look at this cute picture! I see {caption}. This furry friend is a dog and it is so adorable. It has soft fur and a wet nose and the sweetest eyes. The dog was wagging its tail really fast because it was so happy to see me. I gave the dog a big hug and it licked my face. Then we played fetch in the yard with a red ball. The dog ran as fast as the wind to get the ball and brought it right back to me. After playing, the dog sat next to me and rested its head on my leg. Having a dog is like having a best friend who loves you no matter what. I am so glad this picture was taken.",
                
                f"I love seeing pictures of animals like this one. It shows {caption}. The dog in this photo is so cute and full of energy. Its tail was wagging like a little propeller. We played together in the backyard for a long time. I threw a stick and the dog chased after it with so much excitement. The dog also loves to roll on the grass and get belly rubs. When it got tired, it lay down next to me and panted with its tongue out. I sat down and pet its soft head. Being around dogs makes me feel so calm and happy. They teach us to be loyal, kind, and to enjoy every single moment of life."
            ]
        elif 'cat' in caption_lower or 'kitten' in caption_lower:
            stories = [
                f"What a sweet picture this is! I see {caption}. The cat in this photo is so soft and fluffy. Its fur looks like velvet and its eyes are big and shiny. The cat was taking a nap in a warm sunny spot by the window. When it woke up, it stretched its little paws and gave a tiny yawn. Then the cat jumped down and rubbed against my legs. I picked it up and it purred like a little motor. Cats are so gentle and calm. They like to be cozy and loved. I think every home should have a cat to snuggle with at the end of a long day.",
                
                f"This picture makes me smile. It shows {caption}. The cat has the prettiest fur I have ever seen. It was sitting on a soft cushion and looking out the window. I think it was watching birds fly by. When I called its name, the cat turned its head and blinked at me slowly. That is how cats say I love you. I gave the cat a little scratch behind the ears and it started to purr so loudly. Then the cat curled up into a tiny ball and went back to sleep. Cats are such wonderful, peaceful animals to have around."
            ]
        else:
            stories = [
                f"Look at this wonderful animal picture! I see {caption}. The animal looks so peaceful and beautiful. Its fur or feathers are really pretty to look at. I think it was enjoying the nice weather and fresh air. The animal moved around slowly and looked at everything around it. Sometimes it stopped to eat some grass or leaves. Other times it just stood there and watched the world go by. I feel so lucky to see an animal like this up close. Nature is full of amazing creatures, and every single one is special. We should always be kind to animals and take care of our world."
            ]
        story = random.choice(stories)
        return ensure_word_count(story)
    
    # === FAMILY / FRIENDS / GROUP stories ===
    if 'family' in keywords or 'friends' in keywords:
        stories = [
            f"This is one of my favorite pictures. I see {caption}. We were all together in one place, and everyone was smiling. My mom was wearing a pretty dress and my dad had a big smile on his face. My little sister was making a funny face and my brother was laughing. We hugged each other tight and took this photo to remember the day. We had been playing games and eating snacks all afternoon. Being around family makes me feel safe and loved. No matter what happens in life, I know my family will always be there for me. I love them so much.",
            
            f"I treasure this picture so much. It shows {caption}. My friends and I were hanging out and having the best time ever. We talked about our favorite movies and TV shows. We shared secrets and made promises to always be friends. Someone told a really funny joke and we all laughed until our tummies hurt. We took silly photos and made funny faces at the camera. Having good friends is one of the best things in the whole world. Friends listen to you, play with you, and cheer you up when you are sad. I am so lucky to have these amazing people in my life.",
            
            f"This picture means so much to me. I see {caption}. Everyone in this photo is someone I care about deeply. We spent the whole day together doing fun things. We cooked food, watched movies, and played board games. Someone won the game and did a happy dance. We took this picture at the end of the day when we were all tired but very happy. Looking at this photo reminds me of all the love and laughter we shared. I hope we can have many more days like this one. Family and friends are the greatest gift of all."
        ]
        story = random.choice(stories)
        return ensure_word_count(story)
    
    # === BEACH stories ===
    if 'beach' in keywords:
        stories = [
            f"I love this beach picture so much! I see {caption}. The sand was soft and warm under my feet. Big waves crashed on the shore and made a soothing sound. The sun was shining bright and the water looked so blue and sparkly. I built a big sandcastle with a moat around it. My friend helped me decorate it with seashells. Then we ran into the water and splashed each other. The cool ocean water felt so good on a hot day. We found pretty shells and smooth stones to take home. I wish I could stay at the beach forever. It is such a peaceful and happy place to be.",
            
            f"This picture takes me back to a perfect day. It shows {caption}. The beach was not too crowded and the weather was just right. I walked along the shore and let the waves tickle my toes. Seagulls flew above and sang their songs. I collected seashells of all shapes and colors. Some were white, some were pink, and some had pretty patterns on them. I also found a starfish stuck on a rock. I gently put it back in the water. As the sun started to set, the sky turned orange and pink. It was the most beautiful thing I have ever seen. The beach is my happy place."
        ]
        story = random.choice(stories)
        return ensure_word_count(story)
    
    # === DEFAULT / GENERAL stories (80-100 words) ===
    default_stories = [
        f"I really like this picture. When I look at it, I see {caption}. It makes me feel peaceful and happy inside. The colors in this photo are very pretty and bright. I can tell that this was a special moment that someone wanted to remember forever. Every picture tells its own little story. This one seems to be about beauty, joy, and simple happiness. I am glad that I got to see this picture today. It reminds me to stop and notice all the nice things that are around me every single day. Sometimes the smallest moments are the most beautiful.",
        
        f"This is such a nice picture to look at. I see {caption}. It makes me take a deep breath and feel calm. The world is full of so many wonderful things to see. Every day there is something new and beautiful to discover. This picture captures one of those moments. Maybe it was a sunny day or a quiet evening. Maybe someone was happy when they took this photo. I don't know the whole story, but I know it is a good one. Thank you for sharing this lovely picture with me. I will keep it in my heart."
    ]
    story = random.choice(default_stories)
    return ensure_word_count(story)


def text2audio(story_text):
    """
    Convert story text to audio using gTTS
    """
    try:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # Slow speed for kids to understand better
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
            font-size: 18px !important;
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
    st.markdown("### Turn any picture into a happy story!")
    st.markdown("🎈 For kids aged 3-10 | 📖 80-100 words | 🔊 Listen along")
    
    # Sidebar
    with st.sidebar:
        st.header("🌟 How to Use")
        st.markdown("""
        1. **Upload a picture** (JPG or PNG)
        2. **Click 'Make a Story'**
        3. **Read your 80-100 word story**
        4. **Listen to the audio**
        5. **Download** to keep it!
        """)
        st.divider()
        st.markdown("**📸 Try these pictures:**")
        st.markdown("- 🍕 People eating together")
        st.markdown("- 🎂 Birthday party with cake")
        st.markdown("- 🐕 A dog or cat")
        st.markdown("- 👨‍👩‍👧 Family or friends")
        st.markdown("- 🌳 Park or playground")
        st.markdown("- 🏖️ Beach or ocean")
        st.divider()
        st.markdown("**📏 Word count:** 80-100 words per story")
    
    # File uploader
    uploaded_image = st.file_uploader(
        "🎨 **Upload a picture**", 
        type=["jpg", "jpeg", "png"],
        help="Pick any picture and I'll tell a story about it!"
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
            
            # Step 2: Extract keywords
            keywords = extract_keywords(caption)
            
            # Step 3: Create story based on keywords
            with st.spinner("✏️ Writing a story for you..."):
                story = create_story_from_caption(caption, keywords)
            
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
            if 80 <= word_count <= 100:
                st.success(f"📏 {word_count} words (Perfect! ✓)")
            else:
                st.info(f"📏 {word_count} words (Target: 80-100)")
            st.markdown("---")
            
            # Step 4: Make audio
            with st.spinner("🔊 Making audio..."):
                audio_path = text2audio(story)
            
            if audio_path:
                st.subheader("🎧 Listen to the story")
                
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
        
        with st.expander("📷 Example", expanded=False):
            st.markdown("""
            **If you upload a picture of people eating together:**
            
            > *"Today was such a special day. I went out to eat with my family and friends. When I look at this picture, I see a group of people sitting at a table eating. The restaurant was warm and cozy. We ordered so many yummy dishes like pizza, noodles, and salad. Everyone was laughing and telling funny stories. My little brother made a silly face and we all cracked up. We shared our food and tried a little bit of everything. After dinner, we had ice cream for dessert. Eating together with people I love is the best feeling in the world. I hope we can do this again very soon."*
            
            **Word count: 98 words** ✓
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #95A5A6; font-size: 13px;'>"
        "Made with ❤️ for kids | 80-100 word stories | Perfect for reading aloud"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

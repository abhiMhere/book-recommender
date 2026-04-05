import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from groq import Groq


# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="BookAI — Smart Book Recommender",
    page_icon="📚",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stApp { background-color: #0f0f1a; }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .hero-sub {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .book-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border: 1px solid #333;
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        transition: transform 0.2s;
        height: 100%;
    }
    .book-card:hover { transform: translateY(-4px); }
    .book-title {
        color: #fff;
        font-weight: 700;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        line-height: 1.3;
    }
    .book-author {
        color: #a78bfa;
        font-size: 0.75rem;
        margin-top: 0.2rem;
    }
    .stat-box {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-num {
        font-size: 2rem;
        font-weight: 800;
        color: #667eea;
    }
    .stat-label {
        color: #888;
        font-size: 0.85rem;
    }
    .section-title {
        color: #fff;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #667eea;
        padding-left: 0.75rem;
    }
    div[data-testid="stSelectbox"] label {
        color: #ccc !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ─────────────────────────────────────────────
@st.cache_resource
def load_data():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    books = pd.read_csv('books_small.csv', low_memory=False)
    books['Book-Title'] = books['Book-Title'].str.lower()
    books['Book-Author'] = books['Book-Author'].str.lower()
    books['Publisher'] = books['Publisher'].str.lower()
    books['tags'] = (books['Book-Title'] + ' ' +
                    books['Book-Author'] + ' ' +
                    books['Publisher'])
    books = books.reset_index(drop=True)
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books['tags'])
    similarity = cosine_similarity(tfidf_matrix)
    return books, similarity
    
    # Gemini Setup
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

def ask_chatbot(user_message):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
           model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """Tu ek expert book recommendation chatbot hai.
                    Tera naam BookBot hai.
                    Hindi aur English dono mein jawab de.
                    Short aur helpful jawab do.
                    Agar user koi genre maange toh 3-5 books 
                    suggest karo with author name."""
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Chatbot error: {str(e)}"

# ─── Helper Functions ──────────────────────────────────────
def recommend_books(book_name, n=10):
    book_name = book_name.lower()
    if book_name not in books['Book-Title'].values:
        return []
    idx = books[books['Book-Title'] == book_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:n+1]
    results = []
    for index, score in scores:
        results.append({
            'title': books.iloc[index]['Book-Title'].title(),
            'author': books.iloc[index]['Book-Author'].title(),
            'image': books.iloc[index]['Image-URL-M'],
            'year': books.iloc[index]['Year-Of-Publication'],
            'score': round(score * 100, 1)
        })
    return results

def get_book_image(url):
    try:
        return url
    except:
        return "https://via.placeholder.com/120x180?text=No+Cover"

# ─── Hero Section ──────────────────────────────────────────
st.markdown('<div class="hero-title">📚 BookAI</div>', 
            unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-Powered Smart Book Recommendation System</div>', 
            unsafe_allow_html=True)

# ─── Stats ─────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('''<div class="stat-box">
        <div class="stat-num">5000+</div>
        <div class="stat-label">Books</div>
    </div>''', unsafe_allow_html=True)
with col2:
    st.markdown('''<div class="stat-box">
        <div class="stat-num">ML</div>
        <div class="stat-label">Content Filter</div>
    </div>''', unsafe_allow_html=True)
with col3:
    st.markdown('''<div class="stat-box">
        <div class="stat-num">DL</div>
        <div class="stat-label">Neural Network</div>
    </div>''', unsafe_allow_html=True)
with col4:
    st.markdown('''<div class="stat-box">
        <div class="stat-num">AI</div>
        <div class="stat-label">Smart Chatbot</div>
    </div>''', unsafe_allow_html=True)

st.markdown("---")

# ─── Search Section ────────────────────────────────────────
st.markdown('<div class="section-title">🔍 Book Search</div>', 
            unsafe_allow_html=True)

book_list = sorted(books['Book-Title'].str.title().tolist())

user_input = st.text_input(
    "📖 Book ka naam likho:",
    placeholder="Jaise: The Da Vinci Code, Harry Potter..."
)

selected_book = "-- Book chuniye --"

if user_input:
    # Pehle local database mein dhundho
    local_matches = [
        books.iloc[i]['Book-Title'].title()
        for i in range(len(books))
        if user_input.lower() in books.iloc[i]['Book-Title'].lower()
        or user_input.lower() in books.iloc[i]['Book-Author'].lower()
    ]
    
    if local_matches:
        selected_book = st.selectbox(
            "Yeh books mili — sahi wali chuniye:",
            local_matches
        )
    else:
        # Google Books se dhundho
        st.info("🌐 Google Books se search ho raha hai...")
        google_results = search_google_books(user_input)
        
        if google_results:
            st.markdown(
                '<div class="section-title">🌐 Google Books Results</div>',
                unsafe_allow_html=True
            )
            cols = st.columns(5)
            for i, book in enumerate(google_results[:5]):
                with cols[i]:
                    st.markdown(f'''
                    <div class="book-card">
                        <img src="{book["image"]}" 
                             width="100"
                             style="border-radius:8px; 
                                    height:140px; 
                                    object-fit:cover;">
                        <div class="book-title">{book["title"]}</div>
                        <div class="book-author">{book["author"]}</div>
                        <div style="color:#a78bfa; 
                                    font-size:0.75rem;
                                    margin-top:0.3rem;">
                            {book["year"]}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)

            if len(google_results) > 5:
                cols2 = st.columns(5)
                for i, book in enumerate(google_results[5:10]):
                    with cols2[i]:
                        st.markdown(f'''
                        <div class="book-card">
                            <img src="{book["image"]}"
                                 width="100"
                                 style="border-radius:8px;
                                        height:140px;
                                        object-fit:cover;">
                            <div class="book-title">{book["title"]}</div>
                            <div class="book-author">{book["author"]}</div>
                            <div style="color:#a78bfa;
                                        font-size:0.75rem;
                                        margin-top:0.3rem;">
                                {book["year"]}
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
        else:
            st.error("Koi book nahi mili! Alag naam try karo.")

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
with col_btn1:
    search_btn = st.button("🔍 Recommend Karo", 
                           use_container_width=True,
                           type="primary")
with col_btn2:
    lucky_btn = st.button("🎲 Random Book", 
                          use_container_width=True)

# Random book
if lucky_btn:
    selected_book = np.random.choice(book_list)
    st.info(f"Random book chuni: **{selected_book}**")

# ─── Recommendations ───────────────────────────────────────
if search_btn and selected_book != "-- Book chuniye --":
    recommendations = recommend_books(selected_book)
    
    if recommendations:
        st.markdown(
            f'<div class="section-title">✨ "{selected_book}" jaisi books</div>',
            unsafe_allow_html=True
        )
        
        cols = st.columns(5)
        for i, book in enumerate(recommendations[:5]):
            with cols[i]:
                st.markdown(f'''
                <div class="book-card">
                    <img src="{get_book_image(book["image"])}" 
                         width="100" 
                         style="border-radius:8px; height:140px; object-fit:cover;">
                    <div class="book-title">{book["title"]}</div>
                    <div class="book-author">{book["author"]}</div>
                    <div style="color:#4ade80; font-size:0.75rem; 
                                margin-top:0.3rem;">
                        {book["score"]}% match
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        cols2 = st.columns(5)
        for i, book in enumerate(recommendations[5:10]):
            with cols2[i]:
                st.markdown(f'''
                <div class="book-card">
                    <img src="{get_book_image(book["image"])}" 
                         width="100" 
                         style="border-radius:8px; height:140px; object-fit:cover;">
                    <div class="book-title">{book["title"]}</div>
                    <div class="book-author">{book["author"]}</div>
                    <div style="color:#4ade80; font-size:0.75rem; 
                                margin-top:0.3rem;">
                        {book["score"]}% match
                    </div>
                </div>
                ''', unsafe_allow_html=True)
    else:
        st.error("Yeh book database mein nahi mili. Dusri book try karo!")

elif search_btn and selected_book == "-- Book chuniye --":
    st.warning("Pehle koi book chuniye!")

# ─── Popular Books Section ─────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">🔥 Popular Books</div>', 
            unsafe_allow_html=True)

popular = books.head(10)
cols3 = st.columns(5)
for i in range(5):
    with cols3[i]:
        row = popular.iloc[i]
        st.markdown(f'''
        <div class="book-card">
            <img src="{row['Image-URL-M']}" 
                 width="100"
                 style="border-radius:8px; height:140px; object-fit:cover;">
            <div class="book-title">{row['Book-Title'].title()}</div>
            <div class="book-author">{row['Book-Author'].title()}</div>
        </div>
        ''', unsafe_allow_html=True)


# ─── AI Chatbot Section ────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">🤖 BookBot — AI Chatbot</div>',
            unsafe_allow_html=True)

st.markdown("""
<div style='background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
            border: 1px solid #444;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;'>
    <p style='color:#a78bfa; margin:0; font-size:0.9rem;'>
        💡 BookBot se poochho — "mujhe mystery books batao", 
        "best thriller novels kaun se hain", 
        "Paulo Coelho ki books suggest karo"
    </p>
</div>
""", unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chat history dikhao
for chat in st.session_state.chat_history:
    if chat['role'] == 'user':
        st.markdown(f"""
        <div style='text-align:right; margin:0.5rem 0;'>
            <span style='background:#667eea;
                        color:white;
                        padding:0.5rem 1rem;
                        border-radius:18px 18px 4px 18px;
                        font-size:0.9rem;
                        display:inline-block;
                        max-width:70%;'>
                {chat['message']}
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='text-align:left; margin:0.5rem 0;'>
            <span style='background:#2a2a3e;
                        color:#e2e8f0;
                        padding:0.5rem 1rem;
                        border-radius:18px 18px 18px 4px;
                        font-size:0.9rem;
                        display:inline-block;
                        max-width:70%;
                        border:1px solid #444;'>
                🤖 {chat['message']}
            </span>
        </div>
        """, unsafe_allow_html=True)

# Input box
col_chat1, col_chat2 = st.columns([5, 1])
with col_chat1:
    user_message = st.text_input(
        "BookBot se poochho:",
        placeholder="Jaise: mujhe romance novels suggest karo...",
        label_visibility="collapsed"
    )
with col_chat2:
    send_btn = st.button("Send 💬", use_container_width=True)

if send_btn and user_message:
    st.session_state.chat_history.append({
        'role': 'user',
        'message': user_message
    })
    
    with st.spinner("BookBot soch raha hai..."):
        bot_response = ask_chatbot(user_message)
    
    st.session_state.chat_history.append({
        'role': 'bot',
        'message': bot_response
    })
    
    st.rerun()

if st.button("🗑️ Chat Clear Karo"):
    st.session_state.chat_history = []
    st.rerun()

# ─── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#555; font-size:0.85rem; padding:1rem'>
    📚 BookAI — Built with ML + DL + AI | MCA Project 2025
</div>
""", unsafe_allow_html=True)
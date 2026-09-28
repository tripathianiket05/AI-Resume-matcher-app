import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from heapq import nlargest
import string

# Download necessary NLTK data quietly
for corpus in ['punkt', 'stopwords', 'punkt_tab']:
    try:
        if 'punkt' in corpus:
            nltk.data.find(f'tokenizers/{corpus}')
        else:
            nltk.data.find(f'corpora/{corpus}')
    except LookupError:
        nltk.download(corpus, quiet=True)

# A comprehensive list of skills for extraction (can be expanded)
SKILLS_DB = [
    'python', 'java', 'c++', 'c', 'c#', 'javascript', 'html', 'css', 'sql', 
    'nosql', 'mongodb', 'mysql', 'postgresql', 'react', 'angular', 'vue', 
    'django', 'flask', 'spring', 'nodejs', 'express', 'machine learning', 
    'deep learning', 'nlp', 'computer vision', 'tensorflow', 'keras', 
    'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
    'gitlab', 'agile', 'scrum', 'data analysis', 'data science', 'statistics',
    'excel', 'tableau', 'power bi', 'linux', 'bash', 'shell scripting',
    'communication', 'leadership', 'problem solving', 'teamwork'
]

def extract_contact_info(text):
    email = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    phone = re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    
    return {
        "email": email.group(0) if email else "Not found",
        "phone": phone.group(0) if phone else "Not found"
    }

def extract_skills(text):
    text = text.lower()
    found_skills = set()
    for skill in SKILLS_DB:
        # Using word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found_skills.add(skill.title())
    return sorted(list(found_skills))

def calculate_similarity(resume_text, jd_text):
    if not resume_text or not jd_text:
        return 0.0
    
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        vectors = vectorizer.fit_transform([resume_text, jd_text])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return round(similarity * 100, 2)
    except:
        return 0.0

def extract_keywords(text):
    # Simple keyword extraction using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
    try:
        vectorizer.fit([text])
        return list(vectorizer.get_feature_names_out())
    except:
        return []

def get_missing_keywords(resume_text, jd_text):
    jd_keywords = set(extract_keywords(jd_text))
    resume_keywords = set(extract_keywords(resume_text))
    missing = jd_keywords - resume_keywords
    return list(missing)

def generate_summary(text, num_sentences=3):
    # Extractive summarization using word frequencies
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    
    word_frequencies = {}
    for word in words:
        if word not in stop_words and word not in string.punctuation:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
                
    if not word_frequencies:
        return "Not enough text to summarize."
        
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)
        
    sentence_list = sent_tokenize(text)
    sentence_scores = {}
    for sent in sentence_list:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30: # Ignore very long sentences
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
                        
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary if summary else "Not enough meaningful text to summarize."

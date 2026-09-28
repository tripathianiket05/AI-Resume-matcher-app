import streamlit as st
import pdfplumber
import docx2txt
from nlp_utils import (
    extract_contact_info, 
    extract_skills, 
    calculate_similarity, 
    get_missing_keywords,
    generate_summary
)

def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(uploaded_file)
    else:
        return uploaded_file.getvalue().decode("utf-8")

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.title("📄 AI Resume Analyzer")
st.markdown("Analyze resumes, extract skills, and match them against job descriptions using Traditional NLP.")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload Resume")
    uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

with col2:
    st.header("2. Job Description (Optional)")
    job_description = st.text_area("Paste the job description here to check compatibility", height=150)

if uploaded_file is not None:
    with st.spinner('Extracting and analyzing text...'):
        resume_text = extract_text(uploaded_file)
        
        st.divider()
        
        # Section 1: Candidate Overview
        st.subheader("👤 Candidate Overview")
        contact_info = extract_contact_info(resume_text)
        st.write(f"**Email:** {contact_info['email']}")
        st.write(f"**Phone:** {contact_info['phone']}")
        
        st.markdown("### 📝 Brief Summary")
        summary = generate_summary(resume_text)
        st.info(summary)
        
        # Section 2: Skills Extraction
        st.subheader("🛠️ Extracted Skills")
        skills = extract_skills(resume_text)
        if skills:
            st.write(", ".join(skills))
        else:
            st.write("No specific skills identified from our predefined list.")
            
        # Section 3: Job Description Matching
        if job_description.strip():
            st.divider()
            st.subheader("📊 Job Description Match")
            
            match_score = calculate_similarity(resume_text, job_description)
            
            # Display metric
            st.metric(label="Match Score", value=f"{match_score}%")
            
            # Missing keywords
            missing_keywords = get_missing_keywords(resume_text, job_description)
            if missing_keywords:
                st.warning(f"**Missing Keywords / Areas for Improvement:** {', '.join(missing_keywords)}")
            else:
                st.success("Great! Your resume contains most of the key terms from the job description.")
                
            # Progress bar based on match
            st.progress(int(match_score))
            
            if match_score >= 80:
                st.success("Excellent match! You are highly recommended for this role.")
            elif match_score >= 50:
                st.info("Good match, but consider adding some of the missing keywords to improve your chances.")
            else:
                st.error("Low match. Your resume may need significant tailoring for this specific role.")

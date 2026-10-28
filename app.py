
import streamlit as st
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Analyzer Agent", page_icon="📄")

st.title("📄 Resume Analyzer Agent")
st.write("Analyze your resume against a job description.")

resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the job description here")

def extract_text(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def find_skills(text):
    skills = [
        "Python", "Java", "SQL", "Excel", "Power BI",
        "Machine Learning", "Data Analysis", "Statistics",
        "Communication", "HTML", "CSS", "JavaScript",
        "MongoDB", "Flask", "Git", "Pandas"
    ]
    text = text.lower()
    return [skill for skill in skills if skill.lower() in text]

if st.button("Analyze Resume"):
    if not resume_file or not job_description.strip():
        st.warning("Please upload your resume and enter a job description.")
    else:
        try:
            resume_text = extract_text(resume_file)
            if not resume_text.strip():
                st.error("No readable text found. Please upload a text-based PDF.")
            else:
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform(
                    [resume_text, job_description]
                )
                score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

                resume_skills = find_skills(resume_text)
                required_skills = find_skills(job_description)
                missing = [
                    skill for skill in required_skills
                    if skill not in resume_skills
                ]

                st.subheader("Analysis Results")
                st.metric("Estimated Text Match", f"{score:.1f}%")
                st.caption("This is a text-similarity estimate, not a hiring probability.")

                st.write("### Skills Found in Your Resume")
                st.write(", ".join(resume_skills) or "No listed skills detected.")

                st.write("### Skills to Consider Adding")
                st.write(", ".join(missing) or "No missing skills detected from our skill list.")

                st.write("### Resume Improvement Tips")
                st.write("- Highlight relevant projects and achievements.")
                st.write("- Add measurable results where accurate.")
                st.write("- Include relevant skills you genuinely possess.")

                st.write("### Interview Practice Questions")
                for skill in required_skills[:5]:
                    st.write(f"- Explain your experience with {skill}.")
                st.write("- Tell me about a project relevant to this role.")
        except Exception as e:
            st.error(f"Unable to analyze the resume: {e}")

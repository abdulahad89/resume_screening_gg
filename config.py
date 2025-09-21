import os
import streamlit as st
from typing import List, Dict, Any

# API Configuration - Google AI Only
def get_google_ai_token():
    """Get Google AI Pro API key"""
    token = None
    try:
        token = st.secrets["GOOGLE_AI_API_KEY"]
        if token and len(token) > 10:
            return token
    except Exception:
        pass
    
    token = os.getenv("GOOGLE_AI_API_KEY", "")
    if token and len(token) > 10:
        return token
    
    return ""

GOOGLE_AI_API_KEY = get_google_ai_token()

# ✅ GOOGLE AI MODELS - Gemini Family Only
GOOGLE_AI_MODELS = {
    "gemini_pro": "gemini-1.5-pro-latest",               # Best quality, most capable
    "gemini_flash": "gemini-1.5-flash-latest",           # Faster, still excellent  
    "gemini_pro_002": "gemini-1.5-pro-002",             # Stable version
    "flash_20_lite": "gemini-2.0-flash-lite",
    "flash_20": "gemini-2.0-flash",
    "flash_25_lite": "gemini-2.5-flash-lite"
}

# Model selection - All Google AI
MAIN_MODEL = GOOGLE_AI_MODELS["gemini_pro"]            # Main model for all tasks
SCORING_MODEL = GOOGLE_AI_MODELS["gemini_pro"]         # For relevance scoring
CHATBOT_MODEL = GOOGLE_AI_MODELS["gemini_pro"]         # For chatbot
DATA_EXTRACTION_MODEL = GOOGLE_AI_MODELS["gemini_pro"] # For data extraction

# File processing settings
UPLOAD_FOLDER = "./data/uploads"
CACHE_PATH = "./data/cache"
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB

# Scoring configuration - Simplified for Google AI only
SCORING_WEIGHTS = {
    "relevance_score": 0.70,     # Primary: Gemini relevance scoring
    "experience_match": 0.20,    # Experience alignment 
    "skills_match": 0.10         # Skills extraction and matching
}

# Google AI settings - optimized for different tasks
GEMINI_SETTINGS = {
    "data_extraction": {
        "temperature": 0.1,      # Very focused for data extraction
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 2000,  # Longer for structured data
        "timeout": 45
    },
    "relevance_scoring": {
        "temperature": 0.2,      # Slightly more creative for analysis
        "top_p": 0.9,
        "top_k": 50,
        "max_output_tokens": 500,   # Medium length for scoring
        "timeout": 30
    },
    "chatbot": {
        "temperature": 0.3,      # More creative for conversations
        "top_p": 0.95,
        "top_k": 60,
        "max_output_tokens": 400,   # Good length for chat
        "timeout": 25,
        "system_instruction": """You are an expert HR consultant and resume screening specialist. 
        Provide helpful, accurate, and professional advice about candidates, hiring, and recruitment. 
        Be specific and actionable in your responses. Base your analysis on the resume data provided."""
    }
}

# Thresholds
MIN_SCORE_THRESHOLD = 0.3
TOP_CANDIDATES = 15

# Job templates optimized for Gemini analysis
JOB_TEMPLATES = {
    "Data Scientist": """
    Data Scientist position requiring:
    - Strong Python programming and SQL skills
    - Machine learning experience (TensorFlow, PyTorch, scikit-learn)
    - Statistical analysis and data visualization capabilities  
    - 3+ years experience in data science or analytics
    - Master's degree in relevant field preferred
    - Experience with cloud platforms (AWS, Azure, GCP)
    - Strong problem-solving and communication skills
    """,
    
    "Software Engineer": """
    Software Engineer role requiring:
    - Proficiency in programming languages (Python, JavaScript, Java, or similar)
    - Web development experience with modern frameworks
    - Database knowledge and API development experience
    - 2+ years software development experience
    - Bachelor's degree in Computer Science or related field
    - Experience with version control (Git) and agile methodologies
    - Strong analytical and problem-solving abilities
    """,
    
    "Marketing Manager": """
    Marketing Manager position requiring:
    - 5+ years marketing experience with team leadership
    - Digital marketing expertise (SEO, SEM, social media, email)
    - Analytics tools experience (Google Analytics, marketing automation)
    - Campaign management and budget oversight experience
    - Bachelor's degree in Marketing, Business, or related field
    - Excellent communication and project management skills
    - Experience with CRM systems and marketing technologies
    """,
    
    "Product Manager": """
    Product Manager role requiring:
    - 3+ years product management or related experience
    - Experience with product lifecycle management
    - Strong analytical skills and data-driven decision making
    - Agile/Scrum methodology experience
    - Technical background or ability to work with engineering teams
    - Market research and competitive analysis skills
    - Excellent communication and stakeholder management abilities
    """
}

# Google AI Prompts - Optimized for different tasks
GEMINI_PROMPTS = {
    "data_extraction": """You are an expert resume parser. Extract structured information from this resume text.

**Resume Text:**
{resume_text}

**Instructions:** Extract the following information in JSON format:

```json
{{
    "candidate_name": "Full name of candidate",
    "contact_info": {{
        "email": "email address",
        "phone": "phone number",
        "location": "city, state/country",
        "linkedin": "linkedin profile"
    }},
    "professional_summary": "Brief 2-3 sentence summary of candidate",
    "work_experience": [
        {{
            "position": "Job title",
            "company": "Company name", 
            "duration": "Time period",
            "key_responsibilities": ["responsibility 1", "responsibility 2"],
            "achievements": ["achievement 1", "achievement 2"]
        }}
    ],
    "education": [
        {{
            "degree": "Degree type and field",
            "institution": "School name",
            "graduation_year": "Year or expected year",
            "gpa": "If mentioned"
        }}
    ],
    "technical_skills": ["skill1", "skill2", "skill3"],
    "certifications": ["cert1", "cert2"],
    "total_years_experience": 5,
    "key_achievements": ["Notable accomplishment 1", "Notable accomplishment 2"]
}}
```

Be thorough and accurate. If information is not available, use null or empty array.""",

    "relevance_scoring": """You are an expert HR consultant. Score this candidate's relevance for the job position.

**Job Requirements:**
{job_description}

**Candidate Information:**
{candidate_data}

**Evaluation Criteria:**
1. **Technical Skills Match (30%)** - How well do candidate's technical skills align?
2. **Experience Relevance (35%)** - How relevant is their work experience?
3. **Education & Qualifications (15%)** - Educational background alignment
4. **Career Progression (10%)** - Growth and advancement in career
5. **Cultural & Soft Skills Fit (10%)** - Communication, leadership, teamwork

**Instructions:**
Provide a relevance score from 0-100 and detailed analysis.

**Response Format:**
```json
{{
    "overall_score": 85,
    "category_scores": {{
        "technical_skills": 90,
        "experience_relevance": 80,
        "education": 85,
        "career_progression": 85,
        "soft_skills": 80
    }},
    "strengths": ["Strength 1", "Strength 2", "Strength 3"],
    "concerns": ["Potential concern 1", "Potential concern 2"],
    "recommendation": "Strong Match / Good Match / Moderate Match / Poor Match",
    "key_reasons": "2-3 sentence explanation of the score",
    "interview_focus": ["Topic 1 to explore", "Topic 2 to explore"]
}}
```""",

    "detailed_analysis": """You are a senior HR consultant. Provide a comprehensive analysis of this candidate.

**Job Position:** {job_title}
**Job Requirements:** {job_description}
**Candidate Data:** {candidate_data}

Provide a detailed assessment covering:

**🎯 CANDIDATE OVERVIEW**
- Brief candidate summary
- Years of experience and career level
- Current position and industry background

**✅ KEY STRENGTHS**
- Top 3-4 strengths that align with the role
- Specific examples from their background
- Unique value propositions

**⚠️ AREAS FOR CONSIDERATION**
- Skills gaps or missing requirements
- Experience areas that may need development
- Any concerns about role fit

**📊 COMPETENCY ANALYSIS**
- Technical capabilities assessment
- Leadership and management experience (if applicable)
- Industry knowledge and domain expertise

**💬 INTERVIEW RECOMMENDATIONS**
- Key topics to explore in interviews
- Specific questions to ask
- Areas to probe deeper

**🎯 OVERALL RECOMMENDATION**
- Hire recommendation level
- Best role fit within organization
- Onboarding considerations

Be specific, actionable, and professional in your analysis.""",

    "experience_analysis": """Analyze the work experience from this candidate data.

**Candidate Data:** {candidate_data}

Provide analysis in this format:

**Experience Summary:**
- Total years: X years
- Career level: Junior/Mid/Senior/Executive  
- Industry focus: Primary industries
- Role progression: Growth trajectory

**Key Experience Highlights:**
- Most relevant positions for the target role
- Major achievements and impact
- Skills developed through experience
- Leadership and management experience

**Experience Quality Score: X/10**
**Relevance to Target Role: High/Medium/Low**

Keep analysis concise but insightful."""
}

# Directory creation
def create_directories():
    """Create necessary directories"""
    directories = [
        UPLOAD_FOLDER,
        CACHE_PATH,
        "./data/temp_uploads",
        "./logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Model information
def get_model_info():
    """Get model information"""
    return {
        "deployment_type": "🤖 Google AI Only",
        "main_model": {
            "name": "Gemini 1.5 Flash",
            "full_name": MAIN_MODEL,
            "provider": "Google AI Pro",
            "type": "Large Language Model",
            "status": "✅ Google AI Ready" if GOOGLE_AI_API_KEY else "❌ No API Key"
        },
        "capabilities": {
            "data_extraction": "Gemini 1.5 Flash",
            "relevance_scoring": "Gemini 1.5 Flash", 
            "detailed_analysis": "Gemini 1.5 Flash",
            "chatbot": "Gemini 1.5 Flash"
        },
        "advantages": [
            "Single API provider - simplified architecture",
            "Consistent quality across all tasks",
            "Advanced reasoning capabilities",
            "Cost-effective solution",
            "No embedding complexity"
        ]
    }

# Validation
def validate_setup():
    """Validate Google AI setup"""
    issues = []
    
    if not GOOGLE_AI_API_KEY:
        issues.append("❌ Missing Google AI Pro API key")
        issues.append("🔧 Add GOOGLE_AI_API_KEY to .streamlit/secrets.toml")
    elif len(GOOGLE_AI_API_KEY) < 20:
        issues.append("❌ Invalid Google AI API key format")
    else:
        issues.append("✅ Google AI Pro configured")
        issues.append("✅ Using Gemini 1.5 Flash for all tasks")
        issues.append("✅ Simplified single-API architecture")
        issues.append("🎉 Ready for resume screening!")
    
    return issues

# Utility functions
def get_gemini_prompt(template_key: str, **kwargs) -> str:
    """Get formatted Gemini prompt"""
    template = GEMINI_PROMPTS.get(template_key, "")
    return template.format(**kwargs)

# API Health Check
def check_api_health():
    """Check Google AI API health"""
    health = {
        "google_ai": {
            "available": bool(GOOGLE_AI_API_KEY),
            "models": {
                "data_extraction": DATA_EXTRACTION_MODEL,
                "scoring": SCORING_MODEL,
                "chatbot": CHATBOT_MODEL
            },
            "purpose": "All resume screening tasks"
        },
        "overall_status": "ready" if GOOGLE_AI_API_KEY else "incomplete",
        "architecture": "Google AI Only - Simplified"
    }
    
    return health

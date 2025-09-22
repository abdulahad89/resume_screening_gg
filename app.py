import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

# Configure page
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'screening_engine' not in st.session_state:
    st.session_state.screening_engine = None
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'chatbot_history' not in st.session_state:
    st.session_state.chatbot_history = []

def main():
    """Main application function"""
    
    # Header
    st.title("🤖 AI Resume Screening System")
    #st.markdown("### **Google AI Only:** Gemini 1.5 Flash for All Tasks")
    
    # Initialize screening engine with error handling
    if st.session_state.screening_engine is None:
        with st.spinner("🚀 Initializing AI system..."):
            try:
                from models.google_screening_engine import GoogleAIScreeningEngine
                st.session_state.screening_engine = GoogleAIScreeningEngine()
                st.success("✅ AI system initialized!")
            except Exception as e:
                st.error(f"❌ System initialization failed: {e}")
                st.info("💡 Check your API key in secrets.toml")
                return
    
    # Sidebar - System Status
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Resume Screening", 
        "📊 Results & Analysis", 
        "💬 AI Chatbot", 
        "⚙️ System Status"
    ])
    
    with tab1:
        render_resume_screening()
    
    with tab2:
        render_results_analysis()
    
    with tab3:
        render_ai_chatbot()
    
    with tab4:
        render_system_status()

def render_sidebar():
    """Render sidebar with system info"""
    st.sidebar.title("🤖 AI System")
    
    # Quick status check
    if st.session_state.screening_engine:
        try:
            system_status = st.session_state.screening_engine.get_system_status()
            
            if system_status.get('ready_for_screening', False):
                st.sidebar.success("✅ AI Ready")
            else:
                st.sidebar.error("❌ AI Issues")
                
                # Show what's broken
                google_status = system_status.get('google_ai_status', {})
                conn_status = google_status.get('connection_status', {})
                
                if not conn_status.get('connected', False):
                    st.sidebar.write("❌ Connection failed")
                    error = conn_status.get('error', 'Unknown error')
                    st.sidebar.write(f"Error: {error}")
            
            # Model info
            st.sidebar.info("""
            
            **✅ Data Extraction**
            **✅ Relevance Scoring** 
            **✅ Detailed Analysis**
            **✅ Chatbot**
            """)
            
            # Stats
            stats = system_status.get('processing_stats', {})
            if stats.get('total_resumes_processed', 0) > 0:
                st.sidebar.metric("Resumes Processed", stats['total_resumes_processed'])
                st.sidebar.metric("Model Calls", stats['google_ai_calls'])
                
        except Exception as e:
            st.sidebar.error("❌ System Error")
            st.sidebar.write(f"Error: {e}")
    else:
        st.sidebar.warning("⚠️ System not initialized")
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Quick Actions**")
    
    if st.sidebar.button("🔄 Refresh System"):
        st.session_state.screening_engine = None
        st.rerun()
    
    if st.sidebar.button("🧹 Clear Cache"):
        if st.session_state.screening_engine:
            try:
                st.session_state.screening_engine.clear_cache()
                st.sidebar.success("✅ Cache cleared")
            except Exception as e:
                st.sidebar.error(f"❌ Cache clear failed: {e}")

def render_resume_screening():
    """Main resume screening interface"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Upload Resumes")
        
        # Check system status first
        if not st.session_state.screening_engine:
            st.error("❌ System not initialized. Please refresh.")
            return
        
        try:
            system_status = st.session_state.screening_engine.get_system_status()
            
            if not system_status.get('ready_for_screening', False):
                st.warning("⚠️ System not ready. Check API key in System Status.")
                return
                
        except Exception as e:
            st.error(f"❌ System check failed: {e}")
            return
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Select resume files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'doc', 'txt'],
            help="Upload PDF, Word documents, or text files"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            # Show file details
            with st.expander("📁 File Details", expanded=False):
                for file in uploaded_files:
                    st.write(f"• {file.name} ({file.size/1024:.1f} KB)")
    
    with col2:
        st.header("💼 Job Description")
        
        # Job templates
        from config import JOB_TEMPLATES
        
        template_choice = st.selectbox("Choose template:", ["Custom"] + list(JOB_TEMPLATES.keys()))
        
        if template_choice != "Custom":
            job_description = st.text_area(
                "Job Description",
                value=JOB_TEMPLATES[template_choice],
                height=200,
                help="Edit the template or write your own"
            )
        else:
            job_description = st.text_area(
                "Job Description",
                height=200,
                placeholder="Paste job description here...",
                help="Describe the role, required skills, and qualifications"
            )
    
    # Processing section
    st.markdown("---")
    
    if uploaded_files and job_description:
        # col1, col2, col3 = st.columns([1, 1, 1])
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🚀 Analyze with AI", type="primary", use_container_width=True):
                process_resumes(uploaded_files, job_description)
        
        with col2:
            if st.button("📊 Quick Preview", use_container_width=True):
                show_quick_preview(uploaded_files, job_description)
        
        # with col3:
        #     if st.button("💰 Cost Estimate", use_container_width=True):
        #         show_cost_estimate(len(uploaded_files))
    
    elif uploaded_files and not job_description:
        st.warning("⚠️ Please provide a job description to analyze resumes")
    elif not uploaded_files:
        st.info("💡 Upload resume files to get started")

def process_resumes(uploaded_files, job_description):
    """Process uploaded resumes with Google AI"""
    
    st.markdown("### 🤖 Processing with AI...")
    
    start_time = datetime.now()
    
    # Process with Google AI engine
    results = st.session_state.screening_engine.process_multiple_resumes(
        uploaded_files, job_description
    )
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Store results
    st.session_state.last_results = results
    
    # Show summary
    summary = results.get('summary', {})
    
    st.success(f"✅ AI processing completed in {processing_time:.1f} seconds!")
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Processed", summary.get('total_processed', 0))
    
    with col2:
        st.metric("Successful", summary.get('successful', 0))
    
    with col3:
        st.metric("Average Score", f"{summary.get('avg_composite_score', 0):.3f}")
    
    with col4:
        google_ai_calls = summary.get('google_ai_calls', 0)
        st.metric("API Calls", google_ai_calls)
    
    # Show top candidates
    if results.get('results'):
        st.markdown("### 🏆 Top Candidates ")
        
        top_candidates = results['results'][:5]  # Top 5
        
        for i, candidate in enumerate(top_candidates):
            with st.expander(f"#{i+1} {candidate['filename']} - Score: {candidate['scores']['composite_score']:.3f}"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Scores**")
                    scores = candidate['scores']
                    st.write(f"• Relevance : {scores['relevance_score']:.3f}")
                    st.write(f"• Experience: {scores['experience_score']:.3f}")
                    st.write(f"• Skills Match: {scores['skills_score']:.3f}")
                
                with col2:
                    st.markdown("**🤖 Insights**")
                    analysis = candidate.get('analysis', {})
                    
                    # Show candidate summary
                    summary = analysis.get('candidate_summary', 'N/A')
                    st.write(f"**Summary:** {summary}")
                    
                    # Show key highlights
                    highlights = analysis.get('key_highlights', [])
                    if highlights:
                        for highlight in highlights[:3]:
                            st.write(f"• {highlight}")
                
                # Show detailed analysis if available
                detailed = analysis.get('detailed_analysis', '')
                if detailed:
                    st.markdown("**🎯 Detailed Gemini Analysis:**")
                    st.write(detailed[:300] + "..." if len(detailed) > 300 else detailed)
        
        # Export option
        st.markdown("---")
        if st.button("📥 Export Results"):
            csv_path = st.session_state.screening_engine.export_results_to_csv()
            if csv_path:
                st.success(f"✅ Results exported to {csv_path}")
                
                # Offer download
                with open(csv_path, 'rb') as f:
                    st.download_button(
                        "⬇️ Download CSV",
                        f.read(),
                        file_name=f"google_ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

def show_quick_preview(uploaded_files, job_description):
    """Show quick preview"""
    st.info("🔍 Quick Preview - AI Processing")
    
    # Show processing details
    st.write(f"**Files to process:** {len(uploaded_files)}")
    st.write(f"**Job description length:** {len(job_description)} characters")
    
    # Estimated processing time (Google AI is fast)
    estimated_time = len(uploaded_files) * 6  # ~6 seconds per resume with Gemini
    st.write(f"**Estimated processing time:** {estimated_time} seconds")
    
    # Show processing steps
    st.write("**Processing Steps:**")
    st.write("1. 📄 Parse resume text")
    st.write("2. 🤖 Extract structured data ")
    st.write("3. 🎯 Score relevance with Gemini")
    st.write("4. 📊 Generate detailed analysis with Gemini")
    st.write("5. 💬 Prepare chatbot context")
    
    # Show files
    st.markdown("**Files:**")
    for i, file in enumerate(uploaded_files[:3]):
        st.write(f"• {file.name}")
    
    if len(uploaded_files) > 3:
        st.write(f"• ... and {len(uploaded_files) - 3} more files")

def show_cost_estimate(num_files):
    """Show Google AI cost estimates"""
    if st.session_state.screening_engine:
        cost_info = st.session_state.screening_engine.get_cost_estimates(num_files)
        
        st.info("💰 Google AI Cost Estimate")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Estimated Cost", f"${cost_info['estimated_cost_usd']:.4f}")
            st.metric("Cost per Resume", f"${cost_info['cost_per_resume_usd']:.4f}")
        
        with col2:
            breakdown = cost_info['calls_breakdown']
            st.write("**Gemini API Calls:**")
            st.write(f"• Data extraction: {breakdown['data_extraction']}")
            st.write(f"• Relevance scoring: {breakdown['relevance_scoring']}")
            st.write(f"• Detailed analysis: {breakdown['detailed_analysis']}")
            st.write(f"• **Total calls: {breakdown['total_calls']}**")
        
        st.success("✅ " + cost_info['note'])

def render_results_analysis():
    """Render detailed results analysis"""
    
    if not st.session_state.last_results:
        st.info("📊 No results to display. Please process some resumes first.")
        return
    
    results = st.session_state.last_results
    summary = results.get('summary', {})
    
    # Summary metrics
    st.header("📊 Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Resumes", summary.get('total_processed', 0))
    
    with col2:
        success_rate = (summary.get('successful', 0) / max(summary.get('total_processed', 1), 1) * 100)
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        st.metric("Avg Score", f"{summary.get('avg_composite_score', 0):.3f}")
    
    with col4:
        st.metric("API Calls", summary.get('google_ai_calls', 0))
    
    # Google AI Performance
    if 'google_ai_performance' in summary:
        st.markdown("### 🤖 AI Performance")
        
        perf = summary['google_ai_performance']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Successful Extractions", perf.get('successful_extractions', 0))
            st.metric("Successful Scores", perf.get('successful_scores', 0))
        
        with col2:
            st.metric("Error Rate", f"{perf.get('error_rate', 0)*100:.1f}%")
            st.metric("Avg Processing Time", f"{perf.get('avg_processing_time', 0):.1f}s")
        
        with col3:
            total_calls = perf.get('total_calls', 0)
            st.metric("Total API Calls", total_calls)
            
            # if total_calls > 0:
            #     cost_estimate = total_calls * 0.002  # Rough estimate
            #     st.metric("Est. Cost", f"${cost_estimate:.4f}")
    
    # Score distribution
    if 'score_distribution' in summary:
        st.markdown("### 📈 Score Distribution")
        
        dist = summary['score_distribution']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean", f"{dist.get('mean', 0):.3f}")
            st.metric("Median", f"{dist.get('median', 0):.3f}")
        
        with col2:
            st.metric("Min Score", f"{dist.get('min', 0):.3f}")
            st.metric("Max Score", f"{dist.get('max', 0):.3f}")
        
        with col3:
            st.metric("Std Dev", f"{dist.get('std', 0):.3f}")
            quartiles = dist.get('quartiles', {})
            st.metric("Q1-Q3", f"{quartiles.get('q1', 0):.2f} - {quartiles.get('q3', 0):.2f}")
    
    # Google AI insights
    if 'google_ai_insights' in results:
        st.markdown("### 🧠 AI Quality Insights")
        
        insights = results['google_ai_insights']
        
        if 'google_ai_performance' in insights:
            perf_insights = insights['google_ai_performance']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"""
                **🤖 Performance**
                
                Extraction Success: {perf_insights.get('extraction_success_rate', 0)*100:.1f}%
                
                Analysis Success: {perf_insights.get('analysis_success_rate', 0)*100:.1f}%
                
                Reliability: {perf_insights.get('overall_reliability', 'Unknown')}
                """)
            
            with col2:
                quality = insights.get('quality_indicators', {})
                st.info(f"""
                **✅ Quality Indicators**
                
                Data Extraction: {quality.get('structured_data_extraction', 'Unknown')}
                
                Analysis Coverage: {quality.get('detailed_analysis_coverage', 'Unknown')}
                
                Consistency: {quality.get('scoring_consistency', 'Unknown')}
                """)
    
    # Detailed results table
    st.markdown("### 📋 Detailed Google AI Results")
    
    if results.get('results'):
        # Create results DataFrame
        result_data = []
        for result in results['results']:
            scores = result.get('scores', {})
            analysis = result.get('analysis', {})
            extracted_data = analysis.get('extracted_data', {})
            
            result_data.append({
                'Rank': result.get('rank', ''),
                'Filename': result['filename'],
                'Candidate': extracted_data.get('candidate_name', 'Unknown'),
                'Composite Score': scores.get('composite_score', 0),
                'Relevance': scores.get('relevance_score', 0),
                'Experience': scores.get('experience_score', 0),
                'Skills': scores.get('skills_score', 0),
                'Years Exp': extracted_data.get('total_years_experience', 0),
                # 'Gemini Calls': result.get('google_ai_calls', 0),
                # 'Analysis': 'Yes' if analysis.get('detailed_analysis') else 'No'
            })
        
        df = pd.DataFrame(result_data)
        
        # Display with formatting
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Composite Score": st.column_config.ProgressColumn(
                    "Composite Score",
                    help="Overall candidate score",
                    min_value=0,
                    max_value=1,
                ),
                "Relevance (Gemini)": st.column_config.ProgressColumn(
                    "Relevance (Gemini)",
                    help="Gemini relevance score",
                    min_value=0,
                    max_value=1,
                )
            }
        )

def render_ai_chatbot():
    """Render Google AI chatbot interface"""
    
    st.header("💬 Resume Chatbot")
    # st.markdown("**Powered by Google AI Gemini 1.5 Flash**")
    
    if not st.session_state.screening_engine:
        st.error("❌ System not initialized")
        return
    
    # Check if Google AI is ready
    try:
        system_status = st.session_state.screening_engine.get_system_status()
        google_ready = system_status.get('ready_for_screening', False)
        
        if not google_ready:
            st.warning("⚠️ Model not connected. Check your API key.")
            return
            
    except Exception as e:
        st.error(f"❌ System check failed: {e}")
        return
    
    # Resume context selection
    has_results = bool(st.session_state.last_results)
    
    if has_results:
        results = st.session_state.last_results['results']
        resume_options = ["General Questions"] + [r['filename'] for r in results[:10]]
        selected_resume = st.selectbox("Choose resume context:", resume_options)
        
        if selected_resume != "General Questions":
            # Find selected resume data
            resume_data = next((r for r in results if r['filename'] == selected_resume), None)
            if resume_data:
                st.success(f"✅ Using {selected_resume} as context for chatbot")
                
                # Show brief resume info
                with st.expander("📋 Resume Context for model"):
                    analysis = resume_data.get('analysis', {})
                    extracted_data = analysis.get('extracted_data', {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Candidate:**")
                        st.write(f"• Name: {extracted_data.get('candidate_name', 'Unknown')}")
                        st.write(f"• Experience: {extracted_data.get('total_years_experience', 0)} years")
                    
                    with col2:
                        st.write("**Analysis:**")
                        st.write(f"• Composite Score: {resume_data['scores']['composite_score']:.3f}")
                        st.write(f"• Relevance Score: {resume_data['scores']['relevance_score']:.3f}")
    else:
        selected_resume = "General Questions"
        st.info("💡 Process some resumes to get candidate-specific insights from Gemini")
    
    # Chat history
    if st.session_state.chatbot_history:
        st.markdown("### 📜 Chat History")
        
        for i, exchange in enumerate(st.session_state.chatbot_history[-5:]):  # Show last 5
            with st.expander(f"Chat {i+1}: {exchange['question'][:50]}..."):
                st.write(f"**Q:** {exchange['question']}")
                st.write(f"**Model:** {exchange['answer']}")
                st.caption(f"Time: {exchange.get('time', 'Unknown')}")
    
    # Chat input
    st.markdown("### 💬 Ask AI")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask chatbot about resumes or hiring:",
            placeholder="e.g., What are this candidate's main strengths?",
            key="gemini_chat_input"
        )
    
    with col2:
        ask_button = st.button("💬 Ask chatbot", type="primary")
    
    if ask_button and user_question:
        process_gemini_chat(user_question, selected_resume)
    
    # Suggested questions
    st.markdown("### 💡 Suggested Questions for chatbot")
    
    suggested_questions = get_suggested_questions(selected_resume)
    
    cols = st.columns(2)
    for i, suggestion in enumerate(suggested_questions[:6]):
        col = cols[i % 2]
        with col:
            if st.button(f"💭 {suggestion}", key=f"gemini_suggest_{i}"):
                process_gemini_chat(suggestion, selected_resume)
    
    # Clear history
    if st.button("🧹 Clear Chat History"):
        st.session_state.chatbot_history = []
        st.rerun()

def process_gemini_chat(question, selected_resume):
    """Process chatbot query with Google AI"""
    
    # Get resume context if selected
    resume_context = None
    if selected_resume != "General Questions" and st.session_state.last_results:
        results = st.session_state.last_results['results']
        resume_data = next((r for r in results if r['filename'] == selected_resume), None)
        
        if resume_data:
            # Prepare context from extracted data
            analysis = resume_data.get('analysis', {})
            extracted_data = analysis.get('extracted_data', {})
            
            # Create context string
            context_parts = []
            context_parts.append(f"Candidate: {extracted_data.get('candidate_name', 'Unknown')}")
            context_parts.append(f"Experience: {extracted_data.get('total_years_experience', 0)} years")
            context_parts.append(f"Skills: {', '.join(extracted_data.get('technical_skills', [])[:5])}")
            
            # Add score information
            scores = resume_data.get('scores', {})
            context_parts.append(f"Composite Score: {scores.get('composite_score', 0):.3f}")
            context_parts.append(f"Relevance Score: {scores.get('relevance_score', 0):.3f}")
            
            resume_context = "; ".join(context_parts)
    
    # Get response from Google AI
    try:
        with st.spinner("🤖 Chatbot is thinking..."):
            google_ai = st.session_state.screening_engine.google_ai_manager
            response = google_ai.chat_response(question, resume_context)
        
        if response['success']:
            st.success("✅ Response from chatbot:")
            st.write(response['answer'])
            
            # Add to history
            st.session_state.chatbot_history.append({
                'question': question,
                'answer': response['answer'],
                'model': 'Gemini 1.5 Flash',
                'time': datetime.now().strftime("%H:%M:%S"),
                'resume_context': selected_resume
            })
            
            # Auto-refresh to show in history
            st.rerun()
        else:
            st.error(f"❌ Model chat failed: {response.get('error', 'Unknown error')}")
            
    except Exception as e:
        st.error(f"❌ Chat error: {e}")

def get_suggested_questions(selected_resume):
    """Get context-appropriate suggested questions"""
    
    if selected_resume == "General Questions":
        return [
            "How do I evaluate technical skills effectively?",
            "What are red flags to look for in resumes?",
            "How to assess cultural fit during screening?",
            "What questions should I ask in interviews?",
            "How to improve our hiring process?",
            "What makes a strong candidate profile?"
        ]
    else:
        return [
            "What are this candidate's main strengths?",
            "How does their experience align with our needs?",
            "What questions should I ask them in an interview?",
            "Are there any concerns about this candidate?",
            "How does this candidate compare to others?",
            "What role would be best for this candidate?"
        ]

def render_system_status():
    """Render system status and diagnostics"""
    
    st.header("⚙️ System Status")
    
    if not st.session_state.screening_engine:
        st.error("❌ System not initialized")
        
        if st.button("🔄 Try Initialize"):
            st.session_state.screening_engine = None
            st.rerun()
        return
    
    # Get system status
    try:
        system_status = st.session_state.screening_engine.get_system_status()
        
        # Overall status
        if system_status.get('ready_for_screening', False):
            st.success("✅ AI System Ready for Resume Screening")
        else:
            st.error("❌ AI Issues Detected")
        
        # Google AI Status Details
        st.markdown("### 🤖 AI Status")
        
        google_status = system_status.get('google_ai_status', {})
        conn_status = google_status.get('connection_status', {})
        
        if conn_status.get('connected', False):
            st.success("✅ Connected and Ready")
            
            models = google_status.get('models', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **🤖 Active Models**
                
                Main Model: {models.get('main_model', 'Unknown')}
                
                Scoring: {models.get('scoring_model', 'Unknown')}
                
                Chatbot: {models.get('chatbot_model', 'Unknown')}
                """)
            
            # with col2:
            #     if conn_status.get('models_accessible'):
            #         st.info("**📡 Available Models:**")
            #         for model in conn_status['models_accessible']:
            #             st.write(f"• {model}")
        else:
            st.error("❌ Connection Failed")
            error_msg = conn_status.get('error', 'Unknown error')
            st.error(f"**Error:** {error_msg}")
        
        # Performance Stats
        st.markdown("### 📊 Performance Statistics")
        
        stats = system_status.get('processing_stats', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Resumes Processed", stats.get('total_resumes_processed', 0))
        
        with col2:
            st.metric("Avg Processing Time", f"{stats.get('avg_processing_time', 0):.2f}s")
        
        with col3:
            st.metric("Total model Calls", stats.get('google_ai_calls', 0))
        
        with col4:
            google_ai_calls = stats.get('google_ai_calls', 0)
            google_ai_errors = stats.get('google_ai_errors', 0)
            success_rate = ((google_ai_calls - google_ai_errors) / google_ai_calls * 100) if google_ai_calls > 0 else 100
            st.metric("Model Success Rate", f"{success_rate:.1f}%")
        
        # API Usage Breakdown
        if google_ai_calls > 0:
            st.markdown("### 🔄 API Usage Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Successful Operations:**")
                successful_extractions = stats.get('successful_extractions', 0)
                successful_scores = stats.get('successful_scores', 0)
                st.write(f"• Data Extractions: {successful_extractions}")
                st.write(f"• Relevance Scores: {successful_scores}")
                st.write(f"• Total Successful: {successful_extractions + successful_scores}")
            
            with col2:
                st.write("**Performance Metrics:**")
                if google_ai_errors > 0:
                    st.write(f"• Errors: {google_ai_errors}")
                    st.write(f"• Error Rate: {google_ai_errors/google_ai_calls*100:.1f}%")
                else:
                    st.write("✅ No errors recorded")
        
        # Configuration Check
        st.markdown("### 🔧 Configuration")
        
        try:
            from config import GOOGLE_AI_API_KEY
            
            if GOOGLE_AI_API_KEY:
                st.success("✅  API key configured")
                st.info(f"Key length: {len(GOOGLE_AI_API_KEY)} characters")
            else:
                st.error("❌ API key missing")
                st.info("Add GOOGLE_AI_API_KEY to .streamlit/secrets.toml")
                
        except Exception as e:
            st.error(f"Configuration check failed: {e}")
        
        # System Actions
        st.markdown("### 🛠️ System Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🧪 Run Diagnostics"):
                with st.spinner("Running AI diagnostics..."):
                    diagnostics = st.session_state.screening_engine.run_diagnostics()
                    
                    st.json(diagnostics)
        
        with col2:
            if st.button("🔄 Refresh System"):
                st.session_state.screening_engine = None
                st.rerun()
        
        with col3:
            if st.button("🧹 Clear Cache"):
                try:
                    st.session_state.screening_engine.clear_cache()
                    st.success("✅ Cache cleared")
                except Exception as e:
                    st.error(f"❌ Cache clear failed: {e}")
        
        with col4:
            if st.button("📊 Export Status"):
                status_json = json.dumps(system_status, indent=2, default=str)
                st.download_button(
                    "⬇️ Download JSON",
                    status_json,
                    file_name=f"google_ai_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
    except Exception as e:
        st.error(f"❌ System status check failed: {e}")
        st.write("**Error Details:**", str(e))

if __name__ == "__main__":
    main()

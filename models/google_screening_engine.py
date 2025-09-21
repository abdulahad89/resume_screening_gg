import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
import json
from datetime import datetime
import os

class GoogleAIScreeningEngine:
    """Google AI-only resume screening engine using Gemini for all tasks"""
    
    def __init__(self):
        from config import SCORING_WEIGHTS, MIN_SCORE_THRESHOLD, TOP_CANDIDATES
        
        # Initialize components
        self.scoring_weights = SCORING_WEIGHTS
        self.min_threshold = MIN_SCORE_THRESHOLD
        self.top_candidates = TOP_CANDIDATES
        
        # Initialize Google AI manager and parser
        self.google_ai_manager = None
        self.parser = None
        
        # Results storage
        self.last_analysis_results = None
        self.processing_stats = {
            'total_resumes_processed': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0,
            'google_ai_calls': 0,
            'google_ai_errors': 0,
            'successful_extractions': 0,
            'successful_scores': 0
        }
    
    def _initialize_components(self):
        """Lazy initialization of components"""
        if self.google_ai_manager is None:
            from models.google_ai_complete import GoogleAICompleteManager
            self.google_ai_manager = GoogleAICompleteManager()
        
        if self.parser is None:
            from parser import ResumeParser
            self.parser = ResumeParser()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        self._initialize_components()
        
        google_status = self.google_ai_manager.get_status()
        
        return {
            'deployment_type': '🤖 Google AI Only',
            'google_ai_status': google_status,
            'parser_status': {'available': True, 'type': 'Resume Parser'},
            'processing_stats': self.processing_stats.copy(),
            'ready_for_screening': google_status.get('connection_status', {}).get('connected', False),
            'model_info': self._get_model_summary()
        }
    
    def _get_model_summary(self) -> Dict[str, str]:
        """Get summary of active models"""
        return {
            'data_extraction': 'Gemini 1.5 Flash (Google AI)',
            'relevance_scoring': 'Gemini 1.5 Flash (Google AI)',
            'detailed_analysis': 'Gemini 1.5 Flash (Google AI)',
            'chatbot': 'Gemini 1.5 Flash (Google AI)',
            'advantages': 'Single API provider - simplified, consistent, cost-effective'
        }
    
    def process_single_resume(self, file_path: str, filename: str, job_description: str = "") -> Dict[str, Any]:
        """Process a single resume with Google AI-only analysis"""
        self._initialize_components()
        
        start_time = time.time()
        
        result = {
            'filename': filename,
            'status': 'processing',
            'scores': {},
            'analysis': {},
            'error': None,
            'processing_time': 0,
            'google_ai_calls': 0
        }
        
        try:
            # Step 1: Parse resume text
            with st.spinner(f"📄 Parsing {filename}..."):
                parsed_data = self.parser.parse_resume(file_path, filename)
                
                if not parsed_data.get('success', False):
                    result['status'] = 'failed'
                    result['error'] = parsed_data.get('error', 'Parsing failed')
                    return result
                
                resume_text = parsed_data.get('cleaned_text', '')
                if not resume_text:
                    result['status'] = 'failed'
                    result['error'] = 'No text extracted from resume'
                    return result
            
            # Step 2: Extract structured data using Gemini
            with st.spinner("🤖 Extracting data with Gemini..."):
                extraction_result = self.google_ai_manager.extract_resume_data(resume_text)
                result['google_ai_calls'] += 1
                self.processing_stats['google_ai_calls'] += 1
                
                if not extraction_result['success']:
                    # Fallback to parsed data if extraction fails
                    extracted_data = {
                        'candidate_name': 'Unknown',
                        'total_years_experience': 0,
                        'technical_skills': [],
                        'work_experience': [],
                        'education': [],
                        'professional_summary': resume_text[:200] + '...'
                    }
                    st.warning("⚠️ Using basic extraction due to Gemini error")
                else:
                    extracted_data = extraction_result['data']
                    self.processing_stats['successful_extractions'] += 1
            
            # Step 3: Score relevance using Gemini
            relevance_score = 0.0
            relevance_analysis = {}
            
            if job_description:
                with st.spinner("🎯 Scoring relevance with Gemini..."):
                    scoring_result = self.google_ai_manager.score_relevance(
                        job_description, 
                        {'data': extracted_data}
                    )
                    result['google_ai_calls'] += 1
                    self.processing_stats['google_ai_calls'] += 1
                    
                    if scoring_result['success']:
                        relevance_score = scoring_result['score']
                        relevance_analysis = scoring_result['analysis']
                        self.processing_stats['successful_scores'] += 1
                    else:
                        st.warning("⚠️ Relevance scoring failed, using default score")
                        relevance_score = 0.3  # Default fallback score
            
            # Step 4: Calculate experience and skills scores (simple local calculation)
            experience_score = self._calculate_experience_score(extracted_data)
            skills_score = self._calculate_skills_score(extracted_data, job_description)
            
            # Step 5: Calculate composite score
            composite_score = (
                self.scoring_weights['relevance_score'] * relevance_score +
                self.scoring_weights['experience_match'] * experience_score +
                self.scoring_weights['skills_match'] * skills_score
            )
            
            # Step 6: Generate detailed analysis for high-scoring candidates
            detailed_analysis = ""
            if composite_score > 0.5 and job_description:
                with st.spinner("📊 Generating detailed analysis..."):
                    analysis_result = self.google_ai_manager.generate_detailed_analysis(
                        "Position", job_description, {'data': extracted_data}
                    )
                    result['google_ai_calls'] += 1
                    self.processing_stats['google_ai_calls'] += 1
                    
                    if analysis_result['success']:
                        detailed_analysis = analysis_result['analysis']
            
            # Compile comprehensive results
            result.update({
                'status': 'completed',
                'scores': {
                    'composite_score': round(composite_score, 3),
                    'relevance_score': round(relevance_score, 3),
                    'experience_score': round(experience_score, 3),
                    'skills_score': round(skills_score, 3),
                    'breakdown': {
                        'relevance_weight': self.scoring_weights['relevance_score'],
                        'experience_weight': self.scoring_weights['experience_match'],
                        'skills_weight': self.scoring_weights['skills_match']
                    }
                },
                'analysis': {
                    'extracted_data': extracted_data,
                    'relevance_analysis': relevance_analysis,
                    'detailed_analysis': detailed_analysis,
                    'candidate_summary': self._create_candidate_summary(extracted_data),
                    'key_highlights': self._extract_key_highlights(extracted_data),
                    'processing_method': 'Google AI Gemini (Complete)'
                },
                'processing_time': time.time() - start_time
            })
            
            # Update global stats
            self.processing_stats['total_resumes_processed'] += 1
            self.processing_stats['total_processing_time'] += result['processing_time']
            self.processing_stats['avg_processing_time'] = (
                self.processing_stats['total_processing_time'] / 
                self.processing_stats['total_resumes_processed']
            )
            
        except Exception as e:
            result.update({
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time
            })
            st.error(f"❌ Error processing {filename}: {e}")
            self.processing_stats['google_ai_errors'] += result['google_ai_calls']
        
        return result
    
    def process_multiple_resumes(self, uploaded_files: List, job_description: str = "") -> Dict[str, Any]:
        """Process multiple resumes using Google AI"""
        self._initialize_components()
        
        if not uploaded_files:
            return {'error': 'No files uploaded', 'results': []}
        
        total_start_time = time.time()
        results = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Display system status
        system_status = self.get_system_status()
        with st.expander("🔧 System Status", expanded=False):
            if system_status['ready_for_screening']:
                st.success("✅ Google AI Ready for Processing")
                st.info(f"Using: {system_status['model_info']['data_extraction']}")
            else:
                st.error("❌ Google AI not ready - check API key")
                return {'error': 'System not ready', 'results': []}
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
            
            try:
                # Save uploaded file temporarily
                temp_path = self._save_temp_file(uploaded_file)
                
                # Process resume with Google AI
                result = self.process_single_resume(temp_path, uploaded_file.name, job_description)
                results.append(result)
                
                # Clean up temp file
                os.remove(temp_path)
                
                # Show individual result
                if result['status'] == 'completed':
                    score = result['scores']['composite_score']
                    st.write(f"✅ {uploaded_file.name}: Score {score:.2f}")
                else:
                    st.write(f"❌ {uploaded_file.name}: {result.get('error', 'Failed')}")
                
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {e}")
                results.append({
                    'filename': uploaded_file.name,
                    'status': 'error',
                    'error': str(e),
                    'scores': {},
                    'analysis': {},
                    'google_ai_calls': 0
                })
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Sort and rank results
        successful_results = [r for r in results if r['status'] == 'completed']
        successful_results.sort(
            key=lambda x: x['scores'].get('composite_score', 0), 
            reverse=True
        )
        
        # Add rankings
        for i, result in enumerate(successful_results):
            result['rank'] = i + 1
        
        # Calculate Google AI usage summary
        total_google_ai_calls = sum(r.get('google_ai_calls', 0) for r in results)
        
        # Compile comprehensive summary
        summary = {
            'total_processed': len(results),
            'successful': len(successful_results),
            'failed': len(results) - len(successful_results),
            'avg_composite_score': np.mean([
                r['scores'].get('composite_score', 0) 
                for r in successful_results
            ]) if successful_results else 0,
            'score_distribution': self._calculate_score_distribution(successful_results),
            'top_candidates': successful_results[:self.top_candidates],
            'processing_time': time.time() - total_start_time,
            'google_ai_calls': total_google_ai_calls,
            'system_status': system_status,
            'google_ai_performance': {
                'total_calls': total_google_ai_calls,
                'successful_extractions': self.processing_stats['successful_extractions'],
                'successful_scores': self.processing_stats['successful_scores'],
                'error_rate': self.processing_stats['google_ai_errors'] / max(total_google_ai_calls, 1),
                'avg_processing_time': self.processing_stats['avg_processing_time']
            }
        }
        
        # Store results for export
        self.last_analysis_results = {
            'results': results,
            'successful_results': successful_results,
            'summary': summary,
            'job_description': job_description,
            'timestamp': datetime.now().isoformat(),
            'google_ai_only': True
        }
        
        return {
            'results': successful_results,
            'failed_results': [r for r in results if r['status'] != 'completed'],
            'summary': summary,
            'google_ai_insights': self._generate_google_ai_insights(successful_results)
        }
    
    def _calculate_experience_score(self, extracted_data: Dict[str, Any]) -> float:
        """Calculate experience score based on extracted data"""
        try:
            years_exp = extracted_data.get('total_years_experience', 0)
            
            # Simple scoring based on experience
            if years_exp >= 8:
                return 1.0
            elif years_exp >= 5:
                return 0.8
            elif years_exp >= 2:
                return 0.6
            elif years_exp >= 1:
                return 0.4
            else:
                return 0.2
                
        except:
            return 0.3  # Default score
    
    def _calculate_skills_score(self, extracted_data: Dict[str, Any], job_description: str) -> float:
        """Calculate skills matching score"""
        try:
            skills = extracted_data.get('technical_skills', [])
            if not skills or not job_description:
                return 0.3
            
            job_desc_lower = job_description.lower()
            skills_lower = [skill.lower() for skill in skills]
            
            # Count skill matches in job description
            matches = 0
            for skill in skills_lower:
                if skill in job_desc_lower:
                    matches += 1
            
            # Calculate score based on match percentage
            match_ratio = matches / max(len(skills), 1)
            return min(1.0, match_ratio * 1.5)  # Boost the ratio slightly
            
        except:
            return 0.3  # Default score
    
    def _calculate_score_distribution(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate score distribution statistics"""
        if not results:
            return {}
        
        scores = [r['scores']['composite_score'] for r in results]
        
        return {
            'mean': round(np.mean(scores), 3),
            'median': round(np.median(scores), 3),
            'std': round(np.std(scores), 3),
            'min': round(min(scores), 3),
            'max': round(max(scores), 3),
            'quartiles': {
                'q1': round(np.percentile(scores, 25), 3),
                'q3': round(np.percentile(scores, 75), 3)
            }
        }
    
    def _generate_google_ai_insights(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate insights from Google AI analysis"""
        if not results:
            return {}
        
        insights = {
            'google_ai_performance': {},
            'quality_indicators': {},
            'processing_insights': {}
        }
        
        try:
            # Analyze Google AI performance
            total_results = len(results)
            successful_extractions = sum(
                1 for r in results 
                if r['analysis'].get('extracted_data', {}).get('candidate_name') != 'Unknown'
            )
            
            successful_analyses = sum(
                1 for r in results 
                if r['analysis'].get('detailed_analysis', '')
            )
            
            insights['google_ai_performance'] = {
                'extraction_success_rate': round(successful_extractions / total_results, 3),
                'analysis_success_rate': round(successful_analyses / total_results, 3),
                'overall_reliability': 'High' if successful_extractions / total_results > 0.8 else 'Medium',
                'avg_processing_time': round(np.mean([r.get('processing_time', 0) for r in results]), 2)
            }
            
            # Quality indicators
            insights['quality_indicators'] = {
                'structured_data_extraction': 'Excellent' if successful_extractions / total_results > 0.9 else 'Good',
                'detailed_analysis_coverage': f"{round(successful_analyses / total_results * 100)}%",
                'scoring_consistency': 'High - Single model provides consistent results'
            }
            
        except Exception as e:
            insights['error'] = str(e)
        
        return insights
    
    def _create_candidate_summary(self, extracted_data: Dict[str, Any]) -> str:
        """Create a brief candidate summary"""
        name = extracted_data.get('candidate_name', 'Unknown')
        years = extracted_data.get('total_years_experience', 0)
        skills = extracted_data.get('technical_skills', [])[:5]  # Top 5 skills
        
        summary = f"{name} - {years} years experience"
        if skills:
            summary += f" | Key skills: {', '.join(skills)}"
            
        return summary
    
    def _extract_key_highlights(self, extracted_data: Dict[str, Any]) -> List[str]:
        """Extract key highlights from candidate data"""
        highlights = []
        
        # Experience highlights
        years = extracted_data.get('total_years_experience', 0)
        if years > 0:
            highlights.append(f"{years} years of professional experience")
        
        # Skills highlights
        skills = extracted_data.get('technical_skills', [])
        if len(skills) > 5:
            highlights.append(f"Proficient in {len(skills)} technical skills")
        
        # Education highlights
        education = extracted_data.get('education', [])
        if education:
            highlights.append(f"Education: {education[0].get('degree', 'Degree listed')}")
        
        # Achievements highlights
        achievements = extracted_data.get('key_achievements', [])
        if achievements:
            highlights.append(f"{len(achievements)} notable achievements mentioned")
        
        return highlights[:4]  # Top 4 highlights
    
    def _save_temp_file(self, uploaded_file) -> str:
        """Save uploaded file temporarily"""
        temp_dir = "./data/temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return temp_path
    
    def export_results_to_csv(self) -> Optional[str]:
        """Export Google AI analysis results to CSV"""
        if not self.last_analysis_results:
            return None
        
        try:
            results = self.last_analysis_results['successful_results']
            
            # Prepare enhanced CSV data
            csv_data = []
            for result in results:
                scores = result.get('scores', {})
                analysis = result.get('analysis', {})
                extracted_data = analysis.get('extracted_data', {})
                
                row = {
                    'Rank': result.get('rank', ''),
                    'Filename': result['filename'],
                    'Candidate_Name': extracted_data.get('candidate_name', 'Unknown'),
                    'Composite_Score': scores.get('composite_score', 0),
                    'Relevance_Score': scores.get('relevance_score', 0),
                    'Experience_Score': scores.get('experience_score', 0),
                    'Skills_Score': scores.get('skills_score', 0),
                    'Years_Experience': extracted_data.get('total_years_experience', 0),
                    'Top_Skills': ', '.join(extracted_data.get('technical_skills', [])[:5]),
                    'Education': extracted_data.get('education', [{}])[0].get('degree', 'N/A') if extracted_data.get('education') else 'N/A',
                    'Processing_Time_s': round(result.get('processing_time', 0), 2),
                    'Google_AI_Calls': result.get('google_ai_calls', 0),
                    'Processing_Method': 'Google AI Gemini (Complete)',
                    'Analysis_Available': 'Yes' if analysis.get('detailed_analysis') else 'No'
                }
                csv_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            csv_path = f"./data/google_ai_resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False)
            
            return csv_path
            
        except Exception as e:
            st.error(f"CSV export failed: {e}")
            return None
    
    def get_analysis_summary(self) -> Optional[Dict[str, Any]]:
        """Get enhanced summary of last analysis"""
        if not self.last_analysis_results:
            return None
        
        summary = self.last_analysis_results['summary'].copy()
        summary['google_ai_advantages'] = {
            'simplicity': 'Single API provider - no complexity',
            'consistency': 'Same model quality across all tasks',
            'cost_effectiveness': 'No embedding costs, optimized prompts',
            'reliability': 'Google AI enterprise-grade uptime',
            'intelligence': 'Advanced reasoning for better insights'
        }
        
        return summary
    
    def clear_cache(self):
        """Clear Google AI cache"""
        self._initialize_components()
        
        try:
            self.google_ai_manager.clear_cache()
            st.success("✅ Google AI cache cleared")
        except Exception as e:
            st.error(f"❌ Cache clearing failed: {e}")
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        self._initialize_components()
        
        try:
            diagnostics = self.google_ai_manager.run_diagnostics()
            diagnostics.update({
                'system_type': 'Google AI Only (Complete)',
                'processing_stats': self.processing_stats,
                'system_status': self.get_system_status()
            })
            
            # Overall health score
            health_score = 0
            if diagnostics.get('connection_test', {}).get('connected', False):
                health_score += 50
            
            model_tests = diagnostics.get('model_tests', {})
            working_models = sum(1 for test in model_tests.values() if test.get('status') == 'available')
            health_score += (working_models / len(model_tests)) * 50 if model_tests else 0
            
            diagnostics['overall_health'] = {
                'score': round(health_score),
                'rating': 'Excellent' if health_score >= 90 else 'Good' if health_score >= 70 else 'Needs Attention',
                'google_ai_advantage': 'Simplified architecture with single point of failure'
            }
            
            return diagnostics
            
        except Exception as e:
            return {
                'error': str(e),
                'diagnostics_failed': True,
                'system_type': 'Google AI Only'
            }
    
    def get_cost_estimates(self, num_resumes: int) -> Dict[str, Any]:
        """Estimate Google AI costs for processing resumes"""
        # Approximate Google AI calls per resume
        calls_per_resume = {
            'data_extraction': 1,      # Extract structured data
            'relevance_scoring': 1,    # Score relevance
            'detailed_analysis': 0.7,  # Only for high-scoring candidates
            'total_calls': 2.7
        }
        
        # Estimated costs (Google AI Gemini Flash pricing)
        cost_per_1k_tokens = 0.0001  # Very rough estimate
        avg_tokens_per_call = 1500   # Estimate based on prompt + response
        
        total_calls = calls_per_resume['total_calls'] * num_resumes
        total_tokens = total_calls * avg_tokens_per_call
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
        
        return {
            'num_resumes': num_resumes,
            'calls_breakdown': {
                'data_extraction': int(calls_per_resume['data_extraction'] * num_resumes),
                'relevance_scoring': int(calls_per_resume['relevance_scoring'] * num_resumes),
                'detailed_analysis': int(calls_per_resume['detailed_analysis'] * num_resumes),
                'total_calls': int(total_calls)
            },
            'estimated_cost_usd': round(estimated_cost, 4),
            'cost_per_resume_usd': round(estimated_cost / num_resumes, 4) if num_resumes > 0 else 0,
            'advantages': [
                'Single API provider - no complexity',
                'Consistent quality across tasks',
                'No embedding costs',
                'Enterprise-grade reliability'
            ],
            'note': 'Estimates based on Gemini Flash pricing. Actual costs may vary.'
        }
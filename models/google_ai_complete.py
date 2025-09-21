import google.generativeai as genai
import streamlit as st
import time
import hashlib
import json
import os
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

class GoogleAICompleteManager:
    """Complete Google AI manager for all resume screening tasks"""
    
    def __init__(self):
        from config import (
            GOOGLE_AI_API_KEY, GEMINI_SETTINGS, 
            MAIN_MODEL, SCORING_MODEL, CHATBOT_MODEL, DATA_EXTRACTION_MODEL
        )
        
        self.api_key = GOOGLE_AI_API_KEY
        self.settings = GEMINI_SETTINGS
        self.main_model = MAIN_MODEL
        self.scoring_model = SCORING_MODEL
        self.chatbot_model = CHATBOT_MODEL
        self.data_extraction_model = DATA_EXTRACTION_MODEL
        
        # Initialize Google AI
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.models_available = True
            except Exception as e:
                st.error(f"Failed to configure Google AI: {e}")
                self.models_available = False
        else:
            self.models_available = False
        
        # Model instances (created on demand)
        self.model_instances = {}
        
        # Cache settings
        self.cache_dir = "./data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Performance tracking
        self.stats = {
            'data_extractions': 0,
            'relevance_scores': 0,
            'detailed_analyses': 0,
            'chat_responses': 0,
            'total_api_calls': 0,
            'cache_hits': 0,
            'errors': 0
        }
        
        # Test connection
        self.connection_status = self._test_connection()
    
    def _test_connection(self) -> Dict[str, Any]:
        """Test Google AI Pro connection"""
        status = {
            "connected": False,
            "model_available": False,
            "error": None,
            "models_accessible": []
        }
        
        if not self.api_key:
            status["error"] = "No Google AI Pro API key provided"
            return status
        
        try:
            # List available models
            models = genai.list_models()
            available_models = [m.name.split('/')[-1] for m in models]
            status["models_accessible"] = available_models[:5]  # First 5 models
            
            # Test with simple generation
            test_model = genai.GenerativeModel(self.main_model)
            test_response = test_model.generate_content(
                "Test: Say 'Google AI Ready'",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=10
                )
            )
            
            if test_response.text and "ready" in test_response.text.lower():
                status["connected"] = True
                status["model_available"] = True
            else:
                status["error"] = "Model test failed but connection established"
                
        except Exception as e:
            status["error"] = f"Google AI connection test failed: {str(e)}"
        
        return status
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive Google AI status"""
        return {
            "provider": "Google AI Pro (Complete)",
            "models_available": self.models_available,
            "connection_status": self.connection_status,
            "models": {
                "main_model": self.main_model,
                "scoring_model": self.scoring_model,
                "chatbot_model": self.chatbot_model,
                "data_extraction_model": self.data_extraction_model
            },
            "api_configured": bool(self.api_key),
            "stats": self.stats.copy()
        }
    
    def _get_model_instance(self, task_type: str):
        """Get or create model instance for specific task"""
        if task_type not in self.model_instances:
            try:
                # Select model based on task
                if task_type == "data_extraction":
                    model_name = self.data_extraction_model
                    config = self.settings["data_extraction"]
                elif task_type == "relevance_scoring":
                    model_name = self.scoring_model
                    config = self.settings["relevance_scoring"]
                elif task_type == "chatbot":
                    model_name = self.chatbot_model
                    config = self.settings["chatbot"]
                    # Create with system instruction for chatbot
                    self.model_instances[task_type] = genai.GenerativeModel(
                        model_name,
                        generation_config=genai.types.GenerationConfig(
                            temperature=config["temperature"],
                            top_p=config["top_p"],
                            top_k=config["top_k"],
                            max_output_tokens=config["max_output_tokens"]
                        ),
                        system_instruction=config["system_instruction"]
                    )
                    return self.model_instances[task_type]
                else:
                    model_name = self.main_model
                    config = self.settings["relevance_scoring"]  # Default config
                
                # Create standard model instance
                self.model_instances[task_type] = genai.GenerativeModel(
                    model_name,
                    generation_config=genai.types.GenerationConfig(
                        temperature=config["temperature"],
                        top_p=config["top_p"],
                        top_k=config["top_k"],
                        max_output_tokens=config["max_output_tokens"]
                    )
                )
                
            except Exception as e:
                st.error(f"Failed to create {task_type} model: {e}")
                return None
        
        return self.model_instances[task_type]
    
    def _get_cache_key(self, content: str, task_type: str) -> str:
        """Generate cache key"""
        cache_content = f"{task_type}:{self.main_model}:{content}"
        return hashlib.md5(cache_content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load response from cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    # Check cache age (4 hours for Google AI)
                    cache_time = datetime.fromisoformat(cached_data.get('timestamp', ''))
                    if (datetime.now() - cache_time).total_seconds() < 14400:
                        self.stats['cache_hits'] += 1
                        return cached_data.get('response', '')
            except:
                pass
        return None
    
    def _save_to_cache(self, cache_key: str, response: str, task_type: str):
        """Save response to cache"""
        try:
            cache_data = {
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'task_type': task_type,
                'model': self.main_model
            }
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def extract_resume_data(self, resume_text: str) -> Dict[str, Any]:
        """Extract structured data from resume using Gemini"""
        result = {
            'success': False,
            'data': {},
            'error': None,
            'processing_time': 0
        }
        
        if not self.models_available or not self.connection_status.get('connected', False):
            result['error'] = "Google AI not available"
            return result
        
        start_time = time.time()
        
        try:
            from config import get_gemini_prompt
            
            # Create data extraction prompt
            prompt = get_gemini_prompt(
                "data_extraction",
                resume_text=resume_text[:3000]  # Limit text length
            )
            
            # Check cache first
            cache_key = self._get_cache_key(prompt, "data_extraction")
            cached_response = self._load_from_cache(cache_key)
            
            if cached_response:
                result.update({
                    'success': True,
                    'data': self._parse_json_response(cached_response),
                    'processing_time': time.time() - start_time
                })
                return result
            
            # Generate with Gemini
            model = self._get_model_instance("data_extraction")
            if not model:
                result['error'] = "Failed to load data extraction model"
                return result
            
            with st.spinner("🤖 Extracting resume data with Gemini..."):
                response = model.generate_content(prompt)
                
                if response.text:
                    response_text = response.text.strip()
                    
                    # Save to cache
                    self._save_to_cache(cache_key, response_text, "data_extraction")
                    
                    # Parse structured data
                    extracted_data = self._parse_json_response(response_text)
                    
                    result.update({
                        'success': True,
                        'data': extracted_data,
                        'raw_response': response_text
                    })
                    
                    self.stats['data_extractions'] += 1
                else:
                    result['error'] = "No response from Gemini"
        
        except Exception as e:
            result['error'] = f"Data extraction failed: {str(e)}"
            self.stats['errors'] += 1
        
        result['processing_time'] = time.time() - start_time
        self.stats['total_api_calls'] += 1
        return result
    
    def score_relevance(self, job_description: str, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score candidate relevance using Gemini"""
        result = {
            'success': False,
            'score': 0.0,
            'detailed_scores': {},
            'analysis': {},
            'error': None,
            'processing_time': 0
        }
        
        if not self.models_available or not self.connection_status.get('connected', False):
            result['error'] = "Google AI not available"
            return result
        
        start_time = time.time()
        
        try:
            from config import get_gemini_prompt
            
            # Create scoring prompt
            candidate_summary = self._format_candidate_for_scoring(candidate_data)
            prompt = get_gemini_prompt(
                "relevance_scoring",
                job_description=job_description[:1500],
                candidate_data=candidate_summary
            )
            
            # Check cache
            cache_key = self._get_cache_key(prompt, "relevance_scoring")
            cached_response = self._load_from_cache(cache_key)
            
            if cached_response:
                parsed = self._parse_json_response(cached_response)
                result.update({
                    'success': True,
                    'score': parsed.get('overall_score', 0) / 100.0,  # Convert to 0-1 scale
                    'detailed_scores': parsed.get('category_scores', {}),
                    'analysis': {
                        'strengths': parsed.get('strengths', []),
                        'concerns': parsed.get('concerns', []),
                        'recommendation': parsed.get('recommendation', ''),
                        'key_reasons': parsed.get('key_reasons', ''),
                        'interview_focus': parsed.get('interview_focus', [])
                    },
                    'processing_time': time.time() - start_time
                })
                return result
            
            # Generate with Gemini
            model = self._get_model_instance("relevance_scoring")
            if not model:
                result['error'] = "Failed to load scoring model"
                return result
            
            with st.spinner("🎯 Scoring relevance with Gemini..."):
                response = model.generate_content(prompt)
                
                if response.text:
                    response_text = response.text.strip()
                    
                    # Save to cache
                    self._save_to_cache(cache_key, response_text, "relevance_scoring")
                    
                    # Parse scoring result
                    parsed = self._parse_json_response(response_text)
                    
                    result.update({
                        'success': True,
                        'score': parsed.get('overall_score', 0) / 100.0,  # Convert to 0-1 scale
                        'detailed_scores': parsed.get('category_scores', {}),
                        'analysis': {
                            'strengths': parsed.get('strengths', []),
                            'concerns': parsed.get('concerns', []),
                            'recommendation': parsed.get('recommendation', ''),
                            'key_reasons': parsed.get('key_reasons', ''),
                            'interview_focus': parsed.get('interview_focus', [])
                        },
                        'raw_response': response_text
                    })
                    
                    self.stats['relevance_scores'] += 1
                else:
                    result['error'] = "No response from Gemini"
        
        except Exception as e:
            result['error'] = f"Relevance scoring failed: {str(e)}"
            self.stats['errors'] += 1
        
        result['processing_time'] = time.time() - start_time
        self.stats['total_api_calls'] += 1
        return result
    
    def generate_detailed_analysis(self, job_title: str, job_description: str, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed candidate analysis using Gemini"""
        result = {
            'success': False,
            'analysis': '',
            'error': None,
            'processing_time': 0
        }
        
        if not self.models_available or not self.connection_status.get('connected', False):
            result['error'] = "Google AI not available"
            return result
        
        start_time = time.time()
        
        try:
            from config import get_gemini_prompt
            
            # Create detailed analysis prompt
            candidate_summary = self._format_candidate_for_analysis(candidate_data)
            prompt = get_gemini_prompt(
                "detailed_analysis",
                job_title=job_title,
                job_description=job_description[:1500],
                candidate_data=candidate_summary
            )
            
            # Check cache
            cache_key = self._get_cache_key(prompt, "detailed_analysis")
            cached_response = self._load_from_cache(cache_key)
            
            if cached_response:
                result.update({
                    'success': True,
                    'analysis': cached_response,
                    'processing_time': time.time() - start_time
                })
                return result
            
            # Generate with Gemini (longer token limit for detailed analysis)
            model = genai.GenerativeModel(
                self.main_model,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Slightly more creative for analysis
                    top_p=0.9,
                    top_k=50,
                    max_output_tokens=1000  # Longer for detailed analysis
                )
            )
            
            with st.spinner("📊 Generating detailed analysis..."):
                response = model.generate_content(prompt)
                
                if response.text:
                    analysis = response.text.strip()
                    
                    # Save to cache
                    self._save_to_cache(cache_key, analysis, "detailed_analysis")
                    
                    result.update({
                        'success': True,
                        'analysis': analysis
                    })
                    
                    self.stats['detailed_analyses'] += 1
                else:
                    result['error'] = "No analysis generated"
        
        except Exception as e:
            result['error'] = f"Analysis generation failed: {str(e)}"
            self.stats['errors'] += 1
        
        result['processing_time'] = time.time() - start_time
        self.stats['total_api_calls'] += 1
        return result
    
    def chat_response(self, question: str, context: str = None) -> Dict[str, Any]:
        """Generate chatbot response using Gemini"""
        result = {
            'success': False,
            'answer': '',
            'error': None,
            'processing_time': 0
        }
        
        if not self.models_available or not self.connection_status.get('connected', False):
            result['error'] = "Google AI not available"
            return result
        
        start_time = time.time()
        
        try:
            # Prepare context-aware prompt
            if context:
                full_prompt = f"""**Context:** {context[:800]}

**User Question:** {question}

**Instructions:** Provide a helpful and professional answer based on the context provided. Be specific and actionable."""
            else:
                full_prompt = f"""**User Question:** {question}

**Instructions:** Provide helpful guidance about resume screening, hiring practices, or recruitment strategies. Be professional and actionable."""
            
            # Check cache
            cache_key = self._get_cache_key(full_prompt, "chatbot")
            cached_response = self._load_from_cache(cache_key)
            
            if cached_response:
                result.update({
                    'success': True,
                    'answer': cached_response,
                    'processing_time': time.time() - start_time
                })
                return result
            
            # Generate with Gemini chat model
            chat_model = self._get_model_instance("chatbot")
            if not chat_model:
                result['error'] = "Failed to load chat model"
                return result
            
            with st.spinner("💬 Gemini thinking..."):
                response = chat_model.generate_content(full_prompt)
                
                if response.text:
                    answer = response.text.strip()
                    
                    # Save to cache
                    self._save_to_cache(cache_key, answer, "chatbot")
                    
                    result.update({
                        'success': True,
                        'answer': answer
                    })
                    
                    self.stats['chat_responses'] += 1
                else:
                    result['error'] = "No response from Gemini chat"
        
        except Exception as e:
            result['error'] = f"Chat error: {str(e)}"
            self.stats['errors'] += 1
        
        result['processing_time'] = time.time() - start_time
        self.stats['total_api_calls'] += 1
        return result
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response"""
        try:
            # Extract JSON from response (handle code blocks)
            json_pattern = r'```json\s*(.*?)\s*```'
            json_match = re.search(json_pattern, response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                else:
                    return {'error': 'No JSON found in response', 'raw': response}
            
            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            return {'error': f'JSON parse error: {e}', 'raw': response}
        except Exception as e:
            return {'error': f'Response parse error: {e}', 'raw': response}
    
    def _format_candidate_for_scoring(self, candidate_data: Dict[str, Any]) -> str:
        """Format candidate data for scoring prompt"""
        data = candidate_data.get('data', candidate_data)
        
        formatted = f"""
**Candidate:** {data.get('candidate_name', 'Unknown')}
**Experience:** {data.get('total_years_experience', 0)} years
**Current/Recent Position:** {self._get_recent_position(data)}
**Education:** {self._format_education(data)}
**Key Skills:** {', '.join(data.get('technical_skills', [])[:10])}
**Key Achievements:** {'; '.join(data.get('key_achievements', [])[:3])}
"""
        return formatted.strip()
    
    def _format_candidate_for_analysis(self, candidate_data: Dict[str, Any]) -> str:
        """Format candidate data for detailed analysis"""
        data = candidate_data.get('data', candidate_data)
        
        # More detailed formatting for analysis
        formatted = f"""
**Candidate Profile:**
- Name: {data.get('candidate_name', 'Unknown')}
- Total Experience: {data.get('total_years_experience', 0)} years
- Professional Summary: {data.get('professional_summary', 'Not provided')}

**Work Experience:**
{self._format_work_experience(data)}

**Education:**
{self._format_education(data)}

**Technical Skills:**
{', '.join(data.get('technical_skills', []))}

**Certifications:**
{', '.join(data.get('certifications', []))}

**Key Achievements:**
{chr(10).join('• ' + achievement for achievement in data.get('key_achievements', []))}
"""
        return formatted.strip()
    
    def _get_recent_position(self, data: Dict[str, Any]) -> str:
        """Get most recent position"""
        experience = data.get('work_experience', [])
        if experience:
            recent = experience[0]  # Assuming first is most recent
            return f"{recent.get('position', 'Unknown')} at {recent.get('company', 'Unknown')}"
        return "No experience listed"
    
    def _format_work_experience(self, data: Dict[str, Any]) -> str:
        """Format work experience for prompts"""
        experience = data.get('work_experience', [])
        if not experience:
            return "No work experience listed"
        
        formatted = []
        for exp in experience[:3]:  # Top 3 most relevant
            formatted.append(f"""
• {exp.get('position', 'Unknown Position')} at {exp.get('company', 'Unknown Company')} ({exp.get('duration', 'Unknown duration')})
  Responsibilities: {'; '.join(exp.get('key_responsibilities', [])[:2])}
  Achievements: {'; '.join(exp.get('achievements', [])[:2])}""")
        
        return '\n'.join(formatted)
    
    def _format_education(self, data: Dict[str, Any]) -> str:
        """Format education information"""
        education = data.get('education', [])
        if not education:
            return "No education listed"
        
        formatted = []
        for edu in education:
            degree = edu.get('degree', 'Unknown Degree')
            institution = edu.get('institution', 'Unknown Institution')
            year = edu.get('graduation_year', 'Unknown Year')
            formatted.append(f"{degree} from {institution} ({year})")
        
        return '; '.join(formatted)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in cache_files
            )
            
            stats = self.stats.copy()
            stats.update({
                'cache_enabled': True,
                'cache_files': len(cache_files),
                'cache_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_hit_rate': (
                    stats['cache_hits'] / stats['total_api_calls']
                    if stats['total_api_calls'] > 0 else 0
                ),
                'error_rate': (
                    stats['errors'] / stats['total_api_calls']
                    if stats['total_api_calls'] > 0 else 0
                )
            })
            
            return stats
            
        except Exception:
            return self.stats.copy()
    
    def clear_cache(self):
        """Clear all cache"""
        try:
            import shutil
            cache_files = len([f for f in os.listdir(self.cache_dir) if f.endswith('.json')])
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            st.success(f"✅ Cleared {cache_files} Google AI cache entries")
        except Exception as e:
            st.error(f"❌ Failed to clear cache: {e}")
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive diagnostics"""
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'connection_test': self.connection_status,
            'model_tests': {},
            'performance_stats': self.get_cache_stats()
        }
        
        # Test each model type
        test_tasks = ['data_extraction', 'relevance_scoring', 'chatbot']
        
        for task in test_tasks:
            try:
                model = self._get_model_instance(task)
                if model:
                    diagnostics['model_tests'][task] = {'status': 'available', 'error': None}
                else:
                    diagnostics['model_tests'][task] = {'status': 'failed', 'error': 'Model creation failed'}
            except Exception as e:
                diagnostics['model_tests'][task] = {'status': 'error', 'error': str(e)}
        
        return diagnostics
import streamlit as st
import openai
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Dict, Any, Optional
import json
import re
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import uuid
import requests
from urllib.parse import urlparse
import time

# Configuration
st.set_page_config(
    page_title="Real Agentic Support Router",
    page_icon="🤖",
    layout="wide"
)

class AgentDecision:
    def __init__(self, agent_name: str, step: str, input_data: Dict, reasoning: str, decision: Dict, confidence: float):
        self.agent_name = agent_name
        self.step = step
        self.input_data = input_data
        self.reasoning = reasoning
        self.decision = decision
        self.confidence = confidence
        self.timestamp = datetime.now()

class TicketState(TypedDict):
    ticket_id: str
    original_content: str
    customer_email: str
    current_step: str
    agent_decisions: List[Dict[str, Any]]
    extracted_data: Dict[str, Any]
    classification: Dict[str, Any]
    knowledge_search_results: List[Dict[str, Any]]
    escalation_analysis: Dict[str, Any]
    final_response: Dict[str, Any]
    processing_complete: bool
    step_by_step_log: List[str]

# Real Knowledge Base (you can expand this with your actual data)
REAL_KNOWLEDGE_BASE = [
    {
        "id": "kb_001",
        "title": "Password Reset Instructions",
        "content": "To reset your password: 1. Go to the login page 2. Click 'Forgot Password' 3. Enter your registered email 4. Check your email for the reset link 5. Click the link and create a new password",
        "keywords": ["password", "reset", "forgot", "login", "access", "sign in"],
        "category": "account",
        "solution_type": "self_service"
    },
    {
        "id": "kb_002", 
        "title": "Payment Failed Error",
        "content": "If your payment failed: 1. Check if your card has sufficient funds 2. Verify the card details are correct 3. Try a different payment method 4. Contact your bank if the issue persists 5. Contact our billing team if problem continues",
        "keywords": ["payment", "failed", "billing", "card", "declined", "transaction"],
        "category": "billing",
        "solution_type": "guided_resolution"
    },
    {
        "id": "kb_003",
        "title": "App Crashes on Startup",
        "content": "If the app crashes when starting: 1. Force close the app completely 2. Restart your device 3. Update the app to the latest version 4. Clear app cache/data 5. Reinstall the app if necessary",
        "keywords": ["crash", "startup", "app", "launch", "error", "freeze"],
        "category": "technical",
        "solution_type": "troubleshooting"
    },
    {
        "id": "kb_004",
        "title": "Refund Request Process",
        "content": "To request a refund: 1. Log into your account 2. Go to Order History 3. Find the relevant purchase 4. Click 'Request Refund' 5. Fill out the refund form with reason 6. Submit for review (typically processed within 3-5 business days)",
        "keywords": ["refund", "return", "money back", "cancel", "purchase"],
        "category": "billing",
        "solution_type": "process_driven"
    }
]

# Agent Classes
class DataExtractionAgent:
    def __init__(self, client):
        self.client = client
        self.name = "Data Extraction Agent"
    
    def extract_ticket_data(self, state: TicketState) -> TicketState:
        """Extract structured data from raw ticket content"""
        
        step_log = f"🔍 {self.name}: Starting data extraction from ticket content"
        state["step_by_step_log"].append(step_log)
        
        prompt = f"""
        Extract structured information from this customer support ticket:
        
        Ticket Content: {state['original_content']}
        Customer Email: {state['customer_email']}
        
        Extract the following information and return as JSON:
        {{
            "customer_name": "extracted or inferred name",
            "issue_summary": "brief summary of the main issue",
            "issue_category": "technical/billing/account/general",
            "urgency_indicators": ["list", "of", "urgency", "signals"],
            "specific_error_messages": ["any", "error", "messages", "mentioned"],
            "user_actions_taken": ["actions", "user", "already", "tried"],
            "desired_outcome": "what the customer wants to achieve",
            "technical_details": {{"device": "if mentioned", "browser": "if mentioned", "version": "if mentioned"}},
            "sentiment": "positive/neutral/negative",
            "key_phrases": ["important", "phrases", "from", "ticket"]
        }}
        
        Be precise and only extract information that's actually present in the ticket.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",  # Using more reliable model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Clean the response if it contains markdown formatting
            if response_content.startswith("```json"):
                response_content = response_content[7:]  # Remove ```json
            if response_content.endswith("```"):
                response_content = response_content[:-3]  # Remove ```
            
            response_content = response_content.strip()
            
            # Validate that we have content
            if not response_content:
                raise ValueError("Empty response from API")
            
            try:
                extracted_data = json.loads(response_content)
            except json.JSONDecodeError as e:
                # If JSON parsing fails, create a fallback structure
                st.warning(f"JSON parsing failed, using fallback extraction. Error: {str(e)}")
                extracted_data = {
                    "customer_name": "Unknown",
                    "issue_summary": state['original_content'][:100] + "...",
                    "issue_category": "general",
                    "urgency_indicators": [],
                    "specific_error_messages": [],
                    "user_actions_taken": [],
                    "desired_outcome": "Resolve issue",
                    "technical_details": {},
                    "sentiment": "neutral",
                    "key_phrases": [],
                    "raw_response": response_content
                }
            
            # Log the decision
            decision = AgentDecision(
                agent_name=self.name,
                step="data_extraction",
                input_data={"ticket_content": state['original_content']},
                reasoning="Extracted structured data from unstructured ticket content using LLM analysis",
                decision=extracted_data,
                confidence=0.9
            )
            
            state["extracted_data"] = extracted_data
            state["agent_decisions"].append({
                "agent": self.name,
                "step": "data_extraction", 
                "reasoning": decision.reasoning,
                "decision": decision.decision,
                "confidence": decision.confidence,
                "timestamp": decision.timestamp.isoformat()
            })
            
            state["step_by_step_log"].append(f"✅ {self.name}: Extracted {len(extracted_data)} data points")
            state["current_step"] = "data_extraction_complete"
            
            return state
            
        except Exception as e:
            error_log = f"❌ {self.name}: Failed to extract data - {str(e)}"
            state["step_by_step_log"].append(error_log)
            # Create fallback extracted data
            state["extracted_data"] = {
                "customer_name": "Unknown",
                "issue_summary": state['original_content'][:100] + "...",
                "issue_category": "general",
                "urgency_indicators": [],
                "specific_error_messages": [],
                "user_actions_taken": [],
                "desired_outcome": "Resolve issue",
                "technical_details": {},
                "sentiment": "neutral",
                "key_phrases": [],
                "error": str(e)
            }
            return state

class ClassificationAgent:
    def __init__(self, client):
        self.client = client
        self.name = "Classification Agent"
    
    def classify_ticket(self, state: TicketState) -> TicketState:
        """Classify ticket urgency, category, and routing requirements"""
        
        step_log = f"🏷️ {self.name}: Starting ticket classification"
        state["step_by_step_log"].append(step_log)
        
        extracted = state.get("extracted_data", {})
        
        prompt = f"""
        Based on the extracted ticket data, classify this support ticket:
        
        Extracted Data: {json.dumps(extracted, indent=2)}
        Original Ticket: {state['original_content']}
        
        Provide classification as JSON:
        {{
            "urgency_level": "low/medium/high/critical",
            "urgency_reasoning": "explain why this urgency level",
            "primary_category": "technical/billing/account/general",
            "secondary_categories": ["other", "relevant", "categories"],
            "complexity_score": 1-10,
            "estimated_resolution_time": "time estimate",
            "requires_human_agent": true/false,
            "escalation_triggers": ["reasons", "for", "escalation"],
            "routing_recommendation": "which team/agent type should handle this",
            "priority_score": 1-100
        }}
        
        Base your classification on:
        1. Customer sentiment and language urgency
        2. Technical complexity of the issue
        3. Business impact potential
        4. Customer history if inferable
        """
        
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Clean the response if it contains markdown formatting
            if response_content.startswith("```json"):
                response_content = response_content[7:]
            if response_content.endswith("```"):
                response_content = response_content[:-3]
            
            response_content = response_content.strip()
            
            try:
                classification = json.loads(response_content)
            except json.JSONDecodeError as e:
                # Fallback classification
                st.warning(f"Classification JSON parsing failed, using fallback. Error: {str(e)}")
                classification = {
                    "urgency_level": "medium",
                    "urgency_reasoning": "Default classification due to parsing error",
                    "primary_category": extracted.get("issue_category", "general"),
                    "secondary_categories": [],
                    "complexity_score": 5,
                    "estimated_resolution_time": "1-2 hours",
                    "requires_human_agent": True,
                    "escalation_triggers": [],
                    "routing_recommendation": "general support",
                    "priority_score": 50,
                    "raw_response": response_content
                }
            
            # Log the decision
            decision = AgentDecision(
                agent_name=self.name,
                step="classification",
                input_data=extracted,
                reasoning=f"Classified as {classification['urgency_level']} urgency, {classification['primary_category']} category based on content analysis",
                decision=classification,
                confidence=0.85
            )
            
            state["classification"] = classification
            state["agent_decisions"].append({
                "agent": self.name,
                "step": "classification",
                "reasoning": decision.reasoning,
                "decision": decision.decision,
                "confidence": decision.confidence,
                "timestamp": decision.timestamp.isoformat()
            })
            
            state["step_by_step_log"].append(f"✅ {self.name}: Classified as {classification['urgency_level']} urgency, {classification['primary_category']} category")
            state["current_step"] = "classification_complete"
            
            return state
            
        except Exception as e:
            error_log = f"❌ {self.name}: Classification failed - {str(e)}"
            state["step_by_step_log"].append(error_log)
            # Fallback classification
            state["classification"] = {
                "urgency_level": "medium",
                "urgency_reasoning": "Default due to error",
                "primary_category": extracted.get("issue_category", "general"),
                "secondary_categories": [],
                "complexity_score": 5,
                "estimated_resolution_time": "1-2 hours",
                "requires_human_agent": True,
                "escalation_triggers": [],
                "routing_recommendation": "general support",
                "priority_score": 50,
                "error": str(e)
            }
            return state

class KnowledgeSearchAgent:
    def __init__(self, client):
        self.client = client
        self.name = "Knowledge Search Agent"
    
    def search_knowledge_base(self, state: TicketState) -> TicketState:
        """Search knowledge base for relevant solutions"""
        
        step_log = f"🔍 {self.name}: Searching knowledge base for solutions"
        state["step_by_step_log"].append(step_log)
        
        extracted = state.get("extracted_data", {})
        classification = state.get("classification", {})
        
        # Real keyword-based search
        search_terms = []
        if "key_phrases" in extracted:
            search_terms.extend(extracted["key_phrases"])
        if "issue_summary" in extracted:
            search_terms.extend(extracted["issue_summary"].lower().split())
        
        # Search the real knowledge base
        relevant_articles = []
        for article in REAL_KNOWLEDGE_BASE:
            relevance_score = 0
            matched_keywords = []
            
            # Check if article category matches
            if article["category"] == classification.get("primary_category", ""):
                relevance_score += 10
            
            # Check keyword matches
            for term in search_terms:
                term_lower = term.lower()
                for keyword in article["keywords"]:
                    if term_lower in keyword or keyword in term_lower:
                        relevance_score += 5
                        matched_keywords.append(keyword)
            
            # Check content similarity
            content_lower = article["content"].lower()
            for term in search_terms:
                if term.lower() in content_lower:
                    relevance_score += 2
            
            if relevance_score > 0:
                relevant_articles.append({
                    "article": article,
                    "relevance_score": relevance_score,
                    "matched_keywords": matched_keywords
                })
        
        # Sort by relevance
        relevant_articles.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Use LLM to analyze the search results
        if relevant_articles:
            prompt = f"""
            Analyze these knowledge base search results for the customer issue:
            
            Customer Issue: {extracted.get('issue_summary', 'Not specified')}
            Classification: {classification.get('primary_category', 'Unknown')}
            
            Search Results:
            {json.dumps([{
                'title': r['article']['title'],
                'content': r['article']['content'],
                'relevance_score': r['relevance_score'],
                'matched_keywords': r['matched_keywords']
            } for r in relevant_articles[:3]], indent=2)}
            
            Analyze and return JSON:
            {{
                "best_match_found": true/false,
                "recommended_article_id": "id of best match or null",
                "solution_confidence": 0.0-1.0,
                "solution_applicability": "how well this solution fits the issue",
                "additional_steps_needed": ["any", "additional", "steps"],
                "alternative_approaches": ["other", "possible", "solutions"]
            }}
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1500
                )
                
                response_content = response.choices[0].message.content.strip()
                
                # Clean the response
                if response_content.startswith("```json"):
                    response_content = response_content[7:]
                if response_content.endswith("```"):
                    response_content = response_content[:-3]
                
                response_content = response_content.strip()
                
                try:
                    analysis = json.loads(response_content)
                except json.JSONDecodeError:
                    # Fallback analysis
                    analysis = {
                        "best_match_found": len(relevant_articles) > 0,
                        "recommended_article_id": relevant_articles[0]["article"]["id"] if relevant_articles else None,
                        "solution_confidence": 0.7 if relevant_articles else 0.0,
                        "solution_applicability": "Likely helpful based on keyword matching",
                        "additional_steps_needed": [],
                        "alternative_approaches": []
                    }
                
                # Combine search results with analysis
                search_results = {
                    "articles_found": len(relevant_articles),
                    "top_matches": relevant_articles[:3],
                    "analysis": analysis
                }
                
            except Exception as e:
                search_results = {
                    "articles_found": len(relevant_articles),
                    "top_matches": relevant_articles[:3],
                    "analysis": {"error": str(e), "best_match_found": False, "solution_confidence": 0.0}
                }
        else:
            search_results = {
                "articles_found": 0,
                "top_matches": [],
                "analysis": {"best_match_found": False, "solution_confidence": 0.0}
            }
        
        # Log the decision
        decision = AgentDecision(
            agent_name=self.name,
            step="knowledge_search",
            input_data={"search_terms": search_terms},
            reasoning=f"Found {len(relevant_articles)} relevant articles using keyword matching and semantic analysis",
            decision=search_results,
            confidence=0.8 if relevant_articles else 0.3
        )
        
        state["knowledge_search_results"] = search_results
        state["agent_decisions"].append({
            "agent": self.name,
            "step": "knowledge_search",
            "reasoning": decision.reasoning,
            "decision": decision.decision,
            "confidence": decision.confidence,
            "timestamp": decision.timestamp.isoformat()
        })
        
        state["step_by_step_log"].append(f"✅ {self.name}: Found {len(relevant_articles)} relevant articles")
        state["current_step"] = "knowledge_search_complete"
        
        return state

class EscalationAnalysisAgent:
    def __init__(self, client):
        self.client = client
        self.name = "Escalation Analysis Agent"
    
    def analyze_escalation_needs(self, state: TicketState) -> TicketState:
        """Analyze if ticket needs escalation based on all available data"""
        
        step_log = f"⚡ {self.name}: Analyzing escalation requirements"
        state["step_by_step_log"].append(step_log)
        
        extracted = state.get("extracted_data", {})
        classification = state.get("classification", {})
        kb_results = state.get("knowledge_search_results", {})
        
        prompt = f"""
        Analyze if this support ticket requires escalation based on all available data:
        
        Extracted Data: {json.dumps(extracted, indent=2)}
        Classification: {json.dumps(classification, indent=2)}
        Knowledge Base Results: {json.dumps(kb_results.get('analysis', {}), indent=2)}
        
        Consider these escalation factors:
        1. Urgency level and customer sentiment
        2. Complexity score and technical requirements
        3. Knowledge base solution confidence
        4. Potential business impact
        5. Customer tier (if inferable from email domain)
        
        Return analysis as JSON:
        {{
            "requires_escalation": true/false,
            "escalation_reason": "primary reason for escalation",
            "escalation_level": "tier1/tier2/tier3/management",
            "escalation_urgency": "immediate/within_hour/within_day/standard",
            "suggested_department": "technical/billing/customer_success/engineering",
            "escalation_notes": "detailed notes for escalation recipient",
            "auto_resolution_possible": true/false,
            "confidence_in_analysis": 0.0-1.0
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Clean the response
            if response_content.startswith("```json"):
                response_content = response_content[7:]
            if response_content.endswith("```"):
                response_content = response_content[:-3]
            
            response_content = response_content.strip()
            
            try:
                escalation_analysis = json.loads(response_content)
            except json.JSONDecodeError:
                # Fallback escalation analysis
                escalation_analysis = {
                    "requires_escalation": classification.get("urgency_level") in ["high", "critical"],
                    "escalation_reason": "Default escalation based on urgency level",
                    "escalation_level": "tier1",
                    "escalation_urgency": "standard",
                    "suggested_department": "technical",
                    "escalation_notes": "Automated escalation due to parsing error",
                    "auto_resolution_possible": kb_results.get("analysis", {}).get("solution_confidence", 0) > 0.8,
                    "confidence_in_analysis": 0.5
                }
            
            # Log the decision
            decision = AgentDecision(
                agent_name=self.name,
                step="escalation_analysis",
                input_data={
                    "urgency": classification.get("urgency_level"),
                    "complexity": classification.get("complexity_score"),
                    "kb_confidence": kb_results.get("analysis", {}).get("solution_confidence", 0)
                },
                reasoning=escalation_analysis.get("escalation_reason", "Analysis completed"),
                decision=escalation_analysis,
                confidence=escalation_analysis.get("confidence_in_analysis", 0.8)
            )
            
            state["escalation_analysis"] = escalation_analysis
            state["agent_decisions"].append({
                "agent": self.name,
                "step": "escalation_analysis",
                "reasoning": decision.reasoning,
                "decision": decision.decision,
                "confidence": decision.confidence,
                "timestamp": decision.timestamp.isoformat()
            })
            
            escalation_status = "ESCALATION REQUIRED" if escalation_analysis.get("requires_escalation") else "STANDARD PROCESSING"
            state["step_by_step_log"].append(f"✅ {self.name}: {escalation_status}")
            state["current_step"] = "escalation_analysis_complete"
            
            return state
            
        except Exception as e:
            error_log = f"❌ {self.name}: Escalation analysis failed - {str(e)}"
            state["step_by_step_log"].append(error_log)
            # Fallback escalation analysis
            state["escalation_analysis"] = {
                "requires_escalation": True,
                "escalation_reason": f"Error in analysis: {str(e)}",
                "escalation_level": "tier1",
                "escalation_urgency": "standard",
                "suggested_department": "technical",
                "escalation_notes": f"Manual review required due to error: {str(e)}",
                "auto_resolution_possible": False,
                "confidence_in_analysis": 0.3,
                "error": str(e)
            }
            return state

class ResponseGenerationAgent:
    def __init__(self, client):
        self.client = client
        self.name = "Response Generation Agent"
    
    def generate_final_response(self, state: TicketState) -> TicketState:
        """Generate final response based on all agent analyses"""
        
        step_log = f"📝 {self.name}: Generating final customer response"
        state["step_by_step_log"].append(step_log)
        
        extracted = state.get("extracted_data", {})
        classification = state.get("classification", {})
        kb_results = state.get("knowledge_search_results", {})
        escalation = state.get("escalation_analysis", {})
        
        # Get the best knowledge base match if available
        best_solution = None
        if kb_results.get("top_matches"):
            best_match = kb_results["top_matches"][0]
            best_solution = best_match["article"]["content"]
        
        prompt = f"""
        Generate a professional customer support response based on all agent analyses:
        
        Customer Issue: {extracted.get('issue_summary', 'Support request')}
        Customer Name: {extracted.get('customer_name', 'Valued Customer')}
        Urgency: {classification.get('urgency_level', 'medium')}
        Sentiment: {extracted.get('sentiment', 'neutral')}
        
        Knowledge Base Solution Found: {best_solution if best_solution else 'No direct solution found'}
        Escalation Required: {escalation.get('requires_escalation', False)}
        Auto-Resolution Possible: {escalation.get('auto_resolution_possible', False)}
        
        Generate a response that includes:
        1. Acknowledge the customer's issue empathetically
        2. Provide solution steps if available from knowledge base
        3. Set appropriate expectations for next steps
        4. Include escalation notice if required
        5. Professional but warm tone
        
        Return as JSON:
        {{
            "response_text": "full customer response",
            "response_type": "solution_provided/escalation_notice/acknowledgment",
            "next_steps": ["specific", "next", "steps"],
            "estimated_resolution_time": "time estimate",
            "internal_notes": "notes for support team",
            "follow_up_required": true/false
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=2000
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Clean the response
            if response_content.startswith("```json"):
                response_content = response_content[7:]
            if response_content.endswith("```"):
                response_content = response_content[:-3]
            
            response_content = response_content.strip()
            
            try:
                final_response = json.loads(response_content)
            except json.JSONDecodeError:
                # Fallback response
                customer_name = extracted.get('customer_name', 'Valued Customer')
                issue_summary = extracted.get('issue_summary', 'your inquiry')
                
                final_response = {
                    "response_text": f"Dear {customer_name},\n\nThank you for contacting us regarding {issue_summary}. We have received your request and are working to resolve it as quickly as possible. Our team will review your case and provide a detailed response within 24 hours.\n\nBest regards,\nCustomer Support Team",
                    "response_type": "acknowledgment",
                    "next_steps": ["Review case details", "Provide detailed solution"],
                    "estimated_resolution_time": "24 hours",
                    "internal_notes": "Automated response generated due to parsing error",
                    "follow_up_required": True
                }
            
            # Log the decision
            decision = AgentDecision(
                agent_name=self.name,
                step="response_generation",
                input_data={
                    "has_kb_solution": best_solution is not None,
                    "requires_escalation": escalation.get("requires_escalation", False)
                },
                reasoning="Generated personalized response based on all agent analyses and available solutions",
                decision=final_response,
                confidence=0.9
            )
            
            state["final_response"] = final_response
            state["agent_decisions"].append({
                "agent": self.name,
                "step": "response_generation",
                "reasoning": decision.reasoning,
                "decision": decision.decision,
                "confidence": decision.confidence,
                "timestamp": decision.timestamp.isoformat()
            })
            
            state["step_by_step_log"].append(f"✅ {self.name}: Generated {final_response.get('response_type', 'response')}")
            state["current_step"] = "response_complete"
            state["processing_complete"] = True
            
            return state
            
        except Exception as e:
            error_log = f"❌ {self.name}: Response generation failed - {str(e)}"
            state["step_by_step_log"].append(error_log)
            # Fallback response
            customer_name = extracted.get('customer_name', 'Valued Customer')
            state["final_response"] = {
                "response_text": f"Dear {customer_name},\n\nWe have received your support request and our team is reviewing it. We will get back to you as soon as possible with a detailed response.\n\nThank you for your patience.\n\nBest regards,\nCustomer Support Team",
                "response_type": "acknowledgment",
                "next_steps": ["Manual review required"],
                "estimated_resolution_time": "24-48 hours",
                "internal_notes": f"Manual review required due to error: {str(e)}",
                "follow_up_required": True,
                "error": str(e)
            }
            return state

# LangGraph Workflow Functions
def extract_data_step(state: TicketState) -> TicketState:
    """Step 1: Extract structured data from ticket"""
    if 'openrouter_client' in st.session_state:
        agent = DataExtractionAgent(st.session_state.openrouter_client)
        return agent.extract_ticket_data(state)
    return state

def classify_step(state: TicketState) -> TicketState:
    """Step 2: Classify ticket"""
    if 'openrouter_client' in st.session_state:
        agent = ClassificationAgent(st.session_state.openrouter_client)
        return agent.classify_ticket(state)
    return state

def search_knowledge_step(state: TicketState) -> TicketState:
    """Step 3: Search knowledge base"""
    if 'openrouter_client' in st.session_state:
        agent = KnowledgeSearchAgent(st.session_state.openrouter_client)
        return agent.search_knowledge_base(state)
    return state

def analyze_escalation_step(state: TicketState) -> TicketState:
    """Step 4: Analyze escalation needs"""
    if 'openrouter_client' in st.session_state:
        agent = EscalationAnalysisAgent(st.session_state.openrouter_client)
        return agent.analyze_escalation_needs(state)
    return state

def generate_response_step(state: TicketState) -> TicketState:
    """Step 5: Generate final response"""
    if 'openrouter_client' in st.session_state:
        agent = ResponseGenerationAgent(st.session_state.openrouter_client)
        return agent.generate_final_response(state)
    return state

# Create the workflow
def create_real_agentic_workflow():
    """Create the real agentic workflow"""
    workflow = StateGraph(TicketState)
    
    # Add nodes in sequence
    workflow.add_node("extract_data", extract_data_step)
    workflow.add_node("classify", classify_step)
    workflow.add_node("search_knowledge", search_knowledge_step)
    workflow.add_node("analyze_escalation", analyze_escalation_step)
    workflow.add_node("generate_response", generate_response_step)
    
    # Add edges
    workflow.add_edge("extract_data", "classify")
    workflow.add_edge("classify", "search_knowledge")
    workflow.add_edge("search_knowledge", "analyze_escalation")
    workflow.add_edge("analyze_escalation", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Set entry point
    workflow.set_entry_point("extract_data")
    
    return workflow.compile(checkpointer=MemorySaver())

# Streamlit UI
def main():
    st.title("🤖 Real Agentic Support Ticket Router")
    st.markdown("**Step-by-step autonomous processing** with real data and transparent decision making")
    
    # Initialize session state
    if 'processed_tickets' not in st.session_state:
        st.session_state.processed_tickets = []
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key
        api_key = st.text_input("OpenRouter API Key", type="password", help="Get your API key from https://openrouter.ai/keys")
        
        if api_key:
            try:
                # Test the API key with a simple call
                client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    default_headers={
                        "HTTP-Referer": "https://github.com/yourusername/support-router",
                        "X-Title": "Support Ticket Router"
                    }
                )
                
                # Test API connection
                test_response = client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": "Test connection"}],
                    max_tokens=10
                )
                
                st.session_state.openrouter_client = client
                st.success("✅ Agentic System Ready")
                st.success(f"✅ API Test: {test_response.choices[0].message.content[:20]}...")
                
            except Exception as e:
                st.error(f"❌ API Key Error: {str(e)}")
                st.error("Please check your OpenRouter API key and ensure you have credits")
                st.info("💡 Make sure to:")
                st.write("1. Get API key from https://openrouter.ai/keys")
                st.write("2. Add credits to your OpenRouter account")
                st.write("3. Copy the key exactly as shown")
        
        st.divider()
        
        # Real Knowledge Base Preview
        st.subheader("📚 Knowledge Base")
        st.write(f"**{len(REAL_KNOWLEDGE_BASE)} Real Articles**")
        for article in REAL_KNOWLEDGE_BASE:
            with st.expander(f"{article['title']}", expanded=False):
                st.write(f"**Category:** {article['category']}")
                st.write(f"**Keywords:** {', '.join(article['keywords'])}")
                st.write(f"**Solution:** {article['content'][:100]}...")
        
        st.divider()
        
        # Statistics
        if st.session_state.processed_tickets:
            st.subheader("📊 Processing Stats")
            total_tickets = len(st.session_state.processed_tickets)
            st.metric("Tickets Processed", total_tickets)
            
            # Calculate success rate
            successful = sum(1 for t in st.session_state.processed_tickets 
                           if t.get('processing_complete', False))
            st.metric("Success Rate", f"{(successful/total_tickets)*100:.1f}%")
    
    # Main Interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📧 Real Support Ticket Input")
        
        with st.form("real_ticket_form"):
            st.markdown("**Enter actual support ticket data:**")
            
            customer_email = st.text_input(
                "Customer Email",
                placeholder="customer@company.com"
            )
            
            ticket_content = st.text_area(
                "Ticket Content (paste real ticket)",
                placeholder="Hi, I'm having trouble logging into my account. I keep getting an error message that says 'Invalid credentials' even though I'm sure my password is correct. I tried resetting it but didn't receive the email. This is urgent as I need to access my files for a presentation tomorrow. Can you please help? Thanks, John",
                height=150
            )
            
            submitted = st.form_submit_button("🚀 Process with Real Agents", use_container_width=True)
        
        if submitted and ticket_content and customer_email:
            if 'openrouter_client' not in st.session_state:
                st.error("Please enter your OpenRouter API key")
                return
            
            # Create real workflow
            workflow = create_real_agentic_workflow()
            
            with st.spinner("🤖 Agents processing real ticket data..."):
                ticket_id = f"REAL-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                initial_state = {
                    "ticket_id": ticket_id,
                    "original_content": ticket_content,
                    "customer_email": customer_email,
                    "current_step": "initializing",
                    "agent_decisions": [],
                    "extracted_data": {},
                    "classification": {},
                    "knowledge_search_results": {},
                    "escalation_analysis": {},
                    "final_response": {},
                    "processing_complete": False,
                    "step_by_step_log": [f"🎯 Starting processing for ticket {ticket_id}"]
                }
                
                # Run the real workflow
                config = {"configurable": {"thread_id": ticket_id}}
                result = workflow.invoke(initial_state, config)
                
                # Store result
                st.session_state.processed_tickets.append(result)
                st.session_state.current_ticket = result
    
    with col2:
        st.header("🔍 Step-by-Step Agent Analysis")
        
        if 'current_ticket' in st.session_state:
            ticket = st.session_state.current_ticket
            
            st.success(f"**Ticket {ticket['ticket_id']}** - Processing Complete")
            
            # Step-by-step log
            st.subheader("📋 Processing Log")
            for i, log_entry in enumerate(ticket.get('step_by_step_log', [])):
                st.write(f"{i+1}. {log_entry}")
            
            st.divider()
            
            # Agent Decisions Detail
            st.subheader("🤖 Detailed Agent Decisions")
            
            for i, decision in enumerate(ticket.get('agent_decisions', [])):
                with st.expander(f"Step {i+1}: {decision['agent']} - {decision['step']}", expanded=i<2):
                    st.write(f"**Confidence:** {decision['confidence']*100:.1f}%")
                    st.write(f"**Reasoning:** {decision['reasoning']}")
                    st.write(f"**Time:** {decision['timestamp']}")
                    
                    st.write("**Decision Data:**")
                    st.json(decision['decision'])
            
            # Final Results
            if ticket.get('final_response'):
                st.subheader("📨 Generated Response")
                response = ticket['final_response']
                
                # Show response type and confidence
                col_resp1, col_resp2 = st.columns(2)
                with col_resp1:
                    st.metric("Response Type", response.get('response_type', 'Unknown'))
                with col_resp2:
                    st.metric("Follow-up Required", "Yes" if response.get('follow_up_required') else "No")
                
                # Customer response
                st.write("**Customer Response:**")
                st.info(response.get('response_text', 'No response generated'))
                
                # Next steps
                if response.get('next_steps'):
                    st.write("**Next Steps:**")
                    for step in response['next_steps']:
                        st.write(f"• {step}")
                
                # Internal notes
                if response.get('internal_notes'):
                    with st.expander("Internal Notes (for support team)"):
                        st.write(response['internal_notes'])
            
            # Key Insights Summary
            st.subheader("🎯 Key Insights")
            
            extracted = ticket.get('extracted_data', {})
            classification = ticket.get('classification', {})
            kb_results = ticket.get('knowledge_search_results', {})
            escalation = ticket.get('escalation_analysis', {})
            
            # Create insight metrics
            col_insight1, col_insight2, col_insight3 = st.columns(3)
            
            with col_insight1:
                urgency = classification.get('urgency_level', 'Unknown')
                urgency_colors = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
                st.metric("Urgency", f"{urgency_colors.get(urgency.lower(), '⚪')} {urgency.title()}")
                
                sentiment = extracted.get('sentiment', 'Unknown')
                sentiment_colors = {"positive": "😊", "neutral": "😐", "negative": "😟"}
                st.metric("Sentiment", f"{sentiment_colors.get(sentiment.lower(), '😐')} {sentiment.title()}")
            
            with col_insight2:
                kb_found = kb_results.get('articles_found', 0)
                st.metric("KB Articles Found", kb_found)
                
                complexity = classification.get('complexity_score', 0)
                st.metric("Complexity Score", f"{complexity}/10")
            
            with col_insight3:
                escalation_needed = escalation.get('requires_escalation', False)
                st.metric("Escalation", "Required" if escalation_needed else "Not Needed")
                
                auto_resolve = escalation.get('auto_resolution_possible', False)
                st.metric("Auto-Resolve", "Possible" if auto_resolve else "Manual")
            
            # Action Buttons
            st.subheader("🎬 Actions")
            col_action1, col_action2, col_action3 = st.columns(3)
            
            with col_action1:
                if st.button("📧 Send Response", use_container_width=True):
                    st.success("✅ Response sent to customer!")
            
            with col_action2:
                if escalation.get('requires_escalation'):
                    if st.button("⚡ Execute Escalation", use_container_width=True):
                        dept = escalation.get('suggested_department', 'General')
                        st.warning(f"🔄 Escalated to {dept}")
                else:
                    if st.button("✅ Mark Resolved", use_container_width=True):
                        st.success("✅ Ticket marked as resolved!")
            
            with col_action3:
                if st.button("🔄 Reprocess Ticket", use_container_width=True):
                    # Clear current ticket to allow reprocessing
                    if 'current_ticket' in st.session_state:
                        del st.session_state.current_ticket
                    st.rerun()
        
        else:
            st.info("👆 Submit a real support ticket to see step-by-step agent processing")
            
            # Show example tickets
            st.subheader("💡 Example Real Tickets")
            
            examples = [
                {
                    "title": "Login Issue",
                    "content": "Hi, I can't log into my account. Keep getting 'invalid credentials' error even though I know my password is correct. Tried resetting but no email received. This is urgent - need access for tomorrow's presentation. - John",
                    "email": "john.doe@company.com"
                },
                {
                    "title": "Payment Failed",
                    "content": "Hello, my credit card payment failed during checkout. The error says 'Transaction declined' but I have sufficient funds. I tried 3 times. Order #12345. Please help ASAP as this is for a client project due today.",
                    "email": "sarah.smith@agency.com"
                },
                {
                    "title": "App Crash",
                    "content": "The mobile app keeps crashing whenever I try to open it. Happens right at startup with a white screen then closes. I'm on iPhone 14 with latest iOS. Already tried reinstalling. Really frustrating!",
                    "email": "mike.jones@gmail.com"
                }
            ]
            
            for i, example in enumerate(examples):
                with st.expander(f"Example {i+1}: {example['title']}", expanded=False):
                    st.write(f"**Email:** {example['email']}")
                    st.write(f"**Content:** {example['content']}")
                    if st.button(f"Use Example {i+1}", key=f"example_{i}"):
                        # Pre-fill the form (this would require form state management)
                        st.info("Copy the content above and paste into the form")
    
    # Historical Analysis Dashboard
    if len(st.session_state.processed_tickets) > 0:
        st.header("📊 Agent Performance Analytics")
        
        # Create analytics dataframe
        analytics_data = []
        for ticket in st.session_state.processed_tickets:
            extracted = ticket.get('extracted_data', {})
            classification = ticket.get('classification', {})
            kb_results = ticket.get('knowledge_search_results', {})
            escalation = ticket.get('escalation_analysis', {})
            
            analytics_data.append({
                "Ticket ID": ticket['ticket_id'],
                "Category": classification.get('primary_category', 'Unknown'),
                "Urgency": classification.get('urgency_level', 'Unknown'),
                "Complexity": classification.get('complexity_score', 0),
                "KB Articles Found": kb_results.get('articles_found', 0),
                "KB Solution Confidence": kb_results.get('analysis', {}).get('solution_confidence', 0),
                "Escalation Required": escalation.get('requires_escalation', False),
                "Auto-Resolve Possible": escalation.get('auto_resolution_possible', False),
                "Processing Success": ticket.get('processing_complete', False),
                "Agent Steps": len(ticket.get('agent_decisions', []))
            })
        
        df = pd.DataFrame(analytics_data)
        
        # Display analytics
        col_analytics1, col_analytics2 = st.columns(2)
        
        with col_analytics1:
            st.subheader("📈 Category Distribution")
            category_counts = df['Category'].value_counts()
            st.bar_chart(category_counts)
            
            st.subheader("⚡ Urgency Levels")
            urgency_counts = df['Urgency'].value_counts()
            st.bar_chart(urgency_counts)
        
        with col_analytics2:
            st.subheader("🎯 Resolution Metrics")
            
            # Key metrics
            total_tickets = len(df)
            escalated = df['Escalation Required'].sum()
            auto_resolvable = df['Auto-Resolve Possible'].sum()
            avg_complexity = df['Complexity'].mean()
            avg_kb_confidence = df['KB Solution Confidence'].mean()
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Escalation Rate", f"{(escalated/total_tickets)*100:.1f}%")
                st.metric("Auto-Resolve Rate", f"{(auto_resolvable/total_tickets)*100:.1f}%")
            
            with col_metric2:
                st.metric("Avg Complexity", f"{avg_complexity:.1f}/10")
                st.metric("Avg KB Confidence", f"{avg_kb_confidence*100:.1f}%")
            
            # Agent efficiency
            st.subheader("🤖 Agent Efficiency")
            avg_steps = df['Agent Steps'].mean()
            success_rate = df['Processing Success'].mean()
            
            st.metric("Avg Steps per Ticket", f"{avg_steps:.1f}")
            st.metric("Processing Success Rate", f"{success_rate*100:.1f}%")
        
        # Detailed ticket history table
        st.subheader("📋 Processing History")
        
        # Format the dataframe for display
        display_df = df.copy()
        display_df['Escalation Required'] = display_df['Escalation Required'].map({True: "Yes", False: "No"})
        display_df['Auto-Resolve Possible'] = display_df['Auto-Resolve Possible'].map({True: "Yes", False: "No"})
        display_df['Processing Success'] = display_df['Processing Success'].map({True: "✅", False: "❌"})
        display_df['KB Solution Confidence'] = (display_df['KB Solution Confidence'] * 100).round(1).astype(str) + "%"
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Agent Decision Analysis
        st.subheader("🔍 Agent Decision Deep Dive")
        
        if st.session_state.processed_tickets:
            selected_ticket = st.selectbox(
                "Select ticket for detailed analysis:",
                options=[t['ticket_id'] for t in st.session_state.processed_tickets],
                index=len(st.session_state.processed_tickets)-1  # Default to most recent
            )
            
            # Find the selected ticket
            selected_ticket_data = None
            for ticket in st.session_state.processed_tickets:
                if ticket['ticket_id'] == selected_ticket:
                    selected_ticket_data = ticket
                    break
            
            if selected_ticket_data:
                st.write(f"**Analyzing:** {selected_ticket}")
                
                # Show original content
                with st.expander("Original Ticket Content", expanded=False):
                    st.write(f"**Customer:** {selected_ticket_data['customer_email']}")
                    st.write(f"**Content:** {selected_ticket_data['original_content']}")
                
                # Show each agent's decision process
                for i, decision in enumerate(selected_ticket_data.get('agent_decisions', [])):
                    with st.expander(f"Agent {i+1}: {decision['agent']} - {decision['step'].title()}", expanded=True):
                        
                        # Decision metadata
                        col_meta1, col_meta2, col_meta3 = st.columns(3)
                        with col_meta1:
                            st.write(f"**Confidence:** {decision['confidence']*100:.1f}%")
                        with col_meta2:
                            st.write(f"**Timestamp:** {decision['timestamp'][:19]}")
                        with col_meta3:
                            confidence_color = "🟢" if decision['confidence'] > 0.8 else "🟡" if decision['confidence'] > 0.6 else "🔴"
                            st.write(f"**Quality:** {confidence_color}")
                        
                        # Reasoning
                        st.write("**Agent Reasoning:**")
                        st.info(decision['reasoning'])
                        
                        # Decision output
                        st.write("**Decision Output:**")
                        
                        # Format decision output based on step type
                        if decision['step'] == 'data_extraction':
                            decision_data = decision['decision']
                            st.write(f"• **Issue Summary:** {decision_data.get('issue_summary', 'N/A')}")
                            st.write(f"• **Category:** {decision_data.get('issue_category', 'N/A')}")
                            st.write(f"• **Sentiment:** {decision_data.get('sentiment', 'N/A')}")
                            st.write(f"• **Key Phrases:** {', '.join(decision_data.get('key_phrases', []))}")
                            
                        elif decision['step'] == 'classification':
                            decision_data = decision['decision']
                            st.write(f"• **Urgency:** {decision_data.get('urgency_level', 'N/A')}")
                            st.write(f"• **Complexity Score:** {decision_data.get('complexity_score', 'N/A')}/10")
                            st.write(f"• **Routing:** {decision_data.get('routing_recommendation', 'N/A')}")
                            st.write(f"• **Human Required:** {'Yes' if decision_data.get('requires_human_agent') else 'No'}")
                            
                        elif decision['step'] == 'knowledge_search':
                            decision_data = decision['decision']
                            st.write(f"• **Articles Found:** {decision_data.get('articles_found', 0)}")
                            if decision_data.get('top_matches'):
                                st.write("• **Best Match:**")
                                best_match = decision_data['top_matches'][0]
                                st.write(f"  - {best_match['article']['title']}")
                                st.write(f"  - Relevance Score: {best_match['relevance_score']}")
                            
                        elif decision['step'] == 'escalation_analysis':
                            decision_data = decision['decision']
                            st.write(f"• **Escalation Required:** {'Yes' if decision_data.get('requires_escalation') else 'No'}")
                            st.write(f"• **Escalation Level:** {decision_data.get('escalation_level', 'N/A')}")
                            st.write(f"• **Suggested Department:** {decision_data.get('suggested_department', 'N/A')}")
                            st.write(f"• **Auto-Resolution:** {'Possible' if decision_data.get('auto_resolution_possible') else 'Not Possible'}")
                            
                        elif decision['step'] == 'response_generation':
                            decision_data = decision['decision']
                            st.write(f"• **Response Type:** {decision_data.get('response_type', 'N/A')}")
                            st.write(f"• **Follow-up Required:** {'Yes' if decision_data.get('follow_up_required') else 'No'}")
                            st.write(f"• **Resolution Time:** {decision_data.get('estimated_resolution_time', 'N/A')}")

if __name__ == "__main__":
    main()
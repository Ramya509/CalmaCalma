__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
import tempfile
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import json

# Import CrewAI components
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

# Environment setup - Set your API keys here
os.environ["SERPER_API_KEY"] = "59b49b33b3013d2294c8ed1e740afdc2e33c592d"
os.environ["GEMINI_API_KEY"] = "AIzaSyBNDeEt_mq7ZM_QA3L_7ScjoNjOSw794LA"

# Define Pydantic model for structured output
class MentalHealthAssessment(BaseModel):
    phq9_score: int = Field(description="Total PHQ-9 score (0-27)")
    severity: str = Field(description="Depression severity (None, Mild, Moderate, Moderately Severe, Severe)")
    key_concerns: List[str] = Field(description="List of identified mental health concerns")
    recommendations: List[str] = Field(description="List of personalized recommendations")
    resources: List[str] = Field(description="Helpful resources for the user")
    follow_up_needed: bool = Field(description="Whether professional follow-up is recommended")

# Initialize search tool
search_tool = SerperDevTool()

# Set up LLM
my_llm = LLM(
    model='gemini/gemini-2.0-flash',
    api_key=os.environ.get("GEMINI_API_KEY")
)

def create_agents_and_tasks(conversation_history):
    """Create agents and tasks based on current conversation state"""
    
    # Define knowledge sources
    knowledge_sources = []
    
    # Try to add default knowledge sources if they exist
    try:
        default_sources = ["PHQ-9.pdf"]
        existing_sources = [s for s in default_sources if os.path.exists(s)]
        if existing_sources:
            pdf_source = PDFKnowledgeSource(file_paths=existing_sources)
            knowledge_sources.append(pdf_source)
    except Exception as e:
        st.error(f"Error loading default knowledge sources: {e}")
    
    # Add any uploaded resources
    if 'uploaded_resources' in st.session_state and len(st.session_state.uploaded_resources) > 0:
        for file_path in st.session_state.uploaded_resources:
            try:
                if file_path.endswith('.pdf'):
                    pdf_source = PDFKnowledgeSource(file_paths=[file_path])
                    knowledge_sources.append(pdf_source)
                elif file_path.endswith('.txt'):
                    text_source = TextFileKnowledgeSource(file_paths=[file_path])
                    knowledge_sources.append(text_source)
            except Exception as e:
                st.error(f"Error loading uploaded resource {file_path}: {e}")
    
    # Convert conversation history to a format the agents can use
    conversation_context = ""
    for message in conversation_history:
        role = "User" if message["role"] == "user" else message["agent"]
        conversation_context += f"{role}: {message['content']}\n\n"
    
    # Define agents
    questioning_agent = Agent(
        role="Mental Health Interviewer",
        goal="Conduct thorough and empathetic mental health screenings and interviews",
        backstory="""You are a compassionate interviewer trained in mental health assessments. 
        You know how to ask questions sensitively to understand a person's emotional state.
        You guide users through the PHQ-9 depression screening and follow-up with relevant
        contextual questions. You adapt to their responses and tailor your questions accordingly.
        You are skilled at making people feel comfortable sharing their feelings.""",
        tools=[search_tool],
        knowledge_sources=knowledge_sources,
        verbose=True,
        allow_delegation=True,
        llm=my_llm
    )

    analyzer_agent = Agent(
        role="Mental Health Analyzer",
        goal="Accurately evaluate mental health states based on responses and identify patterns",
        backstory="""You are an expert in psychological analysis with years of experience 
        interpreting mental health assessments. You can identify patterns in responses 
        and determine potential mental health conditions. You're skilled at quantifying 
        PHQ-9 scores and analyzing qualitative responses to form comprehensive evaluations.
        You always consider the full context of a person's situation before drawing conclusions.""",
        tools=[search_tool],
        knowledge_sources=knowledge_sources,
        verbose=True,
        allow_delegation=True,
        llm=my_llm
    )

    advising_agent = Agent(
        role="Mental Health Advisor",
        goal="Provide personalized, evidence-based mental health recommendations",
        backstory="""You are a compassionate mental health advisor with expertise in 
        various therapeutic approaches. You create personalized recommendations based on 
        a person's specific situation and mental health evaluation. You balance professional 
        advice with empathy, ensuring your guidance is both helpful and supportive.
        You always include a mix of immediate coping strategies and longer-term approaches.
        You know when to suggest professional intervention and how to communicate this sensitively.""",
        tools=[search_tool],
        knowledge_sources=knowledge_sources,
        verbose=True,
        allow_delegation=True,
        llm=my_llm
    )

    # Determine the current stage of the conversation
    user_message_count = len([m for m in conversation_history if m["role"] == "user"])
    
    # Initial conversation or screening phase
    if user_message_count <= 10:
        task = Task(
            description=f"""Based on the conversation so far: 

{conversation_context}

Conduct a personalized PHQ-9 depression screening assessment with the user. 
If you're just starting, introduce the PHQ-9 screening and ask the first question.
If you're in the middle of the screening, acknowledge their previous response and ask the next appropriate PHQ-9 question.
Tailor your questions based on their previous answers.
Be empathetic and sensitive in your questioning.
Don't use generic or repetitive responses.
Each question should be personalized and contextual to their situation.
Don't remind them of scoring at every question - only mention it if it seems they need guidance.
Your goal is to make this feel like a natural conversation, not a robotic questionnaire.""",
            expected_output="The next personalized question in the PHQ-9 assessment that feels natural and conversational.",
            agent=questioning_agent
        )
        
        crew = Crew(
            agents=[questioning_agent],
            tasks=[task],
            verbose=True,
            manager_llm=my_llm
        )
        
        return crew, "interviewing"
        
    # Analysis phase
    elif user_message_count <= 15:
        analysis_task = Task(
            description=f"""Based on the conversation so far: 

{conversation_context}

You're now in the follow-up phase after the PHQ-9 screening. Ask 3-5 follow-up questions 
that are deeply personalized to the user's specific situation and responses so far.
Focus on understanding their specific context, challenges, and resources.
Each question should build on their previous answers and help create a comprehensive 
picture of their mental health situation.
Be warm, empathetic, and conversational - avoid clinical or generic questions.
If you've already asked enough follow-up questions, let them know you'll now analyze their responses.""",
            expected_output="A thoughtful, personalized follow-up question or a transition to the analysis phase.",
            agent=questioning_agent
        )
        
        crew = Crew(
            agents=[questioning_agent],
            tasks=[analysis_task],
            verbose=True,
            manager_llm=my_llm
        )
        
        return crew, "follow_up"
        
    # Analysis is complete, now provide analysis results
    elif user_message_count == 16:
        analysis_task = Task(
            description=f"""Based on the full conversation so far: 

{conversation_context}

Analyze all the user's responses comprehensively. Calculate a PHQ-9 score based on their 
responses to the depression screening questions. Identify key patterns, concerns, and 
potential mental health conditions evident in their responses.

Your analysis should be thoughtful, personalized, and consider the full context of the 
conversation, not just the screening answers. Identify specific symptoms, triggers, 
and patterns that emerged during your conversation with them.

Explain what their PHQ-9 score means in terms of depression severity:
0-4: None/Minimal depression
5-9: Mild depression
10-14: Moderate depression
15-19: Moderately Severe depression
20-27: Severe depression

End by letting them know you'll now pass this analysis to the Advisor who will provide recommendations.""",
            expected_output="A comprehensive, personalized analysis of the user's mental health state.",
            agent=analyzer_agent
        )
        
        crew = Crew(
            agents=[analyzer_agent],
            tasks=[analysis_task],
            verbose=True,
            manager_llm=my_llm
        )
        
        return crew, "analyzing"
    
    # Recommendation phase
    else:
        recommendation_task = Task(
            description=f"""Based on the full conversation so far: 

{conversation_context}

Create highly personalized mental health recommendations for this specific user.
Your recommendations should directly address the concerns, symptoms, and context they've shared.
Include specific actionable advice, helpful resources, and self-care strategies.
If their severity level indicates professional help is needed, clearly state this in a supportive way.

Ensure recommendations are evidence-based and realistic for them to implement based on what 
you know about their situation.
Include a mix of immediate coping strategies and longer-term approaches.
Provide specific resources like hotlines, websites, books, or apps that might help.

Your recommendations should be compassionate, non-judgmental, and hope-focused.""",
            expected_output="Comprehensive, personalized recommendations including actionable advice, resources, and clear guidance.",
            agent=advising_agent,
            output_pydantic=MentalHealthAssessment
        )
        
        crew = Crew(
            agents=[advising_agent],
            tasks=[recommendation_task],
            verbose=True,
            manager_llm=my_llm
        )
        
        return crew, "recommending"

def process_user_input(user_input, conversation_history):
    """Process user input using actual CrewAI implementation"""
    
    try:
        # Create appropriate agents and tasks based on conversation stage
        crew, stage = create_agents_and_tasks(conversation_history)
        
        # Run the crew with the user input as context
        result = crew.kickoff(inputs={"user_input": user_input})
        
        # Format the response based on the stage
        if stage == "interviewing" or stage == "follow_up":
            bot_response = {
                "role": "assistant",
                "agent": "Interviewer",
                "content": result.raw
            }
            return bot_response, False, None
            
        elif stage == "analyzing":
            bot_response = {
                "role": "assistant",
                "agent": "Analyzer",
                "content": result.raw
            }
            return bot_response, False, None
            
        elif stage == "recommending":
            # Extract the structured output
            assessment_result = result.pydantic
            
            bot_response = {
                "role": "assistant",
                "agent": "Advisor",
                "content": result.raw
            }
            return bot_response, True, assessment_result
            
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        # Fallback response in case of error
        return {
            "role": "assistant",
            "agent": "System",
            "content": "I apologize, but I encountered an error processing your response. Could you please try again?"
        }, False, None

def main():
    # Set up Streamlit page
    st.set_page_config(
        page_title="Mental Health Chatbot",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # App Header
    st.title("Mental Health Chatbot")
    st.subheader("Your AI-powered mental health companion")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message from the system
        st.session_state.messages.append({
            "role": "assistant", 
            "agent": "Interviewer",
            "content": """Hi there! I'm your mental health interviewer. I'd like to start with a depression screening called PHQ-9, followed by some additional questions to better understand your situation. This will help us provide personalized recommendations for you. Everything you share is confidential. Are you ready to begin?"""
        })
    
    if 'assessment_complete' not in st.session_state:
        st.session_state.assessment_complete = False
    
    if 'assessment_result' not in st.session_state:
        st.session_state.assessment_result = None
        
    if 'uploaded_resources' not in st.session_state:
        st.session_state.uploaded_resources = []

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/mental-health.png", width=150)
        st.header("Upload Resources")
        st.write("Upload mental health books or resources for the AI to learn from.")
        
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name
            
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            
            # Add to resources
            if temp_file_path not in st.session_state.uploaded_resources:
                st.session_state.uploaded_resources.append(temp_file_path)
        
        st.header("About")
        st.write("""Mental Health Chatbot provides an interactive mental health screening, 
                 analysis, and personalized recommendations through AI-powered conversations.""")
        
        st.header("Disclaimer")
        st.warning("""This application is not a substitute for professional medical advice, 
                    diagnosis, or treatment. Always seek the advice of your physician or other 
                    qualified health provider with any questions you may have regarding a medical condition.""")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.write(f"**{message['agent']}**")
            st.write(message["content"])
    
    # Display assessment results if complete
    if st.session_state.assessment_complete and st.session_state.assessment_result:
        result = st.session_state.assessment_result
        with st.container():
            st.success("### Mental Health Assessment Results")
            st.write(f"**PHQ-9 Score:** {result.phq9_score} - {result.severity} Depression")
            
            st.write("**Key Concerns:**")
            for concern in result.key_concerns:
                st.write(f"- {concern}")
            
            st.write("**Recommendations:**")
            for rec in result.recommendations:
                st.write(f"- {rec}")
            
            st.write("**Helpful Resources:**")
            for resource in result.resources:
                st.write(f"- {resource}")
            
            if result.follow_up_needed:
                st.info("**Follow-up with a professional is recommended.**")

    # User input area
    if not st.session_state.assessment_complete:
        user_input = st.chat_input("Type your response here...")
        if user_input:
            # Add user message to chat history
            with st.chat_message("user"):
                st.write(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Processing your response..."):
                # Process the user input with real CrewAI implementation
                bot_response, assessment_complete, assessment_result = process_user_input(
                    user_input, st.session_state.messages
                )
                
                # Add agent response to chat history
                st.session_state.messages.append(bot_response)
                
                # Update assessment status if completed
                if assessment_complete:
                    st.session_state.assessment_complete = True
                    st.session_state.assessment_result = assessment_result
            
            # Show agent response
            with st.chat_message("assistant"):
                st.write(f"**{bot_response['agent']}**")
                st.write(bot_response['content'])
            
            # Don't rerun - let the next interaction happen naturally
    else:
        # Assessment is complete, show reset button
        if st.button("Start New Assessment"):
            st.session_state.messages = []
            st.session_state.assessment_complete = False
            st.session_state.assessment_result = None
            
            # Add welcome message for new assessment
            st.session_state.messages.append({
                "role": "assistant", 
                "agent": "Interviewer",
                "content": """Hi there! I'm your mental health interviewer. I'd like to start with a depression screening called PHQ-9, followed by some additional questions to better understand your situation. This will help us provide personalized recommendations for you. Everything you share is confidential. Are you ready to begin?"""
            })
            
            st.rerun()

if __name__ == "__main__":
    main()

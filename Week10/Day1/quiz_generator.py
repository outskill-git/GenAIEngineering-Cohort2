import json
import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ValidationError
from chatbot_openrouter import chat_openrouter
from chatbot_models import Quiz, QuizQuestion, insert_quiz, insert_question, get_quiz, get_quiz_questions, list_quizzes


class QuizGenerationResponse(BaseModel):
    quiz: str
    questions: List[QuizQuestion]


def generate_quiz(topic: str, model: str = "openrouter/cypher-alpha:free", temperature: float = 0.7) -> tuple[Optional[Quiz], Optional[str], Optional[str]]:
    """
    Generate a quiz using LLM with structured responses.
    
    Args:
        topic: The topic for the quiz
        model: The LLM model to use
        temperature: Temperature for generation
        
    Returns:
        tuple: (Quiz object, raw JSON string, error message)
    """
    
    system_prompt = """You are a quiz generator. Create a 5-question multiple choice quiz on the given topic.

You must respond with a JSON object in exactly this format:
{
  "quiz": "Quiz title here",
  "questions": [
    {
      "id": 1,
      "question": "Question text here",
      "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
      "correct_answer": 0
    }
  ]
}

Rules:
- Each question should have exactly 4 options
- correct_answer should be 0, 1, 2, or 3 (index of the correct option)
- Make questions challenging but fair
- Ensure questions are relevant to the topic
- Use clear, unambiguous language
- Make sure only one option is clearly correct

ONLY respond with the JSON object, no additional text."""

    user_prompt = f"Generate a 5-question multiple choice quiz on the topic: {topic}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response, metadata = chat_openrouter(messages, model=model, temperature=temperature)
        
        if not response:
            return None, None, "No response from LLM"
            
        # Parse JSON response
        try:
            quiz_data = json.loads(response)
            
            # Validate the structure
            quiz_response = QuizGenerationResponse(**quiz_data)
            
            # Create Quiz object
            quiz = Quiz(
                quiz=quiz_response.quiz,
                questions=quiz_response.questions
            )
            
            return quiz, response, None
            
        except json.JSONDecodeError as e:
            return None, response, f"Failed to parse JSON: {str(e)}"
        except ValidationError as e:
            return None, response, f"Invalid quiz structure: {str(e)}"
        except Exception as e:
            return None, response, f"Unexpected error: {str(e)}"
            
    except Exception as e:
        return None, None, f"API call failed: {str(e)}"


def save_quiz_to_db(quiz: Quiz, topic: str, raw_json: str) -> int:
    """
    Save a quiz to the database.
    
    Args:
        quiz: The Quiz object
        topic: The topic of the quiz
        raw_json: The raw JSON response from LLM
        
    Returns:
        int: The quiz ID
    """
    
    # Insert quiz
    quiz_id = insert_quiz(quiz.quiz, topic, raw_json)
    
    # Insert questions
    for i, question in enumerate(quiz.questions):
        insert_question(
            quiz_id=quiz_id,
            question_text=question.question,
            option_1=question.options[0],
            option_2=question.options[1],
            option_3=question.options[2],
            option_4=question.options[3],
            correct_answer=question.correct_answer,
            question_order=i
        )
    
    return quiz_id


def load_quiz_from_db(quiz_id: int) -> Optional[Quiz]:
    """
    Load a quiz from the database.
    
    Args:
        quiz_id: The ID of the quiz to load
        
    Returns:
        Quiz object or None if not found
    """
    
    quiz_row = get_quiz(quiz_id)
    if not quiz_row:
        return None
        
    questions_rows = get_quiz_questions(quiz_id)
    
    questions = []
    for q in questions_rows:
        questions.append(QuizQuestion(
            id=q["id"],
            question=q["question_text"],
            options=[q["option_1"], q["option_2"], q["option_3"], q["option_4"]],
            correct_answer=q["correct_answer"]
        ))
    
    return Quiz(
        quiz=quiz_row["title"],
        questions=questions
    )


def get_all_quizzes() -> List[Dict[str, Any]]:
    """
    Get all quizzes from the database.
    
    Returns:
        List of quiz dictionaries
    """
    
    quizzes = list_quizzes()
    return [dict(quiz) for quiz in quizzes]
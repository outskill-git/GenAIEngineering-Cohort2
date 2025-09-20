import gradio as gr
import json
from typing import List, Dict, Any, Optional
from chatbot_models import init_db, insert_quiz_attempt, insert_quiz_response
from quiz_generator import generate_quiz, save_quiz_to_db, load_quiz_from_db, get_all_quizzes


def build_quiz_ui():
    """Build the quiz Gradio interface"""
    
    # Initialize database
    init_db()
    
    def get_quiz_options():
        """Get all quizzes for the sidebar"""
        quizzes = get_all_quizzes()
        options = []
        for quiz in quizzes:
            title = quiz["title"]
            created_at = quiz["created_at"]
            options.append([title, created_at, quiz["id"]])
        return options
    
    def create_new_quiz(topic: str, model: str, temperature: float):
        """Create a new quiz"""
        if not topic.strip():
            return "Please enter a topic", None, None, None, None, gr.update(), None
            
        try:
            # Generate quiz
            quiz, raw_json, error = generate_quiz(topic, model, temperature)
            
            if error:
                return f"Error generating quiz: {error}", None, None, None, None, gr.update(), None
            
            if not quiz:
                return "Failed to generate quiz", None, None, None, None, gr.update(), None
                
            # Save to database
            quiz_id = save_quiz_to_db(quiz, topic, raw_json)
            
            # Prepare quiz display
            quiz_html = f"<h2>{quiz.quiz}</h2><p><strong>Topic:</strong> {topic}</p>"
            questions_data = []
            choices_components = []
            
            for i, question in enumerate(quiz.questions):
                questions_data.append({
                    "id": question.id,
                    "question": question.question,
                    "options": question.options,
                    "correct_answer": question.correct_answer
                })
                
                # Create choices for each question
                choices_components.append(question.options)
            
            # Update sidebar
            updated_options = get_quiz_options()
            
            return (
                f"Quiz '{quiz.quiz}' created successfully!",
                f"<h2>{quiz.quiz}</h2><p><strong>Topic:</strong> {topic}</p>",
                json.dumps(questions_data, indent=2),
                json.dumps(quiz.model_dump() if hasattr(quiz, 'model_dump') else quiz.__dict__, indent=2),
                quiz_id,
                updated_options,
                choices_components
            )
            
        except Exception as e:
            return f"Error creating quiz: {str(e)}", None, None, None, None, gr.update(), None
    
    def load_quiz(evt: gr.SelectData, quiz_options):
        """Load a quiz from the sidebar"""
        if not quiz_options or evt.index[0] >= len(quiz_options):
            return "Invalid quiz selection", None, None, None, None, None
            
        quiz_id = quiz_options[evt.index[0]][2]
        
        try:
            # Load quiz from database
            quiz = load_quiz_from_db(quiz_id)
            if not quiz:
                return "Quiz not found", None, None, None, None, None
                
            questions_data = []
            choices_components = []
            
            for i, question in enumerate(quiz.questions):
                questions_data.append({
                    "id": question.id,
                    "question": question.question,
                    "options": question.options,
                    "correct_answer": question.correct_answer
                })
                
                # Create choices for each question
                choices_components.append(question.options)
            
            return (
                f"Quiz '{quiz.quiz}' loaded successfully!",
                f"<h2>{quiz.quiz}</h2>",
                json.dumps(questions_data, indent=2),
                json.dumps(quiz.model_dump() if hasattr(quiz, 'model_dump') else quiz.__dict__, indent=2),
                quiz_id,
                choices_components
            )
            
        except Exception as e:
            return f"Error loading quiz: {str(e)}", None, None, None, None, None
    
    def submit_quiz(quiz_id, *answers):
        """Submit quiz answers and calculate score"""
        if not quiz_id:
            return "No quiz selected", None
            
        try:
            # Load quiz from database
            quiz = load_quiz_from_db(quiz_id)
            if not quiz:
                return "Quiz not found", None
                
            # Calculate score
            score = 0
            total_questions = len(quiz.questions)
            results = []
            
            for i, question in enumerate(quiz.questions):
                if i < len(answers) and answers[i] is not None:
                    selected_answer = question.options.index(answers[i]) if answers[i] in question.options else -1
                    is_correct = selected_answer == question.correct_answer
                    if is_correct:
                        score += 1
                        
                    results.append({
                        "question": question.question,
                        "selected": answers[i],
                        "correct": question.options[question.correct_answer],
                        "is_correct": is_correct
                    })
                else:
                    results.append({
                        "question": question.question,
                        "selected": "No answer",
                        "correct": question.options[question.correct_answer],
                        "is_correct": False
                    })
            
            # Save attempt to database
            attempt_id = insert_quiz_attempt(quiz_id, score, total_questions)
            
            # Save individual responses
            for i, result in enumerate(results):
                question_id = quiz.questions[i].id
                selected_idx = quiz.questions[i].options.index(result["selected"]) if result["selected"] in quiz.questions[i].options else -1
                insert_quiz_response(attempt_id, question_id, selected_idx, result["is_correct"])
            
            # Create results display
            results_html = f"<h2>Quiz Results</h2>"
            results_html += f"<p><strong>Score: {score}/{total_questions} ({score/total_questions*100:.1f}%)</strong></p>"
            
            for i, result in enumerate(results):
                status = "✅ Correct" if result["is_correct"] else "❌ Incorrect"
                results_html += f"<div style='margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>"
                results_html += f"<p><strong>Question {i+1}:</strong> {result['question']}</p>"
                results_html += f"<p><strong>Your answer:</strong> {result['selected']}</p>"
                results_html += f"<p><strong>Correct answer:</strong> {result['correct']}</p>"
                results_html += f"<p><strong>Status:</strong> {status}</p>"
                results_html += f"</div>"
            
            return f"Quiz completed! Score: {score}/{total_questions}", results_html
            
        except Exception as e:
            return f"Error submitting quiz: {str(e)}", None
    
    def show_new_quiz_modal():
        """Show the new quiz modal"""
        return gr.update(visible=True)
    
    def hide_new_quiz_modal():
        """Hide the new quiz modal"""
        return gr.update(visible=False)
    
    def update_quiz_display(quiz_html, choices_data, questions_data_str):
        """Update the quiz display with new data"""
        if not quiz_html:
            return (gr.update(visible=False), gr.update(), 
                   *[gr.update(visible=False) for _ in range(5)],  # question_blocks
                   *[gr.update() for _ in range(5)],              # question_htmls  
                   *[gr.update(visible=False) for _ in range(5)])  # answer_choices
            
        # Parse questions data
        try:
            questions_data = json.loads(questions_data_str) if questions_data_str else []
        except:
            questions_data = []
            
        # Update question blocks, htmls and choices
        block_updates = []
        html_updates = []
        choice_updates = []
        
        for i in range(5):
            if choices_data and i < len(choices_data) and i < len(questions_data):
                question_html = f"<h3>Question {i+1}</h3><p>{questions_data[i]['question']}</p>"
                block_updates.append(gr.update(visible=True))
                html_updates.append(gr.update(value=question_html))
                choice_updates.append(gr.update(
                    choices=choices_data[i],
                    visible=True,
                    value=None
                ))
            else:
                block_updates.append(gr.update(visible=False))
                html_updates.append(gr.update())
                choice_updates.append(gr.update(visible=False))
        
        return (gr.update(visible=True), gr.update(value=quiz_html), 
               *block_updates, *html_updates, *choice_updates)
    
    with gr.Blocks(title="Quiz App", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 Quiz Generator & Taker")
        
        with gr.Row():
            # Sidebar
            with gr.Column(scale=1):
                gr.Markdown("## Quiz History")
                new_quiz_btn = gr.Button("📝 New Quiz", variant="primary", size="lg")
                quiz_sidebar = gr.Dataframe(
                    headers=["Title", "Created", "ID"],
                    datatype=["str", "str", "str"],
                    visible=True,
                    interactive=True,
                    wrap=True,
                    label="Quiz History"
                )
                
            # Main content
            with gr.Column(scale=3):
                status_msg = gr.Markdown("Welcome! Click 'New Quiz' to create a quiz or select an existing one from the sidebar.")
                
                # Quiz display section
                with gr.Group(visible=False) as quiz_display:
                    quiz_header = gr.HTML()
                    
                    # Create individual question blocks
                    question_blocks = []
                    question_htmls = []
                    answer_choices = []
                    for i in range(5):  # Maximum 5 questions
                        with gr.Group(visible=False) as question_group:
                            question_html = gr.HTML()
                            choice = gr.Radio(
                                label="Choose your answer:",
                                visible=True,
                                interactive=True
                            )
                            answer_choices.append(choice)
                            question_blocks.append(question_group)
                            question_htmls.append(question_html)
                    
                    with gr.Row():
                        submit_btn = gr.Button("📊 Submit Quiz", variant="primary", size="lg")
                        clear_btn = gr.Button("🔄 Clear Answers", variant="secondary")
                    
                    # Results section
                    results_display = gr.HTML()
                    
                    # JSON display
                    with gr.Accordion("Show JSON Response", open=False):
                        json_display = gr.JSON(label="LLM Response")
        
        # New Quiz Modal
        with gr.Row(visible=False) as new_quiz_modal:
            with gr.Column():
                gr.Markdown("## Create New Quiz")
                topic_input = gr.Textbox(
                    label="Quiz Topic",
                    placeholder="Gen AI Engineering",
                    value="Gen AI Engineering"
                )
                with gr.Row():
                    model_select = gr.Dropdown(
                        choices=[
                            "openrouter/cypher-alpha:free",
                            "openai/gpt-4.1",
                            "openai/gpt-4.1-mini",
                            "openai/gpt-4.1-nano",
                            "anthropic/claude-sonnet-4",
                            "anthropic/claude-3.5-sonnet"
                        ],
                        value="openrouter/cypher-alpha:free",
                        label="Model"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                with gr.Row():
                    generate_btn = gr.Button("🎯 Generate Quiz", variant="primary")
                    cancel_btn = gr.Button("❌ Cancel", variant="secondary")
        
        # Hidden states
        current_quiz_id = gr.State()
        quiz_questions_data = gr.State()
        quiz_sidebar_data = gr.State()
        quiz_choices_data = gr.State()
        
        # Load initial sidebar data
        demo.load(lambda: get_quiz_options(), outputs=[quiz_sidebar])
        
        # Event handlers
        new_quiz_btn.click(
            show_new_quiz_modal,
            outputs=[new_quiz_modal]
        )
        
        cancel_btn.click(
            hide_new_quiz_modal,
            outputs=[new_quiz_modal]
        )
        
        generate_btn.click(
            create_new_quiz,
            inputs=[topic_input, model_select, temperature_slider],
            outputs=[status_msg, quiz_header, quiz_questions_data, json_display, current_quiz_id, quiz_sidebar, quiz_choices_data]
        ).then(
            hide_new_quiz_modal,
            outputs=[new_quiz_modal]
        ).then(
            update_quiz_display,
            inputs=[quiz_header, quiz_choices_data, quiz_questions_data],
            outputs=[quiz_display, quiz_header] + question_blocks + question_htmls + answer_choices
        )
        
        quiz_sidebar.select(
            load_quiz,
            inputs=[quiz_sidebar_data],
            outputs=[status_msg, quiz_header, quiz_questions_data, json_display, current_quiz_id, quiz_choices_data]
        ).then(
            update_quiz_display,
            inputs=[quiz_header, quiz_choices_data, quiz_questions_data],
            outputs=[quiz_display, quiz_header] + question_blocks + question_htmls + answer_choices
        )
        
        submit_btn.click(
            submit_quiz,
            inputs=[current_quiz_id] + answer_choices,
            outputs=[status_msg, results_display]
        )
        
        clear_btn.click(
            lambda: [None] * 5,
            outputs=answer_choices
        )
        
        # Update sidebar data state when sidebar updates
        quiz_sidebar.change(
            lambda x: x,
            inputs=[quiz_sidebar],
            outputs=[quiz_sidebar_data]
        )
    
    return demo


if __name__ == "__main__":
    demo = build_quiz_ui()
    demo.launch()
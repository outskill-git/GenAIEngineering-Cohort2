"""
Dynamic Interactive Story Generator using LangGraph + GPT-4o + Streamlit
Generates completely dynamic stories with AI-powered narrative and choices
"""

import streamlit as st
import json
import openai
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
import requests
from datetime import datetime
import re

# Configure page
st.set_page_config(
    page_title="🌟 AI Story Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Story State Definition
class StoryState(TypedDict):
    story_id: str
    character_name: str
    story_theme: str
    story_genre: str
    current_scene: str
    story_context: str
    character_traits: List[str]
    inventory: List[str]
    relationships: Dict[str, str]
    choices_made: List[Dict[str, Any]]
    story_progression: int
    current_narrative: str
    available_choices: List[Dict[str, str]]
    story_complete: bool
    mood: str
    world_state: Dict[str, Any]

class AIStoryGenerator:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        """Initialize with OpenRouter API credentials"""
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = "openai/gpt-4o"
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for dynamic story generation"""
        workflow = StateGraph(StoryState)
        
        # Add nodes for different story phases
        workflow.add_node("initialize_story", self._initialize_story)
        workflow.add_node("generate_scene", self._generate_scene)
        workflow.add_node("process_choice", self._process_choice)
        workflow.add_node("update_world_state", self._update_world_state)
        workflow.add_node("generate_ending", self._generate_ending)
        
        # Define the flow - simplified to prevent recursion
        workflow.set_entry_point("initialize_story")
        workflow.add_edge("initialize_story", "generate_scene")
        
        # Generate scene leads directly to END (waiting for user input)
        workflow.add_edge("generate_scene", END)
        
        # These will be called separately when user makes a choice
        workflow.add_edge("process_choice", "update_world_state")
        workflow.add_edge("update_world_state", "generate_scene")
        workflow.add_edge("generate_ending", END)
        
        return workflow.compile()
    
    def _call_gpt4o(self, prompt: str, system_prompt: str = None) -> str:
        """Make API call to GPT-4o via OpenRouter"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error calling GPT-4o: {str(e)}")
            return "Error generating content. Please check your API key and try again."
    
    def _extract_json_from_response(self, response: str) -> dict:
        """Extract JSON from response, handling cases where JSON is embedded in text"""
        try:
            # First try to parse the entire response as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON within the response using regex
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            
            # If no valid JSON found, return None
            return None
    
    def _initialize_story(self, state: StoryState) -> StoryState:
        """Initialize story with AI-generated opening"""
        system_prompt = f"""You are a master storyteller creating an interactive {state['story_genre']} story with the theme: {state['story_theme']}.

Create an engaging opening for a character named {state['character_name']}.

IMPORTANT: Your response must be valid JSON in this exact format:
{{
    "narrative": "The opening story text with rich descriptions and atmosphere",
    "context": "Brief summary of the current situation",
    "character_traits": ["trait1", "trait2", "trait3"],
    "inventory": ["item1", "item2", "item3"],
    "world_state": {{"location": "starting_location", "time_of_day": "time", "weather": "weather_condition"}},
    "mood": "current_story_mood"
}}

Make the opening compelling and set up the adventure. Include vivid descriptions and create intrigue."""
        
        prompt = f"""Create an opening scene for an interactive story:
- Character: {state['character_name']}
- Genre: {state['story_genre']}
- Theme: {state['story_theme']}

Generate an engaging opening that introduces the character and sets up the adventure."""
        
        response = self._call_gpt4o(prompt, system_prompt)
        
        # Extract JSON from response
        story_data = self._extract_json_from_response(response)
        
        if story_data:
            state.update({
                "current_narrative": story_data.get("narrative", "Your adventure begins..."),
                "story_context": story_data.get("context", "A new adventure unfolds."),
                "character_traits": story_data.get("character_traits", ["brave", "curious"]),
                "inventory": story_data.get("inventory", ["basic supplies"]),
                "world_state": story_data.get("world_state", {}),
                "mood": story_data.get("mood", "mysterious"),
                "current_scene": "opening",
                "story_progression": 1
            })
        else:
            # Enhanced fallback with readable narrative
            fallback_narrative = f"""The story of {state['character_name']} begins in a world of {state['story_genre'].lower()}, where {state['story_theme'].lower()} awaits.

As dawn breaks over an unfamiliar landscape, {state['character_name']} finds themselves at the threshold of an extraordinary adventure. The air is thick with possibility, and every shadow seems to whisper of secrets yet to be discovered.

What path will {state['character_name']} choose in this tale of wonder and discovery?"""
            
            state.update({
                "current_narrative": fallback_narrative,
                "story_context": f"Beginning adventure for {state['character_name']}",
                "character_traits": ["brave", "curious", "determined"],
                "inventory": ["traveler's pack", "water flask", "mysterious compass"],
                "world_state": {"location": "mysterious crossroads", "time": "dawn", "weather": "misty"},
                "mood": "mysterious",
                "current_scene": "opening",
                "story_progression": 1
            })
        
        return state
    
    def _generate_scene(self, state: StoryState) -> StoryState:
        """Generate new scene content with AI"""
        
        # Build context from previous choices
        choice_history = ""
        if state["choices_made"]:
            recent_choices = state["choices_made"][-3:]  # Last 3 choices
            choice_history = "Recent player choices: " + "; ".join([f"{c['choice_text']}" for c in recent_choices])
        
        system_prompt = f"""You are continuing an interactive {state['story_genre']} story with theme: {state['story_theme']}.

CONTEXT:
- Character: {state['character_name']}
- Traits: {', '.join(state['character_traits'])}
- Inventory: {', '.join(state['inventory'])}
- Current mood: {state['mood']}
- World state: {json.dumps(state['world_state'])}
- Story progression: {state['story_progression']}/10
- {choice_history}

Your response MUST be valid JSON in this exact format:
{{
    "narrative": "Compelling scene description continuing the story",
    "choices": [
        {{"id": "choice1", "text": "First choice option", "consequences": "potential outcome"}},
        {{"id": "choice2", "text": "Second choice option", "consequences": "potential outcome"}},
        {{"id": "choice3", "text": "Third choice option", "consequences": "potential outcome"}}
    ],
    "scene_type": "action/dialogue/exploration/puzzle/combat",
    "updated_mood": "current emotional tone"
}}

Create meaningful choices that impact the story. Each choice should feel significant."""
        
        prompt = f"""Continue the story from: {state['story_context']}

Current situation: {state.get('current_narrative', '')[-500:]}  # Last 500 chars

Generate the next scene with 3 meaningful choices for the player."""
        
        response = self._call_gpt4o(prompt, system_prompt)
        
        # Extract JSON from response
        scene_data = self._extract_json_from_response(response)
        
        if scene_data:
            state.update({
                "current_narrative": scene_data.get("narrative", "The story continues..."),
                "available_choices": scene_data.get("choices", [
                    {"id": "continue", "text": "Continue forward", "consequences": "unknown"}
                ]),
                "current_scene": scene_data.get("scene_type", "exploration"),
                "mood": scene_data.get("updated_mood", state["mood"])
            })
        else:
            # Enhanced fallback with readable content
            fallback_narrative = f"""The adventure continues as {state['character_name']} faces new challenges. The path ahead is uncertain, but determination guides each step forward.

What will {state['character_name']} do next in this unfolding tale?"""
            
            state.update({
                "current_narrative": fallback_narrative,
                "available_choices": [
                    {"id": "continue", "text": "Continue the adventure", "consequences": "Move forward with courage"},
                    {"id": "explore", "text": "Explore the area thoroughly", "consequences": "Discover hidden secrets"},
                    {"id": "rest", "text": "Take a moment to rest and think", "consequences": "Regain strength and clarity"}
                ],
                "current_scene": "exploration"
            })
        
        return state
    
    def _process_choice(self, state: StoryState) -> StoryState:
        """Process the user's choice with AI interpretation"""
        if not state.get("pending_choice"):
            return state
        
        choice_id = state["pending_choice"]
        chosen_option = None
        
        for choice in state["available_choices"]:
            if choice["id"] == choice_id:
                chosen_option = choice
                break
        
        if not chosen_option:
            return state
        
        # Use AI to determine consequences
        system_prompt = f"""You are processing a player choice in an interactive {state['story_genre']} story.

CONTEXT:
- Character: {state['character_name']} with traits: {', '.join(state['character_traits'])}
- Current inventory: {', '.join(state['inventory'])}
- Player chose: {chosen_option['text']}
- Potential consequences: {chosen_option.get('consequences', 'unknown')}

Your response MUST be valid JSON:
{{
    "consequence_narrative": "What happens as a result of this choice",
    "trait_changes": ["new_trait1", "new_trait2"],
    "inventory_changes": {{"add": ["item1"], "remove": ["item2"]}},
    "relationship_updates": {{"character_name": "relationship_status"}},
    "context_update": "Brief summary of new situation"
}}

Make consequences meaningful and logical."""
        
        prompt = f"""Process this choice: {chosen_option['text']}
Current story context: {state['story_context']}
Determine realistic consequences."""
        
        response = self._call_gpt4o(prompt, system_prompt)
        
        # Extract JSON from response
        consequence_data = self._extract_json_from_response(response)
        
        if consequence_data:
            # Update character traits
            new_traits = consequence_data.get("trait_changes", [])
            for trait in new_traits:
                if trait not in state["character_traits"]:
                    state["character_traits"].append(trait)
            
            # Update inventory
            inventory_changes = consequence_data.get("inventory_changes", {})
            for item in inventory_changes.get("add", []):
                if item not in state["inventory"]:
                    state["inventory"].append(item)
            for item in inventory_changes.get("remove", []):
                if item in state["inventory"]:
                    state["inventory"].remove(item)
            
            # Update relationships
            relationships = consequence_data.get("relationship_updates", {})
            state["relationships"].update(relationships)
            
            # Add choice to history
            state["choices_made"].append({
                "choice_id": choice_id,
                "choice_text": chosen_option["text"],
                "consequence": consequence_data.get("consequence_narrative", ""),
                "scene": state["current_scene"]
            })
            
            # Update context
            state["story_context"] = consequence_data.get("context_update", state["story_context"])
            
        else:
            # Fallback processing with readable consequence
            consequence_text = f"{state['character_name']}'s choice to {chosen_option['text'].lower()} leads to new developments in the story."
            
            state["choices_made"].append({
                "choice_id": choice_id,
                "choice_text": chosen_option["text"],
                "consequence": consequence_text,
                "scene": state["current_scene"]
            })
        
        # Clear pending choice
        state["pending_choice"] = None
        state["story_progression"] += 1
        
        return state
    
    def _update_world_state(self, state: StoryState) -> StoryState:
        """Update world state based on story progression"""
        system_prompt = f"""Update the world state for this {state['story_genre']} story.

CONTEXT:
- Current world state: {json.dumps(state['world_state'])}
- Recent choice: {state['choices_made'][-1]['choice_text'] if state['choices_made'] else 'None'}
- Story progression: {state['story_progression']}/10

Your response MUST be valid JSON:
{{
    "world_state": {{"location": "current_location", "time_of_day": "time", "weather": "condition", "npcs_present": ["npc1"], "threats": ["threat1"]}},
    "story_context": "Updated situation summary"
}}"""
        
        prompt = "Update the world state based on recent events and story progression."
        
        response = self._call_gpt4o(prompt, system_prompt)
        
        # Extract JSON from response
        update_data = self._extract_json_from_response(response)
        
        if update_data:
            state["world_state"].update(update_data.get("world_state", {}))
            state["story_context"] = update_data.get("story_context", state["story_context"])
        
        return state
    
    def _check_story_completion(self, state: StoryState) -> StoryState:
        """Check if story should end"""
        if state["story_progression"] >= 8:  # Story ends after 8 scenes
            state["story_complete"] = True
        return state
    
    def start_new_story(self, character_name: str, story_theme: str, story_genre: str) -> StoryState:
        """Start a new story using the graph"""
        initial_state = StoryState(
            story_id=f"story_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            character_name=character_name,
            story_theme=story_theme,
            story_genre=story_genre,
            current_scene="",
            story_context="",
            character_traits=[],
            inventory=[],
            relationships={},
            choices_made=[],
            story_progression=0,
            current_narrative="",
            available_choices=[],
            story_complete=False,
            mood="",
            world_state={}
        )
        
        # Use the graph to initialize and generate first scene
        return self.graph.invoke(initial_state)
    
    def continue_story_after_choice(self, state: StoryState, choice_id: str) -> StoryState:
        """Continue story after user makes a choice - separate method to avoid recursion"""
        # Set the pending choice
        state["pending_choice"] = choice_id
        
        # Check if story should end
        if state["story_progression"] >= 8:
            # Generate ending
            return self._generate_ending(state)
        
        # Process choice -> update world -> generate new scene
        state = self._process_choice(state)
        state = self._update_world_state(state)
        state = self._generate_scene(state)
        
        return state
    
    def _generate_ending(self, state: StoryState) -> StoryState:
        """Generate story ending with AI"""
        system_prompt = f"""Create an epic conclusion for this {state['story_genre']} story.

STORY SUMMARY:
- Character: {state['character_name']}
- Final traits: {', '.join(state['character_traits'])}
- Final inventory: {', '.join(state['inventory'])}
- Relationships: {json.dumps(state['relationships'])}
- Key choices made: {[c['choice_text'] for c in state['choices_made'][-5:]]}

Your response MUST be valid JSON:
{{
    "ending_narrative": "Epic conclusion that ties together the story elements",
    "final_character_state": "What the character has become",
    "story_resolution": "How the main conflicts were resolved",
    "epilogue": "What happens next for the character"
}}

Make it satisfying and reflect the player's choices throughout the journey."""
        
        prompt = f"Create an ending for the adventure of {state['character_name']} based on their journey."
        
        response = self._call_gpt4o(prompt, system_prompt)
        
        # Extract JSON from response
        ending_data = self._extract_json_from_response(response)
        
        if ending_data:
            ending_text = f"""
🎭 **THE END** 🎭

{ending_data.get('ending_narrative', 'Your adventure comes to an end...')}

**Final Character State:**
{ending_data.get('final_character_state', f'{state["character_name"]} has grown from their experiences.')}

**Story Resolution:**
{ending_data.get('story_resolution', 'The conflicts have been resolved.')}

**Epilogue:**
{ending_data.get('epilogue', 'The future holds new adventures...')}

**Your Journey:**
- Scenes Completed: {state['story_progression']}
- Choices Made: {len(state['choices_made'])}
- Final Traits: {', '.join(state['character_traits'])}
- Final Inventory: {', '.join(state['inventory'])}
            """
        else:
            # Fallback ending
            ending_text = f"""
🎭 **THE END** 🎭

The adventure of {state['character_name']} comes to a memorable conclusion. Through {len(state['choices_made'])} pivotal choices, they have evolved into someone truly remarkable.

**Final Character State:**
{state['character_name']} has developed these remarkable traits: {', '.join(state['character_traits'])}

**Your Journey:**
- Scenes Completed: {state['story_progression']}
- Choices Made: {len(state['choices_made'])}
- Final Inventory: {', '.join(state['inventory'])}

The story ends, but the memories of this adventure will live on forever.
            """
        
        state.update({
            "current_narrative": ending_text,
            "available_choices": [],
            "story_complete": True
        })
        
        return state

# Streamlit UI
def main():
    st.title("🌟 AI-Powered Interactive Story Generator")
    st.markdown("*Experience dynamic stories powered by GPT-4o and LangGraph*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Get your API key from https://openrouter.ai/keys"
        )
        
        if not api_key:
            st.warning("Please enter your OpenRouter API key to begin!")
            st.info("👆 Enter your API key above, then configure your story settings and click 'Start New Story'")
            st.stop()
        
        st.success("✅ API Key entered! Configure your story below.")
        
        st.header("📖 Story Settings")
        
        # Story configuration
        character_name = st.text_input("Character Name", value="Hero")
        
        story_genre = st.selectbox(
            "Story Genre",
            ["Fantasy", "Sci-Fi", "Mystery", "Horror", "Adventure", "Romance", "Thriller", "Western", "Cyberpunk", "Steampunk"]
        )
        
        story_theme = st.text_input(
            "Story Theme",
            value="A quest for redemption",
            help="Describe the main theme or conflict"
        )
        
        # New Story button
        new_story = st.button("🚀 Start New Story", type="primary")
        
        # Story progress
        if 'story_state' in st.session_state:
            st.header("📊 Progress")
            progress = st.session_state.story_state.get('story_progression', 0) / 10
            st.progress(progress)
            st.write(f"Scene {st.session_state.story_state.get('story_progression', 0)}/10")
    
    # Main content area
    if not api_key:
        # Show welcome message when no API key
        st.info("🔑 Please enter your OpenRouter API key in the sidebar to begin creating stories!")
        st.markdown("""
        ### Welcome to the AI Story Generator!
        
        This app creates dynamic, interactive stories using GPT-4o and LangGraph. To get started:
        
        1. **Enter your OpenRouter API key** in the sidebar
        2. **Configure your story settings** (character name, genre, theme)
        3. **Click "Start New Story"** to begin your adventure
        
        Your choices will shape the narrative in real-time!
        """)
        return
    
    # Initialize story generator only when API key is provided
    if 'generator' not in st.session_state and api_key:
        st.session_state.generator = AIStoryGenerator(api_key)
    
    # Start new story only when button is pressed
    if new_story:
        if not character_name.strip():
            st.error("Please enter a character name!")
            return
        if not story_theme.strip():
            st.error("Please enter a story theme!")
            return
            
        with st.spinner("🎭 Generating your unique story..."):
            try:
                st.session_state.story_state = st.session_state.generator.start_new_story(
                    character_name, story_theme, story_genre
                )
                st.success("✨ Your story has been generated!")
            except Exception as e:
                st.error(f"Error generating story: {str(e)}")
                return
    
    # Display current story only if it exists
    if 'story_state' in st.session_state:
        story_state = st.session_state.story_state
        
        # Main story area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Story narrative
            st.markdown("### 📚 Current Scene")
            narrative_container = st.container()
            with narrative_container:
                st.markdown(story_state.get('current_narrative', ''))
            
            # Choices
            if story_state.get('available_choices') and not story_state.get('story_complete'):
                st.markdown("### 🤔 What do you choose?")
                
                choice_cols = st.columns(len(story_state['available_choices']))
                
                for i, choice in enumerate(story_state['available_choices']):
                    with choice_cols[i]:
                        if st.button(
                            choice['text'],
                            key=f"choice_{i}",
                            help=choice.get('consequences', 'Unknown consequences'),
                            use_container_width=True
                        ):
                            with st.spinner("🎲 Processing your choice..."):
                                try:
                                    st.session_state.story_state = st.session_state.generator.continue_story_after_choice(
                                        story_state, choice['id']
                                    )
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error processing choice: {str(e)}")
            
            elif story_state.get('story_complete'):
                st.success("🎉 Story Complete!")
                if st.button("📖 Start Another Adventure"):
                    del st.session_state.story_state
                    st.rerun()
        
        with col2:
            # Character info
            st.markdown("### 👤 Character")
            st.write(f"**Name:** {story_state.get('character_name', 'Unknown')}")
            st.write(f"**Mood:** {story_state.get('mood', 'Unknown')}")
            
            # Traits
            if story_state.get('character_traits'):
                st.markdown("**Traits:**")
                for trait in story_state['character_traits']:
                    st.write(f"• {trait}")
            
            # Inventory
            if story_state.get('inventory'):
                st.markdown("**Inventory:**")
                for item in story_state['inventory']:
                    st.write(f"• {item}")
            
            # World state
            if story_state.get('world_state'):
                st.markdown("### 🌍 World State")
                for key, value in story_state['world_state'].items():
                    st.write(f"**{key.title()}:** {value}")
            
            # Choice history
            if story_state.get('choices_made'):
                st.markdown("### 📜 Recent Choices")
                recent_choices = story_state['choices_made'][-3:]
                for choice in reversed(recent_choices):
                    with st.expander(f"Scene {choice.get('scene', 'Unknown')}"):
                        st.write(f"**Choice:** {choice['choice_text']}")
                        if choice.get('consequence'):
                            st.write(f"**Result:** {choice['consequence']}")
    
    else:
        # Show instructions when API key is provided but no story exists
        st.markdown("""
        ### Ready to Create Your Story! ✨
        
        Your API key is configured. Now:
        
        1. **Customize your character name** in the sidebar
        2. **Choose your preferred genre** (Fantasy, Sci-Fi, etc.)
        3. **Set your story theme** (what's the main conflict or goal?)
        4. **Click "Start New Story"** to begin!
        
        Each choice you make will dynamically shape your unique adventure.
        """)

if __name__ == "__main__":
    main()

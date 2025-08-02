
import flet as ft
import flet_ads as ads

# AdMob: Initialize ads
banner_ad = ft.Banner(
    ad_unit_id="ca-app-pub-3940256099942544/6300978111",  # Replace with your real Ad Unit ID
    width=320,
    height=50
)

interstitial_ad = ft.InterstitialAd(
    ad_unit_id="ca-app-pub-3940256099942544/1033173712",  # Replace with your real Ad Unit ID
    on_dismissed=lambda e: print("Interstitial closed"),
)

rewarded_ad = ft.RewardedAd(
    ad_unit_id="ca-app-pub-3940256099942544/5224354917",  # Replace with your real Ad Unit ID
    on_user_earned_reward=lambda e: print("User earned reward!"),
)

def show_interstitial_ad(page):
    interstitial_ad.load()
    interstitial_ad.show()

def show_rewarded_ad(page):
    rewarded_ad.load()
    rewarded_ad.show()

import flet as ft
import json
import requests
import asyncio
import platform
import os
from typing import Dict, Optional, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Use environment variables for security
OPENROUTER_API_KEY = os.getenv("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")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"

# Color scheme
PRIMARY_COLOR = "#1FAB78"
SECONDARY_COLOR = "#F2F2F2"
ACCENT_COLOR = "#E64C4C"

class DataManager:
    """Centralized data management with caching and validation."""
    
    def __init__(self):
        self._proverbs: Optional[Dict[str, str]] = None
        self._dictionary: Optional[Dict[str, str]] = None
        self._cultural_data: Optional[Dict[str, any]] = None
        self._cultural_keywords = {
            'imvunulo', 'izingubo', 'amasiko', 'isiko', 'amasiko neziko', 
            'ubuntu', 'ukusina', 'umgido', 'ukwemula', 'umhlanga', 
            'reed dance', 'traditional', 'culture', 'custom', 'ceremony',
            'wedding', 'umshado', 'funeral', 'umngcwabo', 'initiation',
            'ukwaluka', 'coming of age', 'ancestors', 'amadlozi',
            'traditional dress', 'beadwork', 'amaqhiya', 'izidwaba'
        }
    
    def load_json_file(self, filename: str, description: str) -> Dict[str, str]:
        """Load and validate JSON file with error handling."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                raise ValueError(f"{description} must be a dictionary")
            
            # Validate that all values are strings
            for key, value in data.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError(f"Invalid data format in {description}")
            
            logger.info(f"Successfully loaded {len(data)} entries from {filename}")
            return data
            
        except FileNotFoundError:
            logger.error(f"File {filename} not found")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filename}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return {}
    
    @property
    def proverbs(self) -> Dict[str, str]:
        """Lazy load proverbs with caching."""
        if self._proverbs is None:
            self._proverbs = self.load_json_file('izaga_nezisho.json', 'proverbs')
        return self._proverbs
    
    @property
    def dictionary(self) -> Dict[str, str]:
        """Lazy load dictionary with caching."""
        if self._dictionary is None:
            self._dictionary = self.load_json_file('zulu_dictionary.json', 'dictionary')
        return self._dictionary
    
    @property
    def cultural_data(self) -> Dict[str, any]:
        """Lazy load cultural data with caching."""
        if self._cultural_data is None:
            self._cultural_data = self.load_cultural_data()
        return self._cultural_data
    
    def load_cultural_data(self) -> Dict[str, any]:
        """Load and combine all cultural data files."""
        cultural_files = {
            'imvunulo': 'imvunulo_data.json',
            'amasiko': 'amasiko_data.json', 
            'ceremonies': 'ceremonies_data.json',
            'beadwork': 'beadwork_data.json',
            'ubuntu': 'ubuntu_philosophy.json'
        }
        
        combined_data = {}
        
        for category, filename in cultural_files.items():
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    combined_data[category] = data
                    logger.info(f"Loaded cultural data: {category}")
            except FileNotFoundError:
                logger.warning(f"Cultural data file not found: {filename}")
                combined_data[category] = {}
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {filename}: {e}")
                combined_data[category] = {}
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                combined_data[category] = {}
        
        return combined_data
    
    def detect_cultural_context(self, text: str) -> Dict[str, any]:
        """Detect cultural keywords in text and return relevant context."""
        text_lower = text.lower()
        detected_contexts = {}
        
        # Check for cultural keywords
        for keyword in self._cultural_keywords:
            if keyword in text_lower:
                # Find relevant cultural data
                for category, data in self.cultural_data.items():
                    if self._is_relevant_to_keyword(keyword, category, data):
                        if category not in detected_contexts:
                            detected_contexts[category] = []
                        detected_contexts[category].append({
                            'keyword': keyword,
                            'data': data
                        })
        
        return detected_contexts
    
    def _is_relevant_to_keyword(self, keyword: str, category: str, data: Dict) -> bool:
        """Check if cultural data is relevant to the detected keyword."""
        relevance_map = {
            'imvunulo': ['imvunulo', 'izingubo', 'traditional dress', 'beadwork', 'amaqhiya', 'izidwaba'],
            'amasiko': ['amasiko', 'isiko', 'culture', 'custom', 'ubuntu', 'traditional'],
            'ceremonies': ['umshado', 'wedding', 'umngcwabo', 'funeral', 'umhlanga', 'reed dance', 'ceremony'],
            'beadwork': ['beadwork', 'imvunulo', 'traditional dress'],
            'ubuntu': ['ubuntu', 'philosophy', 'culture', 'ancestors', 'amadlozi']
        }
        
        return keyword in relevance_map.get(category, [])
    
    def format_cultural_context(self, contexts: Dict[str, any]) -> str:
        """Format detected cultural contexts into a comprehensive string."""
        if not contexts:
            return ""
        
        formatted_sections = []
        
        for category, context_list in contexts.items():
            if not context_list:
                continue
                
            section_title = category.replace('_', ' ').title()
            formatted_sections.append(f"\n=== {section_title} ===")
            
            for context in context_list:
                data = context['data']
                formatted_sections.append(self._format_category_data(category, data))
        
        return "\n".join(formatted_sections)
    
    def _format_category_data(self, category: str, data: Dict) -> str:
        """Format specific category data based on its structure."""
        if category == 'imvunulo':
            return self._format_imvunulo_data(data)
        elif category == 'amasiko':
            return self._format_amasiko_data(data)
        elif category == 'ceremonies':
            return self._format_ceremonies_data(data)
        elif category == 'beadwork':
            return self._format_beadwork_data(data)
        elif category == 'ubuntu':
            return self._format_ubuntu_data(data)
        else:
            return str(data)
    
    def _format_imvunulo_data(self, data: Dict) -> str:
        """Format traditional attire data."""
        sections = []
        
        if 'men_attire' in data:
            sections.append("Imvunulo Yamadoda (Men's Traditional Attire):")
            for item, description in data['men_attire'].items():
                sections.append(f"  • {item}: {description}")
        
        if 'women_attire' in data:
            sections.append("\nImvunulo Yabesifazane (Women's Traditional Attire):")
            for item, description in data['women_attire'].items():
                sections.append(f"  • {item}: {description}")
        
        if 'occasions' in data:
            sections.append("\nIzikhathi Zokugqoka (Occasions for Traditional Dress):")
            for occasion, details in data['occasions'].items():
                sections.append(f"  • {occasion}: {details}")
        
        return "\n".join(sections)
    
    def _format_amasiko_data(self, data: Dict) -> str:
        """Format customs and traditions data."""
        sections = []
        
        if 'customs' in data:
            sections.append("Amasiko (Customs):")
            for custom, description in data['customs'].items():
                sections.append(f"  • {custom}: {description}")
        
        if 'values' in data:
            sections.append("\nAmanani (Values):")
            for value, explanation in data['values'].items():
                sections.append(f"  • {value}: {explanation}")
        
        return "\n".join(sections)
    
    def _format_ceremonies_data(self, data: Dict) -> str:
        """Format ceremonies data."""
        sections = []
        
        for ceremony_type, details in data.items():
            sections.append(f"{ceremony_type.replace('_', ' ').title()}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    sections.append(f"  • {key}: {value}")
            else:
                sections.append(f"  • {details}")
        
        return "\n".join(sections)
    
    def _format_beadwork_data(self, data: Dict) -> str:
        """Format beadwork information."""
        sections = []
        
        if 'colors' in data:
            sections.append("Imibala Yamabhidi (Bead Colors and Meanings):")
            for color, meaning in data['colors'].items():
                sections.append(f"  • {color}: {meaning}")
        
        if 'patterns' in data:
            sections.append("\nAmaphethini (Patterns):")
            for pattern, description in data['patterns'].items():
                sections.append(f"  • {pattern}: {description}")
        
        return "\n".join(sections)
    
    def _format_ubuntu_data(self, data: Dict) -> str:
        """Format Ubuntu philosophy data."""
        sections = []
        
        if 'definition' in data:
            sections.append(f"Ubuntu Definition: {data['definition']}")
        
        if 'principles' in data:
            sections.append("\nUbuntu Principles:")
            for principle in data['principles']:
                sections.append(f"  • {principle}")
        
        if 'sayings' in data:
            sections.append("\nUbuntu Sayings:")
            for saying, meaning in data['sayings'].items():
                sections.append(f"  • {saying}: {meaning}")
        
        return "\n".join(sections)

class APIClient:
    """Centralized API client with proper error handling."""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required")
        self.api_key = api_key
        self.base_url = OPENROUTER_API_URL
    
    async def generate_text(self, prompt: str, model: str = "deepseek/deepseek-r1-0528:free") -> str:
        """Generate text using OpenRouter API with proper async handling."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            # Use asyncio's run_in_executor for blocking requests
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(self.base_url, headers=headers, json=data, timeout=30)
            )
            
            response.raise_for_status()
            result = response.json()
            
            if 'choices' not in result or not result['choices']:
                raise ValueError("Invalid API response format")
            
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.Timeout:
            raise Exception("Request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error: {str(e)}")
        except KeyError as e:
            raise Exception(f"Invalid API response: missing {str(e)}")
        except Exception as e:
            raise Exception(f"API error: {str(e)}")

class GenerationManager:
    """Manage generation limits and ad logic."""
    
    def __init__(self):
        self.generation_count = 0
        self.free_generations = 2
    
    def can_generate_free(self) -> bool:
        """Check if user can generate content for free."""
        return self.generation_count < self.free_generations
    
    def increment_count(self):
        """Increment generation counter."""
        self.generation_count += 1
    
    def reset_after_ad(self):
        """Reset generation capability after watching ad (simulation)."""
        # In real implementation, this would be called after successful ad view
        pass

class BaseScreen(ft.UserControl):
    """Base class for all screens with common functionality."""
    
    def __init__(self, page: ft.Page):
        super().__init__()
        self.page = page
    
    def show_error_dialog(self, title: str, message: str):
        """Show standardized error dialog."""
        dialog = ft.AlertDialog(
            title=ft.Text(title, font_family="NotoSans"),
            content=ft.Text(message, font_family="NotoSans"),
            actions=[
                ft.TextButton(
                    "OK", 
                    on_click=lambda e: self.close_dialog()
                )
            ]
        )
        self.page.dialog = dialog
        dialog.open = True
        self.page.update()
    
    def close_dialog(self):
        """Close current dialog."""
        if self.page.dialog:
            self.page.dialog.open = False
            self.page.update()
    
    def create_banner_placeholder(self) -> ft.Container:
        """Create banner ad placeholder."""
        return ft.Container(
            content=ft.Text(
                "Banner Ad Space",
                text_align=ft.TextAlign.CENTER,
                color=ft.colors.GREY_600
            ),
            height=50,
            bgcolor=ft.colors.GREY_200,
            border_radius=5,
            alignment=ft.alignment.center
        )

class LoadingScreen(BaseScreen):
    """Loading screen with proper async handling."""
    
    def __init__(self, page: ft.Page, on_complete: Callable):
        super().__init__(page)
        self.on_complete = on_complete
    
    def build(self):
        return ft.Container(
            content=ft.Column(
                [
                    ft.Icon(
                        ft.icons.BOOK,
                        size=100,
                        color=PRIMARY_COLOR
                    ),
                    ft.Text(
                        "isiZulu AI Writer",
                        font_family="NotoSans",
                        size=24,
                        color=PRIMARY_COLOR,
                        text_align=ft.TextAlign.CENTER
                    ),
                    ft.ProgressRing(color=PRIMARY_COLOR),
                    ft.Text(
                        "Loading...",
                        font_family="NotoSans",
                        size=16,
                        color=ft.colors.GREY_600,
                        text_align=ft.TextAlign.CENTER
                    )
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=20
            ),
            bgcolor=SECONDARY_COLOR,
            padding=20,
            expand=True
        )
    
    async def did_mount_async(self):
        """Simulate loading time and complete."""
        await asyncio.sleep(2)
        await self.on_complete()

class MainScreen(BaseScreen):
    """Main navigation screen."""
    
    def __init__(self, page: ft.Page, navigate_to: Callable):
        super().__init__(page)
        self.navigate_to = navigate_to
    
    def build(self):
        return ft.Container(
            content=ft.Column(
                [
                    self.create_banner_placeholder(),
                    ft.Text(
                        "isiZulu AI Writer",
                        font_family="NotoSans",
                        size=28,
                        color=PRIMARY_COLOR,
                        text_align=ft.TextAlign.CENTER,
                        weight=ft.FontWeight.BOLD
                    ),
                    ft.Text(
                        "Create beautiful content in isiZulu",
                        font_family="NotoSans",
                        size=16,
                        color=ft.colors.GREY_600,
                        text_align=ft.TextAlign.CENTER
                    ),
                    ft.Container(height=20),  # Spacer
                    *self.create_navigation_buttons()
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=15
            ),
            bgcolor=SECONDARY_COLOR,
            padding=20,
            expand=True
        )
    
    def create_navigation_buttons(self):
        """Create navigation buttons with consistent styling."""
        buttons = [
            ("Write Essay", ft.icons.EDIT, "essay"),
            ("Write Letter", ft.icons.MAIL, "letter"),
            ("Zulu Dictionary", ft.icons.BOOK, "dictionary"),
            ("Izaga Nezisho", ft.icons.FORMAT_QUOTE, "proverbs"),
            ("Cultural Info", ft.icons.ACCOUNT_BALANCE, "cultural"),
            ("Translation", ft.icons.TRANSLATE, "translation")
        ]
        
        return [
            ft.ElevatedButton(
                content=ft.Row(
                    [ft.Icon(icon), ft.Text(text, font_family="NotoSans")],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=10
                ),
                bgcolor=PRIMARY_COLOR,
                color=ft.colors.WHITE,
                width=250,
                height=50,
                on_click=lambda e, route=route: asyncio.create_task(self.navigate_to(route))
            )
            for text, icon, route in buttons
        ]

class EssayScreen(BaseScreen):
    """Essay generation screen with proper async handling."""
    
    def __init__(self, page: ft.Page, data_manager: DataManager, 
                 api_client: APIClient, generation_manager: GenerationManager):
        super().__init__(page)
        self.data_manager = data_manager
        self.api_client = api_client
        self.generation_manager = generation_manager
        self.is_generating = False
        
        self.topic_input = ft.TextField(
            hint_text="Enter essay topic (e.g., 'Ubuntu philosophy')",
            font_family="NotoSans",
            text_size=16,
            bgcolor=ft.colors.WHITE,
            border_color=PRIMARY_COLOR
        )
        
        self.length_input = ft.TextField(
            hint_text="Enter word count (e.g., 500)",
            font_family="NotoSans",
            text_size=16,
            bgcolor=ft.colors.WHITE,
            border_color=PRIMARY_COLOR,
            keyboard_type=ft.KeyboardType.NUMBER
        )
        
        self.result_text = ft.Text(
            "Your generated essay will appear here...",
            font_family="NotoSans",
            size=14,
            color=ft.colors.GREY_600,
            selectable=True
        )
        
        self.generate_button = ft.ElevatedButton(
            content=ft.Row(
                [ft.Icon(ft.icons.CREATE), ft.Text("Generate Essay", font_family="NotoSans")],
                alignment=ft.MainAxisAlignment.CENTER
            ),
            bgcolor=ACCENT_COLOR,
            color=ft.colors.WHITE,
            on_click=self.on_generate_click
        )
    
    def on_generate_click(self, e):
        """Handle generate button click."""
        if not self.is_generating:
            asyncio.create_task(self.generate_essay())
    
    async def generate_essay(self):
        """Generate essay with proper validation and error handling."""
        if self.is_generating:
            return
        
        self.is_generating = True
        self.update_ui_generating(True)
        
        try:
            # Validate inputs
            topic = self.topic_input.value.strip()
            length_str = self.length_input.value.strip()
            
            if not topic:
                self.show_error_dialog("Input Error", "Please enter an essay topic.")
                return
            
            if not length_str or not length_str.isdigit():
                self.show_error_dialog("Input Error", "Please enter a valid word count (numbers only).")
                return
            
            length = int(length_str)
            if length < 100 or length > 2000:
                self.show_error_dialog("Input Error", "Word count must be between 100 and 2000.")
                return
            
            # Check if user can generate
            if not self.generation_manager.can_generate_free():
                self.show_ad_required_dialog()
                return
            
            # Create prompt with proverbs
            proverbs_text = self.format_proverbs()
            prompt = self.create_essay_prompt(topic, length, proverbs_text)
            
            # Generate essay
            result = await self.api_client.generate_text(prompt)
            
            self.result_text.value = result
            self.result_text.color = ft.colors.BLACK
            self.generation_manager.increment_count()
            
        except Exception as e:
            self.show_error_dialog("Generation Error", str(e))
        finally:
            self.is_generating = False
            self.update_ui_generating(False)
    
    def format_proverbs(self) -> str:
        """Format proverbs for inclusion in prompt."""
        proverbs = self.data_manager.proverbs
        if not proverbs:
            return "No proverbs available."
        
        # Limit to first 10 proverbs to avoid prompt being too long
        items = list(proverbs.items())[:10]
        return "\n".join([f"• {proverb}: {meaning}" for proverb, meaning in items])
    
    def create_essay_prompt(self, topic: str, length: int, proverbs_text: str) -> str:
        """Create a well-structured prompt for essay generation with cultural context."""
        # Detect cultural context
        cultural_contexts = self.data_manager.detect_cultural_context(topic)
        cultural_info = self.data_manager.format_cultural_context(cultural_contexts)
        
        base_prompt = f"""Write a well-structured essay in isiZulu on the topic '{topic}' with approximately {length} words.

Requirements:
- Include a clear introduction, body paragraphs, and conclusion
- Use proper isiZulu grammar and vocabulary
- Incorporate relevant isiZulu proverbs (izaga nezisho) where appropriate
- Maintain cultural authenticity and respect
- Draw from traditional Zulu knowledge and practices"""

        if cultural_info:
            base_prompt += f"""

IMPORTANT CULTURAL CONTEXT TO INCORPORATE:
{cultural_info}

Please use this cultural information to enrich your essay with authentic details, proper terminology, and cultural significance. Ensure the essay reflects deep understanding of Zulu traditions and customs."""

        base_prompt += f"""

Available proverbs to consider:
{proverbs_text}

Please write the essay entirely in isiZulu, ensuring it flows naturally and educates the reader about the topic while showcasing the beauty of the isiZulu language and culture."""

        return base_prompt
    
    def show_ad_required_dialog(self):
        """Show dialog when ad viewing is required."""
        message = "You've used your free generations. In a full version, you would watch an ad to continue."
        self.show_error_dialog("Free Limit Reached", message)
    
    def update_ui_generating(self, generating: bool):
        """Update UI state during generation."""
        if generating:
            self.generate_button.content = ft.Row(
                [ft.ProgressRing(width=16, height=16), ft.Text("Generating...", font_family="NotoSans")],
                alignment=ft.MainAxisAlignment.CENTER
            )
            self.generate_button.disabled = True
            self.result_text.value = "Generating your essay... Please wait."
            self.result_text.color = ft.colors.BLUE
        else:
            self.generate_button.content = ft.Row(
                [ft.Icon(ft.icons.CREATE), ft.Text("Generate Essay", font_family="NotoSans")],
                alignment=ft.MainAxisAlignment.CENTER
            )
            self.generate_button.disabled = False
        
        self.page.update()
    
    def build(self):
        return ft.Container(
            content=ft.Column(
                [
                    self.create_banner_placeholder(),
                    ft.Text(
                        "Essay Generator",
                        font_family="NotoSans",
                        size=24,
                        color=PRIMARY_COLOR,
                        weight=ft.FontWeight.BOLD
                    ),
                    self.topic_input,
                    self.length_input,
                    self.generate_button,
                    ft.Container(
                        content=ft.Column([
                            ft.Text(
                                "Generated Essay:",
                                font_family="NotoSans",
                                size=16,
                                weight=ft.FontWeight.BOLD,
                                color=PRIMARY_COLOR
                            ),
                            self.result_text
                        ], spacing=10),
                        expand=True,
                        padding=15,
                        bgcolor=ft.colors.WHITE,
                        border_radius=10,
                        border=ft.border.all(1, PRIMARY_COLOR)
                    )
                ],
                spacing=15,
                scroll=ft.ScrollMode.AUTO
            ),
            bgcolor=SECONDARY_COLOR,
            padding=20,
            expand=True
        )

# Similar improvements would be made to other screens...

class LetterScreen(BaseScreen):
    """Letter generation screen with cultural context awareness."""
    
    def __init__(self, page: ft.Page, data_manager: DataManager, 
                 api_client: APIClient, generation_manager: GenerationManager):
        super().__init__(page)
        self.data_manager = data_manager
        self.api_client = api_client
        self.generation_manager = generation_manager
        self.is_generating = False
        
        self.recipient_input = ft.TextField(
            hint_text="Enter recipient name",
            font_family="NotoSans",
            text_size=16,
            bgcolor=ft.colors.WHITE,
            border_color=PRIMARY_COLOR
        )
        
        self.purpose_input = ft.TextField(
            hint_text="Enter letter purpose (e.g., 'wedding invitation', 'condolences')",
            font_family="NotoSans",
            text_size=16,
            bgcolor=ft.colors.WHITE,
            border_color=PRIMARY_COLOR,
            multiline=True
        )
        
        self.tone_dropdown = ft.Dropdown(
            hint_text="Select tone",
            options=[
                ft.dropdown.Option("formal", "Formal (Okunesithunzi)"),
                ft.dropdown.Option("informal", "Informal (Okungekho sithunzi)"),
                ft.dropdown.Option("respectful", "Respectful (Okuhloniphayo)"),
                ft.dropdown.Option("ceremonial", "Ceremonial (Kwemicimbi)")
            ],
            bgcolor=ft.colors.WHITE,
            border_color=PRIMARY_COLOR
        )
        
        self.result_text = ft.Text(
            "Your generated letter will appear here...",
            font_family="NotoSans",
            size=14,
            color=ft.colors.GREY_600,
            selectable=True
        )
        
        self.generate_button = ft.ElevatedButton(
            content=ft.Row(
                [ft.Icon(ft.icons.MAIL), ft.Text("Generate Letter", font_family="NotoSans")],
                alignment=ft.MainAxisAlignment.CENTER
            ),
            bgcolor=ACCENT_COLOR,
            color=ft.colors.WHITE,
            on_click=self.on_generate_click
        )
    
    def on_generate_click(self, e):
        """Handle generate button click."""
        if not self.is_generating:
            asyncio.create_task(self.generate_letter())
    
    async def generate_letter(self):
        """Generate letter with cultural context."""
        if self.is_generating:
            return
        
        self.is_generating = True
        self.update_ui_generating(True)
        
        try:
            # Validate inputs
            recipient = self.recipient_input.value.strip()
            purpose = self.purpose_input.value.strip()
            tone = self.tone_dropdown.value
            
            if not recipient or not purpose or not tone:
                self.show_error_dialog("Input Error", "Please fill in all fields.")
                return
            
            # Check generation limit
            if not self.generation_manager.can_generate_free():
                self.show_ad_required_dialog()
                return
            
            # Create prompt with cultural context
            prompt = self.create_letter_prompt(recipient, purpose, tone)
            
            # Generate letter
            result = await self.api_client.generate_text(prompt)
            
            self.result_text.value = result
            self.result_text.color = ft.colors.BLACK
            self.generation_manager.increment_count()
            
        except Exception as e:
            self.show_error_dialog("Generation Error", str(e))
        finally:
            self.is_generating = False
            self.update_ui_generating(False)
    
    def create_letter_prompt(self, recipient: str, purpose: str, tone: str) -> str:
        """Create letter prompt with cultural context."""
        # Detect cultural context from purpose
        cultural_contexts = self.data_manager.detect_cultural_context(purpose)
        cultural_info = self.data_manager.format_cultural_context(cultural_contexts)
        
        tone_descriptions = {
            'formal': 'formal and respectful, using proper Zulu honorifics and etiquette',
            'informal': 'friendly and casual while maintaining cultural respect',
            'respectful': 'deeply respectful, incorporating traditional forms of address',
            'ceremonial': 'ceremonial and traditional, appropriate for cultural events'
        }
        
        base_prompt = f"""Write a {tone_descriptions.get(tone, tone)} letter in isiZulu to {recipient} for the purpose of {purpose}.

Requirements:
- Use appropriate Zulu letter structure and formatting
- Include proper greetings and closings
- Maintain cultural sensitivity and authenticity
- Use correct honorifics and forms of address
- Incorporate relevant cultural expressions"""

        if cultural_info:
            base_prompt += f"""

CULTURAL CONTEXT TO CONSIDER:
{cultural_info}

Please incorporate relevant cultural elements, proper terminology, and traditional practices mentioned above into the letter to make it culturally authentic and meaningful."""

        base_prompt += """

The letter should be entirely in isiZulu and demonstrate proper understanding of Zulu communication customs and etiquette."""

        return base_prompt
    
    def show_ad_required_dialog(self):
        """Show dialog when ad viewing is required."""
        message = "You've used your free generations. In a full version, you would watch an ad to continue."
        self.show_error_dialog("Free Limit Reached", message)
    
    def update_ui_generating(self, generating: bool):
        """Update UI state during generation."""
        if generating:
            self.generate_button.content = ft.Row(
                [ft.ProgressRing(width=16, height=16), ft.Text("Generating...", font_family="NotoSans")],
                alignment=ft.MainAxisAlignment.CENTER
            )
            self.generate_button.disabled = True
            self.result_text.value = "Generating your letter... Please wait."
            self.result_text.color = ft.colors.BLUE
        else:
            self.generate_button.content = ft.Row(
                [ft.Icon(ft.icons.MAIL), ft.Text("Generate Letter", font_family="NotoSans")],
                alignment=ft.MainAxisAlignment.CENTER
            )
            self.generate_button.disabled = False
        
        self.page.update()
    
    def build(self):
        return ft.Container(
            content=ft.Column(
                [
                    self.create_banner_placeholder(),
                    ft.Text(
                        "Letter Generator",
                        font_family="NotoSans",
                        size=24,
                        color=PRIMARY_COLOR,
                        weight=ft.FontWeight.BOLD
                    ),
                    self.recipient_input,
                    self.purpose_input,
                    self.tone_dropdown,
                    self.generate_button,
                    ft.Container(
                        content=ft.Column([
                            ft.Text(
                                "Generated Letter:",
                                font_family="NotoSans",
                                size=16,
                                weight=ft.FontWeight.BOLD,
                                color=PRIMARY_COLOR
                            ),
                            self.result_text
                        ], spacing=10),
                        expand=True,
                        padding=15,
                        bgcolor=ft.colors.WHITE,
                        border_radius=10,
                        border=ft.border.all(1, PRIMARY_COLOR)
                    )
                ],
                spacing=15,
                scroll=ft.ScrollMode.AUTO
            ),
            bgcolor=SECONDARY_COLOR,
            padding=20,
            expand=True
        )

class CulturalInfoScreen(BaseScreen):
    """Screen to browse cultural information."""
    
    def __init__(self, page: ft.Page, data_manager: DataManager):
        super().__init__(page)
        self.data_manager = data_manager
        
        self.category_dropdown = ft.Dropdown(
            hint_text="Select cultural category",
            options=[
                ft.dropdown.Option("imvunulo", "Imvunulo (Traditional Attire)"),
                ft.dropdown.Option("amasiko", "Amasiko (Customs & Traditions)"),
                ft.dropdown.Option("ceremonies", "Ceremonies & Rituals"),
                ft.dropdown.Option("beadwork", "Beadwork & Symbols"),
                ft.dropdown.Option("ubuntu", "Ubuntu Philosophy")
            ],
            bgcolor=ft.colors.WHITE,
            border_color=PRIMARY_COLOR,
            on_change=self.on_category_change
        )
        
        self.info_display = ft.Column(
            [
                ft.Text(
                    "Select a category above to explore Zulu cultural information",
                    font_family="NotoSans",
                    size=16,
                    color=ft.colors.GREY_600,
                    text_align=ft.TextAlign.CENTER
                )
            ],
            scroll=ft.ScrollMode.AUTO
        )
    
    def on_category_change(self, e):
        """Handle category selection change."""
        category = self.category_dropdown.value
        if category:
            self.display_cultural_info(category)
    
    def display_cultural_info(self, category: str):
        """Display information for selected category."""
        self.info_display.controls.clear()
        
        cultural_data = self.data_manager.cultural_data.get(category, {})
        
        if not cultural_data:
            self.info_display.controls.append(
                ft.Text(
                    f"No information available for {category}. Please ensure the data file exists.",
                    font_family="NotoSans",
                    color=ft.colors.RED_400
                )
            )
        else:
            # Add category title
            self.info_display.controls.append(
                ft.Text(
                    category.replace('_', ' ').title(),
                    font_family="NotoSans",
                    size=24,
                    color=PRIMARY_COLOR,
                    weight=ft.FontWeight.BOLD
                )
            )
            
            # Format and display the information
            formatted_info = self.data_manager._format_category_data(category, cultural_data)
            
            # Split by lines and create Text widgets
            for line in formatted_info.split('\n'):
                if line.strip():
                    if line.startswith('==='):
                        # Section headers
                        self.info_display.controls.append(
                            ft.Text(
                                line.replace('=', '').strip(),
                                font_family="NotoSans",
                                size=20,
                                color=ACCENT_COLOR,
                                weight=ft.FontWeight.BOLD
                            )
                        )
                    elif line.strip().endswith(':'):
                        # Subsection headers
                        self.info_display.controls.append(
                            ft.Text(
                                line.strip(),
                                font_family="NotoSans",
                                size=18,
                                color=PRIMARY_COLOR,
                                weight=ft.FontWeight.W_500
                            )
                        )
                    else:
                        # Regular content
                        self.info_display.controls.append(
                            ft.Text(
                                line.strip(),
                                font_family="NotoSans",
                                size=14,
                                color=ft.colors.BLACK
                            )
                        )
                else:
                    # Add spacing for empty lines
                    self.info_display.controls.append(ft.Container(height=10))
        
        self.page.update()
    
    def build(self):
        return ft.Container(
            content=ft.Column(
                [
                    self.create_banner_placeholder(),
                    ft.Text(
                        "Cultural Information",
                        font_family="NotoSans",
                        size=24,
                        color=PRIMARY_COLOR,
                        weight=ft.FontWeight.BOLD,
                        text_align=ft.TextAlign.CENTER
                    ),
                    ft.Text(
                        "Explore Zulu culture, traditions, and customs",
                        font_family="NotoSans",
                        size=16,
                        color=ft.colors.GREY_600,
                        text_align=ft.TextAlign.CENTER
                    ),
                    self.category_dropdown,
                    ft.Container(
                        content=self.info_display,
                        expand=True,
                        padding=15,
                        bgcolor=ft.colors.WHITE,
                        border_radius=10,
                        border=ft.border.all(1, PRIMARY_COLOR)
                    )
                ],
                spacing=15
            ),
            bgcolor=SECONDARY_COLOR,
            padding=20,
            expand=True
        )

class IsiZuluApp:
    """Main application controller."""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.generation_manager = GenerationManager()
        
        # Initialize API client with validation
        if not OPENROUTER_API_KEY:
            logger.warning("No API key provided. Text generation will not work.")
            self.api_client = None
        else:
            try:
                self.api_client = APIClient(OPENROUTER_API_key)
            except ValueError as e:
                logger.error(f"Invalid API key: {e}")
                self.api_client = None
    
    async def navigate_to(self, page: ft.Page, screen_name: str):
        """Navigate to specified screen with proper cleanup."""
        page.controls.clear()
        
        try:
            if screen_name == "main":
                screen = MainScreen(page, lambda route: self.navigate_to(page, route))
            elif screen_name == "essay":
                screen = EssayScreen(page, self.data_manager, self.api_client, self.generation_manager)
            elif screen_name == "letter":
                screen = LetterScreen(page, self.data_manager, self.api_client, self.generation_manager)
            elif screen_name == "cultural":
                screen = CulturalInfoScreen(page, self.data_manager)
            # Add other screens here...
            else:
                logger.error(f"Unknown screen: {screen_name}")
                return
            
            page.add(screen)
            page.update()
            
        except Exception as e:
            logger.error(f"Navigation error: {e}")
            # Fallback to main screen
            if screen_name != "main":
                await self.navigate_to(page, "main")

def main(page: ft.Page):
    """Main application entry point."""
    page.title = "isiZulu AI Writer"
    page.bgcolor = SECONDARY_COLOR
    page.theme_mode = ft.ThemeMode.LIGHT
    
    # Configure fonts if available
    page.fonts = {
        "NotoSans": "assets/fonts/NotoSans-Regular.ttf"
    }
    
    app = IsiZuluApp()
    
    async def start_app():
        """Start the application after loading screen."""
        await app.navigate_to(page, "main")
    
    # Show loading screen first
    loading_screen = LoadingScreen(page, start_app)
    page.add(loading_screen)
    page.update()

# Run the application
if __name__ == "__main__":
    ft.app(target=main, assets_dir="assets")

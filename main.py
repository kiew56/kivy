import flet as ft
import json
import requests
import platform
import os
from typing import Dict, Optional, Callable
import logging

# Optional flet_ads import
try:
    import flet_ads as ads
    FLET_ADS_AVAILABLE = True
except ImportError:
    FLET_ADS_AVAILABLE = False
    ads = None
    logging.warning("flet_ads not available, using placeholder for ads")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Color scheme
PRIMARY_COLOR = "#1FAB78"
SECONDARY_COLOR = "#F2F2F2"
ACCENT_COLOR = "#E64C4C"

class AdManager:
    """Centralized AdMob ad management with fallback if flet_ads is unavailable."""
    
    def __init__(self, page: ft.Page):
        self.page = page
        self.generation_manager = None
        
        if FLET_ADS_AVAILABLE:
            # Platform-specific ad unit IDs (using test IDs)
            self.banner_id = (
                "ca-app-pub-3940256099942544/6300978111"
                if page.platform == ft.PagePlatform.ANDROID
                else "ca-app-pub-3940256099942544/2934735716"
            )
            self.interstitial_id = (
                "ca-app-pub-3940256099942544/1033173712"
                if page.platform == ft.PagePlatform.ANDROID
                else "ca-app-pub-3940256099942544/4411468910"
            )
            self.banner_ad = None
            self.current_interstitial = None
            self._create_ads()
        else:
            logger.info("AdMob disabled, using placeholder")
            self.banner_ad = None
            self.current_interstitial = None
    
    def _create_ads(self):
        """Create ad instances with error handling."""
        try:
            if FLET_ADS_AVAILABLE:
                self.banner_ad = ads.BannerAd(
                    unit_id=self.banner_id,
                    width=320,
                    height=50,
                    on_load=lambda e: logger.info("Banner ad loaded"),
                    on_error=lambda e: logger.error(f"Banner ad error: {e.data}"),
                    on_click=lambda e: logger.info("Banner ad clicked"),
                    on_impression=lambda e: logger.info("Banner ad impression")
                )
                self._create_new_interstitial()
            else:
                logger.warning("Skipping ad creation: flet_ads not available")
        except Exception as e:
            logger.error(f"Error creating ads: {e}")
            self.banner_ad = None
            self.current_interstitial = None
            self.show_error_snackbar("Failed to initialize ads")
    
    def _create_new_interstitial(self):
        """Create a new interstitial ad instance."""
        try:
            if FLET_ADS_AVAILABLE:
                self.current_interstitial = ads.InterstitialAd(
                    unit_id=self.interstitial_id,
                    on_load=lambda e: logger.info("Interstitial ad loaded"),
                    on_error=lambda e: logger.error(f"Interstitial ad error: {e.data}"),
                    on_open=lambda e: logger.info("Interstitial ad opened"),
                    on_close=self._on_interstitial_close,
                    on_impression=lambda e: logger.info("Interstitial ad impression"),
                    on_click=lambda e: logger.info("Interstitial ad clicked")
                )
                if self.current_interstitial and self.current_interstitial not in self.page.overlay:
                    self.page.overlay.append(self.current_interstitial)
            else:
                logger.warning("Skipping interstitial creation: flet_ads not available")
        except Exception as e:
            logger.error(f"Error creating interstitial ad: {e}")
            self.current_interstitial = None
            self.show_error_snackbar("Failed to initialize interstitial ad")
    
    def _on_interstitial_close(self, e):
        """Handle interstitial ad close."""
        logger.info("Interstitial ad closed")
        if hasattr(e.control, '_is_reward_ad') and e.control._is_reward_ad:
            self._grant_reward()
        if e.control in self.page.overlay:
            self.page.overlay.remove(e.control)
        self._create_new_interstitial()
        self.page.update()
    
    def _grant_reward(self):
        """Grant reward after watching interstitial ad."""
        logger.info("User earned reward from interstitial ad")
        if self.generation_manager:
            self.generation_manager.grant_free_generations(5)
        self.page.snack_bar = ft.SnackBar(
            content=ft.Text("🎉 You earned 5 free generations!", color=ft.colors.WHITE),
            bgcolor=ft.colors.GREEN,
            duration=3000
        )
        self.page.snack_bar.open = True
        self.page.update()
    
    def show_interstitial(self):
        """Show interstitial ad."""
        if FLET_ADS_AVAILABLE and self.current_interstitial:
            try:
                self.current_interstitial._is_reward_ad = False
                self.current_interstitial.show()
            except Exception as e:
                logger.error(f"Failed to show interstitial ad: {e}")
                self.show_error_snackbar("Failed to show ad")
        else:
            logger.info("No interstitial ad available")
    
    def show_reward_interstitial(self):
        """Show interstitial ad as reward ad."""
        if FLET_ADS_AVAILABLE and self.current_interstitial:
            try:
                self.current_interstitial._is_reward_ad = True
                self.current_interstitial.show()
            except Exception as e:
                logger.error(f"Failed to show reward ad: {e}")
                self._simulate_reward()
        else:
            self._simulate_reward()
    
    def _simulate_reward(self):
        """Simulate reward for testing."""
        logger.info("Simulating reward (testing mode)")
        if self.generation_manager:
            self.generation_manager.grant_free_generations(5)
        self.page.snack_bar = ft.SnackBar(
            content=ft.Text("🎉 Reward simulated! You earned 5 free generations!", color=ft.colors.WHITE),
            bgcolor=ft.colors.BLUE,
            duration=3000
        )
        self.page.snack_bar.open = True
        self.page.update()
    
    def show_error_snackbar(self, message: str):
        """Show error snackbar."""
        self.page.snack_bar = ft.SnackBar(
            content=ft.Text(f"Error: {message}", color=ft.colors.WHITE),
            bgcolor=ft.colors.RED,
            duration=3000
        )
        self.page.snack_bar.open = True
        self.page.update()
    
    def create_banner_container(self) -> ft.Container:
        """Create a container with banner ad or placeholder."""
        if FLET_ADS_AVAILABLE and self.banner_ad:
            return ft.Container(
                content=self.banner_ad,
                width=320,
                height=50,
                bgcolor=ft.colors.TRANSPARENT,
                alignment=ft.alignment.center
            )
        return ft.Container(
            content=ft.Text("Ad Space Unavailable", size=12, color=ft.colors.GREY_400),
            width=320,
            height=50,
            bgcolor=ft.colors.GREY_100,
            alignment=ft.alignment.center,
            border_radius=5
        )

class GenerationManager:
    """Enhanced generation management with reward system."""
    
    def __init__(self):
        self.generation_count = 0
        self.free_generations = 2
        self.total_earned_generations = 0
    
    def can_generate_free(self) -> bool:
        return self.generation_count < self.free_generations
    
    def increment_count(self):
        self.generation_count += 1
    
    def grant_free_generations(self, count: int):
        self.free_generations += count
        self.total_earned_generations += count
        logger.info(f"Granted {count} free generations. Total available: {self.get_remaining_generations()}")
    
    def get_remaining_generations(self) -> int:
        return max(0, self.free_generations - self.generation_count)
    
    def reset_daily(self):
        self.generation_count = 0
        self.free_generations = 2

class DataManager:
    """Centralized data management with caching and validation."""
    
    def __init__(self):
        self._proverbs: Optional[Dict[str, str]] = None
        self._dictionary: Optional[Dict[str, str]] = None
        self._cultural_keywords = {
            'imvunulo', 'izingubo', 'amasiko', 'isiko', 'ubuntu', 'ukusina',
            'umgido', 'ukwemula', 'umhlanga', 'reed dance', 'traditional',
            'culture', 'custom', 'ceremony', 'wedding', 'umshado', 'funeral',
            'umngcwabo', 'initiation', 'ukwaluka', 'coming of age', 'ancestors',
            'amadlozi', 'traditional dress', 'beadwork', 'amaqhiya', 'izidwaba'
        }
    
    def load_json_file(self, filename: str, description: str) -> Dict[str, str]:
        """Load and validate JSON file with error handling."""
        try:
            with open(f"assets/{filename}", 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"{description} must be a dictionary")
            logger.info(f"Successfully loaded {len(data)} entries from {filename}")
            return data
        except FileNotFoundError:
            logger.warning(f"File {filename} not found - using fallback data")
            return self._get_fallback_data(description)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filename}: {e}")
            return self._get_fallback_data(description)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return self._get_fallback_data(description)
    
    def _get_fallback_data(self, description: str) -> Dict[str, str]:
        if "proverbs" in description.lower():
            return {
                "Ubuntu ngumuntu ngabantu": "A person is a person through other people - emphasizing our interconnectedness",
                "Ukuzila kuyasiza": "Restraint helps - showing the value of self-control",
                "Inhliziyo yami ithi": "My heart says - expressing inner conviction",
                "Akukho siphako esingenasici": "There is no gift without a reason - everything has a purpose",
                "Umuntu ngumuntu ngabanye abantu": "A person becomes human through other people",
                "Isandla siyagezana": "Hands wash each other - mutual assistance",
                "Indlovu ayikhali": "An elephant does not boast - true strength is humble"
            }
        elif "dictionary" in description.lower():
            return {
                "ubuntu": "humanity, humanness, compassion",
                "sawubona": "hello, we see you (greeting)",
                "ngiyabonga": "thank you",
                "hamba kahle": "go well (farewell)",
                "sala kahle": "stay well (farewell to one staying)",
                "ngiyakuthanda": "I love you",
                "amasiko": "customs, traditions",
                "isiko": "custom, tradition",
                "amadlozi": "ancestors",
                "umhlanga": "reed dance ceremony",
                "imvunulo": "traditional attire"
            }
        return {}
    
    @property
    def proverbs(self) -> Dict[str, str]:
        if self._proverbs is None:
            self._proverbs = self.load_json_file('izaga_nezisho.json', 'proverbs')
        return self._proverbs
    
    @property
    def dictionary(self) -> Dict[str, str]:
        if self._dictionary is None:
            self._dictionary = self.load_json_file('zulu_dictionary.json', 'dictionary')
        return self._dictionary

class APIClient:
    """Centralized API client with error handling."""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required")
        self.api_key = api_key
        self.base_url = OPENROUTER_API_URL
    
    async def generate_text(self, prompt: str, model: str = "deepseek/deepseek-r1-0528:free") -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://isizulu-ai-writer.app",
            "X-Title": "isiZulu AI Writer"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.7
        }
        try:
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

class BaseScreen:
    """Base class for all screens with common functionality."""
    
    def __init__(self, page: ft.Page, ad_manager: AdManager):
        self.page = page
        self.ad_manager = ad_manager
    
    def show_error_dialog(self, title: str, message: str):
        try:
            dialog = ft.AlertDialog(
                title=ft.Text(title, font_family="NotoSans" if self.page.fonts else None),
                content=ft.Text(message, font_family="NotoSans" if self.page.fonts else None),
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
        except Exception as e:
            logger.error(f"Error showing dialog: {e}")
            self.page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Error: {message}", color=ft.colors.WHITE),
                bgcolor=ft.colors.RED,
                duration=3000
            )
            self.page.snack_bar.open = True
            self.page.update()
    
    def close_dialog(self):
        if self.page.dialog:
            self.page.dialog.open = False
            self.page.update()
    
    def show_reward_dialog(self, generation_manager: GenerationManager):
        remaining = generation_manager.get_remaining_generations()
        try:
            dialog = ft.AlertDialog(
                title=ft.Text("Watch Ad to Continue", font_family="NotoSans" if self.page.fonts else None, color=PRIMARY_COLOR),
                content=ft.Column([
                    ft.Text(f"You have {remaining} free generations remaining.", font_family="NotoSans" if self.page.fonts else None),
                    ft.Container(height=10),
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Icon(ft.icons.PLAY_CIRCLE_FILLED, color=ft.colors.GREEN, size=40),
                                ft.Column([
                                    ft.Text("Watch a short ad", font_family="NotoSans" if self.page.fonts else None, size=16, weight=ft.FontWeight.BOLD),
                                    ft.Text("Get 5 more generations!", font_family="NotoSans" if self.page.fonts else None, size=14, color=ft.colors.GREEN)
                                ], spacing=0)
                            ], alignment=ft.MainAxisAlignment.CENTER, spacing=15),
                            ft.Container(height=10),
                            ft.Text("🎁 Reward: 5 Free Generations",
                                   font_family="NotoSans" if self.page.fonts else None,
                                   size=14,
                                   color=ft.colors.ORANGE_600,
                                   text_align=ft.TextAlign.CENTER)
                        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        bgcolor=ft.colors.GREEN_50,
                        border_radius=10,
                        padding=15
                    )
                ], tight=True),
                actions=[
                    ft.TextButton(
                        "Maybe Later",
                        on_click=lambda e: self.close_dialog()
                    ),
                    ft.ElevatedButton(
                        content=ft.Row([
                            ft.Icon(ft.icons.PLAY_ARROW, size=18),
                            ft.Text("Watch Ad")
                        ], alignment=ft.MainAxisAlignment.CENTER, spacing=5),
                        bgcolor=ft.colors.GREEN,
                        color=ft.colors.WHITE,
                        on_click=lambda e: self.watch_reward_ad()
                    )
                ]
            )
            self.page.dialog = dialog
            dialog.open = True
            self.page.update()
        except Exception as e:
            logger.error(f"Error showing reward dialog: {e}")
            self.page.snack_bar = ft.SnackBar(
                content=ft.Text("Error: Unable to show reward dialog", color=ft.colors.WHITE),
                bgcolor=ft.colors.RED,
                duration=3000
            )
            self.page.snack_bar.open = True
            self.page.update()
    
    def watch_reward_ad(self):
        self.close_dialog()
        self.page.snack_bar = ft.SnackBar(
            content=ft.Text("Loading ad... Please wait", color=ft.colors.WHITE),
            bgcolor=ft.colors.BLUE,
            duration=2000
        )
        self.page.snack_bar.open = True
        self.page.update()
        self.ad_manager.show_reward_interstitial()

class MainScreen(BaseScreen):
    """Main navigation screen with generation counter."""
    
    def __init__(self, page: ft.Page, ad_manager: AdManager, navigate_to: Callable, generation_manager: GenerationManager):
        super().__init__(page, ad_manager)
        self.navigate_to = navigate_to
        self.generation_manager = generation_manager
    
    def build(self):
        try:
            return ft.Container(
                content=ft.Column(
                    [
                        self.ad_manager.create_banner_container(),
                        ft.Container(height=10),
                        ft.Text(
                            "isiZulu AI Writer",
                            font_family="NotoSans" if self.page.fonts else None,
                            size=28,
                            color=PRIMARY_COLOR,
                            text_align=ft.TextAlign.CENTER,
                            weight=ft.FontWeight.BOLD
                        ),
                        ft.Text(
                            "Create beautiful content in isiZulu",
                            font_family="NotoSans" if self.page.fonts else None,
                            size=16,
                            color=ft.colors.GREY_600,
                            text_align=ft.TextAlign.CENTER
                        ),
                        ft.Container(height=15),
                        ft.Container(
                            content=ft.Column([
                                ft.Row([
                                    ft.Icon(ft.icons.BOLT, color=ft.colors.ORANGE, size=24),
                                    ft.Text(
                                        f"Free Generations: {self.generation_manager.get_remaining_generations()}",
                                        font_family="NotoSans" if self.page.fonts else None,
                                        size=16,
                                        color=ft.colors.BLACK87,
                                        weight=ft.FontWeight.BOLD
                                    )
                                ], alignment=ft.MainAxisAlignment.CENTER, spacing=8),
                                ft.Container(height=5),
                                ft.Row([
                                    ft.ElevatedButton(
                                        content=ft.Row([
                                            ft.Icon(ft.icons.PLAY_CIRCLE_FILLED, size=20, color=ft.colors.WHITE),
                                            ft.Text("Watch Ad for +5", color=ft.colors.WHITE, size=12)
                                        ], alignment=ft.MainAxisAlignment.CENTER, spacing=5),
                                        bgcolor=ft.colors.GREEN,
                                        height=35,
                                        on_click=lambda e: self.show_reward_dialog(self.generation_manager)
                                    )
                                ], alignment=ft.MainAxisAlignment.CENTER)
                            ], spacing=5, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                            bgcolor=ft.colors.AMBER_50,
                            padding=15,
                            border_radius=15,
                            border=ft.border.all(2, ft.colors.ORANGE_200)
                        ),
                        ft.Container(height=25),
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
        except Exception as e:
            logger.error(f"Error building MainScreen: {e}")
            return ft.Container(
                content=ft.Text("Error: Unable to load main screen", color=ft.colors.RED),
                bgcolor=SECONDARY_COLOR,
                padding=20,
                expand=True
            )
    
    def create_navigation_buttons(self):
        buttons = [
            ("Write Essay", ft.icons.EDIT, "essay"),
            ("Write Letter", ft.icons.MAIL, "letter"),
            ("Zulu Dictionary", ft.icons.BOOK, "dictionary"),
            ("Izaga Nezisho", ft.icons.FORMAT_QUOTE, "proverbs"),
            ("Translation", ft.icons.TRANSLATE, "translation")
        ]
        return [
            ft.ElevatedButton(
                content=ft.Row(
                    [ft.Icon(icon), ft.Text(text, font_family="NotoSans" if self.page.fonts else None)],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=10
                ),
                bgcolor=PRIMARY_COLOR,
                color=ft.colors.WHITE,
                width=250,
                height=50,
                on_click=lambda e, route=route: self.page.run_task(self.navigate_to, route)
            )
            for text, icon, route in buttons
        ]

class EssayScreen(BaseScreen):
    """Essay generation screen with ad integration."""
    
    def __init__(self, page: ft.Page, ad_manager: AdManager, data_manager: DataManager,
                 api_client: APIClient, generation_manager: GenerationManager, navigate_to: Callable):
        super().__init__(page, ad_manager)
        self.data_manager = data_manager
        self.api_client = api_client
        self.generation_manager = generation_manager
        self.navigate_to = navigate_to
        self.is_generating = False
        
        self.topic_input = ft.TextField(
            hint_text="Enter essay topic (e.g., 'Ubuntu philosophy')",
            font_family="NotoSans" if page.fonts else None,
            text_size=16,
            bgcolor=ft.colors.WHITE,
            border_color=PRIMARY_COLOR
        )
        self.length_input = ft.TextField(
            hint_text="Enter word count (e.g., 500)",
            font_family="NotoSans" if page.fonts else None,
            text_size=16,
            bgcolor=ft.colors.WHITE,
            border_color=PRIMARY_COLOR,
            keyboard_type=ft.KeyboardType.NUMBER
        )
        self.result_text = ft.Text(
            "Your generated essay will appear here...",
            font_family="NotoSans" if page.fonts else None,
            size=14,
            color=ft.colors.GREY_600,
            selectable=True
        )
        self.generate_button = ft.ElevatedButton(
            content=ft.Row(
                [ft.Icon(ft.icons.CREATE), ft.Text("Generate Essay", font_family="NotoSans" if page.fonts else None)],
                alignment=ft.MainAxisAlignment.CENTER
            ),
            bgcolor=ACCENT_COLOR,
            color=ft.colors.WHITE,
            on_click=self.on_generate_click
        )
    
    def on_generate_click(self, e):
        if not self.is_generating:
            self.page.run_task(self.generate_essay)
    
    async def generate_essay(self):
        if self.is_generating:
            return
        self.is_generating = True
        self.update_ui_generating(True)
        try:
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
            if not self.generation_manager.can_generate_free():
                self.show_reward_dialog(self.generation_manager)
                return
            proverbs_text = self.format_proverbs()
            prompt = self.create_essay_prompt(topic, length, proverbs_text)
            if self.api_client:
                result = await self.api_client.generate_text(prompt)
            else:
                result = f"""Isingeniso (Introduction):
Ubuntu nguqondiso olukhulu oluvela emiqondweni yethu yendabuko. Leli qondiso lithi "Ubuntu ngumuntu ngabantu" - lokhu kusho ukuthi umuntu uba ngumuntu ngokusiza kwabanye abantu.

Umzimba (Body):
Ubuntu bungaphezu nje kokuphila - buyindlela yokuphila. Lapho sikhuluma ngo-{topic}, sibona ukuthi...

Isiphetho (Conclusion):
Ngenxa yalokhu, {topic} kuyaqondakala ukuthi kuyingxenye ebalulekile ye-Ubuntu nendlela yethu yokuphila.

[Demo: Full {length}-word essay about '{topic}' would be generated here using AI with cultural context and proverbs like: {list(self.data_manager.proverbs.keys())[0] if self.data_manager.proverbs else 'Ubuntu ngumuntu ngabantu'}]"""
            self.result_text.value = result
            self.result_text.color = ft.colors.BLACK
            self.generation_manager.increment_count()
            if self.generation_manager.generation_count % 4 == 0:
                self.ad_manager.show_interstitial()
        except Exception as e:
            self.show_error_dialog("Generation Error", str(e))
        finally:
            self.is_generating = False
            self.update_ui_generating(False)
    
    def format_proverbs(self) -> str:
        proverbs = self.data_manager.proverbs
        if not proverbs:
            return "Ubuntu ngumuntu ngabantu - A person is a person through other people"
        items = list(proverbs.items())[:5]
        return "\n".join([f"• {proverb}: {meaning}" for proverb, meaning in items])
    
    def create_essay_prompt(self, topic: str, length: int, proverbs_text: str) -> str:
        return f"""Write a well-structured essay in isiZulu on the topic '{topic}' with approximately {length} words.

Requirements:
- Include a clear introduction (Isingeniso), body paragraphs (Umzimba), and conclusion (Isiphetho)
- Use proper isiZulu grammar and vocabulary
- Incorporate relevant isiZulu proverbs (izaga nezisho) where appropriate
- Maintain cultural authenticity and respect
- Draw from traditional Zulu knowledge and practices

Available proverbs to consider:
{proverbs_text}

Please write the essay entirely in isiZulu, ensuring it flows naturally and educates the reader about the topic while showcasing the beauty of the isiZulu language and culture."""
    
    def update_ui_generating(self, generating: bool):
        try:
            if generating:
                self.generate_button.content = ft.Row(
                    [ft.ProgressRing(width=16, height=16), ft.Text("Generating...", font_family="NotoSans" if self.page.fonts else None)],
                    alignment=ft.MainAxisAlignment.CENTER
                )
                self.generate_button.disabled = True
                self.result_text.value = "Generating your essay... Please wait."
                self.result_text.color = ft.colors.BLUE
            else:
                self.generate_button.content = ft.Row(
                    [ft.Icon(ft.icons.CREATE), ft.Text("Generate Essay", font_family="NotoSans" if self.page.fonts else None)],
                    alignment=ft.MainAxisAlignment.CENTER
                )
                self.generate_button.disabled = False
            self.page.update()
        except Exception as e:
            logger.error(f"Error updating UI: {e}")
            self.result_text.value = "Error: Unable to update UI"
            self.result_text.color = ft.colors.RED
            self.page.update()
    
    def build(self):
        try:
            return ft.Container(
                content=ft.Column(
                    [
                        self.ad_manager.create_banner_container(),
                        ft.Container(height=10),
                        ft.Row([
                            ft.IconButton(
                                ft.icons.ARROW_BACK,
                                icon_color=PRIMARY_COLOR,
                                on_click=lambda e: self.page.run_task(self.navigate_to, "main")
                            ),
                            ft.Text(
                                "Essay Generator",
                                font_family="NotoSans" if self.page.fonts else None,
                                size=24,
                                color=PRIMARY_COLOR,
                                weight=ft.FontWeight.BOLD
                            )
                        ], alignment=ft.MainAxisAlignment.START),
                        ft.Container(
                            content=ft.Row([
                                ft.Icon(ft.icons.BOLT, color=ft.colors.ORANGE, size=16),
                                ft.Text(
                                    f"Remaining: {self.generation_manager.get_remaining_generations()} generations",
                                    font_family="NotoSans" if self.page.fonts else None,
                                    size=12,
                                    color=ft.colors.GREY_600
                                )
                            ], alignment=ft.MainAxisAlignment.CENTER),
                            bgcolor=ft.colors.YELLOW_50,
                            padding=8,
                            border_radius=8
                        ),
                        self.topic_input,
                        self.length_input,
                        self.generate_button,
                        ft.Container(
                            content=ft.Column([
                                ft.Text(
                                    "Generated Essay:",
                                    font_family="NotoSans" if self.page.fonts else None,
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
        except Exception as e:
            logger.error(f"Error building EssayScreen: {e}")
            return ft.Container(
                content=ft.Text("Error: Unable to load essay screen", color=ft.colors.RED),
                bgcolor=SECONDARY_COLOR,
                padding=20,
                expand=True
            )

class LetterScreen(BaseScreen):
    """Letter generation screen with ad integration."""
    
    def __init__(self, page: ft.Page, ad_manager: AdManager, data_manager: DataManager,
                 api_client: APIClient, generation_manager: GenerationManager, navigate_to: Callable):
        super().__init__(page, ad_manager)
        self.data_manager = data_manager
        self.api_client = api_client
        self.generation_manager = generation_manager
        self.navigate_to = navigate_to
        self.is_generating = False
        
        self.recipient_input = ft.TextField(
            hint_text="Enter recipient name",
            font_family="NotoSans" if page.fonts else None,
            text_size=16,
            bgcolor=ft.colors.WHITE,
            border_color=PRIMARY_COLOR
        )
        self.purpose_input = ft.TextField(
            hint_text="Enter letter purpose (e.g., 'wedding invitation', 'condolences')",
            font_family="NotoSans" if page.fonts else None,
            text_size=16,
            bgcolor=ft.colors.WHITE,
            border_color=PRIMARY_COLOR,
            multiline=True,
            max_lines=3
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
            font_family="NotoSans" if page.fonts else None,
            size=14,
            color=ft.colors.GREY_600,
            selectable=True
        )
        self.generate_button = ft.ElevatedButton(
            content=ft.Row(
                [ft.Icon(ft.icons.MAIL), ft.Text("Generate Letter", font_family="NotoSans" if page.fonts else None)],
                alignment=ft.MainAxisAlignment.CENTER
            ),
            bgcolor=ACCENT_COLOR,
            color=ft.colors.WHITE,
            on_click=self.on_generate_click
        )
    
    def on_generate_click(self, e):
        if not self.is_generating:
            self.page.run_task(self.generate_letter)
    
    async def generate_letter(self):
        if self.is_generating:
            return
        self.is_generating = True
        self.update_ui_generating(True)
        try:
            recipient = self.recipient_input.value.strip()
            purpose = self.purpose_input.value.strip()
            tone = self.tone_dropdown.value
            if not recipient or not purpose or not tone:
                self.show_error_dialog("Input Error", "Please fill in all fields.")
                return
            if not self.generation_manager.can_generate_free():
                self.show_reward_dialog(self.generation_manager)
                return
            prompt = self.create_letter_prompt(recipient, purpose, tone)
            if self.api_client:
                result = await self.api_client.generate_text(prompt)
            else:
                result = f"""Sawubona {recipient},

Ngiyethemba ukuthi uyaphila futhi ukhongolose. Ngibhala le ncwadi ukuze...

[Demo: Complete {tone} letter to {recipient} for {purpose} would be generated here in proper isiZulu with cultural greetings and closing]

Ngiyabonga kakhulu.

Sala kahle,
[Umlobi]"""
            self.result_text.value = result
            self.result_text.color = ft.colors.BLACK
            self.generation_manager.increment_count()
            if self.generation_manager.generation_count % 4 == 0:
                self.ad_manager.show_interstitial()
        except Exception as e:
            self.show_error_dialog("Generation Error", str(e))
        finally:
            self.is_generating = False
            self.update_ui_generating(False)
    
    def create_letter_prompt(self, recipient: str, purpose: str, tone: str) -> str:
        tone_descriptions = {
            'formal': 'formal and respectful, using proper Zulu honorifics and etiquette',
            'informal': 'friendly and casual while maintaining cultural respect',
            'respectful': 'deeply respectful, incorporating traditional forms of address',
            'ceremonial': 'ceremonial and traditional, appropriate for cultural events'
        }
        return f"""Write a {tone_descriptions.get(tone, tone)} letter in isiZulu to {recipient} for the purpose of {purpose}.

Requirements:
- Use appropriate Zulu letter structure and formatting
- Include proper greetings (Sawubona/Sanibonani) and closings (Sala kahle/Hamba kahle)
- Maintain cultural sensitivity and authenticity
- Use correct honorifics and forms of address
- Incorporate relevant cultural expressions

The letter should be entirely in isiZulu and demonstrate proper understanding of Zulu communication customs and etiquette."""
    
    def update_ui_generating(self, generating: bool):
        try:
            if generating:
                self.generate_button.content = ft.Row(
                    [ft.ProgressRing(width=16, height=16), ft.Text("Generating...", font_family="NotoSans" if self.page.fonts else None)],
                    alignment=ft.MainAxisAlignment.CENTER
                )
                self.generate_button.disabled = True
                self.result_text.value = "Generating your letter... Please wait."
                self.result_text.color = ft.colors.BLUE
            else:
                self.generate_button.content = ft.Row(
                    [ft.Icon(ft.icons.MAIL), ft.Text("Generate Letter", font_family="NotoSans" if self.page.fonts else None)],
                    alignment=ft.MainAxisAlignment.CENTER
                )
                self.generate_button.disabled = False
            self.page.update()
        except Exception as e:
            logger.error(f"Error updating UI: {e}")
            self.result_text.value = "Error: Unable to update UI"
            self.result_text.color = ft.colors.RED
            self.page.update()
    
    def build(self):
        try:
            return ft.Container(
                content=ft.Column(
                    [
                        self.ad_manager.create_banner_container(),
                        ft.Container(height=10),
                        ft.Row([
                            ft.IconButton(
                                ft.icons.ARROW_BACK,
                                icon_color=PRIMARY_COLOR,
                                on_click=lambda e: self.page.run_task(self.navigate_to, "main")
                            ),
                            ft.Text(
                                "Letter Generator",
                                font_family="NotoSans" if self.page.fonts else None,
                                size=24,
                                color=PRIMARY_COLOR,
                                weight=ft.FontWeight.BOLD
                            )
                        ], alignment=ft.MainAxisAlignment.START),
                        ft.Container(
                            content=ft.Row([
                                ft.Icon(ft.icons.BOLT, color=ft.colors.ORANGE, size=16),
                                ft.Text(
                                    f"Remaining: {self.generation_manager.get_remaining_generations()} generations",
                                    font_family="NotoSans" if self.page.fonts else None,
                                    size=12,
                                    color=ft.colors.GREY_600
                                )
                            ], alignment=ft.MainAxisAlignment.CENTER),
                            bgcolor=ft.colors.YELLOW_50,
                            padding=8,
                            border_radius=8
                        ),
                        self.recipient_input,
                        self.purpose_input,
                        self.tone_dropdown,
                        self.generate_button,
                        ft.Container(
                            content=ft.Column([
                                ft.Text(
                                    "Generated Letter:",
                                    font_family="NotoSans" if self.page.fonts else None,
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
        except Exception as e:
            logger.error(f"Error building LetterScreen: {e}")
            return ft.Container(
                content=ft.Text("Error: Unable to load letter screen", color=ft.colors.RED),
                bgcolor=SECONDARY_COLOR,
                padding=20,
                expand=True
            )

class DictionaryScreen(BaseScreen):
    """Dictionary screen for Zulu words."""
    
    def __init__(self, page: ft.Page, ad_manager: AdManager, data_manager: DataManager, navigate_to: Callable):
        super().__init__(page, ad_manager)
        self.data_manager = data_manager
        self.navigate_to = navigate_to
        self.search_input = ft.TextField(
            hint_text="Search for a Zulu word...",
            font_family="NotoSans" if page.fonts else None,
            text_size=16,
            bgcolor=ft.colors.WHITE,
            border_color=PRIMARY_COLOR,
            on_change=self.on_search_change
        )
        self.results_column = ft.Column(
            [
                ft.Text(
                    "Enter a word to search the dictionary",
                    font_family="NotoSans" if page.fonts else None,
                    size=16,
                    color=ft.colors.GREY_600,
                    text_align=ft.TextAlign.CENTER
                )
            ],
            scroll=ft.ScrollMode.AUTO
        )
    
    def on_search_change(self, e):
        search_term = self.search_input.value.strip().lower()
        self.search_dictionary(search_term)
    
    def search_dictionary(self, search_term: str):
        try:
            self.results_column.controls.clear()
            if not search_term:
                self.results_column.controls.append(
                    ft.Text(
                        "Enter a word to search the dictionary",
                        font_family="NotoSans" if self.page.fonts else None,
                        size=16,
                        color=ft.colors.GREY_600,
                        text_align=ft.TextAlign.CENTER
                    )
                )
            else:
                dictionary = self.data_manager.dictionary
                matches = []
                for word, definition in dictionary.items():
                    if search_term in word.lower() or search_term in definition.lower():
                        matches.append((word, definition))
                if matches:
                    self.results_column.controls.append(
                        ft.Text(
                            f"Found {len(matches)} result(s):",
                            font_family="NotoSans" if self.page.fonts else None,
                            size=16,
                            color=PRIMARY_COLOR,
                            weight=ft.FontWeight.BOLD
                        )
                    )
                    for word, definition in matches[:20]:
                        self.results_column.controls.append(
                            ft.Container(
                                content=ft.Column([
                                    ft.Text(
                                        word,
                                        font_family="NotoSans" if self.page.fonts else None,
                                        size=18,
                                        color=ACCENT_COLOR,
                                        weight=ft.FontWeight.BOLD
                                    ),
                                    ft.Text(
                                        definition,
                                        font_family="NotoSans" if self.page.fonts else None,
                                        size=14,
                                        color=ft.colors.BLACK
                                    )
                                ], spacing=5),
                                padding=10,
                                margin=5,
                                bgcolor=ft.colors.WHITE,
                                border_radius=10,
                                border=ft.border.all(1, ft.colors.GREY_300)
                            )
                        )
                else:
                    self.results_column.controls.append(
                        ft.Text(
                            f"No results found for '{search_term}'",
                            font_family="NotoSans" if self.page.fonts else None,
                            size=16,
                            color=ft.colors.RED_400,
                            text_align=ft.TextAlign.CENTER
                        )
                    )
            self.page.update()
        except Exception as e:
            logger.error(f"Error searching dictionary: {e}")
            self.results_column.controls = [
                ft.Text(
                    "Error: Unable to search dictionary",
                    font_family="NotoSans" if self.page.fonts else None,
                    color=ft.colors.RED,
                    text_align=ft.TextAlign.CENTER
                )
            ]
            self.page.update()
    
    def build(self):
        try:
            return ft.Container(
                content=ft.Column(
                    [
                        self.ad_manager.create_banner_container(),
                        ft.Container(height=10),
                        ft.Row([
                            ft.IconButton(
                                ft.icons.ARROW_BACK,
                                icon_color=PRIMARY_COLOR,
                                on_click=lambda e: self.page.run_task(self.navigate_to, "main")
                            ),
                            ft.Text(
                                "Zulu Dictionary",
                                font_family="NotoSans" if self.page.fonts else None,
                                size=24,
                                color=PRIMARY_COLOR,
                                weight=ft.FontWeight.BOLD
                            )
                        ], alignment=ft.MainAxisAlignment.START),
                        ft.Text(
                            "Search for Zulu words and their meanings",
                            font_family="NotoSans" if self.page.fonts else None,
                            size=16,
                            color=ft.colors.GREY_600,
                            text_align=ft.TextAlign.CENTER
                        ),
                        self.search_input,
                        ft.Container(
                            content=self.results_column,
                            expand=True,
                            padding=10
                        )
                    ],
                    spacing=15
                ),
                bgcolor=SECONDARY_COLOR,
                padding=20,
                expand=True
            )
        except Exception as e:
            logger.error(f"Error building DictionaryScreen: {e}")
            return ft.Container(
                content=ft.Text("Error: Unable to load dictionary screen", color=ft.colors.RED),
                bgcolor=SECONDARY_COLOR,
                padding=20,
                expand=True
            )

class ProverbsScreen(BaseScreen):
    """Screen to display Zulu proverbs."""
    
    def __init__(self, page: ft.Page, ad_manager: AdManager, data_manager: DataManager, navigate_to: Callable):
        super().__init__(page, ad_manager)
        self.data_manager = data_manager
        self.navigate_to = navigate_to
        self.proverbs_column = ft.Column(scroll=ft.ScrollMode.AUTO)
        self.load_proverbs()
    
    def load_proverbs(self):
        try:
            proverbs = self.data_manager.proverbs
            if not proverbs:
                self.proverbs_column.controls.append(
                    ft.Text(
                        "No proverbs available. Please check the data file.",
                        font_family="NotoSans" if self.page.fonts else None,
                        color=ft.colors.RED_400,
                        text_align=ft.TextAlign.CENTER
                    )
                )
            else:
                for proverb, meaning in proverbs.items():
                    self.proverbs_column.controls.append(
                        ft.Container(
                            content=ft.Column([
                                ft.Text(
                                    proverb,
                                    font_family="NotoSans" if self.page.fonts else None,
                                    size=18,
                                    color=ACCENT_COLOR,
                                    weight=ft.FontWeight.BOLD
                                ),
                                ft.Text(
                                    meaning,
                                    font_family="NotoSans" if self.page.fonts else None,
                                    size=14,
                                    color=ft.colors.BLACK
                                )
                            ], spacing=5),
                            padding=15,
                            margin=5,
                            bgcolor=ft.colors.WHITE,
                            border_radius=10,
                            border=ft.border.all(1, ft.colors.GREY_300)
                        )
                    )
        except Exception as e:
            logger.error(f"Error loading proverbs: {e}")
            self.proverbs_column.controls = [
                ft.Text(
                    "Error: Unable to load proverbs",
                    font_family="NotoSans" if self.page.fonts else None,
                    color=ft.colors.RED,
                    text_align=ft.TextAlign.CENTER
                )
            ]
    
    def build(self):
        try:
            return ft.Container(
                content=ft.Column(
                    [
                        self.ad_manager.create_banner_container(),
                        ft.Container(height=10),
                        ft.Row([
                            ft.IconButton(
                                ft.icons.ARROW_BACK,
                                icon_color=PRIMARY_COLOR,
                                on_click=lambda e: self.page.run_task(self.navigate_to, "main")
                            ),
                            ft.Text(
                                "Izaga Nezisho",
                                font_family="NotoSans" if self.page.fonts else None,
                                size=24,
                                color=PRIMARY_COLOR,
                                weight=ft.FontWeight.BOLD
                            )
                        ], alignment=ft.MainAxisAlignment.START),
                        ft.Text(
                            "Zulu Proverbs and Sayings",
                            font_family="NotoSans" if self.page.fonts else None,
                            size=16,
                            color=ft.colors.GREY_600,
                            text_align=ft.TextAlign.CENTER
                        ),
                        ft.Container(
                            content=self.proverbs_column,
                            expand=True,
                            padding=10
                        )
                    ],
                    spacing=15
                ),
                bgcolor=SECONDARY_COLOR,
                padding=20,
                expand=True
            )
        except Exception as e:
            logger.error(f"Error building ProverbsScreen: {e}")
            return ft.Container(
                content=ft.Text("Error: Unable to load proverbs screen", color=ft.colors.RED),
                bgcolor=SECONDARY_COLOR,
                padding=20,
                expand=True
            )

class TranslationScreen(BaseScreen):
    """Translation screen with ad integration."""
    
    def __init__(self, page: ft.Page, ad_manager: AdManager, api_client: APIClient,
                 generation_manager: GenerationManager, navigate_to: Callable):
        super().__init__(page, ad_manager)
        self.api_client = api_client
        self.generation_manager = generation_manager
        self.navigate_to = navigate_to
        self.is_translating = False
        
        self.source_input = ft.TextField(
            hint_text="Enter text to translate...",
            font_family="NotoSans" if page.fonts else None,
            text_size=16,
            bgcolor=ft.colors.WHITE,
            border_color=PRIMARY_COLOR,
            multiline=True,
            max_lines=5
        )
        self.direction_dropdown = ft.Dropdown(
            hint_text="Select translation direction",
            options=[
                ft.dropdown.Option("en_to_zu", "English to isiZulu"),
                ft.dropdown.Option("zu_to_en", "isiZulu to English")
            ],
            bgcolor=ft.colors.WHITE,
            border_color=PRIMARY_COLOR
        )
        self.result_text = ft.Text(
            "Translation will appear here...",
            font_family="NotoSans" if page.fonts else None,
            size=14,
            color=ft.colors.GREY_600,
            selectable=True
        )
        self.translate_button = ft.ElevatedButton(
            content=ft.Row(
                [ft.Icon(ft.icons.TRANSLATE), ft.Text("Translate", font_family="NotoSans" if page.fonts else None)],
                alignment=ft.MainAxisAlignment.CENTER
            ),
            bgcolor=ACCENT_COLOR,
            color=ft.colors.WHITE,
            on_click=self.on_translate_click
        )
    
    def on_translate_click(self, e):
        if not self.is_translating:
            self.page.run_task(self.translate_text)
    
    async def translate_text(self):
        if self.is_translating:
            return
        self.is_translating = True
        self.update_ui_translating(True)
        try:
            source_text = self.source_input.value.strip()
            direction = self.direction_dropdown.value
            if not source_text or not direction:
                self.show_error_dialog("Input Error", "Please enter text and select translation direction.")
                return
            if not self.generation_manager.can_generate_free():
                self.show_reward_dialog(self.generation_manager)
                return
            prompt = self.create_translation_prompt(source_text, direction)
            if self.api_client:
                result = await self.api_client.generate_text(prompt)
            else:
                lang_from = "English" if direction == "en_to_zu" else "isiZulu"
                lang_to = "isiZulu" if direction == "en_to_zu" else "English"
                if direction == "en_to_zu":
                    result = f"Sawubona - Demo translation from {lang_from} to {lang_to} would appear here with proper cultural context."
                else:
                    result = f"Hello - Demo translation from {lang_from} to {lang_to} would appear here."
            self.result_text.value = result
            self.result_text.color = ft.colors.BLACK
            self.generation_manager.increment_count()
        except Exception as e:
            self.show_error_dialog("Translation Error", str(e))
        finally:
            self.is_translating = False
            self.update_ui_translating(False)
    
    def create_translation_prompt(self, text: str, direction: str) -> str:
        if direction == "en_to_zu":
            return f"""Translate the following English text to isiZulu. Ensure the translation is culturally appropriate and uses proper Zulu grammar:

English text: {text}

Please provide only the isiZulu translation."""
        else:
            return f"""Translate the following isiZulu text to English. Maintain the cultural context and meaning:

isiZulu text: {text}

Please provide only the English translation."""
    
    def update_ui_translating(self, translating: bool):
        try:
            if translating:
                self.translate_button.content = ft.Row(
                    [ft.ProgressRing(width=16, height=16), ft.Text("Translating...", font_family="NotoSans" if self.page.fonts else None)],
                    alignment=ft.MainAxisAlignment.CENTER
                )
                self.translate_button.disabled = True
                self.result_text.value = "Translating... Please wait."
                self.result_text.color = ft.colors.BLUE
            else:
                self.translate_button.content = ft.Row(
                    [ft.Icon(ft.icons.TRANSLATE), ft.Text("Translate", font_family="NotoSans" if self.page.fonts else None)],
                    alignment=ft.MainAxisAlignment.CENTER
                )
                self.translate_button.disabled = False
            self.page.update()
        except Exception as e:
            logger.error(f"Error updating UI: {e}")
            self.result_text.value = "Error: Unable to update UI"
            self.result_text.color = ft.colors.RED
            self.page.update()
    
    def build(self):
        try:
            return ft.Container(
                content=ft.Column(
                    [
                        self.ad_manager.create_banner_container(),
                        ft.Container(height=10),
                        ft.Row([
                            ft.IconButton(
                                ft.icons.ARROW_BACK,
                                icon_color=PRIMARY_COLOR,
                                on_click=lambda e: self.page.run_task(self.navigate_to, "main")
                            ),
                            ft.Text(
                                "Translation",
                                font_family="NotoSans" if self.page.fonts else None,
                                size=24,
                                color=PRIMARY_COLOR,
                                weight=ft.FontWeight.BOLD
                            )
                        ], alignment=ft.MainAxisAlignment.START),
                        ft.Container(
                            content=ft.Row([
                                ft.Icon(ft.icons.BOLT, color=ft.colors.ORANGE, size=16),
                                ft.Text(
                                    f"Remaining: {self.generation_manager.get_remaining_generations()} generations",
                                    font_family="NotoSans" if self.page.fonts else None,
                                    size=12,
                                    color=ft.colors.GREY_600
                                )
                            ], alignment=ft.MainAxisAlignment.CENTER),
                            bgcolor=ft.colors.YELLOW_50,
                            padding=8,
                            border_radius=8
                        ),
                        self.source_input,
                        self.direction_dropdown,
                        self.translate_button,
                        ft.Container(
                            content=ft.Column([
                                ft.Text(
                                    "Translation:",
                                    font_family="NotoSans" if self.page.fonts else None,
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
        except Exception as e:
            logger.error(f"Error building TranslationScreen: {e}")
            return ft.Container(
                content=ft.Text("Error: Unable to load translation screen", color=ft.colors.RED),
                bgcolor=SECONDARY_COLOR,
                padding=20,
                expand=True
            )

class IsiZuluApp:
    """Main application controller with proper AdMob integration."""
    
    def __init__(self, page: ft.Page):
        self.page = page
        self.data_manager = DataManager()
        self.generation_manager = GenerationManager()
        self.ad_manager = AdManager(page)
        self.current_screen = None
        self.ad_manager.generation_manager = self.generation_manager
        if not OPENROUTER_API_KEY or "e810205cba9f8ab53c36335c95dd4e55803c0be4c9a1f910a488f9a3eae18bfc" in OPENROUTER_API_KEY:
            logger.warning("No valid API key provided. Using demo mode.")
            self.api_client = None
        else:
            try:
                self.api_client = APIClient(OPENROUTER_API_KEY)
            except ValueError as e:
                logger.error(f"Invalid API key: {e}")
                self.api_client = None
    
    async def navigate_to(self, screen_name: str):
        self.page.controls.clear()
        try:
            logger.info(f"Navigating to {screen_name}")
            if screen_name == "main":
                screen = MainScreen(
                    self.page,
                    self.ad_manager,
                    self.navigate_to,
                    self.generation_manager
                )
            elif screen_name == "essay":
                screen = EssayScreen(
                    self.page,
                    self.ad_manager,
                    self.data_manager,
                    self.api_client,
                    self.generation_manager,
                    self.navigate_to
                )
            elif screen_name == "letter":
                screen = LetterScreen(
                    self.page,
                    self.ad_manager,
                    self.data_manager,
                    self.api_client,
                    self.generation_manager,
                    self.navigate_to
                )
            elif screen_name == "dictionary":
                screen = DictionaryScreen(
                    self.page,
                    self.ad_manager,
                    self.data_manager,
                    self.navigate_to
                )
            elif screen_name == "proverbs":
                screen = ProverbsScreen(
                    self.page,
                    self.ad_manager,
                    self.data_manager,
                    self.navigate_to
                )
            elif screen_name == "translation":
                screen = TranslationScreen(
                    self.page,
                    self.ad_manager,
                    self.api_client,
                    self.generation_manager,
                    self.navigate_to
                )
            else:
                logger.warning(f"Unknown screen: {screen_name}, defaulting to main")
                await self.navigate_to("main")
                return
            self.current_screen = screen
            self.page.add(screen.build())
            self.page.update()
        except Exception as e:
            logger.error(f"Navigation error to {screen_name}: {e}")
            self.page.controls.clear()
            self.page.add(
                ft.Container(
                    content=ft.Text(f"Error: Failed to load {screen_name} screen", color=ft.colors.RED),
                    bgcolor=SECONDARY_COLOR,
                    padding=20,
                    expand=True
                )
            )
            self.page.update()
            if screen_name != "main":
                await self.navigate_to("main")

def main(page: ft.Page):
    """Main application entry point with proper AdMob setup."""
    try:
        logger.info("Starting main function")
        page.title = "isiZulu AI Writer"
        page.bgcolor = SECONDARY_COLOR
        page.theme_mode = ft.ThemeMode.LIGHT
        page.vertical_alignment = ft.MainAxisAlignment.START
        try:
            page.fonts = {
                "NotoSans": "assets/fonts/NotoSans-Regular.ttf"
            }
            logger.info("Custom font loaded")
        except Exception as e:
            logger.warning(f"Failed to load custom font: {e}")
            page.fonts = None
        app = IsiZuluApp(page)
        logger.info("App initialized")
        page.run_task(app.navigate_to, "main")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        page.add(
            ft.Container(
                content=ft.Text("Error: Application failed to start", color=ft.colors.RED),
                bgcolor=SECONDARY_COLOR,
                padding=20,
                expand=True
            )
        )
        page.update()

if __name__ == "__main__":
    ft.app(target=main, assets_dir="assets")

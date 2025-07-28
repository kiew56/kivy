from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.graphics import Color, Rectangle
from kivy.core.text import LabelBase
from kivy.core.window import Window
import json
import requests
import asyncio
import platform

# AdMob imports for Android
try:
    from jnius import autoclass
    Activity = autoclass('org.kivy.android.PythonActivity')
    MobileAds = autoclass('com.google.android.gms.ads.MobileAds')
    BannerAdView = autoclass('com.google.android.gms.ads.AdView')
    AdRequest = autoclass('com.google.android.gms.ads.AdRequest')
    RewardedAd = autoclass('com.google.android.gms.ads.rewarded.RewardedAd')
    RewardedAdLoadCallback = autoclass('com.google.android.gms.ads.rewarded.RewardedAdLoadCallback')
    OnUserEarnedRewardListener = autoclass('com.google.android.gms.ads.OnUserEarnedRewardListener')
    IS_ANDROID = True
except ImportError:
    IS_ANDROID = False

# Register custom fonts
LabelBase.register(name='NotoSans', fn_regular='NotoSans-Regular.ttf')
LabelBase.register(name='FontAwesome', fn_regular='fontawesome-webfont.ttf')

# AdMob configuration (replace with your AdMob IDs)
BANNER_AD_UNIT_ID = "YOUR_BANNER_AD_UNIT_ID"
REWARDED_AD_UNIT_ID = "YOUR_REWARDED_AD_UNIT_ID"

# OpenRouter API configuration
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Color scheme
PRIMARY_COLOR = (0.12, 0.67, 0.47, 1)  # Greenish teal
SECONDARY_COLOR = (0.95, 0.95, 0.95, 1)  # Light gray
ACCENT_COLOR = (0.9, 0.3, 0.3, 1)  # Coral red

# Global generation counter
GENERATION_COUNT = 0

class LoadingScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        with self.canvas.before:
            Color(*SECONDARY_COLOR)
            self.rect = Rectangle(size=Window.size, pos=self.pos)
            self.bind(pos=self.update_rect, size=self.update_rect)
        
        # Logo (replace 'logo.png' with your actual logo path)
        self.logo = Image(source='logo.png', size_hint=(None, None), size=(dp(150), dp(150)), pos_hint={'center_x': 0.5})
        self.loading_label = Label(
            text='[font=NotoSans]Loading isiZulu AI Writer...[/font]',
            markup=True,
            font_size=dp(20),
            color=PRIMARY_COLOR,
            size_hint=(1, 0.2)
        )
        self.layout.add_widget(self.logo)
        self.layout.add_widget(self.loading_label)
        self.add_widget(self.layout)
    
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    
    def on_enter(self):
        Clock.schedule_once(self.switch_to_main, 2)  # Show loading screen for 2 seconds
    
    def switch_to_main(self, dt):
        self.manager.current = 'main'

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        with self.canvas.before:
            Color(*SECONDARY_COLOR)
            self.rect = Rectangle(size=Window.size, pos=self.pos)
            self.bind(pos=self.update_rect, size=self.update_rect)
        
        # Banner Ad
        if IS_ANDROID:
            self.banner_ad = BannerAdView(Activity.mActivity)
            self.banner_ad.setAdUnitId(BANNER_AD_UNIT_ID)
            self.banner_ad.setAdSize(autoclass('com.google.android.gms.ads.AdSize').BANNER)
            ad_request = AdRequest.Builder().build()
            self.banner_ad.loadAd(ad_request)
            self.banner_ad_view = self.banner_ad  # Direct use of AdView
        else:
            self.banner_ad_view = Label(
                text='[font=NotoSans]Banner Ad Placeholder[/font]',
                markup=True,
                font_size=dp(16),
                color=(0, 0, 0, 1),
                size_hint=(1, 0.1)
            )
        self.layout.add_widget(self.banner_ad_view)
        
        # Title
        self.title = Label(
            text='[font=NotoSans]isiZulu AI Writer[/font]',
            markup=True,
            font_size=dp(24),
            color=PRIMARY_COLOR,
            size_hint=(1, 0.2)
        )
        
        # Navigation buttons with FontAwesome icons
        self.essay_btn = Button(
            text='[font=FontAwesome]\uf044[/font] [font=NotoSans]Write Essay[/font]',
            markup=True,
            font_size=dp(18),
            background_color=PRIMARY_COLOR,
            background_normal='',
            size_hint=(1, 0.15)
        )
        self.letter_btn = Button(
            text='[font=FontAwesome]\uf0e0[/font] [font=NotoSans]Write Letter[/font]',
            markup=True,
            font_size=dp(18),
            background_color=PRIMARY_COLOR,
            background_normal='',
            size_hint=(1, 0.15)
        )
        self.dict_btn = Button(
            text='[font=FontAwesome]\uf02d[/font] [font=NotoSans]Zulu Dictionary[/font]',
            markup=True,
            font_size=dp(18),
            background_color=PRIMARY_COLOR,
            background_normal='',
            size_hint=(1, 0.15)
        )
        self.proverb_btn = Button(
            text='[font=FontAwesome]\uf10d[/font] [font=NotoSans]Izaga Nezisho[/font]',
            markup=True,
            font_size=dp(18),
            background_color=PRIMARY_COLOR,
            background_normal='',
            size_hint=(1, 0.15)
        )
        self.translate_btn = Button(
            text='[font=FontAwesome]\uf0ac[/font] [font=NotoSans]Translation[/font]',
            markup=True,
            font_size=dp(18),
            background_color=PRIMARY_COLOR,
            background_normal='',
            size_hint=(1, 0.15)
        )
        
        self.essay_btn.bind(on_press=self.go_to_essay)
        self.letter_btn.bind(on_press=self.go_to_letter)
        self.dict_btn.bind(on_press=self.go_to_dictionary)
        self.proverb_btn.bind(on_press=self.go_to_proverbs)
        self.translate_btn.bind(on_press=self.go_to_translation)
        
        self.layout.add_widget(self.title)
        self.layout.add_widget(self.essay_btn)
        self.layout.add_widget(self.letter_btn)
        self.layout.add_widget(self.dict_btn)
        self.layout.add_widget(self.proverb_btn)
        self.layout.add_widget(self.translate_btn)
        
        self.add_widget(self.layout)
    
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    
    def go_to_essay(self, instance):
        self.manager.current = 'essay'
    
    def go_to_letter(self, instance):
        self.manager.current = 'letter'
    
    def go_to_dictionary(self, instance):
        self.manager.current = 'dictionary'
    
    def go_to_proverbs(self, instance):
        self.manager.current = 'proverbs'
    
    def go_to_translation(self, instance):
        self.manager.current = 'translation'

class EssayScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.can_generate = False
        if IS_ANDROID:
            self.rewarded_ad = None
            self.load_rewarded_ad()
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        with self.canvas.before:
            Color(*SECONDARY_COLOR)
            self.rect = Rectangle(size=Window.size, pos=self.pos)
            self.bind(pos=self.update_rect, size=self.update_rect)
        
        # Banner Ad
        if IS_ANDROID:
            self.banner_ad = BannerAdView(Activity.mActivity)
            self.banner_ad.setAdUnitId(BANNER_AD_UNIT_ID)
            self.banner_ad.setAdSize(autoclass('com.google.android.gms.ads.AdSize').BANNER)
            ad_request = AdRequest.Builder().build()
            self.banner_ad.loadAd(ad_request)
            self.banner_ad_view = self.banner_ad
        else:
            self.banner_ad_view = Label(
                text='[font=NotoSans]Banner Ad Placeholder[/font]',
                markup=True,
                font_size=dp(16),
                color=(0, 0, 0, 1),
                size_hint=(1, 0.1)
            )
        self.layout.add_widget(self.banner_ad_view)
        
        self.topic_input = TextInput(
            hint_text='Enter essay topic',
            font_name='NotoSans',
            font_size=dp(16),
            background_color=(1, 1, 1, 1),
            foreground_color=(0, 0, 0, 1),
            size_hint=(1, 0.1)
        )
        self.length_input = TextInput(
            hint_text='Enter word count (e.g., 500)',
            font_name='NotoSans',
            font_size=dp(16),
            background_color=(1, 1, 1, 1),
            foreground_color=(0, 0, 0, 1),
            size_hint=(1, 0.1)
        )
        self.generate_btn = Button(
            text='[font=FontAwesome]\uf061[/font] [font=NotoSans]Generate Essay[/font]',
            markup=True,
            font_size=dp(18),
            background_color=ACCENT_COLOR,
            background_normal='',
            size_hint=(1, 0.1)
        )
        self.result_label = Label(
            text='',
            font_name='NotoSans',
            font_size=dp(16),
            color=(0, 0, 0, 1),
            size_hint=(1, None),
            text_size=(Window.width - dp(40), None),
            halign='left',
            valign='top'
        )
        self.scroll = ScrollView(size_hint=(1, 0.6))
        self.scroll.add_widget(self.result_label)
        
        self.generate_btn.bind(on_press=self.generate_essay)
        
        self.layout.add_widget(self.topic_input)
        self.layout.add_widget(self.length_input)
        self.layout.add_widget(self.generate_btn)
        self.layout.add_widget(self.scroll)
        self.add_widget(self.layout)
        
        self.load_proverbs()
    
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    
    def load_proverbs(self):
        try:
            with open('izaga_nezisho.json', 'r', encoding='utf-8') as f:
                self.proverbs = json.load(f)
        except Exception as e:
            popup = Popup(
                title='Error',
                content=Label(
                    text=f'Error loading proverbs: {str(e)}',
                    font_name='NotoSans',
                    font_size=dp(16)
                ),
                size_hint=(0.8, 0.4)
            )
            popup.open()
            self.proverbs = {}
    
    def load_rewarded_ad(self):
        if IS_ANDROID:
            class CustomRewardedAdLoadCallback(RewardedAdLoadCallback):
                def onAdLoaded(self, rewarded_ad):
                    self.rewarded_ad = rewarded_ad
                    print("Rewarded ad loaded")
                
                def onAdFailedToLoad(self, load_ad_error):
                    print(f"Rewarded ad failed to load: {load_ad_error}")
                    popup = Popup(
                        title='Ad Error',
                        content=Label(
                            text='Failed to load ad. Please try again.',
                            font_name='NotoSans',
                            font_size=dp(16)
                        ),
                        size_hint=(0.8, 0.4)
                    )
                    popup.open()
            
            RewardedAd.load(
                Activity.mActivity,
                REWARDED_AD_UNIT_ID,
                AdRequest.Builder().build(),
                CustomRewardedAdLoadCallback()
            )
    
    def on_user_earned_reward(self, *args):
        self.can_generate = True
        if platform.system() == "Emscripten":
            asyncio.ensure_future(self.generate_essay_async())
        else:
            asyncio.run(self.generate_essay_async())
    
    def check_and_generate(self):
        global GENERATION_COUNT
        if GENERATION_COUNT < 2:
            self.can_generate = True
            if platform.system() == "Emscripten":
                asyncio.ensure_future(self.generate_essay_async())
            else:
                asyncio.run(self.generate_essay_async())
        else:
            if IS_ANDROID:
                if self.rewarded_ad:
                    class CustomRewardListener(OnUserEarnedRewardListener):
                        def onUserEarnedReward(self, reward_item):
                            self.on_user_earned_reward()
                    
                    self.rewarded_ad.show(Activity.mActivity, CustomRewardListener())
                else:
                    self.load_rewarded_ad()
                    popup = Popup(
                        title='Ad Loading',
                        content=Label(
                            text='Ad is loading, please try again.',
                            font_name='NotoSans',
                            font_size=dp(16)
                        ),
                        size_hint=(0.8, 0.4)
                    )
                    popup.open()
            else:
                print("Simulating rewarded ad on non-Android platform")
                self.can_generate = True
                if platform.system() == "Emscripten":
                    asyncio.ensure_future(self.generate_essay_async())
                else:
                    asyncio.run(self.generate_essay_async())
    
    async def generate_essay_async(self):
        global GENERATION_COUNT
        if not self.can_generate:
            return
        
        topic = self.topic_input.text
        length = self.length_input.text
        if not topic or not length:
            popup = Popup(
                title='Error',
                content=Label(
                    text='Please fill in all fields',
                    font_name='NotoSans',
                    font_size=dp(16)
                ),
                size_hint=(0.8, 0.4)
            )
            popup.open()
            return
        
        # Prepare proverbs for inclusion in the prompt
        proverbs_text = "\n".join([f"{proverb}: {meaning}" for proverb, meaning in self.proverbs.items()])
        prompt = (
            f"Write a well-structured essay in isiZulu on the topic '{topic}' with approximately {length} words. "
            f"Incorporate relevant isiZulu proverbs (izaga nezisho) to enhance the cultural context and structure of the essay. "
            f"Here are some proverbs to consider:\n{proverbs_text}\n"
            f"Ensure the essay has a clear introduction, body, and conclusion, using the proverbs to support your arguments where appropriate."
        )
        
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "deepseek/deepseek-r1:free",
                "messages": [{"role": "user", "content": prompt}]
            }
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            self.result_label.text = result['choices'][0]['message']['content']
            GENERATION_COUNT += 1
            self.can_generate = False
        except Exception as e:
            popup = Popup(
                title='Error',
                content=Label(
                    text=f'Error generating essay: {str(e)}',
                    font_name='NotoSans',
                    font_size=dp(16)
                ),
                size_hint=(0.8, 0.4)
            )
            popup.open()
    
    def generate_essay(self, instance):
        self.check_and_generate()

class LetterScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.can_generate = False
        if IS_ANDROID:
            self.rewarded_ad = None
            self.load_rewarded_ad()
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        with self.canvas.before:
            Color(*SECONDARY_COLOR)
            self.rect = Rectangle(size=Window.size, pos=self.pos)
            self.bind(pos=self.update_rect, size=self.update_rect)
        
        # Banner Ad
        if IS_ANDROID:
            self.banner_ad = BannerAdView(Activity.mActivity)
            self.banner_ad.setAdUnitId(BANNER_AD_UNIT_ID)
            self.banner_ad.setAdSize(autoclass('com.google.android.gms.ads.AdSize').BANNER)
            ad_request = AdRequest.Builder().build()
            self.banner_ad.loadAd(ad_request)
            self.banner_ad_view = self.banner_ad
        else:
            self.banner_ad_view = Label(
                text='[font=NotoSans]Banner Ad Placeholder[/font]',
                markup=True,
                font_size=dp(16),
                color=(0, 0, 0, 1),
                size_hint=(1, 0.1)
            )
        self.layout.add_widget(self.banner_ad_view)
        
        self.recipient_input = TextInput(
            hint_text='Enter recipient name',
            font_name='NotoSans',
            font_size=dp(16),
            background_color=(1, 1, 1, 1),
            foreground_color=(0, 0, 0, 1),
            size_hint=(1, 0.1)
        )
        self.purpose_input = TextInput(
            hint_text='Enter letter purpose',
            font_name='NotoSans',
            font_size=dp(16),
            background_color=(1, 1, 1, 1),
            foreground_color=(0, 0, 0, 1),
            size_hint=(1, 0.1)
        )
        self.tone_input = TextInput(
            hint_text='Enter tone (e.g., formal, informal)',
            font_name='NotoSans',
            font_size=dp(16),
            background_color=(1, 1, 1, 1),
            foreground_color=(0, 0, 0, 1),
            size_hint=(1, 0.1)
        )
        self.generate_btn = Button(
            text='[font=FontAwesome]\uf061[/font] [font=NotoSans]Generate Letter[/font]',
            markup=True,
            font_size=dp(18),
            background_color=ACCENT_COLOR,
            background_normal='',
            size_hint=(1, 0.1)
        )
        self.result_label = Label(
            text='',
            font_name='NotoSans',
            font_size=dp(16),
            color=(0, 0, 0, 1),
            size_hint=(1, None),
            text_size=(Window.width - dp(40), None),
            halign='left',
            valign='top'
        )
        self.scroll = ScrollView(size_hint=(1, 0.5))
        self.scroll.add_widget(self.result_label)
        
        self.generate_btn.bind(on_press=self.generate_letter)
        
        self.layout.add_widget(self.recipient_input)
        self.layout.add_widget(self.purpose_input)
        self.layout.add_widget(self.tone_input)
        self.layout.add_widget(self.generate_btn)
        self.layout.add_widget(self.scroll)
        self.add_widget(self.layout)
    
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    
    def load_rewarded_ad(self):
        if IS_ANDROID:
            class CustomRewardedAdLoadCallback(RewardedAdLoadCallback):
                def onAdLoaded(self, rewarded_ad):
                    self.rewarded_ad = rewarded_ad
                    print("Rewarded ad loaded")
                
                def onAdFailedToLoad(self, load_ad_error):
                    print(f"Rewarded ad failed to load: {load_ad_error}")
                    popup = Popup(
                        title='Ad Error',
                        content=Label(
                            text='Failed to load ad. Please try again.',
                            font_name='NotoSans',
                            font_size=dp(16)
                        ),
                        size_hint=(0.8, 0.4)
                    )
                    popup.open()
            
            RewardedAd.load(
                Activity.mActivity,
                REWARDED_AD_UNIT_ID,
                AdRequest.Builder().build(),
                CustomRewardedAdLoadCallback()
            )
    
    def on_user_earned_reward(self, *args):
        self.can_generate = True
        if platform.system() == "Emscripten":
            asyncio.ensure_future(self.generate_letter_async())
        else:
            asyncio.run(self.generate_letter_async())
    
    def check_and_generate(self):
        global GENERATION_COUNT
        if GENERATION_COUNT < 2:
            self.can_generate = True
            if platform.system() == "Emscripten":
                asyncio.ensure_future(self.generate_letter_async())
            else:
                asyncio.run(self.generate_letter_async())
        else:
            if IS_ANDROID:
                if self.rewarded_ad:
                    class CustomRewardListener(OnUserEarnedRewardListener):
                        def onUserEarnedReward(self, reward_item):
                            self.on_user_earned_reward()
                    
                    self.rewarded_ad.show(Activity.mActivity, CustomRewardListener())
                else:
                    self.load_rewarded_ad()
                    popup = Popup(
                        title='Ad Loading',
                        content=Label(
                            text='Ad is loading, please try again.',
                            font_name='NotoSans',
                            font_size=dp(16)
                        ),
                        size_hint=(0.8, 0.4)
                    )
                    popup.open()
            else:
                print("Simulating rewarded ad on non-Android platform")
                self.can_generate = True
                if platform.system() == "Emscripten":
                    asyncio.ensure_future(self.generate_letter_async())
                else:
                    asyncio.run(self.generate_letter_async())
    
    async def generate_letter_async(self):
        global GENERATION_COUNT
        if not self.can_generate:
            return
        
        recipient = self.recipient_input.text
        purpose = self.purpose_input.text
        tone = self.tone_input.text
        if not recipient or not purpose or not tone:
            popup = Popup(
                title='Error',
                content=Label(
                    text='Please fill in all fields',
                    font_name='NotoSans',
                    font_size=dp(16)
                ),
                size_hint=(0.8, 0.4)
            )
            popup.open()
            return
        
        prompt = f"Write a {tone} letter in isiZulu to {recipient} for the purpose of {purpose}."
        
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "deepseek/deepseek-r1:free",
                "messages": [{"role": "user", "content": prompt}]
            }
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            self.result_label.text = result['choices'][0]['message']['content']
            GENERATION_COUNT += 1
            self.can_generate = False
        except Exception as e:
            popup = Popup(
                title='Error',
                content=Label(
                    text=f'Error generating letter: {str(e)}',
                    font_name='NotoSans',
                    font_size=dp(16)
                ),
                size_hint=(0.8, 0.4)
            )
            popup.open()
    
    def generate_letter(self, instance):
        self.check_and_generate()

class DictionaryScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        with self.canvas.before:
            Color(*SECONDARY_COLOR)
            self.rect = Rectangle(size=Window.size, pos=self.pos)
            self.bind(pos=self.update_rect, size=self.update_rect)
        
        # Banner Ad
        if IS_ANDROID:
            self.banner_ad = BannerAdView(Activity.mActivity)
            self.banner_ad.setAdUnitId(BANNER_AD_UNIT_ID)
            self.banner_ad.setAdSize(autoclass('com.google.android.gms.ads.AdSize').BANNER)
            ad_request = AdRequest.Builder().build()
            self.banner_ad.loadAd(ad_request)
            self.banner_ad_view = self.banner_ad
        else:
            self.banner_ad_view = Label(
                text='[font=NotoSans]Banner Ad Placeholder[/font]',
                markup=True,
                font_size=dp(16),
                color=(0, 0, 0, 1),
                size_hint=(1, 0.1)
            )
        self.layout.add_widget(self.banner_ad_view)
        
        self.search_input = TextInput(
            hint_text='Search Zulu word',
            font_name='NotoSans',
            font_size=dp(16),
            background_color=(1, 1, 1, 1),
            foreground_color=(0, 0, 0, 1),
            size_hint=(1, 0.1)
        )
        self.search_btn = Button(
            text='[font=FontAwesome]\uf002[/font] [font=NotoSans]Search[/font]',
            markup=True,
            font_size=dp(18),
            background_color=ACCENT_COLOR,
            background_normal='',
            size_hint=(1, 0.1)
        )
        self.result_grid = GridLayout(cols=1, size_hint_y=None, spacing=10)
        self.result_grid.bind(minimum_height=self.result_grid.setter('height'))
        self.scroll = ScrollView(size_hint=(1, 0.7))
        self.scroll.add_widget(self.result_grid)
        
        self.search_btn.bind(on_press=self.search_dictionary)
        
        self.layout.add_widget(self.search_input)
        self.layout.add_widget(self.search_btn)
        self.layout.add_widget(self.scroll)
        self.add_widget(self.layout)
        
        self.load_dictionary()
    
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    
    def load_dictionary(self):
        try:
            with open('zulu_dictionary.json', 'r', encoding='utf-8') as f:
                self.dictionary = json.load(f)
        except Exception as e:
            popup = Popup(
                title='Error',
                content=Label(
                    text=f'Error loading dictionary: {str(e)}',
                    font_name='NotoSans',
                    font_size=dp(16)
                ),
                size_hint=(0.8, 0.4)
            )
            popup.open()
            self.dictionary = {}
    
    def search_dictionary(self, instance):
        self.result_grid.clear_widgets()
        search_term = self.search_input.text.lower().strip()
        
        if not search_term:
            for word, meaning in self.dictionary.items():
                self.result_grid.add_widget(Label(
                    text=f"[font=NotoSans]{word}: {meaning}[/font]",
                    markup=True,
                    size_hint_y=None,
                    height=dp(40),
                    text_size=(self.width - dp(20), None),
                    color=(0, 0, 0, 1)
                ))
        else:
            for word, meaning in self.dictionary.items():
                if search_term in word.lower():
                    self.result_grid.add_widget(Label(
                        text=f"[font=NotoSans]{word}: {meaning}[/font]",
                        markup=True,
                        size_hint_y=None,
                        height=dp(40),
                        text_size=(self.width - dp(20), None),
                        color=(0, 0, 0, 1)
                    ))

class ProverbsScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        with self.canvas.before:
            Color(*SECONDARY_COLOR)
            self.rect = Rectangle(size=Window.size, pos=self.pos)
            self.bind(pos=self.update_rect, size=self.update_rect)
        
        # Banner Ad
        if IS_ANDROID:
            self.banner_ad = BannerAdView(Activity.mActivity)
            self.banner_ad.setAdUnitId(BANNER_AD_UNIT_ID)
            self.banner_ad.setAdSize(autoclass('com.google.android.gms.ads.AdSize').BANNER)
            ad_request = AdRequest.Builder().build()
            self.banner_ad.loadAd(ad_request)
            self.banner_ad_view = self.banner_ad
        else:
            self.banner_ad_view = Label(
                text='[font=NotoSans]Banner Ad Placeholder[/font]',
                markup=True,
                font_size=dp(16),
                color=(0, 0, 0, 1),
                size_hint=(1, 0.1)
            )
        self.layout.add_widget(self.banner_ad_view)
        
        self.search_input = TextInput(
            hint_text='Search Izaga Nezisho',
            font_name='NotoSans',
            font_size=dp(16),
            background_color=(1, 1, 1, 1),
            foreground_color=(0, 0, 0, 1),
            size_hint=(1, 0.1)
        )
        self.search_btn = Button(
            text='[font=FontAwesome]\uf002[/font] [font=NotoSans]Search[/font]',
            markup=True,
            font_size=dp(18),
            background_color=ACCENT_COLOR,
            background_normal='',
            size_hint=(1, 0.1)
        )
        self.result_grid = GridLayout(cols=1, size_hint_y=None, spacing=10)
        self.result_grid.bind(minimum_height=self.result_grid.setter('height'))
        self.scroll = ScrollView(size_hint=(1, 0.7))
        self.scroll.add_widget(self.result_grid)
        
        self.search_btn.bind(on_press=self.search_proverbs)
        
        self.layout.add_widget(self.search_input)
        self.layout.add_widget(self.search_btn)
        self.layout.add_widget(self.scroll)
        self.add_widget(self.layout)
        
        self.load_proverbs()
    
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    
    def load_proverbs(self):
        try:
            with open('izaga_nezisho.json', 'r', encoding='utf-8') as f:
                self.proverbs = json.load(f)
        except Exception as e:
            popup = Popup(
                title='Error',
                content=Label(
                    text=f'Error loading proverbs: {str(e)}',
                    font_name='NotoSans',
                    font_size=dp(16)
                ),
                size_hint=(0.8, 0.4)
            )
            popup.open()
            self.proverbs = {}
    
    def search_proverbs(self, instance):
        self.result_grid.clear_widgets()
        search_term = self.search_input.text.lower().strip()
        
        if not search_term:
            for proverb, meaning in self.proverbs.items():
                self.result_grid.add_widget(Label(
                    text=f"[font=NotoSans]{proverb}: {meaning}[/font]",
                    markup=True,
                    size_hint_y=None,
                    height=dp(40),
                    text_size=(self.width - dp(20), None),
                    color=(0, 0, 0, 1)
                ))
        else:
            for proverb, meaning in self.proverbs.items():
                if search_term in proverb.lower():
                    self.result_grid.add_widget(Label(
                        text=f"[font=NotoSans]{proverb}: {meaning}[/font]",
                        markup=True,
                        size_hint_y=None,
                        height=dp(40),
                        text_size=(self.width - dp(20), None),
                        color=(0, 0, 0, 1)
                    ))

class TranslationScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        with self.canvas.before:
            Color(*SECONDARY_COLOR)
            self.rect = Rectangle(size=Window.size, pos=self.pos)
            self.bind(pos=self.update_rect, size=self.update_rect)
        
        # Banner Ad
        if IS_ANDROID:
            self.banner_ad = BannerAdView(Activity.mActivity)
            self.banner_ad.setAdUnitId(BANNER_AD_UNIT_ID)
            self.banner_ad.setAdSize(autoclass('com.google.android.gms.ads.AdSize').BANNER)
            ad_request = AdRequest.Builder().build()
            self.banner_ad.loadAd(ad_request)
            self.banner_ad_view = self.banner_ad
        else:
            self.banner_ad_view = Label(
                text='[font=NotoSans]Banner Ad Placeholder[/font]',
                markup=True,
                font_size=dp(16),
                color=(0, 0, 0, 1),
                size_hint=(1, 0.1)
            )
        self.layout.add_widget(self.banner_ad_view)
        
        self.input_text = TextInput(
            hint_text='Enter text to translate to isiZulu',
            font_name='NotoSans',
            font_size=dp(16),
            background_color=(1, 1, 1, 1),
            foreground_color=(0, 0, 0, 1),
            size_hint=(1, 0.2)
        )
        self.translate_btn = Button(
            text='[font=FontAwesome]\uf0ac[/font] [font=NotoSans]Translate[/font]',
            markup=True,
            font_size=dp(18),
            background_color=ACCENT_COLOR,
            background_normal='',
            size_hint=(1, 0.1)
        )
        self.result_label = Label(
            text='',
            font_name='NotoSans',
            font_size=dp(16),
            color=(0, 0, 0, 1),
            size_hint=(1, None),
            text_size=(Window.width - dp(40), None),
            halign='left',
            valign='top'
        )
        self.scroll = ScrollView(size_hint=(1, 0.6))
        self.scroll.add_widget(self.result_label)
        
        self.translate_btn.bind(on_press=self.translate_text)
        
        self.layout.add_widget(self.input_text)
        self.layout.add_widget(self.translate_btn)
        self.layout.add_widget(self.scroll)
        self.add_widget(self.layout)
    
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    
    async def translate_text_async(self):
        text = self.input_text.text
        if not text:
            popup = Popup(
                title='Error',
                content=Label(
                    text='Please enter text to translate',
                    font_name='NotoSans',
                    font_size=dp(16)
                ),
                size_hint=(0.8, 0.4)
            )
            popup.open()
            return
        
        prompt = f"Translate the following text to isiZulu: {text}"
        
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "deepseek/deepseek-r1:free",
                "messages": [{"role": "user", "content": prompt}]
            }
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            self.result_label.text = result['choices'][0]['message']['content']
        except Exception as e:
            popup = Popup(
                title='Error',
                content=Label(
                    text=f'Error translating text: {str(e)}',
                    font_name='NotoSans',
                    font_size=dp(16)
                ),
                size_hint=(0.8, 0.4)
            )
            popup.open()
    
    def translate_text(self, instance):
        if platform.system() == "Emscripten":
            asyncio.ensure_future(self.translate_text_async())
        else:
            asyncio.run(self.translate_text_async())

class ZuluAIWriterApp(App):
    def build(self):
        if IS_ANDROID:
            MobileAds.initialize(Activity.mActivity)
        sm = ScreenManager(transition=FadeTransition())
        sm.add_widget(LoadingScreen(name='loading'))
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(EssayScreen(name='essay'))
        sm.add_widget(LetterScreen(name='letter'))
        sm.add_widget(DictionaryScreen(name='dictionary'))
        sm.add_widget(ProverbsScreen(name='proverbs'))
        sm.add_widget(TranslationScreen(name='translation'))
        return sm

async def main():
    app = ZuluAIWriterApp()
    app.run()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
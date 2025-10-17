"""
Simplified file creation script - creates vision agent files one by one.
Run this from your project root directory.

Usage: python create_files_simple.py
"""

import os
from pathlib import Path

def create_browser_file():
    """Create browser.py"""
    content = '''"""
Browser manager for Playwright automation with screenshot capabilities.
"""
import os
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright, Browser, Page, BrowserContext

class BrowserManager:
    """Manages Playwright browser lifecycle and page interactions."""
    
    def __init__(self, headless: bool = True, screenshots_dir: str = "screenshots"):
        self.headless = headless
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def start(self):
        """Initialize browser and create a new page."""
        if self._browser is not None:
            return  # Already started
        
        self._playwright = await async_playwright().start()
        
        # Launch browser with reasonable defaults
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled'
            ]
        )
        
        # Create context with realistic viewport
        self._context = await self._browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        
        # Create new page
        self._page = await self._context.new_page()
        
        # Set reasonable timeouts
        self._page.set_default_timeout(30000)  # 30 seconds
        self._page.set_default_navigation_timeout(60000)  # 60 seconds
        
    async def close(self):
        """Close browser and cleanup resources."""
        if self._page:
            await self._page.close()
            self._page = None
        
        if self._context:
            await self._context.close()
            self._context = None
        
        if self._browser:
            await self._browser.close()
            self._browser = None
        
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
    
    @property
    def page(self) -> Page:
        """Get current page instance."""
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        return self._page
    
    async def navigate(self, url: str, wait_until: str = "networkidle") -> Dict[str, Any]:
        """
        Navigate to a URL and wait for page load.
        
        Args:
            url: Target URL
            wait_until: Wait condition ('load', 'domcontentloaded', 'networkidle')
        
        Returns:
            Dict with navigation result and screenshot path
        """
        try:
            # Navigate to URL
            response = await self.page.goto(url, wait_until=wait_until)
            
            # Wait a bit for dynamic content
            await asyncio.sleep(1)
            
            # Capture screenshot
            screenshot_path = await self.capture_screenshot(f"navigate_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            return {
                "success": True,
                "url": self.page.url,
                "title": await self.page.title(),
                "status": response.status if response else None,
                "screenshot": screenshot_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    async def capture_screenshot(self, name: str = None) -> str:
        """
        Capture a screenshot of the current page.
        
        Args:
            name: Optional name for the screenshot file
        
        Returns:
            Path to the saved screenshot
        """
        if name is None:
            name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure .png extension
        if not name.endswith('.png'):
            name += '.png'
        
        screenshot_path = self.screenshots_dir / name
        
        # Capture full page screenshot
        await self.page.screenshot(
            path=str(screenshot_path),
            full_page=True
        )
        
        return str(screenshot_path)
    
    async def get_page_state(self) -> Dict[str, Any]:
        """
        Get current page state information.
        
        Returns:
            Dict with page URL, title, and other metadata
        """
        return {
            "url": self.page.url,
            "title": await self.page.title(),
            "viewport": self.page.viewport_size,
            "cookies": await self._context.cookies() if self._context else []
        }
    
    async def wait_for_selector(self, selector: str, timeout: int = 10000) -> bool:
        """
        Wait for a selector to appear on the page.
        
        Args:
            selector: CSS selector to wait for
            timeout: Timeout in milliseconds
        
        Returns:
            True if selector appeared, False otherwise
        """
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception:
            return False
    
    async def click_element(self, selector: str) -> Dict[str, Any]:
        """
        Click an element by selector.
        
        Args:
            selector: CSS selector of element to click
        
        Returns:
            Dict with click result
        """
        try:
            await self.page.click(selector)
            await asyncio.sleep(0.5)  # Brief wait after click
            
            return {
                "success": True,
                "selector": selector
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "selector": selector
            }
'''
    
    path = Path("ai_platform/systems/netsuite/tools/vision_agent/browser.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"✅ Created: {path}")

def create_perceive_file():
    """Create perceive.py"""
    # Split into parts to avoid string issues
    part1 = '''"""
Vision analysis using Google Gemini Vision API.
Analyzes screenshots to understand page state and identify UI elements.
"""
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

# Use Google Gemini for vision analysis
try:
    from google import genai
    from google.genai import types
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("Warning: Google Generative AI not available for vision analysis")
    print("Install with: pip install google-genai")


class VisionAnalyzer:
    """Analyzes screenshots using Google Gemini Vision to understand UI state."""
    
    def __init__(self, model: str = None):
        """
        Initialize vision analyzer.
        
        Args:
            model: Vision model to use (default: gemini-2.5-flash)
        """
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        if VISION_AVAILABLE:
            # Initialize the client
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("Warning: GOOGLE_API_KEY not set")
                self.client = None
            else:
                self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
            print("Warning: Vision analysis not available. Install google-genai package.")
    
    def load_image_bytes(self, image_path: str) -> bytes:
        """
        Load image file as bytes.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Image bytes
        """
        with open(image_path, 'rb') as f:
            return f.read()
'''
    
    part2 = '''    
    async def analyze_page(
        self, 
        screenshot_path: str, 
        task_context: str = None
    ) -> Dict[str, Any]:
        """
        Analyze a screenshot to understand the page state.
        
        Args:
            screenshot_path: Path to screenshot file
            task_context: Optional context about what we're trying to do
        
        Returns:
            Dict with analysis results
        """
        if not VISION_AVAILABLE or not self.client:
            return self._fallback_analysis(screenshot_path)
        
        try:
            # Build the prompt
            prompt = self._build_analysis_prompt(task_context)
            
            # Load image as bytes
            image_bytes = self.load_image_bytes(screenshot_path)
            
            # Determine MIME type from file extension
            path = Path(screenshot_path)
            mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            
            # Create image part using Part.from_bytes
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )
            
            # Call Gemini Vision API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image_part],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1500,
                )
            )
            
            # Parse response
            content = response.text
            analysis = self._parse_vision_response(content)
            
            return {
                "success": True,
                "analysis": analysis,
                "raw_response": content
            }
            
        except Exception as e:
            print(f"Vision analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback": self._fallback_analysis(screenshot_path)
            }
'''
    
    part3 = '''    
    async def find_element(
        self,
        screenshot_path: str,
        element_description: str
    ) -> Dict[str, Any]:
        """Find a specific element on the page by description."""
        if not VISION_AVAILABLE or not self.client:
            return {"success": False, "error": "Vision not available"}
        
        try:
            # Load image
            image_bytes = self.load_image_bytes(screenshot_path)
            path = Path(screenshot_path)
            mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )
            
            prompt = f"""Analyze this screenshot and locate the following element:
            
Element to find: {element_description}

Provide a JSON response with:
1. found: true/false
2. description: What you see
3. location: rough position (top-left, top-right, center, bottom, etc.)
4. selector_hints: CSS selector suggestions if possible
5. text_content: Any visible text on/near the element

Be specific and accurate."""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image_part],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=800,
                )
            )
            
            content = response.text
            
            return {
                "success": True,
                "element_description": element_description,
                "vision_response": content
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _build_analysis_prompt(self, task_context: str = None) -> str:
        """Build the prompt for page analysis."""
        base_prompt = """Analyze this NetSuite application screenshot and provide:

1. Page Type: Identify what type of page this is
2. Key Elements: List the main UI elements visible
3. Page State: Describe the current state
4. Actionable Elements: Identify clickable/interactable elements
5. Next Actions: Suggest logical next steps

Format your response as structured text with clear sections."""

        if task_context:
            base_prompt += f"\\n\\nTask Context: {task_context}\\nFocus your analysis on elements relevant to this task."
        
        return base_prompt
    
    def _parse_vision_response(self, response: str) -> Dict[str, Any]:
        """Parse the vision model's response into structured data."""
        analysis = {
            "page_type": "unknown",
            "elements": [],
            "state": "unknown",
            "suggestions": [],
            "raw_text": response
        }
        
        # Extract page type
        response_lower = response.lower()
        if "login" in response_lower:
            analysis["page_type"] = "login"
        elif "list" in response_lower or "table" in response_lower:
            analysis["page_type"] = "list"
        elif "form" in response_lower:
            analysis["page_type"] = "form"
        elif "dashboard" in response_lower:
            analysis["page_type"] = "dashboard"
        
        # Look for button mentions
        lines = response.split('\\n')
        for line in lines:
            if any(word in line.lower() for word in ['button', 'click', 'link']):
                analysis["elements"].append(line.strip())
        
        return analysis
    
    def _fallback_analysis(self, screenshot_path: str) -> Dict[str, Any]:
        """Fallback analysis when vision model is unavailable."""
        return {
            "page_type": "unknown",
            "elements": ["Vision analysis unavailable"],
            "state": "Cannot determine without vision model",
            "suggestions": ["Install google-genai package and set GOOGLE_API_KEY"],
            "screenshot_exists": Path(screenshot_path).exists(),
            "screenshot_size": Path(screenshot_path).stat().st_size if Path(screenshot_path).exists() else 0
        }


async def analyze_screenshot(screenshot_path: str, task_context: str = None) -> Dict[str, Any]:
    """Convenience function to analyze a screenshot."""
    analyzer = VisionAnalyzer()
    return await analyzer.analyze_page(screenshot_path, task_context)
'''
    
    content = part1 + part2 + part3
    
    path = Path("ai_platform/systems/netsuite/tools/vision_agent/perceive.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"✅ Created: {path}")

def create_runner_file():
    """Create runner.py - will create in next message due to length"""
    content = '''"""
Vision agent runner - executes plan steps using Playwright + Vision.
"""
import asyncio
from typing import Dict, Any, Optional

from .browser import BrowserManager
from .perceive import VisionAnalyzer


class VisionExecutor:
    """Executes automation steps using browser automation and vision analysis."""
    
    def __init__(self, headless: bool = True, screenshots_dir: str = "screenshots"):
        self.headless = headless
        self.screenshots_dir = screenshots_dir
        self.browser: Optional[BrowserManager] = None
        self.vision = VisionAnalyzer()
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.browser = BrowserManager(
            headless=self.headless,
            screenshots_dir=self.screenshots_dir
        )
        await self.browser.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.browser:
            await self.browser.close()
    
    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single plan step."""
        kind = step.get("kind", "")
        goal = step.get("goal", "")
        args = step.get("args", {})
        
        print(f"Executing {kind}: {goal}")
        
        # Route to appropriate handler
        if kind == "navigate":
            return await self._execute_navigate(goal, args)
        elif kind == "click":
            return await self._execute_click(goal, args)
        elif kind == "fill":
            return await self._execute_fill(goal, args)
        elif kind == "select":
            return await self._execute_select(goal, args)
        elif kind == "verify":
            return await self._execute_verify(goal, args)
        elif kind == "wait":
            return await self._execute_wait(goal, args)
        else:
            return {
                "success": False,
                "error": f"Unknown step kind: {kind}",
                "step": step
            }
    
    async def _execute_navigate(self, goal: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a navigation step."""
        if not self.browser:
            return {"success": False, "error": "Browser not initialized"}
        
        try:
            url = args.get("url")
            
            if url:
                # Direct URL navigation
                nav_result = await self.browser.navigate(url)
                
                if not nav_result["success"]:
                    return {
                        "success": False,
                        "error": f"Navigation failed: {nav_result.get('error')}",
                        "goal": goal
                    }
                
                # Analyze the page after navigation
                screenshot_path = nav_result.get("screenshot")
                vision_analysis = await self.vision.analyze_page(
                    screenshot_path,
                    task_context=f"Navigated to accomplish: {goal}"
                )
                
                return {
                    "success": True,
                    "kind": "navigate",
                    "goal": goal,
                    "url": nav_result["url"],
                    "page_title": nav_result["title"],
                    "screenshot": screenshot_path,
                    "vision_analysis": vision_analysis.get("analysis"),
                    "page_state": await self.browser.get_page_state()
                }
            
            else:
                # Menu navigation - need to find and click menu item
                label = args.get("label", "")
                screenshot_path = await self.browser.capture_screenshot("pre_nav")
                
                element_search = await self.vision.find_element(
                    screenshot_path,
                    f"Navigation menu item or link labeled '{label}'"
                )
                
                return {
                    "success": False,
                    "kind": "navigate",
                    "goal": goal,
                    "error": "Menu navigation requires click implementation",
                    "label": label,
                    "element_search": element_search,
                    "screenshot": screenshot_path,
                    "note": "This will be fully implemented when click action is integrated"
                }
        
        except Exception as e:
            return {
                "success": False,
                "kind": "navigate",
                "goal": goal,
                "error": str(e)
            }
    
    async def _execute_click(self, goal: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a click action."""
        return {
            "success": False,
            "kind": "click",
            "goal": goal,
            "error": "Click action not yet implemented",
            "args": args,
            "note": "Coming in next iteration"
        }
    
    async def _execute_fill(self, goal: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a form fill action."""
        return {
            "success": False,
            "kind": "fill",
            "goal": goal,
            "error": "Fill action not yet implemented",
            "args": args
        }
    
    async def _execute_select(self, goal: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a select/filter action."""
        return {
            "success": False,
            "kind": "select",
            "goal": goal,
            "error": "Select action not yet implemented",
            "args": args
        }
    
    async def _execute_verify(self, goal: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a verification step."""
        if not self.browser:
            return {"success": False, "error": "Browser not initialized"}
        
        screenshot_path = await self.browser.capture_screenshot("verify")
        field = args.get("field", "status")
        expected = args.get("expected_value", "")
        
        analysis = await self.vision.analyze_page(
            screenshot_path,
            task_context=f"Verify that {field} is {expected}"
        )
        
        return {
            "success": True,
            "kind": "verify",
            "goal": goal,
            "screenshot": screenshot_path,
            "verification": analysis.get("analysis"),
            "note": "Manual review of screenshot recommended"
        }
    
    async def _execute_wait(self, goal: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a wait step."""
        timeout = int(args.get("timeout", 5))
        condition = args.get("condition", "page load")
        
        await asyncio.sleep(timeout)
        
        return {
            "success": True,
            "kind": "wait",
            "goal": goal,
            "waited_seconds": timeout,
            "condition": condition
        }


def execute_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous wrapper for execute_step."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            async def run():
                async with VisionExecutor(headless=True) as executor:
                    return await executor.execute_step(step)
            
            result = loop.run_until_complete(run())
            return result
        finally:
            loop.close()
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Execution error: {str(e)}",
            "step": step
        }


async def test_navigation():
    """Test navigation to a URL."""
    async with VisionExecutor(headless=False) as executor:
        step = {
            "kind": "navigate",
            "goal": "Navigate to example.com",
            "args": {"url": "https://example.com"}
        }
        
        result = await executor.execute_step(step)
        print("Navigation result:")
        print(f"  Success: {result['success']}")
        print(f"  URL: {result.get('url')}")
        print(f"  Title: {result.get('page_title')}")
        print(f"  Screenshot: {result.get('screenshot')}")
        
        if result.get('vision_analysis'):
            print(f"  Page type: {result['vision_analysis'].get('page_type')}")


if __name__ == "__main__":
    asyncio.run(test_navigation())
'''
    
    path = Path("ai_platform/systems/netsuite/tools/vision_agent/runner.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"✅ Created: {path}")

def main():
    print("🚀 Creating vision agent files...\n")
    
    try:
        create_browser_file()
        create_perceive_file()
        create_runner_file()
        
        print("\n✅ All files created successfully!")
        print("\nNext steps:")
        print("1. pip install playwright google-genai")
        print("2. python -m playwright install chromium")
        print("3. export GOOGLE_API_KEY=your_key")
        print("4. python ai_platform/systems/netsuite/tools/vision_agent/runner.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
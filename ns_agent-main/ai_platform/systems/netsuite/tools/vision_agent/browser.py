"""
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

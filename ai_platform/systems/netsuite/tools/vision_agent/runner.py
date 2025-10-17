"""
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

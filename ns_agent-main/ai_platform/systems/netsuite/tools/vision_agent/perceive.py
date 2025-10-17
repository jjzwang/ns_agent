"""
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
            base_prompt += f"\n\nTask Context: {task_context}\nFocus your analysis on elements relevant to this task."
        
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
        lines = response.split('\n')
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

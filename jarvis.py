#!/usr/bin/env python3
"""
Qwen3 4B Function Calling Program
This program demonstrates how to use Qwen3 4B model for function calling with various tools.
"""

import json
import requests
import datetime
import math
from typing import Dict, List, Any, Callable
from dataclasses import dataclass


@dataclass
class FunctionDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable


class Qwen3Client:
    """Client for interacting with Qwen3 4B model with function calling"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "qwen3:4b"):
        self.base_url = base_url
        self.model_name = model_name
        self.available_functions = {}
        self.setup_default_functions()
    
    def setup_default_functions(self):
        """Setup default functions for the Scout model"""
        
        # Calculator function
        def calculate(expression: str) -> str:
            """Safely evaluate mathematical expressions"""
            try:
                # Basic safety check - only allow certain characters
                allowed_chars = set('0123456789+-*/().= ')
                if not all(c in allowed_chars for c in expression):
                    return "Error: Invalid characters in expression"
                
                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Get current time function
        def get_current_time(timezone: str = "UTC") -> str:
            """Get current time in specified timezone"""
            now = datetime.datetime.now()
            return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Weather simulation function (placeholder)
        def get_weather(city: str, country: str = "US") -> str:
            """Get weather information for a city (simulated)"""
            # This is a simulation - in real use, you'd call a weather API
            weather_conditions = ["sunny", "cloudy", "rainy", "snowy"]
            import random
            condition = random.choice(weather_conditions)
            temp = random.randint(-10, 35)
            return f"Weather in {city}, {country}: {condition}, {temp}°C"
        
        # Text analysis function
        def analyze_text(text: str, analysis_type: str = "sentiment") -> str:
            """Analyze text for sentiment, length, or word count"""
            if analysis_type == "sentiment":
                # Simple sentiment analysis (placeholder)
                positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
                negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
                
                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                if pos_count > neg_count:
                    return "Sentiment: Positive"
                elif neg_count > pos_count:
                    return "Sentiment: Negative"
                else:
                    return "Sentiment: Neutral"
            
            elif analysis_type == "length":
                return f"Text length: {len(text)} characters"
            
            elif analysis_type == "word_count":
                return f"Word count: {len(text.split())} words"
            
            return "Unknown analysis type"
        
         # Function to get HTTP status code from a URL
        def get_latest_chapter_status(manga: str) -> str:
            """For the given manga, compute the URL, make the API call and based on the HTTP status code respond if the latest chapter is available."""
            try:
                url = ""
                if( manga.lower() == "solo leveling"):
                    url = "https://asuracomic.net/series/solo-leveling-ragnarok-1f84f5a6/chapter/56"
                response = requests.get(url)
                return f"HTTP status code for {url}: {response.status_code}"
            except Exception as e:
                return f"Error fetching {url}: {str(e)}"
        
        # Register functions
        self.register_function(
            "calculate",
            "Perform mathematical calculations",
            {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            },
            calculate
        )
        
        self.register_function(
            "get_current_time",
            "Get the current time",
            {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone (default: UTC)"
                    }
                }
            },
            get_current_time
        )
        
        self.register_function(
            "get_weather",
            "Get weather information for a city",
            {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code (default: US)"
                    }
                },
                "required": ["city"]
            },
            get_weather
        )
        
        self.register_function(
            "analyze_text",
            "Analyze text for various properties",
            {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze"
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis: sentiment, length, or word_count",
                        "enum": ["sentiment", "length", "word_count"]
                    }
                },
                "required": ["text"]
            },
            analyze_text
        )

        self.register_function(
            "get_latest_chapter_status",
            "Get the status of the latest chapter for a given manga",
            {
                "type": "object",
                "properties": {
                    "manga": {
                        "type": "string",
                        "description": "The name of the manga to fetch"
                    }
                },
                "required": ["manga"]
            },
            get_latest_chapter_status
        )
    
    def register_function(self, name: str, description: str, parameters: Dict[str, Any], function: Callable):
        """Register a function for use with the model"""
        self.available_functions[name] = FunctionDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function
        )
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get function definitions in the format expected by the model"""
        definitions = []
        for func_def in self.available_functions.values():
            definitions.append({
                "name": func_def.name,
                "description": func_def.description,
                "parameters": func_def.parameters
            })
        return definitions
    
    def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a function with given arguments"""
        if function_name not in self.available_functions:
            return f"Error: Function '{function_name}' not found"
        
        try:
            func_def = self.available_functions[function_name]
            result = func_def.function(**arguments)
            return str(result)
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"
    
    def chat_with_functions(self, message: str) -> str:
        """Send a message to Qwen 3 4B with function calling enabled"""
        try:
            # Prepare the system prompt with function definitions
            function_specs = self.get_function_definitions()
            system_prompt = f"""You are Qwen 3, a helpful AI assistant with access to tools. 

When you need to use a function, respond with a JSON object in this exact format:
{{
  "function_call": {{
    "name": "function_name",
    "arguments": {{"arg1": "value1", "arg2": "value2"}}
  }}
}}

Available functions:
{json.dumps(function_specs, indent=2)}

If you don't need to call a function, respond normally in plain text.
If a function call response then only respond by formulating the output of the function call.
The response should be brief and to the point, it should be a single line.
Do not give suggestions unless specifically asked."""
            
            # First request to get initial response
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                "stream": False,
                "format": "json" if self._needs_function_call(message) else ""
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                return f"Error: Failed to get response from model (status {response.status_code})"
            
            response_data = response.json()
            assistant_message = response_data.get("message", {}).get("content", "")
            
            # Check if the response contains a function call
            function_call_result = self._parse_function_call(assistant_message)
            
            if function_call_result:
                function_name = function_call_result.get("name")
                function_args = function_call_result.get("arguments", {})
                
                # Execute the function
                execution_result = self.execute_function(function_name, function_args)
                
                # Get final response with function result
                follow_up_payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": """You are Qwen 3.
                            The user asked a question and you called a function.
                            Use the function result to provide a helpful response to the user.
                            Respond in plain text, not JSON.
                            If a function call response then only respond by formulating the output of the function call.
                            The response should be brief and to the point, it should be a single line.
                            Do not give suggestions unless specifically asked.
                            """
                        },
                        {
                            "role": "user",
                            "content": message
                        },
                        {
                            "role": "assistant",
                            "content": f"I'll use the {function_name} function to help answer your question."
                        },
                        {
                            "role": "user",
                            "content": f"Function result: {execution_result}. Please provide a helpful response based on this result."
                        }
                    ],
                    "stream": False
                }
                
                follow_up_response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=follow_up_payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if follow_up_response.status_code == 200:
                    final_response = follow_up_response.json()
                    return final_response.get("message", {}).get("content", "No response")
                else:
                    return f"Function executed: {execution_result}"
            
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _needs_function_call(self, message: str) -> bool:
        """Determine if a message likely needs a function call"""
        function_keywords = [
            "calculate", "compute", "math", "time", "weather", "analyze", "sentiment",
            "what's", "how much", "when is", "what time", "temperature", "forecast", "status code", "status", "solo leveling"
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in function_keywords)
    
    def _parse_function_call(self, response: str) -> dict:
        """Parse function call from model response"""
        try:
            # Try to parse as JSON
            parsed = json.loads(response.strip())
            if "function_call" in parsed:
                return parsed["function_call"]
        except json.JSONDecodeError:
            # Try to extract JSON from text response
            import re
            json_match = re.search(r'\{[^{}]*"function_call"[^{}]*\{[^{}]*\}[^{}]*\}', response)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if "function_call" in parsed:
                        return parsed["function_call"]
                except json.JSONDecodeError:
                    pass
        return None


def main():
    """Main function to demonstrate Qwen3 4B function calling"""
    print("🚀 Qwen3 4B Function Calling Demo")
    print("=" * 50)
    
    # Initialize the client
    client = Qwen3Client()
    
    print(f"Available functions: {', '.join(client.available_functions.keys())}")
    print("\nType 'quit' to exit, 'help' for examples\n")
    
    # Example queries to try
    examples = [
        "What's 15 * 24 + 37?",
        "What time is it?",
        "What's the weather like in New York?",
        "Analyze the sentiment of this text: 'I love this new feature, it's amazing!'",
        "How many words are in this sentence: 'The quick brown fox jumps over the lazy dog'?"
    ]
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("Example queries you can try:")
                for i, example in enumerate(examples, 1):
                    print(f"{i}. {example}")
                print()
                continue
            
            if not user_input:
                continue
            
            print("Qwen: ", end="", flush=True)
            response = client.chat_with_functions(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
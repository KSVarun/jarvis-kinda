#!/usr/bin/env python3
"""
Qwen3 4B Function Calling Program
This program demonstrates how to use Qwen3 4B model for function calling with various tools.
"""

import json
import requests
import datetime
import math
import time
import logging
import os
from sqlalchemy import create_engine, Column, String
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
import re
import aiohttp
import asyncio

model_name = "qwen3:4b"

async def fetch(session, url):
    start_ts = time.perf_counter()
    logging.debug(f"[timing] fetch:start url={url}")
    async with session.get(url) as response:
        # you can also use await response.json() if API returns JSON
        data = await response.text()
        duration_ms = (time.perf_counter() - start_ts) * 1000
        logging.debug(
            f"[timing] fetch:end url={url} status={response.status} bytes={len(data)} duration_ms={duration_ms:.2f}"
        )
        return url, response.status, len(data)
    

default_prompt_instruction = """Respond in plain text, not JSON.
If a function call response then only respond by formulating the output of the function call.
The response should be brief and to the point, it should be a single line.
Do not give suggestions unless specifically asked.
If the URL's are present in the function result, include them in the response."""


@dataclass
class FunctionDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable


class Qwen3Client:
    """Client for interacting with Qwen3 4B model with function calling"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = model_name):
        self.base_url = base_url
        self.model_name = model_name
        self.available_functions = {}
        # Rolling conversation history of alternating user/assistant turns
        # Format: [{"role": "user"|"assistant", "content": str}, ...]
        self.history: List[Dict[str, str]] = []
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
                return f"Error at calculate: {str(e)}"
        
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
        
        async def get_latest_chapter_urls() -> str:
            """For a set of manga, compute the URL, make the API call and based on the HTTP status code respond with the URL if the latest chapter is available."""
            try:
                start_overall = time.perf_counter()
                logging.debug("[timing] get_latest_chapter_urls:start")
                # Build URLs from DB; fallback to empty list if no DB or rows
                try:
                    computed_urls = get_next_chapter_urls_from_db()
                except Exception as url_e:
                    logging.error(f"[db] Failed to compute next chapter URLs: {url_e}")
                    computed_urls = []
                    return "Error: Failed to compute next chapter URLs"

                new_chapter_urls_for_manga = []
                async with aiohttp.ClientSession() as session:
                    targets = computed_urls
                    tasks = [fetch(session, url) for url in targets]
                    gather_start = time.perf_counter()
                    results = await asyncio.gather(*tasks)
                    gather_ms = (time.perf_counter() - gather_start) * 1000
                    logging.debug(f"[timing] get_latest_chapter_urls:gather duration_ms={gather_ms:.2f}")
                    for url, status, size in results:
                        if status == 200:
                            new_chapter_urls_for_manga.append(url)
                overall_ms = (time.perf_counter() - start_overall) * 1000
                logging.debug(f"[timing] get_latest_chapter_urls:end duration_ms={overall_ms:.2f}")
                return f"New chapters available at: {', '.join(new_chapter_urls_for_manga)}" if new_chapter_urls_for_manga else "No new chapters available."
            except Exception as e:
                return f"Error fetching chapters: {str(e)}"
        
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
            "get_latest_chapter_urls",
            "Get the URL of latest chapter of manga's",
            {
                "type": "object",
                "properties": {},
                "required": []
            },
            get_latest_chapter_urls
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
            exec_start = time.perf_counter()
            logging.debug(f"[timing] execute_function:start name={function_name} args={arguments}")
            func_def = self.available_functions[function_name]
            func = func_def.function
            import inspect
            if inspect.iscoroutinefunction(func):
                result = asyncio.run(func(**arguments))
            else:
                result = func(**arguments)
            exec_duration_ms = (time.perf_counter() - exec_start) * 1000
            logging.debug(f"[timing] execute_function:end name={function_name} duration_ms={exec_duration_ms:.2f}")
            return str(result)
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"
    
    def chat_with_functions(self, message: str) -> str:
        """Send a message to Qwen 3 4B with function calling enabled"""
        try:
            total_start = time.perf_counter()
            logging.debug("[timing] chat_with_functions:start")
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

{default_prompt_instruction}
"""
            # First request to get initial response
            assembled_messages: List[Dict[str, str]] = (
                [{"role": "system", "content": system_prompt}] +
                self.history +
                [{"role": "user", "content": message}]
            )
            payload = {
                "model": self.model_name,
                "messages": assembled_messages,
                "stream": False
            }
            if self._needs_function_call(message):
                payload["format"] = "json"
            
            http_start = time.perf_counter()
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            http_duration_ms = (time.perf_counter() - http_start) * 1000
            logging.debug(f"[timing] chat:first_request duration_ms={http_duration_ms:.2f}")
            
            if response.status_code != 200:
                try:
                    err_text = response.text
                except Exception:
                    err_text = "<no body>"
                logging.error(f"[api] chat error status={response.status_code} body={err_text}")
                return f"Error: Failed to get response from model (status {response.status_code})"
            
            response_data = response.json()
            assistant_message = response_data.get("message", {}).get("content", "")
            
            # print(f"[DEBUG] Assistant message: {assistant_message}")
            
            # Check if the response contains a function call
            function_call_result = self._parse_function_call(assistant_message)
            
            if function_call_result:
                function_name = function_call_result.get("name")
                function_args = function_call_result.get("arguments", {})
                
                exec_result_start = time.perf_counter()
                execution_result = self.execute_function(function_name, function_args)
                exec_result_ms = (time.perf_counter() - exec_result_start) * 1000
                logging.debug(f"[timing] chat:function_execution duration_ms={exec_result_ms:.2f}")
                # print(f"[DEBUG] Function '{function_name}' executed with result: {execution_result}")

                # Get final response with function result
                follow_up_payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": f"""You are Qwen 3.
                            The user asked a question and you called a function.
                            Use the function result to provide a helpful response to the user.
                            
                            {default_prompt_instruction}
                            """
                        },
                        *self.history,
                        {"role": "user", "content": message},
                        {
                            "role": "assistant",
                            "content": f"I'll use the {function_name} function to help answer your question."
                        },
                        {
                            "role": "user",
                            "content": f"""Function result: {execution_result}. Please provide a helpful response based on this result.
                            
                            {default_prompt_instruction}
                            """
                        }
                    ],
                    "stream": False
                }
                
                follow_http_start = time.perf_counter()
                follow_up_response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=follow_up_payload,
                    headers={"Content-Type": "application/json"}
                )
                follow_http_duration_ms = (time.perf_counter() - follow_http_start) * 1000
                logging.debug(f"[timing] chat:followup_request duration_ms={follow_http_duration_ms:.2f}")
                
                if follow_up_response.status_code == 200:
                    final_response = follow_up_response.json()
                    total_duration_ms = (time.perf_counter() - total_start) * 1000
                    logging.debug(f"[timing] chat_with_functions:end with_function duration_ms={total_duration_ms:.2f}")
                    final_text = final_response.get("message", {}).get("content", "No response")
                    # Update history with this turn (user + assistant)
                    self.history.append({"role": "user", "content": message})
                    self.history.append({"role": "assistant", "content": final_text})
                    # Trim history to last 10 messages to control context growth
                    if len(self.history) > 10:
                        self.history = self.history[-10:]
                    return final_text
                else:
                    total_duration_ms = (time.perf_counter() - total_start) * 1000
                    logging.debug(f"[timing] chat_with_functions:end followup_failed duration_ms={total_duration_ms:.2f}")
                    return f"Function executed: {execution_result}"
            
            total_duration_ms = (time.perf_counter() - total_start) * 1000
            logging.debug(f"[timing] chat_with_functions:end no_function duration_ms={total_duration_ms:.2f}")
            # Update history when no function is called
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": assistant_message})
            if len(self.history) > 10:
                self.history = self.history[-10:]
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}"
        except Exception as e:
            return f"Error in chat_with_functions: {str(e)}"
    
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
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            parsed = json.loads(response.strip())
            # print(f"[DEBUG] Parsed JSON: {parsed}")
            if "function_call" in parsed:
                return parsed["function_call"]
        except json.JSONDecodeError:
            # Try to extract JSON from text response
            json_match = re.search(r'\{[^{}]*"function_call"[^{}]*\{[^{}]*\}[^{}]*\}', response)
            # print(f"[DEBUG] JSON match: {json_match.group() if json_match else 'None'}")
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if "function_call" in parsed:
                        return parsed["function_call"]
                except json.JSONDecodeError:
                    pass
        return None


Base = declarative_base()


class ComicsRead(Base):
    __tablename__ = "comics_read"
    # Use name as primary key since a comic name should be unique
    name = Column(String, primary_key=True)
    read_chapters = Column(PG_ARRAY(String), nullable=False)
    source_base_url = Column(String, nullable=False)


def get_engine():
    """Create a SQLAlchemy engine from DATABASE_URL env var.
    Example: postgresql+psycopg://user:password@localhost:5432/mydb
    """
    database_url = f"postgresql+psycopg://{os.getenv('POSTGRES_USER_NAME')}:{os.getenv('POSTGRES_PASSWORD_VALUE')}@localhost:5432/{os.getenv('POSTGRES_DB_NAME')}"
    return create_engine(database_url, pool_pre_ping=True)


def get_next_chapter_urls_from_db() -> List[str]:
    """Read comics and compute next-chapter URLs.

    For each row, find the highest chapter number in read_chapters,
    increment by 1, and join with source_base_url.
    Accepts chapter numbers like "55" or "chapter/55" and extracts the
    trailing integer.
    """
    if SessionLocal is None:
        raise RuntimeError("SessionLocal not initialized. Call init_db() first.")

    urls_out: List[str] = []
    with SessionLocal() as session:
        rows = session.query(ComicsRead).all()
        for row in rows:
            # If there are recorded chapters and the list is ordered, use the last one
            if row.read_chapters and len(row.read_chapters) > 0:
                last_chapter = row.read_chapters[-1]
                next_chapter = int(last_chapter) + 1 if int(last_chapter) >= 0 else 1
            else:
                # If none recorded, start from chapter 1
                next_chapter = 1

            base = row.source_base_url
            urls_out.append(f"{base}{next_chapter}")

    logging.debug(f"[db] Computed next-chapter URLs: {urls_out}")
    return urls_out


SessionLocal = None


def init_db():
    """Initialize the database schema (create tables if they don't exist)."""
    engine = get_engine()
    if engine is None:
        return False
    global SessionLocal
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    start_ts = time.perf_counter()
    logging.debug("[db] Creating tables if not exist...")
    Base.metadata.create_all(bind=engine)
    duration_ms = (time.perf_counter() - start_ts) * 1000
    logging.debug(f"[timing] db:init duration_ms={duration_ms:.2f}")
    return True


def main():
    """Main function to demonstrate Qwen3 4B function calling"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.debug("[startup] Program started")
    # Initialize database schema (no-op if DATABASE_URL not set)
    try:
        if init_db():
            logging.info("[db] Initialized schema")
        else:
            logging.info("[db] Schema initialization skipped")
    except Exception as db_e:
        logging.error(f"[db] Initialization failed: {db_e}")
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
            req_start = time.perf_counter()
            response = re.sub(r'<think>.*?</think>', '', client.chat_with_functions(user_input), flags=re.DOTALL)
            req_ms = (time.perf_counter() - req_start) * 1000
            logging.debug(f"[timing] main:request_total duration_ms={req_ms:.2f}")
            print(response.strip())
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error at main: {str(e)}")


if __name__ == "__main__":
    main()
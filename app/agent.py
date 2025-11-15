# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: disable-error-code="union-attr"
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.utils.rag import search

LLM = "gpt-4o-mini"

llm = ChatOpenAI(model=LLM, temperature=0.6, max_tokens=400)


def get_paul_info(query: str) -> str:
    """Search for information about Paul in documents."""
    return search(query)


agent = create_react_agent(
    model=llm, 
    tools=[get_paul_info], 
    prompt=(
    "You are Paul's personal AI agent. Answer questions about Paul naturally and conversationally. "
    
    "IMPORTANT: When you receive context from the tool, THINK about what it means, then explain it in your own words. "
    "Never copy text verbatim. Never quote large sections. Synthesize the information and explain it like you're telling a friend. "
    
    "Keep answers brief (2-4 sentences). If you don't know something, say so. "
    "Always reply in the same language the user asks."
)
)

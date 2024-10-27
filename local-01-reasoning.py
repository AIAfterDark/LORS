import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum
import json
import ollama
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich import print as rprint
import uuid
import os
import re
from textblob import TextBlob
import spacy
from pathlib import Path
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Initialize console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class AgentType(Enum):
    FAST = "fast"
    DEEP = "deep"

@dataclass
class ReasoningAgent:
    agent_type: AgentType
    model_name: str
    temperature: float
    max_tokens: int
    agent_id: str = ""
    
    def __post_init__(self):
        self.agent_id = str(uuid.uuid4())[:8]
    
    async def think(self, prompt: str) -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            console.print(f"[bold cyan]Agent {self.agent_id} ({self.agent_type.value}) starting analysis...[/bold cyan]")
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                }
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            console.print(Panel(
                f"{response['message']['content']}",
                title=f"[bold green]Agent {self.agent_id} ({self.agent_type.value}) - {processing_time:.2f}s[/bold green]",
                border_style="green"
            ))
            
            return {
                'thought': response['message']['content'],
                'processing_time': processing_time,
                'agent_type': self.agent_type.value,
                'agent_id': self.agent_id
            }
        except Exception as e:
            console.print(f"[bold red]Error in Agent {self.agent_id}: {str(e)}[/bold red]")
            raise

class PromptAnalyzer:
    def __init__(self):
        self.complexity_indicators = {
            'technical_terms': set([
                'analyze', 'optimize', 'implement', 'design', 'evaluate',
                'calculate', 'predict', 'compare', 'integrate', 'synthesize',
                'algorithm', 'system', 'architecture', 'framework', 'methodology'
            ]),
            'domain_keywords': {
                'programming': set(['code', 'program', 'function', 'class', 'api', 'database']),
                'math': set(['equation', 'calculate', 'solve', 'formula', 'derivative']),
                'science': set(['experiment', 'hypothesis', 'theory', 'analysis', 'research']),
                'business': set(['strategy', 'market', 'revenue', 'cost', 'profit'])
            }
        }
    
    def analyze_prompt_complexity(self, prompt: str) -> Dict[str, Any]:
        doc = nlp(prompt.lower())
        blob = TextBlob(prompt)
        
        features = {
            'sentence_count': len(list(doc.sents)),
            'avg_sentence_length': sum(len(sent.text.split()) for sent in doc.sents) / len(list(doc.sents)),
            'subjectivity': blob.sentiment.subjectivity,
            'named_entities': len(doc.ents),
            'technical_term_count': sum(1 for word in doc if word.text.lower() in self.complexity_indicators['technical_terms']),
            'domain_complexity': self._analyze_domain_complexity(doc),
            'cognitive_complexity': self._analyze_cognitive_complexity(doc),
            'dependency_depth': self._analyze_dependency_depth(doc)
        }
        
        complexity_score = self._calculate_complexity_score(features)
        
        return {
            'complexity_score': complexity_score,
            'features': features,
            'recommended_scaling': self._get_recommended_scaling(complexity_score)
        }
    
    def _analyze_domain_complexity(self, doc) -> float:
        domain_matches = {
            domain: sum(1 for word in doc if word.text.lower() in keywords)
            for domain, keywords in self.complexity_indicators['domain_keywords'].items()
        }
        return max(domain_matches.values()) if domain_matches else 0
    
    def _analyze_cognitive_complexity(self, doc) -> float:
        cognitive_verbs = sum(1 for token in doc if token.pos_ == "VERB" and token.lemma_ in [
            'analyze', 'evaluate', 'synthesize', 'compare', 'explain'
        ])
        logical_connectors = sum(1 for token in doc if token.dep_ in ['mark', 'cc', 'prep'])
        return (cognitive_verbs + logical_connectors) / len(doc)
    
    def _analyze_dependency_depth(self, doc) -> int:
        def get_depth(token):
            return 1 + max((get_depth(child) for child in token.children), default=0)
        
        return max(get_depth(sent.root) for sent in doc.sents)
    
    def _calculate_complexity_score(self, features: Dict) -> float:
        weights = {
            'sentence_count': 0.1,
            'avg_sentence_length': 0.15,
            'subjectivity': 0.1,
            'named_entities': 0.15,
            'technical_term_count': 0.2,
            'domain_complexity': 0.1,
            'cognitive_complexity': 0.1,
            'dependency_depth': 0.1
        }
        
        normalized_features = {
            'sentence_count': min(features['sentence_count'] / 10, 1),
            'avg_sentence_length': min(features['avg_sentence_length'] / 30, 1),
            'subjectivity': features['subjectivity'],
            'named_entities': min(features['named_entities'] / 5, 1),
            'technical_term_count': min(features['technical_term_count'] / 5, 1),
            'domain_complexity': min(features['domain_complexity'] / 3, 1),
            'cognitive_complexity': min(features['cognitive_complexity'] * 5, 1),
            'dependency_depth': min(features['dependency_depth'] / 10, 1)
        }
        
        return sum(normalized_features[k] * weights[k] for k in weights.keys()) * 100
    
    def _get_recommended_scaling(self, complexity_score: float) -> Dict[str, int]:
        if complexity_score >= 80:
            return {'fast_agents': 5, 'deep_agents': 3}
        elif complexity_score >= 60:
            return {'fast_agents': 4, 'deep_agents': 2}
        elif complexity_score >= 40:
            return {'fast_agents': 3, 'deep_agents': 2}
        else:
            return {'fast_agents': 2, 'deep_agents': 1}

class ResponseManager:
    def __init__(self):
        self.response_dir = Path('responses')
        self.response_dir.mkdir(exist_ok=True)
    
    def generate_title(self, prompt: str) -> str:
        doc = nlp(prompt)
        
        key_phrases = [chunk.text for chunk in doc.noun_chunks][:2]
        main_verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"][:1]
        
        title_components = main_verbs + key_phrases
        title = "_".join(word.lower() for word in " ".join(title_components).split())
        
        title = re.sub(r'[^\w\s-]', '', title)
        title = re.sub(r'[-\s]+', '_', title)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{title}_{timestamp}"
    
    def save_response(self, prompt: str, result: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        title = self.generate_title(prompt)
        file_path = self.response_dir / f"{title}.json"
        
        response_data = {
            'prompt': prompt,
            'timestamp': datetime.now().isoformat(),
            'complexity_analysis': analysis,
            'result': result
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2)
        
        return str(file_path)

class ReasoningOrchestrator:
    def __init__(self):
        self.agent_pools = {
            AgentType.FAST: [],
            AgentType.DEEP: []
        }
        self.prompt_analyzer = PromptAnalyzer()
        self.response_manager = ResponseManager()
    
    def _scale_agents(self, prompt: str) -> Dict[str, Any]:
        analysis = self.prompt_analyzer.analyze_prompt_complexity(prompt)
        scaling = analysis['recommended_scaling']
        
        console.print(f"[bold yellow]Prompt Analysis:[/bold yellow]")
        console.print(f"Complexity Score: {analysis['complexity_score']:.2f}/100")
        console.print("Key Features:")
        for feature, value in analysis['features'].items():
            console.print(f"- {feature}: {value:.2f}")
        console.print(f"Recommended Scaling: {scaling}")
        
        self.agent_pools[AgentType.FAST] = [
            ReasoningAgent(
                AgentType.FAST,
                "llama3.2",
                temperature=0.7,
                max_tokens=150
            ) for _ in range(scaling['fast_agents'])
        ]
        
        self.agent_pools[AgentType.DEEP] = [
            ReasoningAgent(
                AgentType.DEEP,
                "llama3.1",
                temperature=0.9,
                max_tokens=500
            ) for _ in range(scaling['deep_agents'])
        ]
        
        return analysis

    def _combine_thoughts(self, thoughts: List[Dict[str, Any]]) -> Dict[str, Any]:
        combined_thought = ""
        total_time = 0
        agent_ids = []
        
        for thought in thoughts:
            combined_thought += f"\nAgent {thought['agent_id']}: {thought['thought']}"
            total_time += thought['processing_time']
            agent_ids.append(thought['agent_id'])
        
        console.print(Panel(
            f"Combined {len(thoughts)} thoughts from agents: {', '.join(agent_ids)}",
            title=f"[bold yellow]Thought Combination - Avg time: {total_time/len(thoughts):.2f}s[/bold yellow]",
            border_style="yellow"
        ))
        
        return {
            'combined_thought': combined_thought.strip(),
            'avg_processing_time': total_time / len(thoughts),
            'num_agents': len(thoughts),
            'agent_ids': agent_ids
        }

    def _enhance_prompt(self, original_prompt: str, fast_analysis: Dict[str, Any]) -> str:
        enhanced = f"""
        Original prompt: {original_prompt}
        
        Initial fast analysis from {len(fast_analysis['agent_ids'])} agents: 
        {fast_analysis['combined_thought']}
        
        Please provide a deeper, more thorough analysis considering the above context.
        Focus on:
        1. Identifying key concepts and relationships
        2. Exploring potential implications
        3. Generating specific, actionable insights
        """
        
        console.print(Panel(
            enhanced,
            title="[bold cyan]Enhanced Prompt for Deep Analysis[/bold cyan]",
            border_style="cyan"
        ))
        
        return enhanced

    def _synthesize_results(self, fast_analysis: Dict[str, Any], deep_analysis: Dict[str, Any]) -> Dict[str, Any]:
        synthesis = {
            'final_response': f"""
            Quick Analysis Summary:
            (From agents: {', '.join(fast_analysis['agent_ids'])})
            {fast_analysis['combined_thought']}
            
            Deep Analysis Summary:
            (From agents: {', '.join(deep_analysis['agent_ids'])})
            {deep_analysis['combined_thought']}
            """,
            'metrics': {
                'fast_agents_used': fast_analysis['num_agents'],
                'deep_agents_used': deep_analysis['num_agents'],
                'fast_agent_ids': fast_analysis['agent_ids'],
                'deep_agent_ids': deep_analysis['agent_ids'],
                'total_processing_time': fast_analysis['avg_processing_time'] + deep_analysis['avg_processing_time']
            }
        }
        
        console.print(Panel(
            json.dumps(synthesis['metrics'], indent=2),
            title="[bold green]Performance Metrics[/bold green]",
            border_style="green"
        ))
        
        return synthesis

    async def process_prompt(self, prompt: str) -> Dict[str, Any]:
        try:
            console.print("\n[bold magenta]Starting Distributed Reasoning Process[/bold magenta]")
            console.print("=" * 80)
            
            complexity_analysis = self._scale_agents(prompt)
            
            console.print("\n[bold blue]Phase 1: Fast Analysis[/bold blue]")
            fast_thoughts = await asyncio.gather(
                *[agent.think(f"Quick analysis of: {prompt}") for agent in self.agent_pools[AgentType.FAST]]
            )
            
            combined_fast_analysis = self._combine_thoughts(fast_thoughts)
            
            console.print("\n[bold blue]Phase 2: Deep Analysis[/bold blue]")
            enhanced_prompt = self._enhance_prompt(prompt, combined_fast_analysis)
            
            deep_thoughts = await asyncio.gather(
                *[agent.think(enhanced_prompt) for agent in self.agent_pools[AgentType.DEEP]]
            )
            
            final_analysis = self._combine_thoughts(deep_thoughts)
            
            console.print("\n[bold blue]Phase 3: Final Synthesis[/bold blue]")
            result = self._synthesize_results(combined_fast_analysis, final_analysis)
            
            file_path = self.response_manager.save_response(prompt, result, complexity_analysis)
            console.print(f"\n[bold green]Response saved to: {file_path}[/bold green]")
            
            return result
            
        except Exception as e:
            console.print(f"[bold red]Error in reasoning process: {str(e)}[/bold red]")
            raise

async def main(prompt: str):
    try:
        console.print(Panel(
            f"[bold white]{prompt}[/bold white]",
            title="[bold magenta]Input Prompt[/bold magenta]",
            border_style="magenta"
        ))
        
        orchestrator = ReasoningOrchestrator()
        result = await orchestrator.process_prompt(prompt)
        
        console.print("\n[bold green]=== Final Analysis ===[/bold green]")
        console.print(Panel(
            result['final_response'],
            title="[bold green]Combined Analysis from All Agents[/bold green]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[bold red]Error in main execution: {str(e)}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local O1 Reasoning System")
    parser.add_argument("-p", "--prompt", required=True, help="Input prompt for analysis")
    args = parser.parse_args()
    
    asyncio.run(main(args.prompt))

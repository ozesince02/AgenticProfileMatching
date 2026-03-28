"""
Interactive CLI Chat Interface for Matching Agent

Provides conversational interface with natural language intent routing,
operator commands, and real-time feedback.

Main intents:
- search: Find candidates for a job
- filter/refine: Apply constraints and re-rank
- explain: Show reasoning for a candidate
- compare: Side-by-side comparison
- questions: Generate interview questions
- round2/round3: Progress through screening rounds
- finalize: Export final recommendations
"""

import re
import sys
from typing import Optional, Dict, Any, List

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from matching_agent import MatchingAgent, create_agent
from agent_tools import format_candidate_summary, format_comparison_table
from fs_tools import list_files, read_file


class MatchingCLI:
    """Interactive CLI interface for the matching agent."""
    
    def __init__(self):
        self.agent = create_agent()
        self.console = Console() if HAS_RICH else None
        self.running = False
        self.session_messages = []
    
    def print_welcome(self):
        """Show welcome message."""
        msg = """
╔════════════════════════════════════════════════════════════════╗
║         LangGraph Hiring Agent - Interactive CLI              ║
║     AI-Powered Candidate Matching & Multi-Round Screening     ║
╚════════════════════════════════════════════════════════════════╝

Commands:
  Type your query naturally: "Search for ML engineers"
  Or use commands: /help, /new_jd, /show_state, /export, /exit

Examples:
  - Search for Python developers with 5+ years
  - Compare Sneha, Arjun, and Rohan
  - Why did Arjun rank #1?
  - Deep dive on top 5
  - Generate interview questions for Sneha
  
Type /help for full command list.
"""
        if self.console:
            self.console.print(msg, style="cyan")
        else:
            print(msg)
    
    def print_help(self):
        """Show available commands."""
        help_text = """
        
═══════════════════════════════════════════════════════════════
                        HELP & COMMANDS
═══════════════════════════════════════════════════════════════

NATURAL LANGUAGE QUERIES:
  "Search for ML engineers in FinTech"
  "Find Python developers with 5+ years"
  "Show only candidates with Docker and AWS"
  "Compare Sneha, Arjun, and Rohan"
  "Why did Arjun rank #1?"
  "Generate interview questions for Sneha"
  
OPERATOR COMMANDS (prefix with /):
  /new_jd          - Load new job description
  /show_state      - Display current state
  /top N           - Show top N candidates
  /export          - Save results to JSON
  /history         - Show refinement history
  /round2          - Deep analysis on top 5
  /round3          - Final recommendations (Hire/Hold/No-Hire)
  /help            - Show this help
  /exit, /quit     - Exit session
  
═══════════════════════════════════════════════════════════════
"""
        if self.console:
            self.console.print(help_text, style="bright_black")
        else:
            print(help_text)

    def _print_recent_assistant_messages(self, max_messages: int = 3):
        """Print the most recent assistant messages for user-visible actions."""
        assistant_msgs = [m.content for m in self.agent.state.messages if m.role == "assistant"]
        if not assistant_msgs:
            return

        recent = assistant_msgs[-max_messages:]
        block = "\n\n".join(recent)
        if self.console:
            # Keep literal text (e.g., [names]) instead of treating it as Rich markup tags.
            self.console.print(block, style="blue", markup=False)
        else:
            print(block)

    def _safe_auto_export(self):
        """Best-effort session export on exit."""
        try:
            if self.agent.state.messages:
                filepath = self.agent.export_state()
                if self.console:
                    self.console.print(f"✅ Session auto-exported to: {filepath}", style="green")
                else:
                    print(f"✅ Session auto-exported to: {filepath}")
        except Exception as e:
            if self.console:
                self.console.print(f"⚠️ Could not auto-export session: {e}", style="yellow")
            else:
                print(f"⚠️ Could not auto-export session: {e}")
    
    def classify_intent(self, query: str) -> str:
        """Route query to appropriate intent."""
        query_lower = query.lower().strip()
        
        # Commands (start with /)
        if query_lower.startswith("/"):
            return self._handle_operator_command(query_lower)
        
        # Natural language intent classification
        if any(word in query_lower for word in ["search", "find", "candidates", "resumes"]):
            return "SEARCH"
        
        if any(word in query_lower for word in ["compare", "versus", "vs", "side-by-side"]):
            return "COMPARE"
        
        if any(word in query_lower for word in ["why", "explain", "reason", "how", "ranking"]):
            return "EXPLAIN"
        
        if any(word in query_lower for word in ["refine", "filter", "only", "with", "exclude"]):
            return "REFINE"
        
        if any(word in query_lower for word in ["interview", "questions"]):
            return "QUESTIONS"
        
        if any(word in query_lower for word in ["deep dive", "round 2", "round2", "analysis"]):
            return "ROUND2"
        
        if any(word in query_lower for word in ["final", "round 3", "round3", "hire", "decision"]):
            return "ROUND3"
        
        if any(word in query_lower for word in ["load", "job", "jd"]):
            return "LOAD_JOB"
        
        return "UNKNOWN"
    
    def _handle_operator_command(self, command: str) -> str:
        """Handle /commands."""
        cmd = command[1:].lower().strip()
        
        if cmd == "help" or cmd == "h":
            self.print_help()
            return "HELP"
        
        elif cmd == "new_jd":
            self.agent.reset_session()
            if self.console:
                self.console.print("✅ Session reset. Enter new job description:", style="green")
            else:
                print("✅ Session reset. Enter new job description:")
            return "LOAD_JOB"
        
        elif cmd == "show_state":
            state_summary = self.agent.get_state_snapshot()
            if self.console:
                self.console.print(f"[yellow]{state_summary}[/yellow]")
            else:
                print(state_summary)
            return "SHOW_STATE"
        
        elif cmd.startswith("top"):
            try:
                n = int(cmd.split()[-1])
                self._display_top_candidates(n)
            except:
                self._display_top_candidates(5)
            return "SHOW_TOP"
        
        elif cmd == "export":
            filepath = self.agent.export_state()
            msg = f"✅ State exported to: {filepath}"
            if self.console:
                self.console.print(msg, style="green")
            else:
                print(msg)
            return "EXPORT"
        
        elif cmd == "history":
            self._show_refinement_history()
            return "HISTORY"
        
        elif cmd == "round2":
            # Defer execution to main loop to avoid duplicate invocation.
            return "ROUND2"
        
        elif cmd == "round3":
            # Defer execution to main loop to avoid duplicate invocation.
            return "ROUND3"
        
        elif cmd in ["exit", "quit", "q"]:
            return "EXIT"
        
        return "UNKNOWN"
    
    def handle_search(self, query: str):
        """Handle SEARCH intent."""
        # Extract job description from query
        # Try to load file if mentioned
        job_match = re.search(r"from\s+(\S+\.txt)", query, re.IGNORECASE)
        
        if job_match:
            job_file = f"jobs/{job_match.group(1)}"
            result = read_file(job_file)
            if result.get("success"):
                job_text = result.get("content", "")
            else:
                print(f"❌ Could not load {job_file}")
                return
        else:
            # Use free-text job description
            job_text = query
        
        if not job_text.strip():
            print("❌ Please provide a job description. Example: 'Search for ML engineers with Python'")
            return
        
        # Load into agent and run search
        self.agent.load_job_description(job_text)
        
        if self.console:
            with self.console.status("[bold green]Searching..."):
                result = self.agent.run_initial_search()
        else:
            print("Searching...")
            result = self.agent.run_initial_search()
        
        # Display results
        if result["success"]:
            self._display_shortlist(result["shortlist"])
        else:
            print(f"❌ Error: {result['error']}")
    
    def handle_compare(self, query: str):
        """Handle COMPARE intent."""
        # Extract candidate names from free-form compare query (case-insensitive).
        body = re.sub(r"(?i)^\s*compare\s+", "", query).strip()
        parts = re.split(r"(?i)\s*(?:,|and|vs|versus)\s*", body)
        candidates = [p.strip(" .!?:;") for p in parts if p.strip()]
        
        if len(candidates) < 2:
            print("❌ Please specify at least 2 candidates to compare. Example: 'Compare Sneha and Arjun'")
            return
        
        result = self.agent.compare_top_candidates(candidates)
        
        if "error" in result:
            print(f"❌ {result['error']}")
            available = [c.get("candidate_name", "") for c in self.agent.state.shortlist[:5] if c.get("candidate_name")]
            if available:
                print("Available in current shortlist:", ", ".join(available))
            return
        
        # Display comparison table
        self._display_comparison(result.get("comparison", []))
    
    def handle_explain(self, query: str):
        """Handle EXPLAIN intent."""
        # Extract candidate name
        candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query)
        
        if not candidates:
            print("❌ Please specify a candidate. Example: 'Why did Sneha rank #1?'")
            return
        
        candidate = candidates[0]
        result = self.agent.explain_candidate(candidate)
        
        if "error" in result:
            print(f"❌ {result['error']}")
            return
        
        if self.console:
            self.console.print(result.get("explanation", ""), style="blue")
        else:
            print(result.get("explanation", ""))
    
    def handle_refine(self, query: str):
        """Handle REFINE intent."""
        refinements = {}

        # Extract experience requirement
        exp_match = re.search(r"(\d+)\s*\+\s*years", query, re.IGNORECASE)
        if exp_match:
            min_exp = float(exp_match.group(1))
            refinements["min_experience"] = min_exp

        # Extract skill terms from the query while removing command words/experience fragments.
        skill_text = re.sub(r"(?i)\b(filter|refine|show|only|candidates|with|also|by)\b", " ", query)
        skill_text = re.sub(r"(?i)\b\d+\s*\+\s*years?\b", " ", skill_text)
        skill_parts = re.split(r"(?i)\s*(?:,|and)\s*", skill_text)
        skills = []
        for part in skill_parts:
            cleaned = re.sub(r"[^a-zA-Z0-9\-\+\.# ]+", " ", part).strip()
            if not cleaned:
                continue
            # Ignore fragments that are clearly not skill-like.
            if cleaned.lower() in {"years", "year", "minimum", "experience"}:
                continue
            skills.append(cleaned)

        if skills:
            refinements["skill_filter"] = skills
        
        if not refinements:
            print("❌ Could not parse refinement. Example: 'Filter by Python, Docker, with 5+ years'")
            return
        
        result = self.agent.refine_requirements(refinements)
        
        if result.get("success"):
            self._display_shortlist(result.get("shortlist", []))
        else:
            print(f"❌ Error: {result}")
    
    def handle_questions(self, query: str):
        """Handle QUESTIONS intent."""
        query_clean = query.strip()
        # Case-insensitive extraction that supports lowercase and first-name inputs.
        body = re.sub(r"(?i)^\s*generate\s+interview\s+questions\s+for\s+", "", query_clean)
        body = re.sub(r"(?i)^\s*questions\s+for\s+", "", body)
        body = body.strip(" .!?:;")
        candidates = [body] if body else []
        
        if not candidates:
            print("❌ Please specify a candidate. Example: 'Generate interview questions for Sneha'")
            return
        
        candidate = candidates[0]
        result = self.agent.generate_interview_for_candidate(candidate)
        
        if "error" in result:
            print(f"❌ {result['error']}")
            return
        
        msg = f"\n📋 **Interview Questions for {result.get('candidate_name')}**\n\n"
        for i, q in enumerate(result.get("questions", []), 1):
            msg += f"{i}. {q}\n"
        
        if self.console:
            self.console.print(msg, style="green")
        else:
            print(msg)
    
    def handle_load_job(self, query: str):
        """Handle LOAD_JOB intent."""
        # Show available jobs
        jobs = list_files("jobs")
        
        if self.console:
            self.console.print("\n📋 Available Job Descriptions:\n", style="cyan")
        else:
            print("\n📋 Available Job Descriptions:\n")
        
        for i, job in enumerate(jobs[:5], 1):
            print(f"{i}. {job['name']}")
        
        if self.console:
            self.console.print(f"\nTotal: {len(jobs)} jobs\n", style="dim")
        else:
            print(f"\nTotal: {len(jobs)} jobs\n")
        
        # Prompt for choice
        try:
            choice = input("Select job #, or enter custom JD text: ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(jobs):
                job_file = jobs[int(choice) - 1]["name"]
                result = read_file(f"jobs/{job_file}")
                if result.get("success"):
                    job_text = result.get("content", "")
                    self.agent.load_job_description(job_text)
                    self.handle_search(job_text[:100])  # Run initial search
            else:
                self.agent.load_job_description(choice)
                self.handle_search(choice)
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _display_shortlist(self, shortlist: List[Dict]):
        """Display candidates in shortlist."""
        if not shortlist:
            print("⚠️ No candidates found.")
            return
        
        msg = f"\n🎯 **Top {len(shortlist)} Candidates:**\n"
        for i, cand in enumerate(shortlist, 1):
            name = cand.get("candidate_name", "Unknown")
            score = cand.get("match_score", 0)
            skills = ", ".join(cand.get("matched_skills", [])[:3])
            msg += f"{i}. {name} ({score}/100) — {skills}\n"
        
        if self.console:
            self.console.print(msg, style="green")
        else:
            print(msg)
    
    def _display_top_candidates(self, n: int):
        """Display top N candidates."""
        shortlist = self.agent.state.shortlist[:n]
        self._display_shortlist(shortlist)
    
    def _display_comparison(self, comparison: List[Dict]):
        """Display comparison table."""
        msg = "\n📊 **Comparison:**\n"
        
        for cand in comparison:
            msg += f"\n**{cand.get('candidate_name')}**\n"
            msg += f"  Skills: {', '.join(cand.get('matched_skills', []))}\n"
            msg += f"  Experience: {cand.get('experience_years', 0):.1f} years\n"
            msg += f"  Match Score: {cand.get('match_score', 0)}/100\n"
        
        if self.console:
            self.console.print(msg, style="cyan")
        else:
            print(msg)
    
    def _show_refinement_history(self):
        """Display refinement history."""
        if not self.agent.state.refinement_history:
            print("No refinements applied yet.")
            return
        
        msg = "\n📜 **Refinement History:**\n"
        for i, ref in enumerate(self.agent.state.refinement_history, 1):
            msg += f"\n{i}. {ref.description} @ {ref.timestamp}\n"
        
        if self.console:
            self.console.print(msg, style="yellow")
        else:
            print(msg)
    
    def run(self):
        """Main CLI loop."""
        self.running = True
        self.print_welcome()
        
        while self.running:
            try:
                # Prompt
                user_input = input("\n👤 you> ").strip()
                
                if not user_input:
                    continue
                
                # Classify and route
                intent = self.classify_intent(user_input)
                
                if intent == "EXIT":
                    self._safe_auto_export()
                    print("\n✅ Thank you! Bye!")
                    self.running = False
                    break
                
                elif intent == "HELP":
                    continue
                
                elif intent == "SEARCH":
                    self.handle_search(user_input)
                
                elif intent == "COMPARE":
                    self.handle_compare(user_input)
                
                elif intent == "EXPLAIN":
                    self.handle_explain(user_input)
                
                elif intent == "REFINE":
                    self.handle_refine(user_input)
                
                elif intent == "QUESTIONS":
                    self.handle_questions(user_input)
                
                elif intent == "LOAD_JOB":
                    self.handle_load_job(user_input)
                
                elif intent == "ROUND2":
                    result = self.agent.start_round2_analysis()
                    if "error" in result:
                        print(f"❌ {result['error']}")
                    else:
                        self._print_recent_assistant_messages(max_messages=3)
                
                elif intent == "ROUND3":
                    result = self.agent.start_round3_final()
                    if "error" in result:
                        print(f"❌ {result['error']}")
                    else:
                        self._print_recent_assistant_messages(max_messages=6)
                
                elif intent == "UNKNOWN":
                    print("❓ I didn't understand that. Type /help for available commands or try:")
                    print("  'Search for [job]'")
                    print("  'Compare [candidates]'")
                    print("  'Why did [candidate] rank [position]?'")
            
            except KeyboardInterrupt:
                self._safe_auto_export()
                print("\n\n✅ Session interrupted. Bye!")
                self.running = False
            
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Entry point."""
    cli = MatchingCLI()
    cli.run()


if __name__ == "__main__":
    main()

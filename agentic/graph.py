# agentic/graph.py
"""
A lightweight pipeline graph that mimics your original LangGraph agent flow
but works 100% locally and integrates with your existing agentic/*.py files.

Steps:
1. Parse JD
2. Parse resumes
3. Score resumes
4. Generate explanations
5. Produce a final sorted result
"""

from typing import Dict, Any, List
from .parsers import parse_jd, parse_resume
from .scoring import score_resume_against_jd
from .prompts import explanation_for_score


def build_agent_graph():
    """
    Return a callable object with an .invoke(state) method
    to mimic LangGraph's graph.invoke() behavior.
    """

    class GraphWrapper:
        def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:

            jd_raw = state.get("jd_text", "")
            resume_inputs = state.get("resume_paths", [])
            from_file = state.get("from_file", False)

            # Step 1: Parse JD
            jd = parse_jd(jd_raw, from_file=from_file)

            # Step 2: Parse all resumes
            parsed_resumes: List[Dict] = []
            for r in resume_inputs:
                parsed_resumes.append(parse_resume(r, from_file=from_file))

            # Step 3: Score resumes
            scored = []
            for idx, res in enumerate(parsed_resumes):
                scores = score_resume_against_jd(jd, res)
                scored.append({
                    "resume_id": idx,
                    "resume": res,
                    "scores": scores
                })

            # Step 4: Generate explanations
            for s in scored:
                explanation = explanation_for_score(jd, s["resume"], s["scores"])
                s["explanation"] = explanation

            # Step 5: Sort by final score

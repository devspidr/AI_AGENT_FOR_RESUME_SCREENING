
# agentic/agent.py
from typing import List, Dict, Any
import traceback

# Try to import the user's graph builder; if it's not usable we'll fallback.
try:
    from .graph import build_agent_graph
except Exception as e:
    build_agent_graph = None
    _GRAPH_IMPORT_ERROR = e
else:
    _GRAPH_IMPORT_ERROR = None

# Import the standard helpers used by the fallback pipeline
from .parsers import parse_jd, parse_resume
from .scoring import score_resume_against_jd
from .prompts import explanation_for_score


class SimpleAgent:
    def __init__(self):
        self.graph = None
        self._graph_error = None
        # Try to build the user's graph if available
        if build_agent_graph is not None:
            try:
                g = build_agent_graph()
                # Accept either an object with .invoke or a callable that returns a state dict
                if g is None:
                    raise RuntimeError("build_agent_graph() returned None")
                # Graph may be a factory that returns a class; ensure it has invoke
                if hasattr(g, "invoke") and callable(getattr(g, "invoke")):
                    self.graph = g
                elif callable(g):
                    # if build_agent_graph returned a plain function, wrap it
                    class _FuncWrapper:
                        def __init__(self, fn):
                            self._fn = fn
                        def invoke(self, state):
                            return self._fn(state)
                    self.graph = _FuncWrapper(g)
                else:
                    raise RuntimeError("build_agent_graph() returned object without 'invoke' method")
            except Exception as e:
                # record error and fall back
                self._graph_error = e
                self.graph = None
        else:
            # record the import-time error for debugging
            self._graph_error = _GRAPH_IMPORT_ERROR

        # If graph is still None, we will use an internal fallback pipeline
        if self.graph is None:
            # No exception thrown here — fallback pipeline implemented below
            pass

    def _fallback_pipeline(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        A simple deterministic pipeline that mirrors the expected graph steps:
        1) parse JD
        2) parse resumes
        3) score each resume
        4) generate explanation
        5) sort and return in state['full_results']
        """
        jd_raw = state.get("jd_text", "")
        resume_inputs = state.get("resume_paths", [])
        from_file = state.get("from_file", False)

        jd = parse_jd(jd_raw, from_file=from_file)

        parsed_resumes: List[Dict] = []
        for r in resume_inputs:
            parsed_resumes.append(parse_resume(r, from_file=from_file))

        scored = []
        for idx, res in enumerate(parsed_resumes):
            scores = score_resume_against_jd(jd, res)
            explanation = explanation_for_score(jd, res, scores)
            scored.append({
                "resume_id": idx,
                "resume": res,
                "scores": scores,
                "explanation": explanation
            })

        # Sort by final score
        scored.sort(key=lambda x: x["scores"].get("final_score", 0.0), reverse=True)
        state["full_results"] = scored
        return state

    def run(self, jd_text_or_path: str, resumes_text_or_paths: List[str], from_file: bool = False) -> List[Dict[str, Any]]:
        """
        Run the pipeline (graph or fallback) and return a normalized list of result dicts.
        """
        initial_state = {
            "jd_text": jd_text_or_path,
            "resume_paths": resumes_text_or_paths,
            "from_file": from_file
        }

        final_state = None
        # Prefer user's graph if valid
        if self.graph is not None:
            try:
                final_state = self.graph.invoke(initial_state)
            except Exception as e:
                # if graph invocation fails, capture error and fall back
                self._graph_error = e
                final_state = None

        if final_state is None:
            # use fallback deterministic pipeline
            final_state = self._fallback_pipeline(initial_state)

        # Normalize results into list of dicts expected by app UI
        candidates = final_state.get("full_results") or final_state.get("results") or []
        out: List[Dict[str, Any]] = []
        for c in candidates:
            # c expected shape (as produced by fallback): dict with keys resume_id, resume, scores, explanation
            resume_obj = c.get("resume") if isinstance(c, dict) else None
            scores = c.get("scores") if isinstance(c, dict) else {}
            text_preview = ""
            keywords = []
            raw_resume = ""
            name = None
            resume_id = c.get("resume_id", None) if isinstance(c, dict) else None

            if resume_obj:
                # resume_obj might be dict with 'text' or 'raw_text'
                text_preview = (resume_obj.get("text") or resume_obj.get("raw_text") or "")[:1000]
                raw_resume = resume_obj.get("raw_text", "") or resume_obj.get("text", "")
                keywords = resume_obj.get("keywords", []) or []
                name = (resume_obj.get("meta", {}) or {}).get("name") or resume_obj.get("name")
            else:
                # if candidate is an object (dataclass), attempt attribute access
                try:
                    text_preview = (getattr(c, "text_preview", "") or "")[:1000]
                except Exception:
                    text_preview = ""
                keywords = getattr(c, "keywords", []) or []
                name = getattr(c, "name", None)
                raw_resume = getattr(c, "raw_resume", "")

            score_val = 0.0
            if isinstance(scores, dict):
                score_val = scores.get("final_score") or scores.get("composite_score") or scores.get("composite") or 0.0
            else:
                try:
                    score_val = float(scores)
                except Exception:
                    score_val = 0.0

            out_item = {
                "resume_id": resume_id,
                "name": name,
                "score": score_val,
                "scores": scores,
                "explanation": c.get("explanation") if isinstance(c, dict) else getattr(c, "rationale", None),
                "text_preview": text_preview,
                "keywords": keywords,
                "raw_resume": raw_resume
            }
            out.append(out_item)

        # ensure sorted by score (defensive)
        out.sort(key=lambda x: x.get("score", 0.0) or 0.0, reverse=True)
        return out

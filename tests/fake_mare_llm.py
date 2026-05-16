"""Deterministic fake MARE LLM client for role/action workflow tests."""

from __future__ import annotations

import json


class ScriptedMareLLMClient:
    """Return valid JSON outputs for each MARE paper action."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        del temperature, max_tokens
        payload = json.loads(messages[-1]["content"])
        action = str(payload.get("action", "")).strip()
        fragments = [str(item) for item in payload.get("requirement_fragments", []) if str(item).strip()]
        workspace = payload.get("workspace_snapshot", {})

        if action == "SpeakUserStories":
            items = [
                f"As a stakeholder, I need {fragment[:120]}" for fragment in fragments[:4]
            ] or ["As a stakeholder, I need reliable system behavior."]
        elif action == "ProposeQuestion":
            stories = [str(item) for item in workspace.get("user_stories", [])]
            items = [
                f"What acceptance criterion validates story {index}?"
                for index in range(1, min(3, len(stories)) + 1)
            ] or ["What acceptance criterion defines completion?"]
        elif action == "AnswerQuestion":
            questions = [str(item) for item in workspace.get("questions", [])]
            base = fragments[0] if fragments else "requirements remain testable"
            items = [
                f"Answer {index}: The system shall {base[:100]}"
                for index, _ in enumerate(questions, start=1)
            ] or [f"Answer 1: The system shall {base[:100]}"]
        elif action == "WriteReqDraft":
            stories = [str(item) for item in workspace.get("user_stories", [])]
            answers = [str(item) for item in workspace.get("answers", [])]
            items = [
                f"Requirement {index}: The system shall {story[:120]}"
                for index, story in enumerate(stories[:3], start=1)
            ]
            items.extend(answers[:2])
            if not items:
                items = ["Requirement 1: The system shall satisfy stakeholder needs."]
        elif action == "ExtractEntity":
            drafts = " ".join(str(item) for item in workspace.get("requirement_draft", []))
            tokens = [token.strip(".,:;()[]") for token in drafts.split()]
            items = []
            seen: set[str] = set()
            for token in tokens:
                lower = token.lower()
                if len(lower) < 5 or lower in seen:
                    continue
                seen.add(lower)
                items.append(lower)
                if len(items) >= 6:
                    break
            if not items:
                items = ["requirement", "verification", "stakeholder"]
        elif action == "ExtractRelation":
            entities = [str(item) for item in workspace.get("entities", [])]
            items = [
                f"{entities[index]}|refines|{entities[index + 1]}"
                for index in range(max(0, len(entities) - 1))
            ]
            if not items and entities:
                items = [f"{entities[0]}|supports|{entities[0]}"]
            if not items:
                items = ["requirement|supports|verification"]
        elif action == "CheckRequirement":
            items = [
                "Requirements are phrased with shall-statements.",
                "Acceptance criteria should remain measurable and testable.",
            ]
        elif action == "WriteSRS":
            items = [
                "SRS-1 Scope and Stakeholders",
                "SRS-2 Functional and Quality Requirements",
                "SRS-3 Verification and Acceptance",
            ]
        elif action == "WriteCheckReport":
            findings = [str(item) for item in workspace.get("findings", [])]
            items = [
                f"Check report item {index}: {finding}"
                for index, finding in enumerate(findings, start=1)
            ] or ["Check report item 1: Requirements remain verifiable."]
        else:
            items = ["Workflow output unavailable."]

        response = {
            "items": items,
            "summary": f"{action} generated {len(items)} items",
        }
        return json.dumps(response)


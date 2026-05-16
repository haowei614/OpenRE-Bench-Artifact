"""Deterministic fake iReDev LLM client for 6-agent/17-action workflow tests."""

from __future__ import annotations

import json


class ScriptedIredevLLMClient:
    """Return valid JSON outputs for each iReDev paper action."""

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

        if action == "DialogueWithEndUser":
            items = [
                f"What are the key goals for {fragment[:100]}?" for fragment in fragments[:4]
            ] or ["What are the key user goals for this system?"]
        elif action == "RespondEndUser":
            items = [
                f"User goal: {fragment[:120]}" for fragment in fragments[:4]
            ] or ["User goal: reliable system behavior."]
        elif action == "RaiseQuestionEndUser":
            items = ["Could you clarify the priority between performance and accuracy?"]
        elif action == "ConfirmOrRefineEndUser":
            responses = [str(item) for item in workspace.get("enduser_responses", [])]
            items = [
                f"Confirmed: {resp[:100]}" for resp in responses[:3]
            ] or ["Confirmed: requirements are clear."]
        elif action == "WriteInterviewRecords":
            responses = [str(item) for item in workspace.get("enduser_responses", [])]
            items = [
                f"Record {i + 1}: {resp[:120]}" for i, resp in enumerate(responses[:4])
            ] or ["Record 1: User needs documented."]
        elif action == "WriteUserRequirementsList":
            items = [
                f"UR-{i + 1}: The system shall {fragment[:120]}"
                for i, fragment in enumerate(fragments[:6])
            ] or ["UR-1: The system shall satisfy user needs."]
        elif action == "DialogueWithDeployer":
            items = [
                f"What infrastructure constraints apply to {fragment[:80]}?"
                for fragment in fragments[:3]
            ] or ["What deployment environment does the system target?"]
        elif action == "RespondDeployer":
            items = [
                f"Deploy constraint: {fragment[:120]}" for fragment in fragments[:3]
            ] or ["Deploy constraint: standard web server required."]
        elif action == "RaiseQuestionDeployer":
            items = ["What are the expected peak concurrent connections?"]
        elif action == "ConfirmOrRefineDeployer":
            responses = [str(item) for item in workspace.get("deployer_responses", [])]
            items = [
                f"Confirmed: {resp[:100]}" for resp in responses[:3]
            ] or ["Confirmed: deployment constraints are clear."]
        elif action == "WriteOperatingEnvironmentList":
            items = [
                "ENV-1: Web server with HTTPS",
                "ENV-2: Relational database with backups",
                "ENV-3: Authentication framework",
            ]
        elif action == "WriteSystemRequirementsList":
            user_reqs = [str(item) for item in workspace.get("user_requirements", [])]
            items = [
                f"SR-{i + 1}: The system shall {req[:120]}"
                for i, req in enumerate(user_reqs[:6])
            ] or ["SR-1: The system shall meet all stakeholder requirements."]
        elif action == "SelectRequirementModel":
            items = ["Selected model: Use Case Diagram (UML)"]
        elif action == "BuildRequirementModel":
            sys_reqs = [str(item) for item in workspace.get("system_requirements", [])]
            tokens = [t.strip(",.;:()[]") for t in " ".join(sys_reqs).split()]
            entities = [t.lower() for t in tokens if len(t) >= 5][:6]
            items = [f"Model element: {e}" for e in entities] or ["Model element: system"]
        elif action == "WriteSRS":
            items = [
                "SRS-1 Purpose and Scope",
                "SRS-2 Functional Requirements",
                "SRS-3 Non-Functional Requirements",
            ]
        elif action == "Evaluate":
            items = [
                "Finding: Requirements are phrased with shall-statements.",
                "Finding: All requirements should include verifiable acceptance criteria.",
            ]
        elif action == "ConfirmClosure":
            findings = [str(item) for item in workspace.get("review_findings", [])]
            items = [
                f"Closure: {finding[:100]} — resolved"
                for finding in findings
            ] or ["Closure: All findings resolved."]
        else:
            items = ["Workflow output unavailable."]

        response = {
            "items": items,
            "summary": f"{action} generated {len(items)} items",
        }
        return json.dumps(response)

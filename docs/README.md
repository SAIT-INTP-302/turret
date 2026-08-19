# Turret Documentation

This is the technical documentation set for the turret project — the *why*
behind the system, alongside the root [`README.md`](../README.md), which
stays the single source of truth for exact setup/wiring/bring-up commands.

## Contents

| Doc | What's in it |
|---|---|
| [Overview](overview.md) | What the turret is, what it can do, and where the scope boundaries are |
| [Architecture](architecture.md) | Module-by-module walkthrough, the control loop, the config system, the safety model, the test strategy |
| [Stack & Rationale](stack.md) | Every major dependency and design choice, and *why* it was picked over the alternatives |
| [Deployment & Operations](deployment-and-ops.md) | How it actually runs day to day: systemd, the pre-run self-test, live tuning |
| [Show Notes](show-notes.md) | Cue-card talking points for presenting/demoing — not a script |

Start with **Overview** if you're new to the project, **Architecture** if
you're about to change code, **Stack & Rationale** if you're wondering "why
does this use X instead of Y," and **Show Notes** if you're about to demo
it in five minutes.

## Diagrams

All diagrams are authored as [PlantUML](https://plantuml.com/) source in
[`diagrams/`](diagrams/) and rendered to `.svg` (committed alongside the
source, so nothing needs to render on the fly to view them on GitHub).

| Diagram | Source | Rendered |
|---|---|---|
| System context | [`diagrams/system-context.puml`](diagrams/system-context.puml) | [`diagrams/system-context.svg`](diagrams/system-context.svg) |
| Control-loop sequence | [`diagrams/control-loop-sequence.puml`](diagrams/control-loop-sequence.puml) | [`diagrams/control-loop-sequence.svg`](diagrams/control-loop-sequence.svg) |
| Deployment | [`diagrams/deployment.puml`](diagrams/deployment.puml) | [`diagrams/deployment.svg`](diagrams/deployment.svg) |
| Detector backends | [`diagrams/detector-backends.puml`](diagrams/detector-backends.puml) | [`diagrams/detector-backends.svg`](diagrams/detector-backends.svg) |

To re-render after editing a `.puml` file (needs Java + Graphviz; the repo
doesn't otherwise depend on either, so this is a `nix-shell` one-liner
rather than an added dev dependency):

```sh
nix-shell -p plantuml --run "plantuml -tsvg docs/diagrams/*.puml"
```

## See also

- [Root README](../README.md) — setup, wiring, CLI flags, bring-up order

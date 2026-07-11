# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Language

- Converse with the user in **Japanese**.
- Write all code, comments, commit messages, and documentation in **English**.

## Model Cost & Delegation Policy

The primary models (Fable, Opus) are very expensive. To control cost:

- The main agent should limit itself to **planning, orchestration, and high-difficulty implementation only**.
- Delegate implementation work to **subagents** as much as possible (e.g., routine coding, file edits, searches, test runs, and other well-specified tasks).
- Give subagents clear, self-contained instructions so they can complete tasks without repeated round-trips.

---
title: Kaizen OS — Agentic Kernel
emoji: 🖥️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: true
short_description: LLM agent that autonomously manages a simulated OS in real-time
---

# Kaizen OS — The Agentic Kernel

An LLM agent that monitors real system telemetry, detects chaos events, and takes corrective OS management actions — all in real-time with a live dashboard.

**Model**: Qwen2.5-3B-Instruct (upgradeable via `KAIZEN_MODEL_NAME` secret)

## Usage

1. Click the app to open the dashboard
2. Click **Start Episode** in the top bar
3. Watch the agent reason through a chaos event and resolve it

## Space Secrets

| Secret | Value |
|--------|-------|
| `KAIZEN_MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` |
| `KAIZEN_4BIT` | `true` |

## Links

- [GitHub Repository](https://github.com/YOUR_USERNAME/kaizen-os)
- [Hackathon Submission](https://YOUR_SUBMISSION_URL)

@echo off
title ECAS Fable Agent Workspace
cd /d "C:\Code\ECAS\ecas"

echo ====================================================
echo 🤖 Launching Free Local Fable Autonomous Agent...
echo ====================================================

:: --auto-accept-architect applies edits instantly
:: --yes-always auto-approves adding files/docs to the chat loop without prompting you
aider --model ollama/parable/granite4.1-fable:8b --openai-api-base http://127.0.0 --auto-accept-architect --yes-always

pause

---
title: Misinformation Resilience Framework
emoji: 🛡️
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
---

# Misinformation Resilience Framework (MRF)

A fake news detection system built for Law Enforcement Agencies.

## Features
- Weighted ensemble: BERT + Passive Aggressive + Random Forest + Logistic Regression
- Google Fact Check Tools API integration
- Real-time credibility scoring and risk assessment
- FastAPI backend with interactive dashboard

## Usage
- Visit `/app` for the dashboard
- POST to `/predict` with `{"text": "your news article here"}`
- GET `/health` to check system status

## Built by
Rohan | MSc Cybersecurity | NFSU